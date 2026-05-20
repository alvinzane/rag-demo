from __future__ import annotations

import json
import shutil
import socket
import statistics
import subprocess
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import psutil

DEFAULT_MILVUS_HOST = "127.0.0.1"
DEFAULT_MILVUS_PORT = 19530
DEFAULT_ATTU_HOST = "127.0.0.1"
DEFAULT_ATTU_PORT = 8000
DEFAULT_T2_COLLECTION = "rag_demo_t2_vectors"
DEFAULT_DIM = 768
DEFAULT_COUNT = 1_000_000
DEFAULT_BATCH_SIZE = 5_000
DEFAULT_SEED = 42
DEFAULT_TOP_K = 10
IndexType = Literal["IVF_FLAT", "HNSW"]


@dataclass(frozen=True)
class RuntimeCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class DatasetMetadata:
    count: int
    dim: int
    seed: int
    query_count: int
    materialized: bool
    output_dir: str


@dataclass(frozen=True)
class IndexBuildResult:
    collection: str
    index_type: str
    metric_type: str
    rows: int
    elapsed_sec: float
    params: dict[str, int]


@dataclass(frozen=True)
class BenchResult:
    collection: str
    index_type: str
    runs: int
    top_k: int
    qps: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    rows: int
    memory_rss_mb: float
    search_params: dict[str, int]


def check_port(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def docker_compose_command() -> str | None:
    docker = shutil.which("docker")
    if docker:
        try:
            subprocess.run(
                [docker, "compose", "version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return "docker compose"
        except (OSError, subprocess.CalledProcessError):
            pass
    if shutil.which("docker-compose"):
        return "docker-compose"
    return None


def runtime_checks(
    milvus_host: str = DEFAULT_MILVUS_HOST,
    milvus_port: int = DEFAULT_MILVUS_PORT,
    attu_host: str = DEFAULT_ATTU_HOST,
    attu_port: int = DEFAULT_ATTU_PORT,
) -> list[RuntimeCheck]:
    compose = docker_compose_command()
    docker = shutil.which("docker")
    checks = [
        RuntimeCheck("Docker", docker is not None, docker or "missing"),
        RuntimeCheck(
            "Compose",
            compose is not None,
            compose or "missing docker compose/docker-compose",
        ),
        RuntimeCheck(
            "Milvus port",
            check_port(milvus_host, milvus_port),
            f"{milvus_host}:{milvus_port}",
        ),
        RuntimeCheck("Attu port", check_port(attu_host, attu_port), f"{attu_host}:{attu_port}"),
        RuntimeCheck(
            "bpftrace",
            shutil.which("bpftrace") is not None,
            shutil.which("bpftrace") or "missing; install bpftrace and run with sudo",
        ),
    ]
    return checks


def generate_vectors(offset: int, count: int, dim: int, seed: int = DEFAULT_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed + offset)
    vectors = rng.standard_normal((count, dim), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def generate_queries(query_count: int, dim: int, seed: int = DEFAULT_SEED) -> np.ndarray:
    return generate_vectors(offset=10_000_000, count=query_count, dim=dim, seed=seed)


def prepare_dataset(
    output_dir: Path,
    count: int = DEFAULT_COUNT,
    dim: int = DEFAULT_DIM,
    seed: int = DEFAULT_SEED,
    query_count: int = 1_000,
    materialize: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> DatasetMetadata:
    validate_positive("count", count)
    validate_positive("dim", dim)
    validate_positive("query_count", query_count)
    validate_positive("batch_size", batch_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = generate_queries(query_count=query_count, dim=dim, seed=seed)
    np.save(output_dir / "queries.npy", queries)

    if materialize:
        vectors_path = output_dir / "vectors.npy"
        vectors = np.lib.format.open_memmap(
            vectors_path,
            mode="w+",
            dtype=np.float32,
            shape=(count, dim),
        )
        for start in range(0, count, batch_size):
            size = min(batch_size, count - start)
            vectors[start : start + size] = generate_vectors(start, size, dim, seed)
        vectors.flush()

    metadata = DatasetMetadata(
        count=count,
        dim=dim,
        seed=seed,
        query_count=query_count,
        materialized=materialize,
        output_dir=str(output_dir),
    )
    (output_dir / "dataset_meta.json").write_text(
        json.dumps(asdict(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def connect_milvus(host: str = DEFAULT_MILVUS_HOST, port: int = DEFAULT_MILVUS_PORT) -> None:
    from pymilvus import connections

    connections.connect(alias="default", host=host, port=str(port))


def create_or_replace_collection(
    collection_name: str = DEFAULT_T2_COLLECTION,
    dim: int = DEFAULT_DIM,
    reset: bool = False,
) -> Any:
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

    if utility.has_collection(collection_name):
        if not reset:
            return Collection(collection_name)
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="group_id", dtype=DataType.INT64),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description="RAG demo W4-T2 synthetic vectors")
    return Collection(name=collection_name, schema=schema)


def load_vectors(
    collection_name: str = DEFAULT_T2_COLLECTION,
    count: int = DEFAULT_COUNT,
    dim: int = DEFAULT_DIM,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    reset: bool = False,
) -> int:
    validate_positive("count", count)
    validate_positive("dim", dim)
    validate_positive("batch_size", batch_size)
    collection = create_or_replace_collection(collection_name, dim=dim, reset=reset)

    inserted = 0
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        vectors = generate_vectors(start, size, dim, seed)
        ids = list(range(start, start + size))
        group_ids = [item % 100 for item in ids]
        collection.insert([ids, group_ids, vectors.tolist()])
        inserted += size
    collection.flush()
    return inserted


def create_index(
    collection_name: str = DEFAULT_T2_COLLECTION,
    index_type: IndexType = "IVF_FLAT",
    metric_type: str = "COSINE",
    nlist: int = 1024,
    hnsw_m: int = 16,
    ef_construction: int = 200,
) -> IndexBuildResult:
    from pymilvus import Collection

    collection = Collection(collection_name)
    rows = int(collection.num_entities)
    params = index_params(index_type, metric_type, nlist, hnsw_m, ef_construction)
    with suppress(Exception):
        collection.release()
    with suppress(Exception):
        collection.drop_index()

    started = time.perf_counter()
    collection.create_index(field_name="vector", index_params=params)
    collection.load()
    elapsed = time.perf_counter() - started
    return IndexBuildResult(
        collection=collection_name,
        index_type=index_type,
        metric_type=metric_type,
        rows=rows,
        elapsed_sec=elapsed,
        params=params["params"],
    )


def run_benchmark(
    collection_name: str = DEFAULT_T2_COLLECTION,
    index_type: IndexType = "IVF_FLAT",
    runs: int = 1_000,
    dim: int = DEFAULT_DIM,
    top_k: int = DEFAULT_TOP_K,
    seed: int = DEFAULT_SEED,
    nprobe: int = 16,
    ef: int = 64,
    filter_expr: str | None = None,
) -> BenchResult:
    from pymilvus import Collection

    validate_positive("runs", runs)
    validate_positive("dim", dim)
    validate_positive("top_k", top_k)
    collection = Collection(collection_name)
    collection.load()
    rows = int(collection.num_entities)
    queries = generate_queries(query_count=runs, dim=dim, seed=seed)
    params = search_params(index_type, nprobe=nprobe, ef=ef)

    latencies_ms: list[float] = []
    started = time.perf_counter()
    for query in queries:
        query_started = time.perf_counter()
        collection.search(
            data=[query.tolist()],
            anns_field="vector",
            param=params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["group_id"],
        )
        latencies_ms.append((time.perf_counter() - query_started) * 1000)
    elapsed = time.perf_counter() - started

    return BenchResult(
        collection=collection_name,
        index_type=index_type,
        runs=runs,
        top_k=top_k,
        qps=runs / elapsed if elapsed else 0.0,
        avg_ms=statistics.fmean(latencies_ms),
        p50_ms=percentile(latencies_ms, 50),
        p95_ms=percentile(latencies_ms, 95),
        p99_ms=percentile(latencies_ms, 99),
        min_ms=min(latencies_ms),
        max_ms=max(latencies_ms),
        rows=rows,
        memory_rss_mb=psutil.Process().memory_info().rss / 1024 / 1024,
        search_params=params["params"],
    )


def sweep_benchmark(
    collection_name: str,
    index_type: IndexType,
    values: list[int],
    runs: int,
    dim: int,
    top_k: int,
    seed: int,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    for value in values:
        if index_type == "IVF_FLAT":
            results.append(
                run_benchmark(
                    collection_name=collection_name,
                    index_type=index_type,
                    runs=runs,
                    dim=dim,
                    top_k=top_k,
                    seed=seed,
                    nprobe=value,
                )
            )
        else:
            results.append(
                run_benchmark(
                    collection_name=collection_name,
                    index_type=index_type,
                    runs=runs,
                    dim=dim,
                    top_k=top_k,
                    seed=seed,
                    ef=value,
                )
            )
    return results


def write_report(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(item) for item in payload] if isinstance(payload, list) else asdict(payload)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def index_params(
    index_type: IndexType,
    metric_type: str,
    nlist: int,
    hnsw_m: int,
    ef_construction: int,
) -> dict[str, Any]:
    if index_type == "IVF_FLAT":
        return {
            "index_type": "IVF_FLAT",
            "metric_type": metric_type,
            "params": {"nlist": nlist},
        }
    if index_type == "HNSW":
        return {
            "index_type": "HNSW",
            "metric_type": metric_type,
            "params": {"M": hnsw_m, "efConstruction": ef_construction},
        }
    raise ValueError(f"Unsupported index type: {index_type}")


def search_params(index_type: IndexType, nprobe: int, ef: int) -> dict[str, Any]:
    if index_type == "IVF_FLAT":
        return {"metric_type": "COSINE", "params": {"nprobe": nprobe}}
    if index_type == "HNSW":
        return {"metric_type": "COSINE", "params": {"ef": ef}}
    raise ValueError(f"Unsupported index type: {index_type}")


def percentile(values: list[float], percent: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (percent / 100)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
