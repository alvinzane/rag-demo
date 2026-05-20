from __future__ import annotations

import json
import re
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import psutil
import requests

from rag_demo2.milvus_bench import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COUNT,
    DEFAULT_DIM,
    DEFAULT_SEED,
    DEFAULT_TOP_K,
    RuntimeCheck,
    check_port,
    docker_compose_command,
    generate_queries,
    generate_vectors,
    percentile,
    validate_positive,
)

DEFAULT_QDRANT_HOST = "127.0.0.1"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_WEAVIATE_HOST = "127.0.0.1"
DEFAULT_WEAVIATE_PORT = 8081
DEFAULT_T3_COLLECTION = "rag_demo_t3_vectors"
DEFAULT_WEAVIATE_CLASS = "RagDemoT3Vector"
DEFAULT_BUCKET_MOD = 100
Backend = Literal["qdrant", "weaviate"]


@dataclass(frozen=True)
class LoadResult:
    backend: str
    collection: str
    rows: int
    dim: int
    batch_size: int
    elapsed_sec: float
    memory_mb: float


@dataclass(frozen=True)
class PocBenchResult:
    backend: str
    collection: str
    rows: int
    runs: int
    top_k: int
    filtered: bool
    qps: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    memory_mb: float
    api: str


@dataclass(frozen=True)
class ExperienceNote:
    backend: str
    engine: str
    query_api: str
    filter_api: str
    operational_note: str


def runtime_checks(
    qdrant_host: str = DEFAULT_QDRANT_HOST,
    qdrant_port: int = DEFAULT_QDRANT_PORT,
    weaviate_host: str = DEFAULT_WEAVIATE_HOST,
    weaviate_port: int = DEFAULT_WEAVIATE_PORT,
) -> list[RuntimeCheck]:
    docker = _which_docker()
    compose = docker_compose_command()
    return [
        RuntimeCheck("Docker", docker is not None, docker or "missing"),
        RuntimeCheck(
            "Compose",
            compose is not None,
            compose or "missing docker compose/docker-compose",
        ),
        RuntimeCheck(
            "Qdrant HTTP",
            check_port(qdrant_host, qdrant_port),
            qdrant_url(qdrant_host, qdrant_port),
        ),
        RuntimeCheck(
            "Weaviate port",
            check_port(weaviate_host, weaviate_port),
            weaviate_url(weaviate_host, weaviate_port),
        ),
        RuntimeCheck(
            "Weaviate meta",
            _weaviate_meta_ready(weaviate_host, weaviate_port),
            f"{weaviate_url(weaviate_host, weaviate_port)}/v1/meta",
        ),
        RuntimeCheck(
            "Weaviate GraphQL",
            _graphql_ready(weaviate_host, weaviate_port),
            f"{weaviate_url(weaviate_host, weaviate_port)}/v1/graphql",
        ),
    ]


def qdrant_url(host: str = DEFAULT_QDRANT_HOST, port: int = DEFAULT_QDRANT_PORT) -> str:
    return f"http://{host}:{port}"


def weaviate_url(host: str = DEFAULT_WEAVIATE_HOST, port: int = DEFAULT_WEAVIATE_PORT) -> str:
    return f"http://{host}:{port}"


def load_backend(
    backend: Backend,
    count: int = DEFAULT_COUNT,
    dim: int = DEFAULT_DIM,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    reset: bool = False,
    collection: str = DEFAULT_T3_COLLECTION,
    qdrant_host: str = DEFAULT_QDRANT_HOST,
    qdrant_port: int = DEFAULT_QDRANT_PORT,
    weaviate_host: str = DEFAULT_WEAVIATE_HOST,
    weaviate_port: int = DEFAULT_WEAVIATE_PORT,
) -> LoadResult:
    validate_positive("count", count)
    validate_positive("dim", dim)
    validate_positive("batch_size", batch_size)

    started = time.perf_counter()
    if backend == "qdrant":
        base_url = qdrant_url(qdrant_host, qdrant_port)
        _prepare_qdrant_collection(base_url, collection, dim, reset)
        rows = _load_qdrant(base_url, collection, count, dim, batch_size, seed)
    elif backend == "weaviate":
        base_url = weaviate_url(weaviate_host, weaviate_port)
        _prepare_weaviate_class(base_url, collection, dim, reset)
        rows = _load_weaviate(base_url, collection, count, dim, batch_size, seed)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return LoadResult(
        backend=backend,
        collection=collection,
        rows=rows,
        dim=dim,
        batch_size=batch_size,
        elapsed_sec=time.perf_counter() - started,
        memory_mb=_container_memory_mb(container_name(backend)),
    )


def benchmark_backend(
    backend: Backend,
    runs: int = 1_000,
    dim: int = DEFAULT_DIM,
    top_k: int = DEFAULT_TOP_K,
    seed: int = DEFAULT_SEED,
    filtered: bool = True,
    collection: str = DEFAULT_T3_COLLECTION,
    qdrant_host: str = DEFAULT_QDRANT_HOST,
    qdrant_port: int = DEFAULT_QDRANT_PORT,
    weaviate_host: str = DEFAULT_WEAVIATE_HOST,
    weaviate_port: int = DEFAULT_WEAVIATE_PORT,
) -> PocBenchResult:
    validate_positive("runs", runs)
    validate_positive("dim", dim)
    validate_positive("top_k", top_k)

    queries = generate_queries(query_count=runs, dim=dim, seed=seed)
    latencies_ms: list[float] = []
    started = time.perf_counter()
    rows = 0
    if backend == "qdrant":
        base_url = qdrant_url(qdrant_host, qdrant_port)
        rows = _qdrant_count(base_url, collection)
        for index, query in enumerate(queries):
            bucket = _bucket(index) if filtered else None
            query_started = time.perf_counter()
            _search_qdrant(base_url, collection, query.tolist(), top_k, bucket)
            latencies_ms.append((time.perf_counter() - query_started) * 1000)
        api = "REST /collections/{collection}/points/search"
    elif backend == "weaviate":
        base_url = weaviate_url(weaviate_host, weaviate_port)
        rows = _weaviate_count(base_url, collection)
        for index, query in enumerate(queries):
            bucket = _bucket(index) if filtered else None
            query_started = time.perf_counter()
            _search_weaviate(base_url, collection, query.tolist(), top_k, bucket)
            latencies_ms.append((time.perf_counter() - query_started) * 1000)
        api = "GraphQL nearVector + where"
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    elapsed = time.perf_counter() - started

    return PocBenchResult(
        backend=backend,
        collection=collection,
        rows=rows,
        runs=runs,
        top_k=top_k,
        filtered=filtered,
        qps=runs / elapsed if elapsed else 0.0,
        avg_ms=statistics.fmean(latencies_ms),
        p50_ms=percentile(latencies_ms, 50),
        p95_ms=percentile(latencies_ms, 95),
        p99_ms=percentile(latencies_ms, 99),
        min_ms=min(latencies_ms),
        max_ms=max(latencies_ms),
        memory_mb=_container_memory_mb(container_name(backend)),
        api=api,
    )


def experience_notes() -> list[ExperienceNote]:
    return [
        ExperienceNote(
            backend="qdrant",
            engine="Rust vector search engine",
            query_api="HTTP search API with JSON request bodies",
            filter_api="Payload filter uses must/should/must_not clauses",
            operational_note=(
                "Single binary style service, easy local startup and low client overhead."
            ),
        ),
        ExperienceNote(
            backend="weaviate",
            engine="Go service with HNSW index and module system",
            query_api="GraphQL Get query with nearVector",
            filter_api="GraphQL where filter over typed properties",
            operational_note=(
                "Schema-first data model; GraphQL is expressive but query construction is noisier."
            ),
        ),
    ]


def write_report(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(item) for item in payload] if isinstance(payload, list) else asdict(payload)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def container_name(backend: Backend) -> str:
    return f"rag-demo2-{backend}"


def weaviate_graphql_query(
    class_name: str,
    vector: list[float],
    top_k: int,
    bucket: str | None = None,
) -> str:
    vector_literal = "[" + ",".join(f"{value:.8f}" for value in vector) + "]"
    where_clause = ""
    if bucket:
        safe_bucket = re.sub(r"[^A-Za-z0-9_.:-]", "", bucket)
        where_clause = (
            f', where: {{path: ["bucket"], operator: Equal, valueText: "{safe_bucket}"}}'
        )
    return (
        "{ Get { "
        f"{class_name}(nearVector: {{vector: {vector_literal}}}, limit: {top_k}{where_clause}) "
        "{ doc_id bucket source _additional { distance certainty } } "
        "} }"
    )


def qdrant_search_payload(
    vector: list[float],
    top_k: int,
    bucket: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "vector": vector,
        "limit": top_k,
        "with_payload": True,
        "with_vector": False,
    }
    if bucket:
        payload["filter"] = {"must": [{"key": "bucket", "match": {"value": bucket}}]}
    return payload


def _prepare_qdrant_collection(base_url: str, collection: str, dim: int, reset: bool) -> None:
    session = requests.Session()
    if reset:
        session.delete(f"{base_url}/collections/{collection}", timeout=30)
    response = session.get(f"{base_url}/collections/{collection}", timeout=30)
    if response.status_code == 200:
        return
    _raise_unless(response, {404})
    body = {
        "vectors": {"size": dim, "distance": "Cosine"},
        "optimizers_config": {"default_segment_number": 4},
        "hnsw_config": {"m": 16, "ef_construct": 200},
    }
    _request(session, "PUT", f"{base_url}/collections/{collection}", json=body, timeout=120)


def _load_qdrant(
    base_url: str,
    collection: str,
    count: int,
    dim: int,
    batch_size: int,
    seed: int,
) -> int:
    session = requests.Session()
    inserted = 0
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        vectors = generate_vectors(start, size, dim, seed)
        points = [
            {
                "id": item_id,
                "vector": vectors[offset].tolist(),
                "payload": _payload(item_id),
            }
            for offset, item_id in enumerate(range(start, start + size))
        ]
        _request(
            session,
            "PUT",
            f"{base_url}/collections/{collection}/points",
            params={"wait": "true"},
            json={"points": points},
            timeout=300,
        )
        inserted += size
    return inserted


def _search_qdrant(
    base_url: str,
    collection: str,
    vector: list[float],
    top_k: int,
    bucket: str | None,
) -> dict[str, Any]:
    response = _request(
        requests.Session(),
        "POST",
        f"{base_url}/collections/{collection}/points/search",
        json=qdrant_search_payload(vector, top_k, bucket),
        timeout=60,
    )
    return response.json()


def _qdrant_count(base_url: str, collection: str) -> int:
    response = _request(
        requests.Session(),
        "POST",
        f"{base_url}/collections/{collection}/points/count",
        json={"exact": True},
        timeout=60,
    )
    return int(response.json()["result"]["count"])


def _prepare_weaviate_class(base_url: str, class_name: str, dim: int, reset: bool) -> None:
    session = requests.Session()
    if reset:
        session.delete(f"{base_url}/v1/schema/{class_name}", timeout=60)
    response = session.get(f"{base_url}/v1/schema/{class_name}", timeout=30)
    if response.status_code == 200:
        return
    _raise_unless(response, {404})
    schema = {
        "class": class_name,
        "description": "RAG demo W4-T3 vectors with payload filters",
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {
            "distance": "cosine",
            "efConstruction": 200,
            "maxConnections": 16,
        },
        "properties": [
            {"name": "doc_id", "dataType": ["int"]},
            {"name": "bucket", "dataType": ["text"]},
            {"name": "lang", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]},
        ],
    }
    _request(session, "POST", f"{base_url}/v1/schema", json=schema, timeout=120)


def _load_weaviate(
    base_url: str,
    class_name: str,
    count: int,
    dim: int,
    batch_size: int,
    seed: int,
) -> int:
    session = requests.Session()
    inserted = 0
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        vectors = generate_vectors(start, size, dim, seed)
        objects = [
            {
                "class": class_name,
                "id": _uuid_for(item_id),
                "properties": _payload(item_id),
                "vector": vectors[offset].tolist(),
            }
            for offset, item_id in enumerate(range(start, start + size))
        ]
        _request(
            session,
            "POST",
            f"{base_url}/v1/batch/objects",
            json={"objects": objects},
            timeout=300,
        )
        inserted += size
    return inserted


def _search_weaviate(
    base_url: str,
    class_name: str,
    vector: list[float],
    top_k: int,
    bucket: str | None,
) -> dict[str, Any]:
    response = _request(
        requests.Session(),
        "POST",
        f"{base_url}/v1/graphql",
        json={"query": weaviate_graphql_query(class_name, vector, top_k, bucket)},
        timeout=60,
    )
    return response.json()


def _weaviate_count(base_url: str, class_name: str) -> int:
    query = f"{{ Aggregate {{ {class_name} {{ meta {{ count }} }} }} }}"
    response = _request(
        requests.Session(),
        "POST",
        f"{base_url}/v1/graphql",
        json={"query": query},
        timeout=60,
    )
    data = response.json()
    return int(data["data"]["Aggregate"][class_name][0]["meta"]["count"])


def _request(
    session: requests.Session,
    method: str,
    url: str,
    **kwargs: Any,
) -> requests.Response:
    response = session.request(method, url, **kwargs)
    response.raise_for_status()
    return response


def _raise_unless(response: requests.Response, allowed: set[int]) -> None:
    if response.status_code not in allowed:
        response.raise_for_status()


def _payload(item_id: int) -> dict[str, Any]:
    return {
        "doc_id": item_id,
        "bucket": _bucket(item_id),
        "lang": "cn" if item_id % 2 else "en",
        "source": f"synthetic-doc-{item_id % 10_000}",
    }


def _bucket(item_id: int) -> str:
    return f"bucket-{item_id % DEFAULT_BUCKET_MOD}"


def _uuid_for(item_id: int) -> str:
    return str(UUID(int=item_id + 1))


def _container_memory_mb(name: str) -> float:
    try:
        completed = subprocess.run(
            [
                "docker",
                "stats",
                name,
                "--no-stream",
                "--format",
                "{{.MemUsage}}",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return psutil.Process().memory_info().rss / 1024 / 1024

    value = completed.stdout.strip().split("/")[0].strip()
    return _parse_memory_mb(value)


def _parse_memory_mb(value: str) -> float:
    match = re.fullmatch(r"([0-9.]+)([KMG]i?B|B)", value)
    if not match:
        return 0.0
    amount = float(match.group(1))
    unit = match.group(2)
    if unit == "B":
        return amount / 1024 / 1024
    if unit in {"KiB", "KB"}:
        return amount / 1024
    if unit in {"MiB", "MB"}:
        return amount
    if unit in {"GiB", "GB"}:
        return amount * 1024
    return amount


def _graphql_ready(host: str, port: int) -> bool:
    if not check_port(host, port):
        return False
    try:
        response = requests.post(
            f"{weaviate_url(host, port)}/v1/graphql",
            json={"query": "{ __typename }"},
            timeout=3,
        )
    except requests.RequestException:
        return False
    if response.status_code != 200:
        return False
    content_type = response.headers.get("content-type", "")
    return "json" in content_type.lower()


def _weaviate_meta_ready(host: str, port: int) -> bool:
    if not check_port(host, port):
        return False
    try:
        response = requests.get(f"{weaviate_url(host, port)}/v1/meta", timeout=3)
    except requests.RequestException:
        return False
    if response.status_code != 200:
        return False
    try:
        data = response.json()
    except ValueError:
        return False
    return "hostname" in data or "version" in data


def _which_docker() -> str | None:
    try:
        completed = subprocess.run(
            ["sh", "-c", "command -v docker"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None
