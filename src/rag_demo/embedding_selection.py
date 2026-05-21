from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import requests

from rag_demo.config import DEFAULT_OLLAMA_BASE_URL

DEFAULT_T5_DATASET = Path("docs/sample_embedding_eval.jsonl")
DEFAULT_T5_REPORT = Path(".rag/t5/embedding_report.json")
DEFAULT_T5_CACHE = Path(".rag/t5/cache")
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    positive: str
    negatives: list[str]
    language: str
    source: str


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    provider: str
    model: str
    dims: int
    cost_per_1m_tokens_usd: float
    local_deployable: bool
    multilingual: bool
    notes: str


@dataclass(frozen=True)
class CandidateResult:
    name: str
    provider: str
    model: str
    dims: int
    local_deployable: bool
    multilingual: bool
    queries: int
    corpus: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    avg_latency_ms: float
    estimated_tokens: int
    estimated_cost_usd: float
    vector_memory_mb: float
    unavailable: bool
    error: str | None
    notes: str


@dataclass(frozen=True)
class MissCase:
    model: str
    query: str
    language: str
    expected_id: str
    rank: int | None
    top_ids: list[str]


@dataclass(frozen=True)
class EmbeddingReport:
    dataset: str
    top_k: int
    mode: str
    results: list[CandidateResult]
    misses: list[MissCase]
    recommendation: str


DEFAULT_CANDIDATES: dict[str, CandidateSpec] = {
    "bge-m3": CandidateSpec(
        name="bge-m3",
        provider="ollama",
        model="bge-m3:latest",
        dims=1024,
        cost_per_1m_tokens_usd=0.0,
        local_deployable=True,
        multilingual=True,
        notes="Local Ollama model; good default when data privacy and predictable cost matter.",
    ),
    "text-embedding-3-small": CandidateSpec(
        name="text-embedding-3-small",
        provider="openai",
        model="text-embedding-3-small",
        dims=1536,
        cost_per_1m_tokens_usd=0.02,
        local_deployable=False,
        multilingual=True,
        notes="OpenAI small embedding model; low API cost and compact vectors.",
    ),
    "text-embedding-3-large": CandidateSpec(
        name="text-embedding-3-large",
        provider="openai",
        model="text-embedding-3-large",
        dims=3072,
        cost_per_1m_tokens_usd=0.13,
        local_deployable=False,
        multilingual=True,
        notes="OpenAI large embedding model; strongest quality candidate with higher storage cost.",
    ),
}


SAMPLE_CASES = [
    EvalCase(
        id="mteb-en-rag",
        query="How should a RAG assistant answer when the context is insufficient?",
        positive=(
            "A RAG assistant should say that the indexed context is insufficient "
            "instead of inventing an answer."
        ),
        negatives=[
            "Milvus IVF_FLAT performance changes as nprobe increases during vector search.",
            "A release checklist should include rollback ownership and production validation.",
            "Semantic chunking merges adjacent paragraphs when their meaning is close enough.",
        ],
        language="en",
        source="mteb-subset",
    ),
    EvalCase(
        id="mteb-zh-embedding",
        query="为什么团队要比较不同的 Embedding 模型？",
        positive=(
            "团队比较 Embedding 模型是为了权衡召回质量、多语言效果、"
            "成本、向量维度和是否能本地部署。"
        ),
        negatives=[
            "父子分块会检索较小的 child chunk，但返回更完整的 parent section。",
            "Weaviate 通过 GraphQL nearVector 和 where 条件执行过滤检索。",
            "Attu 可以作为 Milvus 的 Web 管理界面查看 collection 和索引。",
        ],
        language="zh",
        source="mteb-subset",
    ),
    EvalCase(
        id="team-qdrant",
        query="Qdrant 在选型 POC 里主要展示什么优势？",
        positive="Qdrant 使用 Rust 内核，过滤检索延迟低，payload filter 的 JSON API 也比较直接。",
        negatives=[
            "OpenAI text-embedding-3-large 默认输出 3072 维向量。",
            "固定分块需要调节 chunk_size 和 chunk_overlap。",
            "Confluence Markdown 导出可以先用 Loader 递归读取再切分。",
        ],
        language="zh",
        source="team-qa",
    ),
    EvalCase(
        id="team-milvus",
        query="Milvus 压测里 nprobe 影响什么？",
        positive=(
            "在 IVF_FLAT 索引中，nprobe 控制查询时扫描的倒排分区数量，"
            "通常会影响召回、延迟和 QPS。"
        ),
        negatives=[
            "GraphQL API 是 Weaviate 在 POC 中的主要开发体验差异点。",
            "BGE-M3 可以通过 Ollama 在本地或内网机器上部署。",
            "MTEB 子集可以覆盖英文和中文检索问题。",
        ],
        language="zh",
        source="team-qa",
    ),
    EvalCase(
        id="mteb-en-parent-child",
        query="What is the parent-child chunking strategy?",
        positive=(
            "Parent-child chunking indexes small child chunks for precise matching "
            "and returns the larger parent section for context."
        ),
        negatives=[
            "The OpenAI embeddings endpoint returns vectors for text inputs.",
            "Qdrant and Weaviate can both store vectors with payload metadata.",
            "A Docker compose file can start Milvus, MinIO, etcd, and Attu.",
        ],
        language="en",
        source="mteb-subset",
    ),
]


class EmbeddingClient:
    def embed(self, texts: list[str], spec: CandidateSpec) -> list[list[float]]:
        raise NotImplementedError


class HashingEmbeddingClient(EmbeddingClient):
    def embed(self, texts: list[str], spec: CandidateSpec) -> list[list[float]]:
        return [hashing_embedding(text, spec.dims) for text in texts]


class OllamaEmbeddingClient(EmbeddingClient):
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed(self, texts: list[str], spec: CandidateSpec) -> list[list[float]]:
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": spec.model, "input": texts},
            timeout=self.timeout,
        )
        if response.status_code == 404:
            return self._legacy_embed(texts, spec)
        response.raise_for_status()
        payload = response.json()
        vectors = payload.get("embeddings")
        if not isinstance(vectors, list):
            raise ValueError(f"Ollama response missing embeddings for {spec.model}")
        return [list(map(float, vector)) for vector in vectors]

    def _legacy_embed(self, texts: list[str], spec: CandidateSpec) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": spec.model, "prompt": text},
                timeout=self.timeout,
            )
            response.raise_for_status()
            vector = response.json().get("embedding")
            if not isinstance(vector, list):
                raise ValueError(f"Ollama response missing embedding for {spec.model}")
            vectors.append(list(map(float, vector)))
        return vectors


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed(self, texts: list[str], spec: CandidateSpec) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embedding models")
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": spec.model, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])
        if len(data) != len(texts):
            raise ValueError(f"OpenAI returned {len(data)} embeddings for {len(texts)} inputs")
        ordered = sorted(data, key=lambda item: item["index"])
        return [list(map(float, item["embedding"])) for item in ordered]


def ensure_sample_dataset(path: Path = DEFAULT_T5_DATASET) -> Path:
    if path.exists():
        return path
    write_dataset(path, SAMPLE_CASES)
    return path


def load_dataset(path: Path) -> list[EvalCase]:
    rows: list[EvalCase] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        try:
            rows.append(
                EvalCase(
                    id=str(payload["id"]),
                    query=str(payload["query"]),
                    positive=str(payload["positive"]),
                    negatives=[str(item) for item in payload.get("negatives", [])],
                    language=str(payload.get("language", "unknown")),
                    source=str(payload.get("source", "team-qa")),
                )
            )
        except KeyError as exc:
            raise ValueError(f"Missing field {exc} in {path}:{line_no}") from exc
    if not rows:
        raise ValueError(f"No eval cases found: {path}")
    return rows


def write_dataset(path: Path, cases: list[EvalCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(case), ensure_ascii=False) for case in cases]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_candidates(value: str) -> list[CandidateSpec]:
    names = list(DEFAULT_CANDIDATES) if value.strip().lower() == "all" else split_csv(value)
    specs: list[CandidateSpec] = []
    for name in names:
        if name not in DEFAULT_CANDIDATES:
            available = ", ".join(DEFAULT_CANDIDATES)
            raise ValueError(f"Unknown embedding candidate: {name}. Available: {available}")
        specs.append(DEFAULT_CANDIDATES[name])
    return specs


def evaluate_embeddings(
    dataset_path: Path,
    candidates: list[CandidateSpec],
    top_k: int = 5,
    real: bool = False,
    skip_unavailable: bool = True,
    cache_dir: Path = DEFAULT_T5_CACHE,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL,
) -> EmbeddingReport:
    cases = load_dataset(dataset_path)
    results: list[CandidateResult] = []
    misses: list[MissCase] = []
    mode = "real" if real else "offline-hashing"

    for spec in candidates:
        try:
            result, model_misses = evaluate_candidate(
                cases=cases,
                spec=spec,
                top_k=top_k,
                client=client_for(spec, real, ollama_base_url, openai_base_url),
                cache_dir=cache_dir,
                mode=mode,
            )
            results.append(result)
            misses.extend(model_misses)
        except Exception as exc:
            if not skip_unavailable:
                raise
            results.append(unavailable_result(spec, cases, top_k, str(exc)))

    return EmbeddingReport(
        dataset=str(dataset_path),
        top_k=top_k,
        mode=mode,
        results=results,
        misses=misses,
        recommendation=recommend(results),
    )


def evaluate_candidate(
    cases: list[EvalCase],
    spec: CandidateSpec,
    top_k: int,
    client: EmbeddingClient,
    cache_dir: Path,
    mode: str,
) -> tuple[CandidateResult, list[MissCase]]:
    corpus_texts, corpus_ids = build_corpus(cases)
    query_texts = [case.query for case in cases]
    texts = query_texts + corpus_texts
    estimated_tokens = sum(estimate_tokens(text) for text in texts)

    start = time.perf_counter()
    vectors = embed_with_cache(texts, spec, client, cache_dir, mode)
    elapsed_ms = (time.perf_counter() - start) * 1000

    query_vectors = np.asarray(vectors[: len(query_texts)], dtype=np.float32)
    corpus_vectors = np.asarray(vectors[len(query_texts) :], dtype=np.float32)
    query_vectors = normalize(query_vectors)
    corpus_vectors = normalize(corpus_vectors)
    scores = query_vectors @ corpus_vectors.T

    hits_1 = 0
    hits_3 = 0
    hits_5 = 0
    reciprocal_ranks: list[float] = []
    misses: list[MissCase] = []

    for row_index, case in enumerate(cases):
        ranking = np.argsort(-scores[row_index])
        positive_id = f"{case.id}:positive"
        ranked_ids = [corpus_ids[index] for index in ranking]
        try:
            rank = ranked_ids.index(positive_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            rank = None
            reciprocal_ranks.append(0.0)

        if rank == 1:
            hits_1 += 1
        if rank is not None and rank <= 3:
            hits_3 += 1
        if rank is not None and rank <= top_k:
            hits_5 += 1
        if rank is None or rank > top_k:
            misses.append(
                MissCase(
                    model=spec.name,
                    query=case.query,
                    language=case.language,
                    expected_id=positive_id,
                    rank=rank,
                    top_ids=ranked_ids[:top_k],
                )
            )

    query_count = len(cases)
    return (
        CandidateResult(
            name=spec.name,
            provider=spec.provider,
            model=spec.model,
            dims=vector_dim(vectors, spec),
            local_deployable=spec.local_deployable,
            multilingual=spec.multilingual,
            queries=query_count,
            corpus=len(corpus_texts),
            recall_at_1=hits_1 / query_count,
            recall_at_3=hits_3 / query_count,
            recall_at_5=hits_5 / query_count,
            mrr=sum(reciprocal_ranks) / query_count,
            avg_latency_ms=elapsed_ms / max(len(texts), 1),
            estimated_tokens=estimated_tokens,
            estimated_cost_usd=estimated_tokens * spec.cost_per_1m_tokens_usd / 1_000_000,
            vector_memory_mb=len(corpus_texts) * vector_dim(vectors, spec) * 4 / 1024 / 1024,
            unavailable=False,
            error=None,
            notes=spec.notes,
        ),
        misses,
    )


def client_for(
    spec: CandidateSpec,
    real: bool,
    ollama_base_url: str,
    openai_base_url: str,
) -> EmbeddingClient:
    if not real:
        return HashingEmbeddingClient()
    if spec.provider == "ollama":
        return OllamaEmbeddingClient(ollama_base_url)
    if spec.provider == "openai":
        return OpenAIEmbeddingClient(base_url=openai_base_url)
    raise ValueError(f"Unsupported provider: {spec.provider}")


def build_corpus(cases: list[EvalCase]) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    ids: list[str] = []
    seen: set[str] = set()
    for case in cases:
        entries = [(f"{case.id}:positive", case.positive)]
        entries.extend(
            (f"{case.id}:negative:{index}", text)
            for index, text in enumerate(case.negatives)
        )
        for item_id, text in entries:
            fingerprint = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            ids.append(item_id)
            texts.append(text)
    return texts, ids


def embed_with_cache(
    texts: list[str],
    spec: CandidateSpec,
    client: EmbeddingClient,
    cache_dir: Path,
    mode: str,
) -> list[list[float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = f"{safe_name(mode)}-{safe_name(spec.name)}-{safe_name(spec.model)}.json"
    cache_path = cache_dir / cache_file
    cached: dict[str, list[float]] = {}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))

    missing_texts = [text for text in texts if text_key(text) not in cached]
    if missing_texts:
        vectors = client.embed(missing_texts, spec)
        for text, vector in zip(missing_texts, vectors, strict=True):
            cached[text_key(text)] = vector
        cache_path.write_text(json.dumps(cached), encoding="utf-8")
    return [cached[text_key(text)] for text in texts]


def unavailable_result(
    spec: CandidateSpec,
    cases: list[EvalCase],
    top_k: int,
    error: str,
) -> CandidateResult:
    _ = top_k
    return CandidateResult(
        name=spec.name,
        provider=spec.provider,
        model=spec.model,
        dims=spec.dims,
        local_deployable=spec.local_deployable,
        multilingual=spec.multilingual,
        queries=len(cases),
        corpus=0,
        recall_at_1=0.0,
        recall_at_3=0.0,
        recall_at_5=0.0,
        mrr=0.0,
        avg_latency_ms=0.0,
        estimated_tokens=0,
        estimated_cost_usd=0.0,
        vector_memory_mb=0.0,
        unavailable=True,
        error=error,
        notes=spec.notes,
    )


def recommend(results: list[CandidateResult]) -> str:
    available = [item for item in results if not item.unavailable]
    if not available:
        return "No candidate completed. Check Ollama/OpenAI credentials or run without --real."
    best_quality = max(
        available,
        key=lambda item: (item.recall_at_5, item.mrr, -item.estimated_cost_usd),
    )
    local = [item for item in available if item.local_deployable]
    if local:
        best_local = max(local, key=lambda item: (item.recall_at_5, item.mrr))
        if best_local.recall_at_5 >= best_quality.recall_at_5 - 0.03:
            return (
                f"Recommend {best_local.name} for team default: near-best Recall@5 "
                "with zero API token cost and local deployability."
            )
    return (
        f"Recommend {best_quality.name} when quality is the priority; validate privacy, "
        "latency, and recurring token/storage cost before production rollout."
    )


def write_report(path: Path, report: EmbeddingReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_report(path: Path) -> EmbeddingReport:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return EmbeddingReport(
        dataset=payload["dataset"],
        top_k=int(payload["top_k"]),
        mode=payload["mode"],
        results=[CandidateResult(**item) for item in payload["results"]],
        misses=[MissCase(**item) for item in payload.get("misses", [])],
        recommendation=payload["recommendation"],
    )


def normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return array / norms


def hashing_embedding(text: str, dim: int) -> list[float]:
    vector = np.zeros(dim, dtype=np.float32)
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        bucket = int.from_bytes(digest[:8], "little") % dim
        vector[bucket] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector.tolist()


def tokenize(text: str) -> list[str]:
    normalized = text.lower()
    tokens: list[str] = []
    current: list[str] = []
    for char in normalized:
        if char.isalnum():
            current.append(char)
            continue
        if current:
            tokens.append("".join(current))
            current = []
        if "\u4e00" <= char <= "\u9fff":
            tokens.append(char)
    if current:
        tokens.append("".join(current))

    char_bigrams = [
        normalized[index : index + 2]
        for index in range(len(normalized) - 1)
        if any("\u4e00" <= c <= "\u9fff" for c in normalized[index : index + 2])
    ]
    filtered = [
        token for token in tokens if len(token) > 1 or "\u4e00" <= token <= "\u9fff"
    ]
    return filtered + char_bigrams


def estimate_tokens(text: str) -> int:
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, math.ceil(ascii_chars / 4 + non_ascii_chars / 2))


def vector_dim(vectors: list[list[float]], spec: CandidateSpec) -> int:
    if vectors:
        return len(vectors[0])
    return spec.dims


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value.lower()).strip("-")


def text_key(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
