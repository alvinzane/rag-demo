from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Literal

import numpy as np

Strategy = Literal["fixed", "semantic", "parent_child"]

DEFAULT_RFC_PATH = Path("docs/sample_rfc.md")
DEFAULT_T4_REPORT = Path(".rag/t4/recall_report.json")
DEFAULT_T4_QUERIES = Path(".rag/t4/queries.jsonl")
MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+(?P<title>[^\n]{2,120})$")
RFC_HEADING_RE = re.compile(r"^(?P<title>\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{2,120})$")
TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
STOPWORDS = {
    "a",
    "about",
    "and",
    "are",
    "as",
    "at",
    "does",
    "for",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "rfc",
    "say",
    "the",
    "to",
    "what",
    "with",
}


@dataclass(frozen=True)
class Section:
    section_id: str
    title: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    strategy: Strategy
    text: str
    section_id: str
    section_title: str
    parent_id: str | None = None


@dataclass(frozen=True)
class QueryCase:
    query: str
    expected_section_id: str
    expected_section_title: str


@dataclass(frozen=True)
class RetrievedItem:
    rank: int
    chunk_id: str
    section_id: str
    section_title: str
    score: float
    preview: str


@dataclass(frozen=True)
class StrategyEval:
    strategy: Strategy
    chunks: int
    indexed_items: int
    queries: int
    top_k: int
    hits: int
    recall_at_k: float
    avg_latency_ms: float
    chunk_size: int
    chunk_overlap: int
    semantic_threshold: float


@dataclass(frozen=True)
class MissCase:
    strategy: Strategy
    query: str
    expected_section: str
    retrieved_sections: list[str]


@dataclass(frozen=True)
class RecallReport:
    rfc_path: str
    generated_at: float
    top_k: int
    query_count: int
    results: list[StrategyEval]
    misses: list[MissCase]


class HashingEmbedder:
    def __init__(self, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError("dim must be greater than 0")
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in TOKEN_RE.findall(text.lower()):
            if token in STOPWORDS:
                continue
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            bucket = value % self.dim
            sign = 1.0 if value & 1 == 0 else -1.0
            vector[bucket] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed(text) for text in texts])


def ensure_sample_rfc(path: Path = DEFAULT_RFC_PATH) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# RFC 9000 Demo: QUIC Transport Notes

## 1. Introduction

QUIC is a secure general-purpose transport protocol. It combines stream
multiplexing, connection migration, congestion control, and TLS based security
in a single transport design.

## 2. Streams

Streams provide ordered byte-stream delivery. Multiple streams can be active at
the same time, so an application can avoid head-of-line blocking between
independent exchanges.

## 3. Packet Numbers

Packet numbers are strictly increasing within each packet number space. They
support loss detection, acknowledgements, and protection against replay.

## 4. Connection Migration

Endpoints can migrate a connection to a new network path when the client changes
IP address or port. Path validation is used before sending large amounts of data
on the new path.

## 5. Flow Control

QUIC uses connection-level and stream-level flow control. Receivers advertise
limits, and senders must not exceed those limits.

## 6. Loss Detection

Loss detection uses acknowledgements, packet thresholds, and time thresholds.
Probe packets are sent when an endpoint needs to elicit acknowledgements.

## 7. Security

TLS protects QUIC packets and authenticates the handshake. Most packet headers
are protected after keys are available.
""",
        encoding="utf-8",
    )


def load_rfc(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"RFC file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"RFC file is empty: {path}")
    return text


def parse_sections(text: str) -> list[Section]:
    matches: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        match = MARKDOWN_HEADING_RE.match(stripped) or RFC_HEADING_RE.match(stripped)
        if match and len(stripped.split()) <= 12:
            title = match.group("title").strip().strip("#").strip()
            matches.append((offset, title))
        offset += len(line)

    if not matches:
        return [Section("s0", "Document", text.strip(), 0, len(text))]

    sections: list[Section] = []
    for index, (start, title) in enumerate(matches):
        end = matches[index + 1][0] if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append(Section(f"s{index}", title, body, start, end))
    return sections


def load_queries(
    path: Path | None,
    sections: list[Section],
    limit: int | None = None,
) -> list[QueryCase]:
    if path and path.exists():
        queries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            queries.append(
                QueryCase(
                    query=payload["query"],
                    expected_section_id=str(payload["expected_section_id"]),
                    expected_section_title=str(payload.get("expected_section_title", "")),
                )
            )
        return queries[:limit] if limit else queries

    candidates = [
        section
        for section in sections
        if len(section.text.split()) >= 12 and not section.title.lower().startswith("rfc ")
    ]
    queries = [
        QueryCase(
            query=f"What does the RFC say about {clean_query_title(section.title)}?",
            expected_section_id=section.section_id,
            expected_section_title=section.title,
        )
        for section in candidates
    ]
    return queries[:limit] if limit else queries


def write_queries(path: Path, queries: list[QueryCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(query), ensure_ascii=False) for query in queries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_query_title(title: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", title).strip()


def build_chunks(
    strategy: Strategy,
    sections: list[Section],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    semantic_threshold: float = 0.42,
    embedder: HashingEmbedder | None = None,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be greater than or equal to 0 and smaller than chunk_size"
        )
    if strategy == "fixed":
        return fixed_chunks(sections, chunk_size, chunk_overlap)
    if strategy == "semantic":
        return semantic_chunks(
            sections,
            chunk_size,
            semantic_threshold,
            embedder or HashingEmbedder(),
        )
    if strategy == "parent_child":
        return parent_child_chunks(sections, chunk_size, chunk_overlap)
    raise ValueError(f"unknown strategy: {strategy}")


def fixed_chunks(sections: list[Section], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = chunk_size - chunk_overlap
    for section in sections:
        text = section.text
        for start in range(0, len(text), step):
            part = text[start : start + chunk_size].strip()
            if not part:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"fixed-{len(chunks)}",
                    strategy="fixed",
                    text=part,
                    section_id=section.section_id,
                    section_title=section.title,
                )
            )
            if start + chunk_size >= len(text):
                break
    return chunks


def semantic_chunks(
    sections: list[Section],
    max_chars: int,
    threshold: float,
    embedder: HashingEmbedder,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n", section.text) if item.strip()]
        if not paragraphs:
            continue
        current = [paragraphs[0]]
        current_len = len(paragraphs[0])
        previous_embedding = embedder.embed(paragraphs[0])
        for paragraph in paragraphs[1:]:
            embedding = embedder.embed(paragraph)
            similarity = float(np.dot(previous_embedding, embedding))
            would_fit = current_len + len(paragraph) + 2 <= max_chars
            if would_fit and similarity >= threshold:
                current.append(paragraph)
                current_len += len(paragraph) + 2
            else:
                chunks.append(_semantic_chunk(chunks, current, section))
                current = [paragraph]
                current_len = len(paragraph)
            previous_embedding = embedding
        if current:
            chunks.append(_semantic_chunk(chunks, current, section))
    return chunks


def _semantic_chunk(existing: list[Chunk], paragraphs: list[str], section: Section) -> Chunk:
    return Chunk(
        chunk_id=f"semantic-{len(existing)}",
        strategy="semantic",
        text="\n\n".join(paragraphs),
        section_id=section.section_id,
        section_title=section.title,
    )


def parent_child_chunks(
    sections: list[Section],
    child_size: int,
    child_overlap: int,
) -> list[Chunk]:
    children: list[Chunk] = []
    step = child_size - child_overlap
    for section in sections:
        parent_id = f"parent-{section.section_id}"
        for start in range(0, len(section.text), step):
            part = section.text[start : start + child_size].strip()
            if not part:
                continue
            children.append(
                Chunk(
                    chunk_id=f"child-{len(children)}",
                    strategy="parent_child",
                    text=part,
                    section_id=section.section_id,
                    section_title=section.title,
                    parent_id=parent_id,
                )
            )
            if start + child_size >= len(section.text):
                break
    return children


def evaluate_strategies(
    rfc_path: Path,
    query_path: Path | None = None,
    strategies: list[Strategy] | None = None,
    top_k: int = 5,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    semantic_threshold: float = 0.42,
    dim: int = 384,
    query_limit: int | None = None,
) -> RecallReport:
    text = load_rfc(rfc_path)
    sections = parse_sections(text)
    queries = load_queries(query_path, sections, query_limit)
    if not queries:
        raise ValueError("No query cases available for evaluation")
    selected = strategies or ["fixed", "semantic", "parent_child"]
    embedder = HashingEmbedder(dim)
    results: list[StrategyEval] = []
    misses: list[MissCase] = []
    for strategy in selected:
        chunks = build_chunks(
            strategy,
            sections,
            chunk_size,
            chunk_overlap,
            semantic_threshold,
            embedder,
        )
        result, strategy_misses = evaluate_strategy(
            strategy,
            chunks,
            queries,
            embedder,
            top_k,
            chunk_size,
            chunk_overlap,
            semantic_threshold,
        )
        results.append(result)
        misses.extend(strategy_misses)
    return RecallReport(
        rfc_path=str(rfc_path),
        generated_at=time.time(),
        top_k=top_k,
        query_count=len(queries),
        results=results,
        misses=misses,
    )


def evaluate_strategy(
    strategy: Strategy,
    chunks: list[Chunk],
    queries: list[QueryCase],
    embedder: HashingEmbedder,
    top_k: int,
    chunk_size: int,
    chunk_overlap: int,
    semantic_threshold: float,
) -> tuple[StrategyEval, list[MissCase]]:
    if not chunks:
        raise ValueError(f"strategy {strategy} produced no chunks")
    matrix = embedder.embed_many([chunk.text for chunk in chunks])
    hits = 0
    latencies: list[float] = []
    misses: list[MissCase] = []
    for query in queries:
        started = time.perf_counter()
        retrieved = retrieve(query.query, chunks, matrix, embedder, top_k)
        latencies.append((time.perf_counter() - started) * 1000)
        retrieved_sections = [item.section_id for item in retrieved]
        if query.expected_section_id in retrieved_sections:
            hits += 1
        else:
            misses.append(
                MissCase(
                    strategy=strategy,
                    query=query.query,
                    expected_section=query.expected_section_title,
                    retrieved_sections=[item.section_title for item in retrieved],
                )
            )
    return (
        StrategyEval(
            strategy=strategy,
            chunks=len(chunks),
            indexed_items=len(chunks),
            queries=len(queries),
            top_k=top_k,
            hits=hits,
            recall_at_k=hits / len(queries),
            avg_latency_ms=sum(latencies) / len(latencies),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            semantic_threshold=semantic_threshold,
        ),
        misses,
    )


def retrieve(
    query: str,
    chunks: list[Chunk],
    matrix: np.ndarray,
    embedder: HashingEmbedder,
    top_k: int = 5,
) -> list[RetrievedItem]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    query_vector = embedder.embed(query)
    scores = matrix @ query_vector
    if chunks and chunks[0].strategy == "parent_child":
        return retrieve_parent_sections(chunks, scores, top_k)
    limit = min(top_k, len(chunks))
    if limit == len(chunks):
        indices = np.argsort(scores)[::-1]
    else:
        indices = np.argpartition(scores, -limit)[-limit:]
        indices = indices[np.argsort(scores[indices])[::-1]]
    return [
        RetrievedItem(
            rank=rank,
            chunk_id=chunks[index].chunk_id,
            section_id=chunks[index].section_id,
            section_title=chunks[index].section_title,
            score=float(scores[index]),
            preview=compact_preview(chunks[index].text),
        )
        for rank, index in enumerate(indices, start=1)
    ]


def retrieve_parent_sections(
    chunks: list[Chunk],
    scores: np.ndarray,
    top_k: int,
) -> list[RetrievedItem]:
    best_by_section: dict[str, tuple[int, float]] = {}
    for index, score in enumerate(scores):
        section_id = chunks[index].section_id
        current = best_by_section.get(section_id)
        if current is None or score > current[1]:
            best_by_section[section_id] = (index, float(score))
    ranked = sorted(best_by_section.values(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        RetrievedItem(
            rank=rank,
            chunk_id=chunks[index].parent_id or chunks[index].chunk_id,
            section_id=chunks[index].section_id,
            section_title=chunks[index].section_title,
            score=score,
            preview=compact_preview(chunks[index].text),
        )
        for rank, (index, score) in enumerate(ranked, start=1)
    ]


def strategy_retrieve(
    rfc_path: Path,
    query: str,
    strategy: Strategy,
    top_k: int = 5,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    semantic_threshold: float = 0.42,
    dim: int = 384,
) -> list[RetrievedItem]:
    sections = parse_sections(load_rfc(rfc_path))
    embedder = HashingEmbedder(dim)
    chunks = build_chunks(
        strategy,
        sections,
        chunk_size,
        chunk_overlap,
        semantic_threshold,
        embedder,
    )
    matrix = embedder.embed_many([chunk.text for chunk in chunks])
    return retrieve(query, chunks, matrix, embedder, top_k)


def compact_preview(text: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def write_report(path: Path, report: RecallReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_report(path: Path) -> RecallReport:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RecallReport(
        rfc_path=payload["rfc_path"],
        generated_at=float(payload["generated_at"]),
        top_k=int(payload["top_k"]),
        query_count=int(payload["query_count"]),
        results=[StrategyEval(**item) for item in payload["results"]],
        misses=[MissCase(**item) for item in payload["misses"]],
    )


def parse_strategy(value: str) -> Strategy:
    normalized = value.lower().replace("-", "_")
    if normalized not in {"fixed", "semantic", "parent_child"}:
        raise ValueError("strategy must be fixed, semantic, or parent_child")
    return normalized  # type: ignore[return-value]


def parse_strategies(value: str) -> list[Strategy]:
    if value.lower() == "all":
        return ["fixed", "semantic", "parent_child"]
    return [parse_strategy(item.strip()) for item in value.split(",") if item.strip()]


def recommended_chunk_sizes(chunk_size: int) -> list[int]:
    low = max(120, math.floor(chunk_size * 0.5))
    high = max(low + 1, math.floor(chunk_size * 1.5))
    return sorted({low, chunk_size, high})
