from pathlib import Path

from rag_demo2.chunking_recall import (
    HashingEmbedder,
    build_chunks,
    evaluate_strategies,
    load_queries,
    parse_sections,
    parse_strategy,
    retrieve,
    write_queries,
)

RFC = """# Demo RFC

## 1. Introduction

QUIC combines secure transport, stream multiplexing, and connection migration.

## 2. Flow Control

Receivers advertise connection-level and stream-level limits. Senders must not
exceed the advertised limits.

## 3. Loss Detection

Loss detection uses acknowledgements, packet thresholds, and time thresholds.
"""


def test_parse_sections_detects_markdown_headings() -> None:
    sections = parse_sections(RFC)

    assert [section.title for section in sections] == [
        "Demo RFC",
        "1. Introduction",
        "2. Flow Control",
        "3. Loss Detection",
    ]


def test_build_chunks_supports_all_strategies() -> None:
    sections = parse_sections(RFC)

    fixed = build_chunks("fixed", sections, chunk_size=80, chunk_overlap=10)
    semantic = build_chunks("semantic", sections, chunk_size=160, chunk_overlap=10)
    parent_child = build_chunks("parent_child", sections, chunk_size=80, chunk_overlap=10)

    assert fixed
    assert semantic
    assert parent_child
    assert parent_child[0].parent_id is not None


def test_parent_child_retrieval_returns_unique_parent_sections() -> None:
    sections = parse_sections(RFC)
    chunks = build_chunks("parent_child", sections, chunk_size=50, chunk_overlap=5)
    embedder = HashingEmbedder(dim=64)
    matrix = embedder.embed_many([chunk.text for chunk in chunks])

    results = retrieve("stream level limits", chunks, matrix, embedder, top_k=3)

    assert len({item.section_id for item in results}) == len(results)


def test_evaluate_strategies_returns_recall_report(tmp_path: Path) -> None:
    rfc_path = tmp_path / "rfc.md"
    rfc_path.write_text(RFC, encoding="utf-8")

    report = evaluate_strategies(
        rfc_path=rfc_path,
        strategies=["fixed", "semantic", "parent_child"],
        top_k=2,
        chunk_size=120,
        chunk_overlap=20,
        dim=64,
    )

    assert report.query_count > 0
    assert {result.strategy for result in report.results} == {"fixed", "semantic", "parent_child"}
    assert all(0 <= result.recall_at_k <= 1 for result in report.results)


def test_query_jsonl_round_trip(tmp_path: Path) -> None:
    sections = parse_sections(RFC)
    queries = load_queries(None, sections)
    path = tmp_path / "queries.jsonl"

    write_queries(path, queries)
    loaded = load_queries(path, sections)

    assert loaded == queries


def test_parse_strategy_accepts_hyphen_alias() -> None:
    assert parse_strategy("parent-child") == "parent_child"
