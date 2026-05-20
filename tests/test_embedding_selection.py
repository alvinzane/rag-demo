from __future__ import annotations

from pathlib import Path

from rag_demo2.embedding_selection import (
    DEFAULT_CANDIDATES,
    SAMPLE_CASES,
    evaluate_embeddings,
    load_dataset,
    parse_candidates,
    read_report,
    write_dataset,
    write_report,
)


def test_parse_candidates_all() -> None:
    specs = parse_candidates("all")
    assert [spec.name for spec in specs] == [
        "bge-m3",
        "text-embedding-3-small",
        "text-embedding-3-large",
    ]


def test_dataset_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    write_dataset(path, SAMPLE_CASES[:2])
    loaded = load_dataset(path)
    assert len(loaded) == 2
    assert loaded[0].id == "mteb-en-rag"
    assert loaded[1].language == "zh"


def test_offline_embedding_evaluation_and_report_roundtrip(tmp_path: Path) -> None:
    dataset = tmp_path / "eval.jsonl"
    report_path = tmp_path / "report.json"
    write_dataset(dataset, SAMPLE_CASES)

    report = evaluate_embeddings(
        dataset_path=dataset,
        candidates=[DEFAULT_CANDIDATES["bge-m3"], DEFAULT_CANDIDATES["text-embedding-3-small"]],
        top_k=5,
        real=False,
        cache_dir=tmp_path / "cache",
    )
    write_report(report_path, report)
    loaded = read_report(report_path)

    assert loaded.mode == "offline-hashing"
    assert len(loaded.results) == 2
    assert all(not item.unavailable for item in loaded.results)
    assert all(item.queries == len(SAMPLE_CASES) for item in loaded.results)
    assert loaded.recommendation
