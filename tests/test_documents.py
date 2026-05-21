from pathlib import Path

import pytest

from rag_demo.documents import discover_markdown_files, load_markdown_documents, split_documents


def test_discover_markdown_files(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n", encoding="utf-8")
    (tmp_path / "b.markdown").write_text("# B\n", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("x\n", encoding="utf-8")

    files = discover_markdown_files(tmp_path)

    assert [path.name for path in files] == ["a.md", "b.markdown"]


def test_load_markdown_documents_raises_for_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No Markdown files"):
        load_markdown_documents(tmp_path)


def test_split_documents_validates_overlap(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\nhello", encoding="utf-8")
    docs = load_markdown_documents(tmp_path)

    with pytest.raises(ValueError, match="smaller"):
        split_documents(docs, chunk_size=100, chunk_overlap=100)


def test_split_documents_adds_chunk_id(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n" + "hello " * 200, encoding="utf-8")
    docs = load_markdown_documents(tmp_path)

    chunks = split_documents(docs, chunk_size=80, chunk_overlap=10)

    assert chunks
    assert chunks[0].metadata["chunk_id"] == 0
    assert chunks[0].metadata["source"].endswith("a.md")
