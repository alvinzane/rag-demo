from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

MARKDOWN_PATTERNS = ("*.md", "*.markdown")


def discover_markdown_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Docs directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Docs path must be a directory: {root}")

    files: list[Path] = []
    for pattern in MARKDOWN_PATTERNS:
        files.extend(root.rglob(pattern))
    return sorted(path for path in files if path.is_file())


def load_markdown_documents(root: Path) -> list[Document]:
    docs: list[Document] = []
    for path in discover_markdown_files(root):
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = str(path)
            docs.append(doc)
    if not docs:
        raise ValueError(f"No Markdown files found under: {root}")
    return docs


def split_documents(
    docs: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    return chunks
