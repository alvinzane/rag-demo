from pathlib import Path

from rag_demo2.rag import IndexMetadata, read_metadata, write_metadata


def test_metadata_roundtrip(tmp_path: Path) -> None:
    metadata = IndexMetadata(
        docs_dir="docs",
        docs_dirs=["docs"],
        document_count=2,
        chunk_count=4,
        chunk_size=800,
        chunk_overlap=120,
        embed_model="nomic-embed-text",
        collection_name="rag_demo_t1",
    )

    write_metadata(tmp_path, metadata)

    assert read_metadata(tmp_path) == metadata
