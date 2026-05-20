from rag_demo2.qdrant_weaviate import (
    _parse_memory_mb,
    experience_notes,
    qdrant_search_payload,
    weaviate_graphql_query,
)


def test_qdrant_search_payload_adds_payload_filter() -> None:
    payload = qdrant_search_payload([0.1, 0.2], top_k=3, bucket="bucket-7")

    assert payload["limit"] == 3
    assert payload["with_payload"] is True
    assert payload["filter"] == {
        "must": [{"key": "bucket", "match": {"value": "bucket-7"}}]
    }


def test_weaviate_graphql_query_contains_near_vector_and_where() -> None:
    query = weaviate_graphql_query("RagDemoT3Vector", [0.1, 0.2], top_k=5, bucket="bucket-1")

    assert "RagDemoT3Vector" in query
    assert "nearVector" in query
    assert "limit: 5" in query
    assert 'path: ["bucket"]' in query
    assert 'valueText: "bucket-1"' in query


def test_parse_memory_mb_handles_docker_units() -> None:
    assert _parse_memory_mb("512KiB") == 0.5
    assert _parse_memory_mb("128MiB") == 128
    assert _parse_memory_mb("1.5GiB") == 1536


def test_experience_notes_cover_both_backends() -> None:
    notes = experience_notes()

    assert {note.backend for note in notes} == {"qdrant", "weaviate"}
    assert any("GraphQL" in note.query_api for note in notes)
