from pathlib import Path

import numpy as np

from rag_demo.milvus_bench import (
    generate_vectors,
    index_params,
    percentile,
    prepare_dataset,
    search_params,
)


def test_generate_vectors_is_deterministic_and_normalized() -> None:
    first = generate_vectors(offset=0, count=4, dim=8, seed=7)
    second = generate_vectors(offset=0, count=4, dim=8, seed=7)

    assert np.allclose(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)


def test_percentile_interpolates() -> None:
    values = [10.0, 20.0, 30.0, 40.0]

    assert percentile(values, 50) == 25.0
    assert percentile(values, 95) == 38.5


def test_index_and_search_params() -> None:
    assert index_params("IVF_FLAT", "COSINE", nlist=128, hnsw_m=16, ef_construction=200) == {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128},
    }
    assert search_params("HNSW", nprobe=8, ef=64) == {
        "metric_type": "COSINE",
        "params": {"ef": 64},
    }


def test_prepare_dataset_writes_metadata_and_queries(tmp_path: Path) -> None:
    metadata = prepare_dataset(tmp_path, count=10, dim=6, seed=3, query_count=5)

    assert metadata.count == 10
    assert metadata.dim == 6
    assert (tmp_path / "dataset_meta.json").exists()
    queries = np.load(tmp_path / "queries.npy")
    assert queries.shape == (5, 6)
