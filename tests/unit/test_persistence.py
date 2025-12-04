"""Tests for persistence utilities including index builders.

TDD: Tests for IVF index builders written before implementation.
"""

from __future__ import annotations

import faiss
import numpy as np
import pytest

from biovector_eval.utils.persistence import (
    build_flat_index,
    build_hnsw_index,
    build_ivf_flat_index,
    build_ivf_pq_index,
    build_pq_index,
    build_sq4_index,
    build_sq8_index,
)


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Generate sample normalized embeddings for testing.

    Returns 1000 random embeddings of dimension 128, normalized.
    """
    np.random.seed(42)
    embeddings = np.random.randn(1000, 128).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


@pytest.fixture
def small_embeddings() -> np.ndarray:
    """Smaller embeddings for quick tests."""
    np.random.seed(42)
    embeddings = np.random.randn(100, 64).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


class TestBuildFlatIndex:
    """Tests for build_flat_index function."""

    def test_flat_index_exact(self, sample_embeddings: np.ndarray) -> None:
        """Flat index should use IndexFlatIP for cosine similarity."""
        index = build_flat_index(sample_embeddings)

        # Should be an exact index (Flat)
        assert isinstance(index, faiss.IndexFlatIP)

    def test_flat_index_returns_all(self, sample_embeddings: np.ndarray) -> None:
        """Flat index should contain all input vectors."""
        index = build_flat_index(sample_embeddings)

        assert index.ntotal == sample_embeddings.shape[0]

    def test_flat_index_dimension(self, sample_embeddings: np.ndarray) -> None:
        """Flat index should have correct dimension."""
        index = build_flat_index(sample_embeddings)

        assert index.d == sample_embeddings.shape[1]

    def test_flat_index_exact_search(self, sample_embeddings: np.ndarray) -> None:
        """Flat index should return exact matches (ground truth)."""
        index = build_flat_index(sample_embeddings)

        # Search for the first vector
        query = sample_embeddings[0:1]
        distances, indices = index.search(query, k=1)

        # Should return itself as closest match
        assert indices[0, 0] == 0
        # Distance should be ~1.0 for normalized vectors (cosine = 1.0)
        assert np.isclose(distances[0, 0], 1.0, atol=1e-5)


class TestBuildHNSWIndex:
    """Tests for build_hnsw_index function."""

    def test_hnsw_index_type(self, sample_embeddings: np.ndarray) -> None:
        """HNSW index should be IndexHNSWFlat."""
        index = build_hnsw_index(sample_embeddings)

        assert isinstance(index, faiss.IndexHNSWFlat)

    def test_hnsw_index_params(self, sample_embeddings: np.ndarray) -> None:
        """HNSW index should use M=32 by default."""
        index = build_hnsw_index(sample_embeddings, hnsw_m=32)

        # Check that vectors are added
        assert index.ntotal == sample_embeddings.shape[0]

    def test_hnsw_ef_construction(self, sample_embeddings: np.ndarray) -> None:
        """HNSW index should use efConstruction=200 by default."""
        index = build_hnsw_index(sample_embeddings, ef_construction=200)

        assert index.hnsw.efConstruction == 200

    def test_hnsw_custom_ef_construction(self, sample_embeddings: np.ndarray) -> None:
        """HNSW index should accept custom efConstruction."""
        index = build_hnsw_index(sample_embeddings, ef_construction=100)

        assert index.hnsw.efConstruction == 100

    def test_hnsw_search_quality(self, sample_embeddings: np.ndarray) -> None:
        """HNSW index should return high-quality results."""
        index = build_hnsw_index(sample_embeddings)

        # Search for the first vector
        query = sample_embeddings[0:1]
        distances, indices = index.search(query, k=1)

        # Should return itself as closest match
        assert indices[0, 0] == 0


class TestBuildSQ8Index:
    """Tests for build_sq8_index (8-bit scalar quantization)."""

    def test_hnsw_sq8_type(self, sample_embeddings: np.ndarray) -> None:
        """SQ8 index should be IndexHNSWSQ."""
        index = build_sq8_index(sample_embeddings)

        assert isinstance(index, faiss.IndexHNSWSQ)

    def test_hnsw_sq8_compression(self, sample_embeddings: np.ndarray) -> None:
        """SQ8 index should compress vectors (8 bits per dimension)."""
        index = build_sq8_index(sample_embeddings)

        # Index should contain all vectors
        assert index.ntotal == sample_embeddings.shape[0]

    def test_hnsw_sq8_search(self, sample_embeddings: np.ndarray) -> None:
        """SQ8 index should return reasonable search results."""
        index = build_sq8_index(sample_embeddings)

        query = sample_embeddings[0:1]
        distances, indices = index.search(query, k=5)

        # First result should be the query itself (index 0)
        assert indices[0, 0] == 0


class TestBuildSQ4Index:
    """Tests for build_sq4_index (4-bit scalar quantization)."""

    def test_hnsw_sq4_type(self, sample_embeddings: np.ndarray) -> None:
        """SQ4 index should be IndexHNSWSQ."""
        index = build_sq4_index(sample_embeddings)

        assert isinstance(index, faiss.IndexHNSWSQ)

    def test_hnsw_sq4_compression(self, sample_embeddings: np.ndarray) -> None:
        """SQ4 index should compress vectors (4 bits per dimension)."""
        index = build_sq4_index(sample_embeddings)

        assert index.ntotal == sample_embeddings.shape[0]

    def test_hnsw_sq4_search(self, sample_embeddings: np.ndarray) -> None:
        """SQ4 index should return reasonable search results."""
        index = build_sq4_index(sample_embeddings)

        query = sample_embeddings[0:1]
        distances, indices = index.search(query, k=5)

        # First result should be the query itself (index 0)
        assert indices[0, 0] == 0


class TestBuildPQIndex:
    """Tests for build_pq_index (product quantization)."""

    def test_hnsw_pq_type(self, sample_embeddings: np.ndarray) -> None:
        """PQ index should be IndexHNSWPQ."""
        index = build_pq_index(sample_embeddings)

        assert isinstance(index, faiss.IndexHNSWPQ)

    def test_hnsw_pq_compression(self, sample_embeddings: np.ndarray) -> None:
        """PQ index should contain all vectors."""
        index = build_pq_index(sample_embeddings)

        assert index.ntotal == sample_embeddings.shape[0]

    def test_hnsw_pq_adjusts_m(self) -> None:
        """PQ should adjust pq_m to divide dimension evenly."""
        # 100 dimension, pq_m=32 doesn't divide evenly
        embeddings = np.random.randn(500, 100).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        # Should not raise, but adjust pq_m internally
        index = build_pq_index(embeddings, pq_m=32)
        assert index.ntotal == 500


# ============================================================================
# IVF Index Tests (TDD - implementation pending)
# ============================================================================


class TestBuildIVFFlatIndex:
    """Tests for build_ivf_flat_index (clustering with exhaustive search)."""

    def test_ivf_flat_type(self, sample_embeddings: np.ndarray) -> None:
        """IVFFlat index should be IndexIVFFlat."""
        index = build_ivf_flat_index(sample_embeddings)

        assert isinstance(index, faiss.IndexIVFFlat)

    def test_ivf_flat_requires_train(self, sample_embeddings: np.ndarray) -> None:
        """IVFFlat index should be trained after construction."""
        index = build_ivf_flat_index(sample_embeddings)

        assert index.is_trained

    def test_ivf_flat_nlist_param(self, sample_embeddings: np.ndarray) -> None:
        """IVFFlat should use specified nlist (number of clusters)."""
        index = build_ivf_flat_index(sample_embeddings, nlist=64)

        assert index.nlist == 64

    def test_ivf_flat_default_nlist(self, sample_embeddings: np.ndarray) -> None:
        """IVFFlat should have reasonable default nlist."""
        index = build_ivf_flat_index(sample_embeddings)

        # Default should be reasonable for dataset size
        # For 1000 vectors, sqrt(1000) ~= 32
        assert index.nlist > 0

    def test_ivf_flat_nprobe_default(self, sample_embeddings: np.ndarray) -> None:
        """IVFFlat should have nprobe=32 by default."""
        index = build_ivf_flat_index(sample_embeddings)

        assert index.nprobe == 32

    def test_ivf_flat_custom_nprobe(self, sample_embeddings: np.ndarray) -> None:
        """IVFFlat should accept custom nprobe."""
        index = build_ivf_flat_index(sample_embeddings, nprobe=16)

        assert index.nprobe == 16

    def test_ivf_flat_contains_vectors(self, sample_embeddings: np.ndarray) -> None:
        """IVFFlat should contain all input vectors."""
        index = build_ivf_flat_index(sample_embeddings)

        assert index.ntotal == sample_embeddings.shape[0]

    def test_ivf_flat_search(self, sample_embeddings: np.ndarray) -> None:
        """IVFFlat should return valid search results."""
        index = build_ivf_flat_index(sample_embeddings)

        query = sample_embeddings[0:1]
        distances, indices = index.search(query, k=5)

        # Should return valid indices
        assert indices.shape == (1, 5)
        assert all(0 <= idx < sample_embeddings.shape[0] for idx in indices[0])

    def test_ivf_flat_inner_product(self, sample_embeddings: np.ndarray) -> None:
        """IVFFlat should use inner product metric for normalized vectors."""
        index = build_ivf_flat_index(sample_embeddings)

        # The metric type should be METRIC_INNER_PRODUCT for cosine similarity
        assert index.metric_type == faiss.METRIC_INNER_PRODUCT


class TestBuildIVFPQIndex:
    """Tests for build_ivf_pq_index (clustering + product quantization)."""

    def test_ivf_pq_type(self, sample_embeddings: np.ndarray) -> None:
        """IVFPQ index should be IndexIVFPQ."""
        index = build_ivf_pq_index(sample_embeddings)

        assert isinstance(index, faiss.IndexIVFPQ)

    def test_ivf_pq_requires_train(self, sample_embeddings: np.ndarray) -> None:
        """IVFPQ index should be trained after construction."""
        index = build_ivf_pq_index(sample_embeddings)

        assert index.is_trained

    def test_ivf_pq_nlist_param(self, sample_embeddings: np.ndarray) -> None:
        """IVFPQ should use specified nlist."""
        index = build_ivf_pq_index(sample_embeddings, nlist=64)

        assert index.nlist == 64

    def test_ivf_pq_pq_m_param(self, sample_embeddings: np.ndarray) -> None:
        """IVFPQ should use specified pq_m (subquantizers)."""
        # 128 dimension, pq_m=16 divides evenly
        index = build_ivf_pq_index(sample_embeddings, pq_m=16)

        # Check that it was built successfully
        assert index.ntotal == sample_embeddings.shape[0]

    def test_ivf_pq_compression(self, sample_embeddings: np.ndarray) -> None:
        """IVFPQ should compress vectors significantly."""
        index = build_ivf_pq_index(sample_embeddings)

        assert index.ntotal == sample_embeddings.shape[0]

    def test_ivf_pq_search(self, sample_embeddings: np.ndarray) -> None:
        """IVFPQ should return valid search results."""
        index = build_ivf_pq_index(sample_embeddings, nlist=32, nprobe=8)

        query = sample_embeddings[0:1]
        distances, indices = index.search(query, k=5)

        # Should return valid indices
        assert indices.shape == (1, 5)
        assert all(0 <= idx < sample_embeddings.shape[0] for idx in indices[0])

    def test_ivf_pq_nprobe_default(self, sample_embeddings: np.ndarray) -> None:
        """IVFPQ should have nprobe=32 by default."""
        index = build_ivf_pq_index(sample_embeddings)

        assert index.nprobe == 32

    def test_ivf_pq_adjusts_pq_m(self) -> None:
        """IVFPQ should adjust pq_m to divide dimension evenly."""
        # 100 dimension, pq_m=32 doesn't divide evenly
        embeddings = np.random.randn(500, 100).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        # Should not raise, but adjust pq_m internally
        index = build_ivf_pq_index(embeddings, pq_m=32)
        assert index.ntotal == 500

    def test_ivf_pq_inner_product(self, sample_embeddings: np.ndarray) -> None:
        """IVFPQ should use inner product metric."""
        index = build_ivf_pq_index(sample_embeddings)

        assert index.metric_type == faiss.METRIC_INNER_PRODUCT


class TestIndexQualityComparison:
    """Tests comparing search quality across index types."""

    def test_flat_vs_hnsw_recall(self, sample_embeddings: np.ndarray) -> None:
        """HNSW should have high recall compared to Flat (ground truth)."""
        flat_index = build_flat_index(sample_embeddings)
        hnsw_index = build_hnsw_index(sample_embeddings)

        # Use first 10 vectors as queries
        queries = sample_embeddings[:10]
        k = 10

        # Get ground truth from Flat
        _, flat_indices = flat_index.search(queries, k)

        # Get HNSW results
        _, hnsw_indices = hnsw_index.search(queries, k)

        # Calculate recall
        recalls = []
        for i in range(len(queries)):
            gt_set = set(flat_indices[i])
            found = len(set(hnsw_indices[i]) & gt_set)
            recalls.append(found / k)

        avg_recall = np.mean(recalls)
        # HNSW should have high recall
        assert avg_recall > 0.8

    def test_ivf_flat_vs_flat_recall(self, sample_embeddings: np.ndarray) -> None:
        """IVFFlat should have reasonable recall compared to Flat."""
        flat_index = build_flat_index(sample_embeddings)
        ivf_index = build_ivf_flat_index(sample_embeddings, nlist=32, nprobe=16)

        queries = sample_embeddings[:10]
        k = 10

        _, flat_indices = flat_index.search(queries, k)
        _, ivf_indices = ivf_index.search(queries, k)

        recalls = []
        for i in range(len(queries)):
            gt_set = set(flat_indices[i])
            found = len(set(ivf_indices[i]) & gt_set)
            recalls.append(found / k)

        avg_recall = np.mean(recalls)
        # IVFFlat should have reasonable recall with nprobe=16
        assert avg_recall > 0.6
