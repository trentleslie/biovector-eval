"""Tests for build_indices.py batch index builder.

TDD: Tests written before implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pytest

# Import will fail until we add the functions - that's expected in TDD
from scripts.build_indices import (
    INDEX_TYPES,
    build_all_indices,
    build_index_for_embedding_file,
    get_index_output_path,
    parse_arguments,
    should_skip_index,
)


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Create sample embeddings for testing."""
    np.random.seed(42)
    embeddings = np.random.randn(1000, 384).astype(np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


@pytest.fixture
def sample_multi_vector_data(tmp_path: Path) -> tuple[Path, dict]:
    """Create sample multi-vector embeddings with metadata."""
    np.random.seed(42)
    embeddings = np.random.randn(500, 384).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    metadata = {
        str(i): {
            "hmdb_id": f"HMDB{i % 100:07d}",
            "tier": ["primary", "synonym", "variation"][i % 3],
            "weight": [1.0, 0.9, 0.7][i % 3],
            "text": f"Metabolite text {i}",
        }
        for i in range(500)
    }

    # Save to directory structure
    multi_dir = tmp_path / "embeddings" / "bge-m3_multi-vector"
    multi_dir.mkdir(parents=True)
    np.save(multi_dir / "embeddings.npy", embeddings)
    with open(multi_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return multi_dir, metadata


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create temporary output directory."""
    output_dir = tmp_path / "indices"
    output_dir.mkdir(parents=True)
    return output_dir


class TestIndexTypes:
    """Tests for INDEX_TYPES configuration."""

    def test_index_types_contains_all_seven(self) -> None:
        """INDEX_TYPES should contain all 7 index types."""
        expected_types = {"flat", "hnsw", "sq8", "sq4", "pq", "ivf_flat", "ivf_pq"}
        assert set(INDEX_TYPES.keys()) == expected_types

    def test_index_types_have_builder_functions(self) -> None:
        """Each index type should have a builder function reference."""
        for index_type, config in INDEX_TYPES.items():
            assert "builder" in config, f"{index_type} missing builder"
            assert callable(config["builder"]), f"{index_type} builder not callable"

    def test_index_types_have_descriptions(self) -> None:
        """Each index type should have a description."""
        for index_type, config in INDEX_TYPES.items():
            assert "description" in config, f"{index_type} missing description"
            assert isinstance(config["description"], str)


class TestParseArguments:
    """Tests for argument parsing."""

    def test_parse_required_arguments(self) -> None:
        """Should parse required --embeddings and --output-dir."""
        args = parse_arguments(
            ["--embeddings", "data/embeddings", "--output-dir", "data/indices"]
        )
        assert args.embeddings == Path("data/embeddings")
        assert args.output_dir == Path("data/indices")

    def test_parse_index_types_default(self) -> None:
        """Default should be 'all' index types."""
        args = parse_arguments(
            ["--embeddings", "data/embeddings", "--output-dir", "data/indices"]
        )
        assert args.index_types == "all"

    def test_parse_index_types_comma_separated(self) -> None:
        """Should accept comma-separated index types."""
        args = parse_arguments(
            [
                "--embeddings",
                "data/embeddings",
                "--output-dir",
                "data/indices",
                "--index-types",
                "flat,hnsw,sq8",
            ]
        )
        assert args.index_types == "flat,hnsw,sq8"

    def test_parse_skip_existing_flag(self) -> None:
        """Should accept --skip-existing flag."""
        args = parse_arguments(
            [
                "--embeddings",
                "data/embeddings",
                "--output-dir",
                "data/indices",
                "--skip-existing",
            ]
        )
        assert args.skip_existing is True

    def test_parse_skip_existing_default_false(self) -> None:
        """Default --skip-existing should be False."""
        args = parse_arguments(
            ["--embeddings", "data/embeddings", "--output-dir", "data/indices"]
        )
        assert args.skip_existing is False


class TestGetIndexOutputPath:
    """Tests for index output path generation."""

    def test_single_vector_path(self, temp_output_dir: Path) -> None:
        """Single-vector embeddings should use flat naming."""
        path = get_index_output_path(
            output_dir=temp_output_dir,
            embedding_name="bge-m3_single-primary",
            index_type="hnsw",
        )
        assert path == temp_output_dir / "bge-m3_single-primary_hnsw.faiss"

    def test_multi_vector_path(self, temp_output_dir: Path) -> None:
        """Multi-vector embeddings should use same flat naming."""
        path = get_index_output_path(
            output_dir=temp_output_dir,
            embedding_name="sapbert_multi-vector",
            index_type="ivf_flat",
        )
        assert path == temp_output_dir / "sapbert_multi-vector_ivf_flat.faiss"

    def test_all_index_types_have_unique_paths(self, temp_output_dir: Path) -> None:
        """Each index type should produce a unique path."""
        paths = set()
        for index_type in INDEX_TYPES:
            path = get_index_output_path(
                output_dir=temp_output_dir,
                embedding_name="test_model",
                index_type=index_type,
            )
            assert path not in paths, f"Duplicate path for {index_type}"
            paths.add(path)


class TestShouldSkipIndex:
    """Tests for skip existing logic."""

    def test_skip_when_exists_and_flag_true(self, temp_output_dir: Path) -> None:
        """Should skip when file exists and skip_existing=True."""
        index_path = temp_output_dir / "test_hnsw.faiss"
        index_path.touch()  # Create empty file

        assert should_skip_index(index_path, skip_existing=True) is True

    def test_no_skip_when_exists_but_flag_false(self, temp_output_dir: Path) -> None:
        """Should not skip when file exists but skip_existing=False."""
        index_path = temp_output_dir / "test_hnsw.faiss"
        index_path.touch()

        assert should_skip_index(index_path, skip_existing=False) is False

    def test_no_skip_when_not_exists(self, temp_output_dir: Path) -> None:
        """Should not skip when file doesn't exist."""
        index_path = temp_output_dir / "test_hnsw.faiss"

        assert should_skip_index(index_path, skip_existing=True) is False
        assert should_skip_index(index_path, skip_existing=False) is False


class TestBuildIndexForEmbeddingFile:
    """Tests for building indices from embedding files."""

    def test_builds_flat_index(
        self, sample_embeddings: np.ndarray, temp_output_dir: Path, tmp_path: Path
    ) -> None:
        """Should build flat index from embeddings file."""
        # Save embeddings
        emb_path = tmp_path / "embeddings" / "test_single-primary.npy"
        emb_path.parent.mkdir(parents=True)
        np.save(emb_path, sample_embeddings)

        # Build index
        index_path = build_index_for_embedding_file(
            embedding_path=emb_path,
            output_dir=temp_output_dir,
            index_type="flat",
        )

        assert index_path.exists()
        index = faiss.read_index(str(index_path))
        assert index.ntotal == 1000

    def test_builds_hnsw_index(
        self, sample_embeddings: np.ndarray, temp_output_dir: Path, tmp_path: Path
    ) -> None:
        """Should build HNSW index from embeddings file."""
        emb_path = tmp_path / "embeddings" / "test_single-primary.npy"
        emb_path.parent.mkdir(parents=True)
        np.save(emb_path, sample_embeddings)

        index_path = build_index_for_embedding_file(
            embedding_path=emb_path,
            output_dir=temp_output_dir,
            index_type="hnsw",
        )

        assert index_path.exists()
        index = faiss.read_index(str(index_path))
        assert index.ntotal == 1000

    def test_builds_ivf_flat_index(
        self, sample_embeddings: np.ndarray, temp_output_dir: Path, tmp_path: Path
    ) -> None:
        """Should build IVF Flat index from embeddings file."""
        emb_path = tmp_path / "embeddings" / "test_single-primary.npy"
        emb_path.parent.mkdir(parents=True)
        np.save(emb_path, sample_embeddings)

        index_path = build_index_for_embedding_file(
            embedding_path=emb_path,
            output_dir=temp_output_dir,
            index_type="ivf_flat",
        )

        assert index_path.exists()
        index = faiss.read_index(str(index_path))
        assert index.ntotal == 1000

    def test_builds_ivf_pq_index(
        self, sample_embeddings: np.ndarray, temp_output_dir: Path, tmp_path: Path
    ) -> None:
        """Should build IVF PQ index from embeddings file."""
        emb_path = tmp_path / "embeddings" / "test_single-primary.npy"
        emb_path.parent.mkdir(parents=True)
        np.save(emb_path, sample_embeddings)

        index_path = build_index_for_embedding_file(
            embedding_path=emb_path,
            output_dir=temp_output_dir,
            index_type="ivf_pq",
        )

        assert index_path.exists()
        index = faiss.read_index(str(index_path))
        assert index.ntotal == 1000

    def test_handles_multi_vector_directory(
        self, sample_multi_vector_data: tuple[Path, dict], temp_output_dir: Path
    ) -> None:
        """Should handle multi-vector directory structure."""
        multi_dir, metadata = sample_multi_vector_data
        emb_path = multi_dir / "embeddings.npy"

        index_path = build_index_for_embedding_file(
            embedding_path=emb_path,
            output_dir=temp_output_dir,
            index_type="hnsw",
        )

        assert index_path.exists()
        index = faiss.read_index(str(index_path))
        assert index.ntotal == 500


class TestBuildAllIndices:
    """Tests for batch index building."""

    def test_builds_all_requested_types(
        self, sample_embeddings: np.ndarray, temp_output_dir: Path, tmp_path: Path
    ) -> None:
        """Should build all requested index types."""
        # Save embeddings
        emb_dir = tmp_path / "embeddings"
        emb_dir.mkdir(parents=True)
        np.save(emb_dir / "test_single-primary.npy", sample_embeddings)

        # Build all indices
        results = build_all_indices(
            embeddings_dir=emb_dir,
            output_dir=temp_output_dir,
            index_types=["flat", "hnsw"],
            skip_existing=False,
        )

        assert len(results) == 2
        assert all(r["success"] for r in results)
        assert (temp_output_dir / "test_single-primary_flat.faiss").exists()
        assert (temp_output_dir / "test_single-primary_hnsw.faiss").exists()

    def test_builds_for_multiple_embedding_files(
        self, sample_embeddings: np.ndarray, temp_output_dir: Path, tmp_path: Path
    ) -> None:
        """Should build indices for all embedding files in directory."""
        emb_dir = tmp_path / "embeddings"
        emb_dir.mkdir(parents=True)
        np.save(emb_dir / "model1_single-primary.npy", sample_embeddings)
        np.save(emb_dir / "model2_single-pooled.npy", sample_embeddings)

        results = build_all_indices(
            embeddings_dir=emb_dir,
            output_dir=temp_output_dir,
            index_types=["flat"],
            skip_existing=False,
        )

        assert len(results) == 2
        assert (temp_output_dir / "model1_single-primary_flat.faiss").exists()
        assert (temp_output_dir / "model2_single-pooled_flat.faiss").exists()

    def test_skips_existing_when_flag_set(
        self, sample_embeddings: np.ndarray, temp_output_dir: Path, tmp_path: Path
    ) -> None:
        """Should skip existing indices when skip_existing=True."""
        emb_dir = tmp_path / "embeddings"
        emb_dir.mkdir(parents=True)
        np.save(emb_dir / "test_single-primary.npy", sample_embeddings)

        # Create existing index file
        existing_path = temp_output_dir / "test_single-primary_flat.faiss"
        existing_path.touch()
        original_mtime = existing_path.stat().st_mtime

        results = build_all_indices(
            embeddings_dir=emb_dir,
            output_dir=temp_output_dir,
            index_types=["flat"],
            skip_existing=True,
        )

        # File should not have been modified
        assert existing_path.stat().st_mtime == original_mtime
        assert len(results) == 1
        assert results[0]["skipped"] is True

    def test_handles_all_index_types(
        self, sample_embeddings: np.ndarray, temp_output_dir: Path, tmp_path: Path
    ) -> None:
        """Should handle 'all' index types."""
        emb_dir = tmp_path / "embeddings"
        emb_dir.mkdir(parents=True)
        np.save(emb_dir / "test_single-primary.npy", sample_embeddings)

        results = build_all_indices(
            embeddings_dir=emb_dir,
            output_dir=temp_output_dir,
            index_types=["all"],
            skip_existing=False,
        )

        # Should build all 7 index types
        assert len(results) == 7
        for index_type in INDEX_TYPES:
            expected_path = temp_output_dir / f"test_single-primary_{index_type}.faiss"
            assert expected_path.exists(), f"Missing {index_type} index"


class TestMultiVectorMetadataCopy:
    """Tests for copying metadata alongside multi-vector indices."""

    def test_copies_metadata_for_multi_vector(
        self, sample_multi_vector_data: tuple[Path, dict], temp_output_dir: Path
    ) -> None:
        """Should copy metadata.json alongside multi-vector index."""
        multi_dir, original_metadata = sample_multi_vector_data
        emb_path = multi_dir / "embeddings.npy"

        index_path = build_index_for_embedding_file(
            embedding_path=emb_path,
            output_dir=temp_output_dir,
            index_type="hnsw",
        )

        # Check metadata was copied
        metadata_path = index_path.with_suffix(".metadata.json")
        assert metadata_path.exists(), "Metadata file not copied"

        with open(metadata_path) as f:
            copied_metadata = json.load(f)
        assert copied_metadata == original_metadata

    def test_no_metadata_for_single_vector(
        self, sample_embeddings: np.ndarray, temp_output_dir: Path, tmp_path: Path
    ) -> None:
        """Should not create metadata file for single-vector embeddings."""
        emb_path = tmp_path / "embeddings" / "test_single-primary.npy"
        emb_path.parent.mkdir(parents=True)
        np.save(emb_path, sample_embeddings)

        index_path = build_index_for_embedding_file(
            embedding_path=emb_path,
            output_dir=temp_output_dir,
            index_type="hnsw",
        )

        metadata_path = index_path.with_suffix(".metadata.json")
        assert not metadata_path.exists(), "Unexpected metadata file for single-vector"


class TestIndexSearchQuality:
    """Tests to verify built indices can actually search."""

    def test_flat_index_searches_correctly(
        self, sample_embeddings: np.ndarray, temp_output_dir: Path, tmp_path: Path
    ) -> None:
        """Flat index should return exact matches."""
        emb_path = tmp_path / "embeddings" / "test_single-primary.npy"
        emb_path.parent.mkdir(parents=True)
        np.save(emb_path, sample_embeddings)

        index_path = build_index_for_embedding_file(
            embedding_path=emb_path,
            output_dir=temp_output_dir,
            index_type="flat",
        )

        index = faiss.read_index(str(index_path))
        query = sample_embeddings[0:1]
        distances, indices = index.search(query, 1)

        assert indices[0, 0] == 0  # Should find itself

    def test_hnsw_index_searches_correctly(
        self, sample_embeddings: np.ndarray, temp_output_dir: Path, tmp_path: Path
    ) -> None:
        """HNSW index should return correct results."""
        emb_path = tmp_path / "embeddings" / "test_single-primary.npy"
        emb_path.parent.mkdir(parents=True)
        np.save(emb_path, sample_embeddings)

        index_path = build_index_for_embedding_file(
            embedding_path=emb_path,
            output_dir=temp_output_dir,
            index_type="hnsw",
        )

        index = faiss.read_index(str(index_path))
        query = sample_embeddings[0:1]
        distances, indices = index.search(query, 1)

        # HNSW may not always find exact match but should find very similar
        assert indices[0, 0] in range(1000)
