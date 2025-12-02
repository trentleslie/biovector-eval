"""Embedding model evaluation for metabolite search."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from biovector_eval.core.metrics import SearchMetrics, evaluate_search_results
from biovector_eval.core.performance import (
    LatencyMetrics,
    MemoryMetrics,
    measure_latency,
    measure_memory,
)
from biovector_eval.metabolites.ground_truth import GroundTruthDataset

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for an embedding model."""

    name: str
    model_id: str
    device: str = "cpu"
    batch_size: int = 512


@dataclass
class EvaluationResult:
    """Complete evaluation result for a model."""

    model_name: str
    model_id: str
    accuracy: SearchMetrics
    latency: LatencyMetrics
    memory: MemoryMetrics
    index_config: dict[str, Any]

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "model": {"name": self.model_name, "id": self.model_id},
            "accuracy": self.accuracy.to_dict(),
            "latency": self.latency.to_dict(),
            "memory": self.memory.to_dict(),
            "index_config": self.index_config,
        }

    def save(self, path: Path | str) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Model configurations for Phase 1
METABOLITE_MODELS: dict[str, ModelConfig] = {
    "baseline": ModelConfig(
        name="BGE-Small (Baseline)",
        model_id="BAAI/bge-small-en-v1.5",
    ),
    "bge-m3": ModelConfig(
        name="BGE-M3",
        model_id="BAAI/bge-m3",
    ),
    "chemberta-mtr": ModelConfig(
        name="ChemBERTa-MTR",
        model_id="DeepChem/ChemBERTa-77M-MTR",
    ),
}


class MetaboliteEvaluator:
    """Evaluate embedding models for metabolite search."""

    def __init__(
        self,
        texts: list[str],
        ids: list[str],
        ground_truth: GroundTruthDataset,
        device: str = "cpu",
    ):
        """Initialize evaluator.

        Args:
            texts: List of metabolite name strings to embed.
            ids: Corresponding HMDB IDs.
            ground_truth: Ground truth dataset for evaluation.
            device: Device for model inference ('cpu' or 'cuda').
        """
        self.texts = texts
        self.ids = ids
        self.ground_truth = ground_truth
        self.device = device
        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._embeddings: np.ndarray | None = None

    def load_model(self, config: ModelConfig) -> SentenceTransformer:
        """Load embedding model."""
        logger.info(f"Loading model: {config.model_id}")
        self._model = SentenceTransformer(
            config.model_id, device=config.device or self.device
        )
        return self._model

    def generate_embeddings(self, batch_size: int = 512) -> np.ndarray:
        """Generate embeddings for all metabolites."""
        if self._model is None:
            raise RuntimeError("Model not loaded")
        logger.info(f"Generating embeddings for {len(self.texts)} metabolites")
        embeddings = self._model.encode(
            self.texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        self._embeddings = embeddings.astype("float32")
        return self._embeddings

    def build_index(
        self,
        index_type: str = "hnsw",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
    ) -> faiss.Index:
        """Build FAISS index from embeddings."""
        if self._embeddings is None:
            raise RuntimeError("Embeddings not generated")

        dimension = self._embeddings.shape[1]
        logger.info(
            f"Building {index_type} index: {dimension}d, {len(self._embeddings)} vectors"
        )

        if index_type == "flat":
            index = faiss.IndexFlatIP(dimension)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, hnsw_m)
            index.hnsw.efConstruction = hnsw_ef_construction
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Normalize for cosine similarity
        embeddings = self._embeddings.copy()
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        self._index = index
        return index

    def search(self, query: str, k: int = 10) -> list[str]:
        """Search for similar metabolites."""
        if self._model is None or self._index is None:
            raise RuntimeError("Model and index must be loaded")

        query_vec = self._model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)
        _, indices = self._index.search(query_vec, k)
        return [self.ids[i] for i in indices[0] if 0 <= i < len(self.ids)]

    def evaluate(
        self,
        config: ModelConfig,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 128,
    ) -> EvaluationResult:
        """Run complete evaluation for a model.

        Args:
            config: Model configuration.
            hnsw_m: HNSW M parameter.
            hnsw_ef_construction: HNSW efConstruction parameter.
            hnsw_ef_search: HNSW efSearch parameter for queries.

        Returns:
            EvaluationResult with all metrics.
        """
        logger.info(f"Evaluating: {config.name}")

        # Load and embed
        self.load_model(config)
        self.generate_embeddings(config.batch_size)

        # Build index
        index = self.build_index("hnsw", hnsw_m, hnsw_ef_construction)
        index.hnsw.efSearch = hnsw_ef_search

        # Evaluate accuracy
        queries = [q.to_dict() for q in self.ground_truth.queries]
        accuracy = evaluate_search_results(queries, self.search, k=10)

        # Evaluate latency
        query_texts = [q.query for q in self.ground_truth.queries]
        latency = measure_latency(self.search, query_texts, k=10, warmup=20)

        # Evaluate memory
        memory = measure_memory(
            len(self._embeddings), self._embeddings.shape[1], hnsw_m=hnsw_m
        )

        return EvaluationResult(
            model_name=config.name,
            model_id=config.model_id,
            accuracy=accuracy,
            latency=latency,
            memory=memory,
            index_config={
                "type": "hnsw",
                "hnsw_m": hnsw_m,
                "hnsw_ef_construction": hnsw_ef_construction,
                "hnsw_ef_search": hnsw_ef_search,
            },
        )

    def compare_models(
        self,
        models: list[ModelConfig],
        output_dir: Path | str,
    ) -> dict[str, EvaluationResult]:
        """Compare multiple models and save results.

        Args:
            models: List of model configurations to compare.
            output_dir: Directory to save results.

        Returns:
            Dictionary mapping model names to results.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, EvaluationResult] = {}
        for config in models:
            result = self.evaluate(config)
            results[config.name] = result
            result.save(output_dir / f"{config.name.lower().replace(' ', '_')}.json")

        # Save summary
        self._save_summary(results, output_dir / "comparison_summary.json")
        return results

    def _save_summary(
        self, results: dict[str, EvaluationResult], path: Path
    ) -> None:
        """Save comparison summary."""
        baseline = results.get("BGE-Small (Baseline)")
        comparison = {}

        for name, result in results.items():
            entry = result.to_dict()
            if baseline and name != "BGE-Small (Baseline)":
                entry["vs_baseline"] = {
                    "recall@1_delta": result.accuracy.recall_at_1
                    - baseline.accuracy.recall_at_1,
                    "recall@5_delta": result.accuracy.recall_at_5
                    - baseline.accuracy.recall_at_5,
                    "mrr_delta": result.accuracy.mrr - baseline.accuracy.mrr,
                    "latency_ratio": result.latency.p95_ms / baseline.latency.p95_ms,
                }
            comparison[name] = entry

        with open(path, "w") as f:
            json.dump(comparison, f, indent=2)
