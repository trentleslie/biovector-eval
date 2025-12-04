"""Vector quantization evaluation for compression/accuracy tradeoffs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import faiss
import numpy as np

from biovector_eval.core.metrics import SearchMetrics, evaluate_search_results

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Supported quantization types."""

    NONE = "none"
    SQ8 = "sq8"  # Scalar quantization, 8-bit
    SQ4 = "sq4"  # Scalar quantization, 4-bit
    PQ = "pq"  # Product quantization


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""

    qtype: QuantizationType
    pq_m: int = 32  # Number of subquantizers
    pq_nbits: int = 8  # Bits per subquantizer
    use_hnsw: bool = True
    hnsw_m: int = 32


@dataclass
class QuantizationResult:
    """Result of quantization evaluation."""

    config: QuantizationConfig
    accuracy: SearchMetrics
    compression_ratio: float
    original_size_mb: float
    compressed_size_mb: float
    accuracy_loss: float  # Relative to unquantized

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "quantization": self.config.qtype.value,
            "compression_ratio": self.compression_ratio,
            "original_size_mb": self.original_size_mb,
            "compressed_size_mb": self.compressed_size_mb,
            "accuracy_loss_pct": self.accuracy_loss * 100,
            "accuracy": self.accuracy.to_dict(),
        }


class QuantizationEvaluator:
    """Evaluate quantization strategies for vector indices."""

    def __init__(self, embeddings: np.ndarray, entity_ids: list[str]):
        """Initialize with embeddings.

        Args:
            embeddings: Float32 embeddings matrix (N x D).
            entity_ids: Corresponding entity IDs.
        """
        self.embeddings = embeddings.astype("float32").copy()
        self.entity_ids = entity_ids
        self.dimension = embeddings.shape[1]
        self.num_vectors = embeddings.shape[0]
        faiss.normalize_L2(self.embeddings)
        self._baseline_index: faiss.Index | None = None

    def build_index(self, config: QuantizationConfig) -> faiss.Index:
        """Build quantized index based on configuration."""
        logger.info(f"Building {config.qtype.value} index")

        if config.qtype == QuantizationType.NONE:
            index = faiss.IndexHNSWFlat(self.dimension, config.hnsw_m)
            index.hnsw.efConstruction = 200
            index.add(self.embeddings)

        elif config.qtype == QuantizationType.SQ8:
            index = faiss.IndexHNSWSQ(
                self.dimension, faiss.ScalarQuantizer.QT_8bit, config.hnsw_m
            )
            index.train(self.embeddings)
            index.add(self.embeddings)

        elif config.qtype == QuantizationType.SQ4:
            index = faiss.IndexHNSWSQ(
                self.dimension, faiss.ScalarQuantizer.QT_4bit, config.hnsw_m
            )
            index.train(self.embeddings)
            index.add(self.embeddings)

        elif config.qtype == QuantizationType.PQ:
            index = faiss.IndexHNSWPQ(self.dimension, config.pq_m, config.hnsw_m)
            index.train(self.embeddings)
            index.add(self.embeddings)

        else:
            raise ValueError(f"Unknown quantization type: {config.qtype}")

        return index

    def _calc_size(self, config: QuantizationConfig) -> float:
        """Calculate index size in MB."""
        n, d = self.num_vectors, self.dimension

        if config.qtype == QuantizationType.NONE:
            vector_bytes = n * d * 4
        elif config.qtype == QuantizationType.SQ8:
            vector_bytes = n * d
        elif config.qtype == QuantizationType.SQ4:
            vector_bytes = n * d * 0.5
        elif config.qtype == QuantizationType.PQ:
            vector_bytes = n * config.pq_m
        else:
            vector_bytes = n * d * 4

        graph_bytes = n * config.hnsw_m * 8 if config.use_hnsw else 0
        return (vector_bytes + graph_bytes) / 1024 / 1024

    def evaluate(
        self,
        config: QuantizationConfig,
        queries: list[dict],
        search_model: Any,
    ) -> QuantizationResult:
        """Evaluate a quantization configuration.

        Args:
            config: Quantization configuration.
            queries: Ground truth queries.
            search_model: Model for encoding queries.

        Returns:
            QuantizationResult with accuracy and compression metrics.
        """
        index = self.build_index(config)
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = 128

        def search_fn(query: str, k: int) -> list[str]:
            vec = search_model.encode([query], convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(vec)
            _, indices = index.search(vec, k)
            return [
                self.entity_ids[i] for i in indices[0] if 0 <= i < len(self.entity_ids)
            ]

        accuracy = evaluate_search_results(queries, search_fn, k=10)

        original_size = self._calc_size(
            QuantizationConfig(qtype=QuantizationType.NONE, hnsw_m=config.hnsw_m)
        )
        compressed_size = self._calc_size(config)

        # Baseline accuracy
        if self._baseline_index is None:
            self._baseline_index = self.build_index(
                QuantizationConfig(qtype=QuantizationType.NONE)
            )
            self._baseline_index.hnsw.efSearch = 128

        def baseline_search(query: str, k: int) -> list[str]:
            vec = search_model.encode([query], convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(vec)
            _, indices = self._baseline_index.search(vec, k)
            return [
                self.entity_ids[i] for i in indices[0] if 0 <= i < len(self.entity_ids)
            ]

        baseline_acc = evaluate_search_results(queries, baseline_search, k=10)
        loss = 0.0
        if baseline_acc.recall_at_5 > 0:
            loss = max(
                0,
                (baseline_acc.recall_at_5 - accuracy.recall_at_5)
                / baseline_acc.recall_at_5,
            )

        return QuantizationResult(
            config=config,
            accuracy=accuracy,
            compression_ratio=(
                original_size / compressed_size if compressed_size > 0 else 1.0
            ),
            original_size_mb=original_size,
            compressed_size_mb=compressed_size,
            accuracy_loss=loss,
        )

    def compare_all(
        self, queries: list[dict], search_model: Any
    ) -> list[QuantizationResult]:
        """Compare all standard quantization configurations."""
        configs = [
            QuantizationConfig(qtype=QuantizationType.NONE),
            QuantizationConfig(qtype=QuantizationType.SQ8),
            QuantizationConfig(qtype=QuantizationType.SQ4),
            QuantizationConfig(qtype=QuantizationType.PQ, pq_m=32),
        ]

        results = []
        for cfg in configs:
            result = self.evaluate(cfg, queries, search_model)
            results.append(result)
            logger.info(
                f"{cfg.qtype.value}: {result.compression_ratio:.1f}x, "
                f"loss={result.accuracy_loss * 100:.1f}%"
            )

        return results
