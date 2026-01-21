"""Base evaluator with shared evaluation logic.

Provides common functionality for evaluating embedding models
across different biological domains.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from biovector_eval.base.entity import BaseEntity
    from biovector_eval.base.ground_truth import GroundTruthDataset
    from biovector_eval.core.metrics import LatencyMetrics, SearchMetrics


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation.

    Attributes:
        model_name: Short name for the model.
        model_id: HuggingFace model identifier.
        device: Device for inference (cpu/cuda/auto).
        batch_size: Batch size for encoding.
        index_type: FAISS index type (flat/hnsw).
    """

    model_name: str
    model_id: str
    device: str = "auto"
    batch_size: int = 512
    index_type: str = "hnsw"


@dataclass
class EvaluationResult:
    """Results from model evaluation.

    Attributes:
        model_name: Name of the evaluated model.
        accuracy: Search accuracy metrics (recall, MRR).
        latency: Latency distribution metrics.
        memory: Memory usage metrics.
        index_config: FAISS index configuration used.
        extra: Additional domain-specific results.
    """

    model_name: str
    accuracy: SearchMetrics
    latency: LatencyMetrics
    memory: dict
    index_config: dict
    extra: dict | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "accuracy": {
                "recall_at_1": self.accuracy.recall_at_1,
                "recall_at_5": self.accuracy.recall_at_5,
                "recall_at_10": self.accuracy.recall_at_10,
                "mrr": self.accuracy.mrr,
            },
            "latency": {
                "p50_ms": self.latency.p50_ms,
                "p95_ms": self.latency.p95_ms,
                "p99_ms": self.latency.p99_ms,
                "mean_ms": self.latency.mean_ms,
            },
            "memory": self.memory,
            "index_config": self.index_config,
            "extra": self.extra,
        }


class BaseEvaluator(ABC):
    """Abstract base class for domain evaluators.

    Provides common evaluation workflow while allowing domains
    to customize entity handling, embedding generation, and
    result interpretation.
    """

    def __init__(
        self,
        entities: list[BaseEntity],
        ground_truth: GroundTruthDataset,
        device: str = "auto",
    ):
        """Initialize evaluator.

        Args:
            entities: List of entities to evaluate.
            ground_truth: Ground truth dataset for evaluation.
            device: Device for model inference.
        """
        self.entities = entities
        self.ground_truth = ground_truth
        self.device = self._resolve_device(device)

        # Cache for loaded models and embeddings
        self._model: SentenceTransformer | None = None
        self._embeddings: np.ndarray | None = None
        self._index: Any = None

    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' device to actual device."""
        if device == "auto":
            from biovector_eval.utils.device import get_best_device

            return get_best_device()
        return device

    @abstractmethod
    def get_texts_for_embedding(self) -> list[str]:
        """Get texts to embed for each entity.

        Subclasses implement domain-specific text extraction.
        For metabolites, this is the name + synonyms.
        For demographics, this might be variable labels.

        Returns:
            List of texts to embed.
        """
        ...

    @abstractmethod
    def get_entity_ids(self) -> list[str]:
        """Get entity IDs in same order as texts.

        Returns:
            List of entity IDs.
        """
        ...

    def load_model(self, config: EvaluationConfig) -> SentenceTransformer:
        """Load embedding model.

        Args:
            config: Model configuration.

        Returns:
            Loaded SentenceTransformer model.
        """
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(config.model_id, device=self.device)
        return self._model

    def generate_embeddings(
        self,
        model: SentenceTransformer,
        batch_size: int = 512,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for all entity texts.

        Args:
            model: SentenceTransformer model.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.

        Returns:
            Numpy array of embeddings.
        """
        import numpy as np

        texts = self.get_texts_for_embedding()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        self._embeddings = np.array(embeddings)
        return self._embeddings

    def build_index(
        self,
        embeddings: np.ndarray,
        index_type: str = "hnsw",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
    ) -> Any:
        """Build FAISS index from embeddings.

        Args:
            embeddings: Embedding vectors.
            index_type: Index type (flat/hnsw).
            hnsw_m: HNSW M parameter.
            hnsw_ef_construction: HNSW ef_construction parameter.

        Returns:
            FAISS index.
        """
        import faiss

        dimension = embeddings.shape[1]

        if index_type == "flat":
            index = faiss.IndexFlatIP(dimension)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, hnsw_m)
            index.hnsw.efConstruction = hnsw_ef_construction
            index.hnsw.efSearch = 128
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        index.add(embeddings.astype("float32"))
        self._index = index
        return index

    def evaluate(self, config: EvaluationConfig) -> EvaluationResult:
        """Run full evaluation for a model.

        Args:
            config: Evaluation configuration.

        Returns:
            EvaluationResult with accuracy, latency, and memory metrics.
        """
        from biovector_eval.core.metrics import (
            LatencyMetrics,
            SearchMetrics,
            evaluate_search_results,
            measure_latency,
        )

        # Load model and generate embeddings
        model = self.load_model(config)
        embeddings = self.generate_embeddings(model, config.batch_size)

        # Build index
        index = self.build_index(embeddings, config.index_type)
        entity_ids = self.get_entity_ids()

        # Define search function
        def search_fn(query: str, k: int) -> list[str]:
            query_emb = model.encode([query], normalize_embeddings=True)
            distances, indices = index.search(query_emb.astype("float32"), k)
            return [entity_ids[i] for i in indices[0] if i < len(entity_ids)]

        # Evaluate accuracy
        queries = [q.to_dict() for q in self.ground_truth.queries]
        accuracy = evaluate_search_results(queries, search_fn, k=10)

        # Measure latency
        query_texts = [q.query for q in self.ground_truth.queries[:100]]
        latency = measure_latency(search_fn, query_texts, k=10, warmup=10)

        # Memory estimation
        memory = {
            "embedding_bytes": embeddings.nbytes,
            "num_vectors": len(embeddings),
            "dimension": embeddings.shape[1],
        }

        return EvaluationResult(
            model_name=config.model_name,
            accuracy=accuracy,
            latency=latency,
            memory=memory,
            index_config={
                "type": config.index_type,
                "vectors": len(embeddings),
            },
        )

    def compare_models(
        self,
        configs: list[EvaluationConfig],
        output_dir: Path | str | None = None,
    ) -> dict[str, EvaluationResult]:
        """Compare multiple models.

        Args:
            configs: List of model configurations to evaluate.
            output_dir: Optional directory to save results.

        Returns:
            Dict mapping model names to results.
        """
        import json

        results = {}
        for config in configs:
            result = self.evaluate(config)
            results[config.model_name] = result

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            summary = {name: r.to_dict() for name, r in results.items()}
            with open(output_dir / "comparison_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

        return results
