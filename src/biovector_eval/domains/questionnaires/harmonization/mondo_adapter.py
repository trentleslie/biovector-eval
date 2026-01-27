"""MONDO (Monarch Disease Ontology) mapping adapter.

Maps survey questions to MONDO disease codes using a hybrid approach:
1. Vector similarity search against MONDO embeddings (primary)
2. LLM fallback for ambiguous cases (optional)

MONDO complements HPO (phenotypes) by providing disease-focused mapping,
useful for questions that ask about disease diagnoses rather than symptoms.

Example:
    from biovector_eval.base.entity import BaseEntity
    from biovector_eval.domains.questionnaires.harmonization.mondo_adapter import (
        MONDOAdapter,
    )

    adapter = MONDOAdapter()
    question = BaseEntity(
        id="q1",
        name="Have you been diagnosed with diabetes?",
        synonyms=[],
        metadata={"source": "Arivale"},
    )
    result = adapter.map_entity(question)
    if result.success:
        print(f"MONDO: {result.mondo_code} - {result.mondo_term}")
    else:
        print(f"Error: {result.error_type} - {result.error_message}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from biovector_eval.base.entity import BaseEntity
from biovector_eval.domains.questionnaires.harmonization.cache import GenOMACache
from biovector_eval.domains.questionnaires.harmonization.ontology_adapter import (
    OntologyResult,
    confidence_to_level,
    determine_match_quality,
)

if TYPE_CHECKING:
    from biovector_eval.domains.questionnaires.harmonization.candidate_pair import (
        CandidatePair,
    )

logger = logging.getLogger(__name__)


@dataclass
class MONDOResult(OntologyResult):
    """Result from MONDO mapping operation.

    Extends OntologyResult with MONDO-specific attributes and
    provides convenience properties for MONDO code/term access.

    Attributes:
        mondo_code: MONDO code (e.g., "MONDO:0005015").
        mondo_term: MONDO term name (e.g., "diabetes mellitus").
        raw_state: Full mapping state dict for debugging.
        model_version: Which embedding model was used.
    """

    # MONDO-specific aliases (stored in base class code/term fields)
    raw_state: dict[str, Any] = field(default_factory=dict)
    model_version: str = "unknown"

    def __post_init__(self) -> None:
        """Set ontology name after init."""
        self.ontology_name = "MONDO"

    @property
    def mondo_code(self) -> str:
        """MONDO code (alias for code)."""
        return self.code

    @property
    def mondo_term(self) -> str:
        """MONDO term name (alias for term)."""
        return self.term


class MONDOAdapter:
    """Hybrid MONDO mapping: vector search + optional LLM refinement.

    This adapter maps survey questions to MONDO disease codes using
    vector similarity search against pre-built MONDO embeddings.
    When embeddings are unavailable, it gracefully degrades to
    returning informative error results.

    The adapter supports:
    - Vector similarity search (requires MONDO index)
    - Result caching to avoid repeated searches
    - Confidence-based filtering
    - LLM fallback for ambiguous cases (optional, future)

    Args:
        mondo_index_path: Path to MONDO FAISS index. If None, vector
            search is disabled and errors are returned gracefully.
        mondo_entities_path: Path to MONDO entities JSON file.
        use_llm_fallback: Enable LLM for ambiguous cases (future).
        use_cache: Enable result caching.
        cache_dir: Directory for cache files.
        min_confidence: Minimum confidence threshold for matches.
        model_version: Model version for cache invalidation.

    Example:
        # Without embeddings (returns graceful error)
        adapter = MONDOAdapter(mondo_index_path=None)
        result = adapter.map_entity(entity)
        print(result.error_type)  # "no_embeddings"

        # With embeddings
        adapter = MONDOAdapter(
            mondo_index_path=Path("data/mondo/mondo.faiss"),
            mondo_entities_path=Path("data/mondo/mondo_entities.json"),
        )
        result = adapter.map_entity(entity)
    """

    def __init__(
        self,
        mondo_index_path: Path | str | None = None,
        mondo_entities_path: Path | str | None = None,
        use_llm_fallback: bool = False,
        use_cache: bool = True,
        cache_dir: str | Path = "data/cache/mondo",
        min_confidence: float = 0.5,
        model_version: str = "sapbert",
    ):
        self.mondo_index_path = Path(mondo_index_path) if mondo_index_path else None
        self.mondo_entities_path = (
            Path(mondo_entities_path) if mondo_entities_path else None
        )
        self.use_llm_fallback = use_llm_fallback
        self.min_confidence = min_confidence
        self.model_version = model_version

        # Initialize cache if enabled
        self.cache: GenOMACache | None = None
        if use_cache:
            self.cache = GenOMACache(
                cache_dir=cache_dir,
                model_version=model_version,
            )

        # Lazy-loaded resources
        self._index: Any = None  # FAISS index
        self._entities: list[BaseEntity] | None = None
        self._entity_lookup: dict[str, BaseEntity] | None = None
        self._embedding_model: Any = None

    def _load_resources(self) -> bool:
        """Lazy-load MONDO index and entities.

        Returns:
            True if resources loaded successfully, False otherwise.
        """
        if self._index is not None:
            return True  # Already loaded

        if self.mondo_index_path is None or not self.mondo_index_path.exists():
            return False

        if self.mondo_entities_path is None or not self.mondo_entities_path.exists():
            return False

        try:
            import faiss

            from biovector_eval.base.loaders import load_entities

            # Load FAISS index
            self._index = faiss.read_index(str(self.mondo_index_path))

            # Load entities
            self._entities = load_entities(self.mondo_entities_path)
            self._entity_lookup = {e.id: e for e in self._entities}

            logger.info(
                f"Loaded MONDO index with {self._index.ntotal} vectors "
                f"and {len(self._entities)} entities"
            )
            return True

        except ImportError:
            logger.warning("FAISS not installed. MONDO vector search unavailable.")
            return False
        except Exception as e:
            logger.error(f"Failed to load MONDO resources: {e}")
            return False

    def _get_embedding_model(self) -> Any:
        """Lazy-load embedding model.

        Returns:
            SentenceTransformer model or None if not available.
        """
        if self._embedding_model is not None:
            return self._embedding_model

        try:
            from sentence_transformers import SentenceTransformer

            # Default to SapBERT for biomedical text
            model_id = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
            self._embedding_model = SentenceTransformer(model_id)
            return self._embedding_model
        except ImportError:
            logger.warning("sentence-transformers not installed")
            return None

    def map_entity(
        self,
        entity: BaseEntity,
        timeout: float = 30.0,
        top_k: int = 5,
    ) -> MONDOResult:
        """Map an entity to a MONDO disease code.

        Args:
            entity: BaseEntity to map.
            timeout: Timeout for mapping operation (currently unused).
            top_k: Number of candidates to retrieve from vector search.

        Returns:
            MONDOResult with MONDO mapping or error information.

        Note:
            Returns graceful error results when embeddings/index are
            unavailable rather than raising exceptions.
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get(entity.name, "mondo")
            if cached:
                return self._parse_cached_result(cached)

        # Check if resources are available
        if not self._load_resources():
            return MONDOResult(
                code="",
                term="",
                confidence=0.0,
                extracted_terms=[],
                success=False,
                error_type="no_embeddings",
                error_message=(
                    "MONDO embeddings not available. "
                    "Run scripts/generate_mondo_embeddings.py to create them."
                ),
                model_version=self.model_version,
            )

        # Type narrowing: _load_resources() sets these to non-None on success
        assert self._entities is not None
        entities = self._entities

        # Get embedding model
        model = self._get_embedding_model()
        if model is None:
            return MONDOResult(
                code="",
                term="",
                confidence=0.0,
                extracted_terms=[],
                success=False,
                error_type="no_embeddings",
                error_message="Embedding model not available. Install sentence-transformers.",
                model_version=self.model_version,
            )

        try:
            import numpy as np

            # Embed query
            query_embedding = model.encode([entity.name], normalize_embeddings=True)
            query_embedding = np.array(query_embedding, dtype=np.float32)

            # Search index
            scores, indices = self._index.search(query_embedding, top_k)

            # Process results
            if len(indices[0]) == 0 or indices[0][0] == -1:
                return MONDOResult(
                    code="",
                    term="",
                    confidence=0.0,
                    extracted_terms=[],
                    success=True,  # Not an error, just no match
                    error_type="no_match",
                    error_message="No MONDO match found above threshold",
                    model_version=self.model_version,
                )

            # Get best match
            best_idx = indices[0][0]
            best_score = float(scores[0][0])

            if best_idx >= len(entities):
                return MONDOResult(
                    code="",
                    term="",
                    confidence=0.0,
                    extracted_terms=[],
                    success=False,
                    error_type="api_error",
                    error_message=f"Index mismatch: index {best_idx} exceeds entity count",
                    model_version=self.model_version,
                )

            best_entity = entities[best_idx]

            # Check confidence threshold
            if best_score < self.min_confidence:
                return MONDOResult(
                    code="",
                    term="",
                    confidence=best_score,
                    extracted_terms=[],
                    success=True,
                    error_type="no_match",
                    error_message=f"Best match score {best_score:.3f} below threshold {self.min_confidence}",
                    model_version=self.model_version,
                )

            result = MONDOResult(
                code=best_entity.id,
                term=best_entity.name,
                confidence=best_score,
                extracted_terms=[],  # Could extract disease-related terms
                success=True,
                model_version=self.model_version,
                raw_state={
                    "top_k_results": [
                        {
                            "id": entities[idx].id if idx < len(entities) else "unknown",
                            "name": entities[idx].name if idx < len(entities) else "unknown",
                            "score": float(scores[0][i]),
                        }
                        for i, idx in enumerate(indices[0])
                        if idx != -1 and idx < len(entities)
                    ]
                },
            )

            # Cache result
            if self.cache:
                self.cache.set(entity.name, "mondo", result.raw_state)

            return result

        except Exception as e:
            logger.error(f"MONDO mapping failed for {entity.id}: {e}")
            return MONDOResult(
                code="",
                term="",
                confidence=0.0,
                extracted_terms=[],
                success=False,
                error_type="api_error",
                error_message=str(e),
                model_version=self.model_version,
            )

    def _parse_cached_result(self, cached: dict) -> MONDOResult:
        """Parse cached result into MONDOResult.

        Args:
            cached: Cached result dict.

        Returns:
            MONDOResult from cached data.
        """
        top_results = cached.get("top_k_results", [])
        if not top_results:
            return MONDOResult(
                code="",
                term="",
                confidence=0.0,
                extracted_terms=[],
                success=True,
                cached=True,
                error_type="no_match",
                error_message="No matches in cached result",
                model_version=self.model_version,
            )

        best = top_results[0]
        return MONDOResult(
            code=best.get("id", ""),
            term=best.get("name", ""),
            confidence=best.get("score", 0.0),
            extracted_terms=[],
            success=True,
            cached=True,
            raw_state=cached,
            model_version=self.model_version,
        )

    def to_candidate_pair(
        self,
        source_entity: BaseEntity,
        result: MONDOResult,
    ) -> "CandidatePair":
        """Convert MONDO result to CandidatePair format.

        This enables MONDO results to be combined with other ontology
        mappings (LOINC, HPO) in a unified CandidatePairDataset.

        Args:
            source_entity: Original BaseEntity that was mapped.
            result: MONDOResult from mapping.

        Returns:
            CandidatePair with MONDO mapping details.
        """
        from biovector_eval.domains.questionnaires.harmonization.candidate_pair import (
            CandidatePair,
        )

        return CandidatePair(
            source_id=source_entity.id,
            source_text=source_entity.name,
            source_metadata=source_entity.metadata,
            target_id=result.mondo_code,
            target_text=result.mondo_term,
            target_metadata={"ontology": "MONDO"},
            similarity_score=result.confidence,
            match_tier="primary",
            harmonization_pathway="q2mondo",
            confidence_level=confidence_to_level(result.confidence),
            mondo_code=result.mondo_code,
            match_quality=determine_match_quality(result.confidence),
            llm_processed=False,  # Vector search, not LLM
        )

    def dry_run(self, entity: BaseEntity) -> dict[str, Any]:
        """Return what would happen without making the actual call.

        Useful for testing and debugging.

        Args:
            entity: Entity to preview mapping for.

        Returns:
            Dict with status information.
        """
        resources_available = self._load_resources()
        cache_key = None
        cached_exists = False

        if self.cache:
            cache_key = self.cache._hash_key(entity.name, "mondo")
            cached_exists = self.cache.has(entity.name, "mondo")

        return {
            "entity_id": entity.id,
            "entity_text": entity.name,
            "resources_available": resources_available,
            "mondo_index_path": str(self.mondo_index_path) if self.mondo_index_path else None,
            "mondo_entities_path": str(self.mondo_entities_path) if self.mondo_entities_path else None,
            "cache_key": cache_key,
            "cached_exists": cached_exists,
            "model_version": self.model_version,
        }


__all__ = [
    "MONDOAdapter",
    "MONDOResult",
]
