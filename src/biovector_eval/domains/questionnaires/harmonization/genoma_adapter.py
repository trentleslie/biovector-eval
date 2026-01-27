"""Adapter to use GenOMA's ontology mapping with biovector-eval entities.

GenOMA is an LLM-based ontology mapping agent that maps survey questions
to HPO (Human Phenotype Ontology) codes. This adapter bridges GenOMA's
LangGraph-based workflow with biovector-eval's BaseEntity format.

The adapter provides:
- Conversion between BaseEntity and GenOMA input format
- Caching to avoid repeated LLM calls
- Hybrid mode to skip LLM when vector similarity is confident
- Error handling with graceful degradation
- Conversion of results to CandidatePair format

Example:
    from biovector_eval.base.entity import BaseEntity
    from biovector_eval.domains.questionnaires.harmonization.genoma_adapter import (
        GenOMAAdapter,
    )

    adapter = GenOMAAdapter()
    question = BaseEntity(
        id="q1",
        name="Do you experience difficulty sleeping?",
        synonyms=[],
        metadata={"source": "Arivale"},
    )
    result = adapter.map_entity(question)
    print(f"HPO: {result.hpo_code} - {result.hpo_term}")
    print(f"Confidence: {result.confidence}")
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from biovector_eval.base.entity import BaseEntity
from biovector_eval.domains.questionnaires.harmonization.cache import GenOMACache

if TYPE_CHECKING:
    from biovector_eval.domains.questionnaires.harmonization.candidate_pair import (
        CandidatePair,
    )

logger = logging.getLogger(__name__)


def _find_project_root() -> Path:
    """Find project root by walking up to pyproject.toml.

    This is more robust than using a fixed parent index (e.g., parents[5])
    because it works regardless of how deeply the file is nested.

    Returns:
        Path to project root directory

    Raises:
        RuntimeError: If no pyproject.toml found in any parent directory
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")


@dataclass
class GenOMAResult:
    """Parsed result from GenOMA mapping.

    Attributes:
        hpo_code: Best matching HPO code (e.g., "HP:0000739")
        hpo_term: Best matching term name (e.g., "Anxiety")
        confidence: Confidence score from 0.0-1.0
        extracted_terms: Medical terms extracted from input text
        raw_state: Full GenOMA state dict for debugging
        success: Whether mapping succeeded
        error_message: Error message if mapping failed
        error_type: Error category (timeout, rate_limit, no_match, api_error)
        timestamp: When this result was generated
        model_version: Which LLM model was used
        cached: Whether this result came from cache
    """

    hpo_code: str
    hpo_term: str
    confidence: float
    extracted_terms: list[str]
    raw_state: dict[str, Any]

    # Error tracking
    success: bool = True
    error_message: str = ""
    error_type: str = ""  # "timeout" | "rate_limit" | "no_match" | "api_error"

    # Provenance
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_version: str = "gpt-4"
    cached: bool = False

    @property
    def has_match(self) -> bool:
        """Check if a valid HPO code was found."""
        return bool(self.hpo_code) and self.success


class GenOMAAdapter:
    """Bridge between biovector-eval BaseEntity and GenOMA's mapping graph.

    This adapter handles:
    1. Converting BaseEntity objects to GenOMA input format
    2. Invoking GenOMA's LangGraph workflow
    3. Caching results to avoid repeated LLM calls
    4. Parsing GenOMA output into structured results
    5. Converting results to CandidatePair format for harmonization

    Args:
        default_field_type: Default field type when not inferable ("radio", "checkbox", "short")
        use_cache: Enable result caching
        cache_dir: Directory for cache files
        hybrid_mode: Skip LLM if vector similarity exceeds threshold
        vector_confidence_threshold: Similarity threshold to skip LLM in hybrid mode
        model_version: Model version for cache invalidation

    Example:
        adapter = GenOMAAdapter()

        # Single entity mapping
        result = adapter.map_entity(question_entity)

        # Batch mapping
        results = adapter.map_entities_batch(questions)

        # Dry run (no API calls)
        preview = adapter.dry_run(question_entity)
    """

    def __init__(
        self,
        default_field_type: str = "radio",
        use_cache: bool = True,
        cache_dir: str | Path = "data/cache/genoma",
        hybrid_mode: bool = True,
        vector_confidence_threshold: float = 0.85,
        model_version: str = "gpt-4",
    ):
        self.default_field_type = default_field_type
        self.use_cache = use_cache
        self.hybrid_mode = hybrid_mode
        self.vector_confidence_threshold = vector_confidence_threshold
        self.model_version = model_version

        # Initialize cache if enabled
        self.cache: GenOMACache | None = None
        if use_cache:
            self.cache = GenOMACache(cache_dir=cache_dir, model_version=model_version)

        # Lazy-load GenOMA graph to avoid import errors when not needed
        self._genoma_graph: Any = None

    def _get_genoma_graph(self) -> Any:
        """Lazy-load the GenOMA mapping graph.

        Returns:
            Compiled LangGraph workflow

        Raises:
            ImportError: If GenOMA dependencies not installed
        """
        if self._genoma_graph is None:
            # Add GenOMA to path if needed
            project_root = _find_project_root()
            genoma_path = project_root / "external" / "genoma" / "src"
            if str(genoma_path) not in sys.path:
                sys.path.insert(0, str(genoma_path))

            try:
                from graph.builder import umls_mapping_graph

                self._genoma_graph = umls_mapping_graph
            except ImportError as e:
                raise ImportError(
                    "GenOMA dependencies not installed. Install with:\n"
                    "  uv add langgraph langchain langchain-openai openai\n\n"
                    "Also ensure OPENAI_API_KEY is set in your environment."
                ) from e

        return self._genoma_graph

    def entity_to_genoma_input(
        self,
        entity: BaseEntity,
        field_type: str | None = None,
    ) -> dict[str, str]:
        """Convert BaseEntity to GenOMA input format.

        Args:
            entity: BaseEntity to convert
            field_type: Override field type (uses inference if None)

        Returns:
            Dict with 'text' and 'field_type' keys for GenOMA
        """
        return {
            "text": entity.name,
            "field_type": field_type or self._infer_field_type(entity),
        }

    def _infer_field_type(self, entity: BaseEntity) -> str:
        """Infer GenOMA field_type from entity metadata.

        Field types affect how GenOMA extracts medical terms:
        - "radio": Single-select questions (default)
        - "checkbox": Multi-select questions
        - "short": Free-text short answer

        Args:
            entity: Entity to infer type for

        Returns:
            Field type string
        """
        # Check if response_options metadata suggests checkbox/radio
        response_options = entity.metadata.get("response_options", [])
        if isinstance(response_options, list) and len(response_options) > 2:
            return "checkbox"
        elif len(entity.name.split()) < 5:
            return "short"
        return self.default_field_type

    def should_invoke_genoma(
        self,
        entity: BaseEntity,
        loinc_similarity: float | None = None,
    ) -> bool:
        """Determine if GenOMA call is needed.

        In hybrid mode, skips LLM call if vector similarity is high enough.
        This optimizes cost by only using LLM for ambiguous cases.

        Args:
            entity: Entity being mapped
            loinc_similarity: Vector similarity score from LOINC search (optional)

        Returns:
            True if GenOMA should be invoked
        """
        if not self.hybrid_mode:
            return True  # Always call in non-hybrid mode

        if loinc_similarity and loinc_similarity >= self.vector_confidence_threshold:
            return False  # LOINC match is confident enough

        return True

    def map_entity(
        self,
        entity: BaseEntity,
        timeout: float = 30.0,
        max_retries: int = 3,
        field_type: str | None = None,
    ) -> GenOMAResult:
        """Run GenOMA mapping on a single entity.

        Args:
            entity: BaseEntity to map
            timeout: Timeout for LLM call in seconds
            max_retries: Number of retry attempts on failure
            field_type: Override field type (uses inference if None)

        Returns:
            GenOMAResult with HPO mapping or error info
        """
        inferred_field_type = field_type or self._infer_field_type(entity)

        # Check cache first
        if self.cache:
            cached = self.cache.get(entity.name, inferred_field_type)
            if cached:
                result = self._parse_result(cached)
                result.cached = True
                return result

        # Get GenOMA graph (lazy-loaded)
        try:
            genoma_graph = self._get_genoma_graph()
        except ImportError as e:
            return GenOMAResult(
                hpo_code="",
                hpo_term="",
                confidence=0.0,
                extracted_terms=[],
                raw_state={},
                success=False,
                error_type="api_error",
                error_message=str(e),
            )

        # Call GenOMA with retries
        genoma_input = self.entity_to_genoma_input(entity, field_type)
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                raw_result = genoma_graph.invoke(genoma_input)

                # Cache successful result
                if self.cache:
                    self.cache.set(entity.name, inferred_field_type, dict(raw_result))

                return self._parse_result(dict(raw_result))

            except TimeoutError:
                logger.warning(f"GenOMA timeout (attempt {attempt + 1}/{max_retries})")
                last_error = TimeoutError(f"Timeout after {max_retries} attempts")
                continue

            except Exception as e:
                # Check for rate limiting (OpenAI-specific)
                error_str = str(e).lower()
                if "rate" in error_str and "limit" in error_str:
                    logger.warning(f"Rate limited: {e}")
                    return GenOMAResult(
                        hpo_code="",
                        hpo_term="",
                        confidence=0.0,
                        extracted_terms=[],
                        raw_state={},
                        success=False,
                        error_type="rate_limit",
                        error_message=str(e),
                    )

                logger.error(f"GenOMA error on attempt {attempt + 1}: {e}")
                last_error = e
                continue

        # All retries exhausted
        return GenOMAResult(
            hpo_code="",
            hpo_term="",
            confidence=0.0,
            extracted_terms=[],
            raw_state={},
            success=False,
            error_type="timeout" if isinstance(last_error, TimeoutError) else "api_error",
            error_message=str(last_error) if last_error else "Unknown error",
        )

    def map_entities_batch(
        self,
        entities: list[BaseEntity],
        batch_size: int = 10,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[GenOMAResult]:
        """Batch process entities through GenOMA (synchronous).

        Note: GenOMA's LangGraph invoke is synchronous. For true async,
        consider using concurrent.futures.ThreadPoolExecutor.

        Args:
            entities: List of BaseEntity objects to map
            batch_size: Not used (reserved for future async batching)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of GenOMAResult objects in same order as input
        """
        results = []
        total = len(entities)

        for i, entity in enumerate(entities):
            result = self.map_entity(entity)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def _parse_result(self, raw_result: dict[str, Any]) -> GenOMAResult:
        """Parse GenOMA output into structured result.

        GenOMA returns a MappingState dict with various fields depending
        on which workflow path was taken.

        Args:
            raw_result: Raw state dict from GenOMA

        Returns:
            Parsed GenOMAResult
        """
        # Check if question was mappable
        if not raw_result.get("is_mappable", True):
            return GenOMAResult(
                hpo_code="",
                hpo_term="",
                confidence=0.0,
                extracted_terms=[],
                raw_state=raw_result,
                success=True,  # Not an error, just not mappable
                error_type="no_match",
                error_message="Question not mappable to medical ontology",
            )

        # Extract best match from validated_mappings or direct fields
        validated = raw_result.get("validated_mappings", [])
        if validated and isinstance(validated, list) and len(validated) > 0:
            best = validated[0]
            hpo_code = best.get("best_match_code", "") or ""
            hpo_term = best.get("best_match_term", "") or ""
            confidence = best.get("confidence", 0.0)
        else:
            # Fallback to direct state fields
            hpo_code = raw_result.get("best_match_code", "") or ""
            hpo_term = raw_result.get("best_match_term", "") or ""
            confidence = raw_result.get("confidence", 0.0)

        extracted_terms = raw_result.get("extracted_terms", [])
        if not isinstance(extracted_terms, list):
            extracted_terms = []

        return GenOMAResult(
            hpo_code=hpo_code,
            hpo_term=hpo_term,
            confidence=float(confidence) if confidence else 0.0,
            extracted_terms=extracted_terms,
            raw_state=raw_result,
            model_version=self.model_version,
        )

    def to_candidate_pair(
        self,
        source_entity: BaseEntity,
        genoma_result: GenOMAResult,
    ) -> "CandidatePair":
        """Convert GenOMA result to CandidatePair format.

        This enables GenOMA results to be combined with vector similarity
        results in a unified CandidatePairDataset.

        Args:
            source_entity: Original BaseEntity that was mapped
            genoma_result: GenOMAResult from mapping

        Returns:
            CandidatePair with HPO mapping details
        """
        # Import here to avoid circular imports
        from biovector_eval.domains.questionnaires.harmonization.candidate_pair import (
            CandidatePair,
        )

        return CandidatePair(
            source_id=source_entity.id,
            source_text=source_entity.name,
            source_metadata=source_entity.metadata,
            target_id=genoma_result.hpo_code,
            target_text=genoma_result.hpo_term,
            target_metadata={"ontology": "HPO"},
            similarity_score=genoma_result.confidence,  # Use LLM confidence
            match_tier="primary",
            harmonization_pathway="q2hpo",  # HPO pathway via GenOMA
            confidence_level=self._confidence_to_level(genoma_result.confidence),
            hpo_code=genoma_result.hpo_code,
            # Extended LLM fields
            llm_confidence=genoma_result.confidence,
            semantic_reasoning=", ".join(genoma_result.extracted_terms),
            extracted_terms=genoma_result.extracted_terms,
            match_quality=self._determine_quality(genoma_result.confidence),
            llm_processed=True,
        )

    def _confidence_to_level(self, confidence: float) -> str:
        """Map confidence score to discrete level.

        Args:
            confidence: Score from 0.0 to 1.0

        Returns:
            "high", "medium", or "low"
        """
        if confidence >= 0.85:
            return "high"
        elif confidence >= 0.50:
            return "medium"
        return "low"

    def _determine_quality(self, confidence: float) -> str:
        """Determine match quality for review workflow.

        Args:
            confidence: Score from 0.0 to 1.0

        Returns:
            Quality classification for review status
        """
        if confidence >= 0.85:
            return "auto_accept"
        elif confidence >= 0.50:
            return "needs_review"
        return "rejected"

    def dry_run(self, entity: BaseEntity) -> dict[str, Any]:
        """Return what would be sent to GenOMA without making the call.

        Useful for testing and debugging without incurring API costs.

        Args:
            entity: Entity to preview mapping for

        Returns:
            Dict with input preview and cache status
        """
        field_type = self._infer_field_type(entity)
        cache_key = None
        cached_exists = False

        if self.cache:
            cache_key = self.cache._hash_key(entity.name, field_type)
            cached_exists = self.cache.has(entity.name, field_type)

        return {
            "input": self.entity_to_genoma_input(entity),
            "would_invoke": self.should_invoke_genoma(entity),
            "cache_key": cache_key,
            "cached_exists": cached_exists,
            "inferred_field_type": field_type,
            "entity_id": entity.id,
        }
