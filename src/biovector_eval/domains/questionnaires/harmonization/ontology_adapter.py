"""Base protocol for ontology adapters (GenOMA, MONDO, etc.).

This module defines the common interface that ontology mapping adapters
must implement, enabling consistent handling of different ontologies
(HPO via GenOMA, MONDO, etc.) in the harmonization pipeline.

The protocol ensures:
- Consistent method signatures across adapters
- Uniform result formatting for CandidatePair conversion
- Graceful error handling for missing resources
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from biovector_eval.base.entity import BaseEntity
    from biovector_eval.domains.questionnaires.harmonization.candidate_pair import (
        CandidatePair,
    )


@dataclass
class OntologyResult:
    """Base result class for ontology mapping operations.

    All ontology adapters should return results that can be converted
    to or extend this base class for consistent handling.

    Attributes:
        code: Ontology code (e.g., "HP:0000739", "MONDO:0005015").
        term: Human-readable term name (e.g., "Anxiety", "diabetes mellitus").
        confidence: Confidence score from 0.0-1.0.
        extracted_terms: Medical terms extracted from input text.
        ontology_name: Name of the ontology (e.g., "HPO", "MONDO").
        success: Whether mapping succeeded.
        cached: Whether this result came from cache.
        error_message: Error message if mapping failed.
        error_type: Error category for programmatic handling.
        timestamp: When this result was generated.
    """

    code: str
    term: str
    confidence: float
    extracted_terms: list[str] = field(default_factory=list)
    ontology_name: str = "unknown"

    # Status
    success: bool = True
    cached: bool = False

    # Error tracking
    error_message: str = ""
    error_type: str = ""  # "no_embeddings", "no_match", "api_error", "timeout"

    # Provenance
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def has_match(self) -> bool:
        """Check if a valid ontology code was found."""
        return bool(self.code) and self.success


@runtime_checkable
class OntologyAdapter(Protocol):
    """Protocol for ontology mapping adapters.

    All ontology adapters (GenOMA for HPO, MONDO adapter, etc.) should
    implement this interface to ensure consistent behavior across the
    harmonization pipeline.

    The key design principle is graceful degradation: adapters should
    return informative error results rather than raising exceptions
    when resources (embeddings, indices, API keys) are unavailable.
    """

    def map_entity(
        self,
        entity: "BaseEntity",
        timeout: float = 30.0,
    ) -> OntologyResult:
        """Map an entity to an ontology code.

        Args:
            entity: BaseEntity to map.
            timeout: Timeout for mapping operation in seconds.

        Returns:
            OntologyResult with mapping or error information.

        Note:
            Implementations should NOT raise exceptions for missing
            resources. Instead, return an OntologyResult with
            success=False and appropriate error_type/error_message.
        """
        ...

    def to_candidate_pair(
        self,
        source_entity: "BaseEntity",
        result: OntologyResult,
    ) -> "CandidatePair":
        """Convert ontology result to CandidatePair format.

        Args:
            source_entity: Original entity that was mapped.
            result: OntologyResult from mapping operation.

        Returns:
            CandidatePair with ontology mapping details.
        """
        ...


def confidence_to_level(confidence: float) -> str:
    """Map confidence score to discrete level.

    Standard confidence thresholds used across all ontology adapters.

    Args:
        confidence: Score from 0.0 to 1.0.

    Returns:
        "high", "medium", or "low".
    """
    if confidence >= 0.85:
        return "high"
    elif confidence >= 0.50:
        return "medium"
    return "low"


def determine_match_quality(confidence: float) -> str:
    """Determine match quality for review workflow.

    Standard quality classification used across all ontology adapters.

    Args:
        confidence: Score from 0.0 to 1.0.

    Returns:
        Quality classification for review status.
    """
    if confidence >= 0.85:
        return "auto_accept"
    elif confidence >= 0.50:
        return "needs_review"
    return "rejected"


__all__ = [
    "OntologyAdapter",
    "OntologyResult",
    "confidence_to_level",
    "determine_match_quality",
]
