"""Domain protocol and configuration for multi-domain support.

This module defines the interface that all biological domains must implement,
enabling the framework to work with metabolites, demographics, questionnaires,
and future domains through a common API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from biovector_eval.base.entity import BaseEntity
    from biovector_eval.base.ground_truth import GroundTruthDataset


@dataclass
class DomainConfig:
    """Configuration for a biological domain.

    Defines paths, column mappings, and domain-specific settings
    that control how data is loaded and processed.

    Attributes:
        name: Domain identifier (e.g., "metabolites", "demographics").
        data_dir: Base directory for domain data.
        entities_file: Path to entities file (relative to data_dir).
        ground_truth_file: Path to ground truth file (relative to data_dir).
        id_column: Column name for entity ID in TSV files.
        name_column: Column name for entity name in TSV files.
        synonym_columns: Column names containing synonyms in TSV files.
        file_format: Data file format ("json" or "tsv").
        extra: Domain-specific extra configuration.
    """

    name: str
    data_dir: Path
    entities_file: str = "entities.json"
    ground_truth_file: str = "ground_truth.json"
    id_column: str = "id"
    name_column: str = "name"
    synonym_columns: list[str] = field(default_factory=list)
    file_format: str = "json"
    extra: dict = field(default_factory=dict)

    @property
    def entities_path(self) -> Path:
        """Full path to entities file."""
        return self.data_dir / self.entities_file

    @property
    def ground_truth_path(self) -> Path:
        """Full path to ground truth file."""
        return self.data_dir / self.ground_truth_file

    @property
    def embeddings_dir(self) -> Path:
        """Directory for domain embeddings."""
        return self.data_dir / "embeddings"

    @property
    def indices_dir(self) -> Path:
        """Directory for domain indices."""
        return self.data_dir / "indices"


@runtime_checkable
class Domain(Protocol):
    """Protocol defining the interface for biological domains.

    Each domain (metabolites, demographics, questionnaires) must implement
    this protocol to work with the evaluation framework.
    """

    @property
    def name(self) -> str:
        """Domain identifier (e.g., 'metabolites')."""
        ...

    @property
    def config(self) -> DomainConfig:
        """Domain configuration."""
        ...

    def load_entities(self, path: Path | None = None) -> list[BaseEntity]:
        """Load entities from file.

        Args:
            path: Optional override path. Uses config default if not provided.

        Returns:
            List of BaseEntity objects.
        """
        ...

    def generate_ground_truth(
        self,
        entities: list,
        **kwargs,
    ) -> GroundTruthDataset:
        """Generate ground truth dataset for this domain.

        Args:
            entities: List of entities to generate queries from.
                     Can be list[BaseEntity] or list[dict] for legacy support.
            **kwargs: Domain-specific generation parameters.

        Returns:
            GroundTruthDataset with queries for evaluation.
        """
        ...

    def get_embedding_models(self) -> dict[str, str]:
        """Get recommended embedding models for this domain.

        Returns:
            Dict mapping model short names to HuggingFace model IDs.
        """
        ...
