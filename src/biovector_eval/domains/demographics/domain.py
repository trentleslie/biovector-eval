"""Demographics domain implementation.

Implements the Domain protocol for demographic variables, supporting
data consolidation from multiple cohort sources (Arivale, Israeli10K, UKBB).
"""

from __future__ import annotations

import logging
from pathlib import Path

from biovector_eval.base.domain import DomainConfig
from biovector_eval.base.entity import BaseEntity
from biovector_eval.base.ground_truth import GroundTruthDataset, GroundTruthQuery
from biovector_eval.base.loaders import load_entities

logger = logging.getLogger(__name__)


# Embedding models suitable for demographic variable text
# These are general-purpose medical/scientific models that work well
# with variable descriptions and demographic terminology.
DEMOGRAPHICS_MODELS = {
    "sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}


class DemographicsDomain:
    """Domain implementation for demographic variables.

    Handles demographic data from multiple cohort sources (Arivale,
    Israeli10K, UK Biobank) for cross-cohort variable harmonization.

    The domain supports:
    - Loading consolidated demographics from JSON
    - Multi-source provenance tracking via metadata
    - Ground truth generation for harmonization evaluation
    - Biomedical embedding models for demographic text
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        entities_file: str = "demographics.json",
    ):
        """Initialize demographics domain.

        Args:
            data_dir: Base directory for demographics data.
                     Defaults to data/demographics/ in project root.
            entities_file: Name of consolidated demographics file.
        """
        if data_dir is None:
            # Default to data/demographics/ relative to project root
            current = Path.cwd()
            while current != current.parent:
                if (current / "pyproject.toml").exists():
                    data_dir = current / "data" / "demographics"
                    break
                current = current.parent
            else:
                data_dir = Path("data/demographics")

        self._data_dir = Path(data_dir)
        self._config = DomainConfig(
            name="demographics",
            data_dir=self._data_dir,
            entities_file=entities_file,
            ground_truth_file="ground_truth.json",
            id_column="id",
            name_column="name",
            synonym_columns=[],
            file_format="json",
        )

    @property
    def name(self) -> str:
        """Domain identifier."""
        return "demographics"

    @property
    def config(self) -> DomainConfig:
        """Domain configuration."""
        return self._config

    @property
    def processed_dir(self) -> Path:
        """Directory for processed data files."""
        return self._data_dir / "processed"

    @property
    def raw_dir(self) -> Path:
        """Directory for raw data files."""
        return self._data_dir / "raw"

    def load_entities(self, path: Path | str | None = None) -> list[BaseEntity]:
        """Load demographic entities from file.

        Args:
            path: Optional override path. Uses config default if not provided.

        Returns:
            List of BaseEntity objects representing demographic variables.
        """
        if path is None:
            # Try processed directory first
            processed_path = self.processed_dir / self._config.entities_file
            if processed_path.exists():
                path = processed_path
            else:
                # Fall back to default path
                path = self._config.entities_path

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Demographics file not found at {path}. "
                "Please run scripts/consolidate_demographics.py first."
            )

        return load_entities(path)

    def load_entities_by_source(
        self,
        source: str,
        path: Path | str | None = None,
    ) -> list[BaseEntity]:
        """Load entities filtered by source cohort.

        Args:
            source: Source cohort name (e.g., "Arivale", "Israeli10K", "UKBB").
            path: Optional override path for entities file.

        Returns:
            List of BaseEntity objects from the specified source.
        """
        all_entities = self.load_entities(path)
        return [
            e for e in all_entities if e.metadata.get("source_questionnaire") == source
        ]

    def get_sources(self, path: Path | str | None = None) -> list[str]:
        """Get list of unique source cohorts in the dataset.

        Args:
            path: Optional override path for entities file.

        Returns:
            Sorted list of source cohort names.
        """
        all_entities = self.load_entities(path)
        sources = {
            e.metadata.get("source_questionnaire", "unknown") for e in all_entities
        }
        return sorted(sources)

    def generate_ground_truth(
        self,
        entities: list,
        **kwargs,
    ) -> GroundTruthDataset:
        """Generate ground truth dataset for demographics.

        For demographic harmonization, ground truth can be generated from:
        - Exact text matches (same variable in different cohorts)
        - Near-duplicate detection within cohorts
        - Known harmonization mappings (if available)

        Args:
            entities: List of demographic entities.
            **kwargs: Generation parameters:
                - seed: Random seed (default: 42)
                - exact: Number of exact match queries
                - synonym: Number of synonym-based queries

        Returns:
            GroundTruthDataset with queries for evaluation.
        """
        import random

        seed = kwargs.get("seed", 42)
        exact_count = kwargs.get("exact", 50)
        synonym_count = kwargs.get("synonym", 25)

        random.seed(seed)

        # Convert to BaseEntity if needed
        if entities and not isinstance(entities[0], BaseEntity):
            entities = [BaseEntity.from_dict(e) for e in entities]

        queries: list[GroundTruthQuery] = []

        # Generate exact match queries (using variable description)
        entities_with_names = [e for e in entities if e.name]
        if entities_with_names:
            sampled = random.sample(
                entities_with_names, min(exact_count, len(entities_with_names))
            )
            for entity in sampled:
                queries.append(
                    GroundTruthQuery(
                        query=entity.name,
                        expected=[entity.id],
                        category="exact_match",
                        notes=f"Source: {entity.metadata.get('source_questionnaire', 'unknown')}",
                    )
                )

        # Generate synonym-based queries
        entities_with_synonyms = [e for e in entities if e.synonyms]
        if entities_with_synonyms:
            sampled = random.sample(
                entities_with_synonyms,
                min(synonym_count, len(entities_with_synonyms)),
            )
            for entity in sampled:
                synonym = random.choice(entity.synonyms)
                queries.append(
                    GroundTruthQuery(
                        query=synonym,
                        expected=[entity.id],
                        category="synonym_match",
                        notes=f"Original: {entity.name}",
                    )
                )

        return GroundTruthDataset(queries=queries)

    def get_embedding_models(self) -> dict[str, str]:
        """Get recommended embedding models for demographics.

        Returns:
            Dict mapping model short names to HuggingFace model IDs.
        """
        return DEMOGRAPHICS_MODELS.copy()
