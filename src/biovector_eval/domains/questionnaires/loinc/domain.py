"""LOINC domain implementation.

Implements the Domain protocol for LOINC (Logical Observation Identifiers
Names and Codes) entities for questionnaire harmonization.
"""

from __future__ import annotations

from pathlib import Path

from biovector_eval.base.domain import DomainConfig
from biovector_eval.base.entity import BaseEntity
from biovector_eval.base.ground_truth import GroundTruthDataset, GroundTruthQuery
from biovector_eval.base.loaders import load_entities
from biovector_eval.domains.questionnaires.loinc.parser import LOINCParser
from biovector_eval.domains.questionnaires.models import LOINC_MODELS


class LOINCDomain:
    """Domain implementation for LOINC codes.

    Handles LOINC data for questionnaire-to-LOINC mapping with support for:
    - LOINC Table Core CSV parsing
    - Multi-vector strategies using 6-part LOINC structure
    - Survey-specific code filtering
    - Consumer-friendly name integration
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        entities_file: str = "loinc_entities.json",
        survey_only: bool = True,
    ):
        """Initialize LOINC domain.

        Args:
            data_dir: Base directory for LOINC data.
                     Defaults to data/questionnaires/ in project root.
            entities_file: Name of processed entities file.
            survey_only: If True, only load survey-related LOINC codes.
        """
        if data_dir is None:
            # Default to data/questionnaires/ relative to project root
            current = Path.cwd()
            while current != current.parent:
                if (current / "pyproject.toml").exists():
                    data_dir = current / "data" / "questionnaires"
                    break
                current = current.parent
            else:
                data_dir = Path("data/questionnaires")

        self._data_dir = Path(data_dir)
        self._survey_only = survey_only
        self._config = DomainConfig(
            name="loinc",
            data_dir=self._data_dir,
            entities_file=entities_file,
            ground_truth_file="loinc_ground_truth.json",
            id_column="LOINC_NUM",
            name_column="LONG_COMMON_NAME",
            synonym_columns=["SHORTNAME", "COMPONENT"],
            file_format="json",
            extra={"survey_only": survey_only},
        )

    @property
    def name(self) -> str:
        """Domain identifier."""
        return "loinc"

    @property
    def config(self) -> DomainConfig:
        """Domain configuration."""
        return self._config

    @property
    def raw_loinc_path(self) -> Path:
        """Path to raw LOINC CSV."""
        return (
            self._data_dir / "raw" / "loinc" / "LoincTableCore" / "LoincTableCore.csv"
        )

    @property
    def consumer_names_path(self) -> Path:
        """Path to consumer names CSV."""
        return (
            self._data_dir
            / "raw"
            / "loinc"
            / "AccessoryFiles"
            / "ConsumerName"
            / "ConsumerName.csv"
        )

    def load_entities(self, path: Path | str | None = None) -> list[BaseEntity]:
        """Load LOINC entities from file.

        If the processed JSON file doesn't exist, will parse from raw CSV.

        Args:
            path: Optional override path. Uses config default if not provided.

        Returns:
            List of BaseEntity objects.
        """
        if path is None:
            path = self._config.entities_path

        path = Path(path)

        # If processed file exists, load it
        if path.exists():
            return load_entities(path)

        # Otherwise, parse from raw CSV and save
        return self._parse_and_save()

    def _parse_and_save(self) -> list[BaseEntity]:
        """Parse LOINC CSV and save processed entities.

        Returns:
            List of parsed BaseEntity objects.
        """
        if not self.raw_loinc_path.exists():
            raise FileNotFoundError(
                f"LOINC CSV not found at {self.raw_loinc_path}. "
                "Please extract LOINC data to data/questionnaires/raw/loinc/"
            )

        parser = LOINCParser(
            self.raw_loinc_path,
            self.consumer_names_path if self.consumer_names_path.exists() else None,
        )

        if self._survey_only:
            entities = parser.parse_survey_codes()
        else:
            entities = parser.parse(active_only=True)

        # Save to processed directory
        output_path = self._config.entities_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        parser.save_entities(entities, output_path)

        return entities

    def refresh_entities(self) -> list[BaseEntity]:
        """Force re-parse LOINC CSV and update processed file.

        Returns:
            List of freshly parsed BaseEntity objects.
        """
        return self._parse_and_save()

    def generate_ground_truth(
        self,
        entities: list,
        **kwargs,
    ) -> GroundTruthDataset:
        """Generate ground truth dataset for LOINC.

        For LOINC, ground truth typically comes from:
        - Known synonym mappings in the LOINC hierarchy
        - Consumer name to long common name mappings
        - Component variations

        Args:
            entities: List of LOINC entities.
            **kwargs: Generation parameters:
                - seed: Random seed (default: 42)
                - exact: Number of exact match queries
                - synonym: Number of synonym match queries

        Returns:
            GroundTruthDataset with queries for evaluation.
        """
        import random

        seed = kwargs.get("seed", 42)
        exact_count = kwargs.get("exact", 100)
        synonym_count = kwargs.get("synonym", 100)

        random.seed(seed)

        # Convert to BaseEntity if needed
        if entities and not isinstance(entities[0], BaseEntity):
            entities = [BaseEntity.from_dict(e) for e in entities]

        queries: list[GroundTruthQuery] = []

        # Generate exact match queries (using long common name)
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
                        notes=f"Source: long_common_name",
                    )
                )

        # Generate synonym match queries
        entities_with_synonyms = [e for e in entities if e.synonyms]
        if entities_with_synonyms:
            sampled = random.sample(
                entities_with_synonyms, min(synonym_count, len(entities_with_synonyms))
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
        """Get recommended embedding models for LOINC.

        Returns:
            Dict mapping model short names to HuggingFace model IDs.
        """
        return {name: config.model_id for name, config in LOINC_MODELS.items()}
