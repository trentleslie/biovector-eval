"""Metabolite domain implementation.

Implements the Domain protocol for metabolite/biochemical entities
using HMDB data format.
"""

from __future__ import annotations

from pathlib import Path

from biovector_eval.base.domain import DomainConfig
from biovector_eval.base.entity import BaseEntity
from biovector_eval.base.ground_truth import GroundTruthDataset
from biovector_eval.base.loaders import load_entities
from biovector_eval.domains.metabolites.ground_truth import HMDBGroundTruthGenerator
from biovector_eval.domains.metabolites.models import METABOLITE_MODELS


class MetaboliteDomain:
    """Domain implementation for metabolite/biochemical entities.

    Handles HMDB-format metabolite data with support for:
    - JSON format with hmdb_id, name, synonyms fields
    - Ground truth generation with exact, synonym, fuzzy, and edge case queries
    - Specialized embedding models for biochemical text
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        entities_file: str = "metabolites.json",
    ):
        """Initialize metabolite domain.

        Args:
            data_dir: Base directory for metabolite data.
                     Defaults to data/metabolites/ in project root.
            entities_file: Name of entities file.
        """
        if data_dir is None:
            # Default to data/metabolites/ relative to project root
            # Try to find project root by looking for pyproject.toml
            current = Path.cwd()
            while current != current.parent:
                if (current / "pyproject.toml").exists():
                    data_dir = current / "data" / "metabolites"
                    break
                current = current.parent
            else:
                data_dir = Path("data/metabolites")

        self._config = DomainConfig(
            name="metabolites",
            data_dir=Path(data_dir),
            entities_file=entities_file,
            ground_truth_file="ground_truth.json",
            id_column="hmdb_id",  # Legacy HMDB format uses hmdb_id
            name_column="name",
            synonym_columns=[],  # Synonyms are in a list field, not columns
            file_format="json",
        )

    @property
    def name(self) -> str:
        """Domain identifier."""
        return "metabolites"

    @property
    def config(self) -> DomainConfig:
        """Domain configuration."""
        return self._config

    def load_entities(self, path: Path | str | None = None) -> list[BaseEntity]:
        """Load metabolites from file.

        Args:
            path: Optional override path. Uses config default if not provided.

        Returns:
            List of BaseEntity objects.
        """
        if path is None:
            path = self._config.entities_path
        return load_entities(path)

    def load_entities_legacy(self, path: Path | str | None = None) -> list[dict]:
        """Load metabolites in legacy dict format for backward compatibility.

        Args:
            path: Optional override path.

        Returns:
            List of metabolite dicts with hmdb_id, name, synonyms keys.
        """
        import json

        if path is None:
            path = self._config.entities_path
        with open(path) as f:
            return json.load(f)

    def generate_ground_truth(
        self,
        entities: list,
        **kwargs,
    ) -> GroundTruthDataset:
        """Generate ground truth dataset for metabolites.

        Args:
            entities: List of entities (BaseEntity or legacy dicts).
            **kwargs: Generation parameters:
                - seed: Random seed for reproducibility (default: 42)
                - exact: Number of exact match queries (default: 150)
                - synonym: Number of synonym match queries (default: 150)
                - fuzzy: Number of fuzzy match queries (default: 100)
                - greek: Number of Greek letter edge cases (default: 100)
                - numeric: Number of numeric prefix edge cases (default: 100)
                - special: Number of special prefix edge cases (default: 100)

        Returns:
            GroundTruthDataset with queries for evaluation.
        """
        # Extract parameters with defaults
        seed = kwargs.get("seed", 42)
        exact = kwargs.get("exact", 150)
        synonym = kwargs.get("synonym", 150)
        fuzzy = kwargs.get("fuzzy", 100)
        greek = kwargs.get("greek", 100)
        numeric = kwargs.get("numeric", 100)
        special = kwargs.get("special", 100)

        # Convert BaseEntity to dict format for generator
        if entities and isinstance(entities[0], BaseEntity):
            metabolites: list[dict] = []
            for e in entities:
                metabolites.append(
                    {
                        "hmdb_id": e.id,
                        "name": e.name,
                        "synonyms": e.synonyms,
                    }
                )
        else:
            metabolites = entities

        generator = HMDBGroundTruthGenerator(metabolites, seed=seed)
        return generator.generate_dataset(
            exact=exact,
            synonym=synonym,
            fuzzy=fuzzy,
            greek=greek,
            numeric=numeric,
            special=special,
        )

    def get_embedding_models(self) -> dict[str, str]:
        """Get recommended embedding models for metabolites.

        Returns:
            Dict mapping model short names to HuggingFace model IDs.
        """
        return {name: config.model_id for name, config in METABOLITE_MODELS.items()}
