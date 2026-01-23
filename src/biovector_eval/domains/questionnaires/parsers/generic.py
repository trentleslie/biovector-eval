"""Generic questionnaire parser with configurable column mappings.

Handles parsing of questionnaire data from various sources (NHANES,
UK Biobank, custom surveys) into the standardized BaseEntity format.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from biovector_eval.base.entity import BaseEntity


@dataclass
class ColumnMapping:
    """Configuration for mapping source columns to entity fields.

    Attributes:
        id_column: Column name for question ID/code.
        text_column: Column name for question text (becomes entity name).
        synonym_columns: Column names that contain alternative text.
        metadata_columns: Column names to include in metadata.
        source_name: Name of the questionnaire source (e.g., "NHANES").
        year_column: Optional column for year/version.
        response_column: Optional column for response options.
        id_prefix: Optional prefix to add to IDs for uniqueness.
        id_generator: Optional function to generate custom IDs from row data.
            When provided, this function takes the row dict and returns the
            complete entity ID, bypassing id_column/id_prefix logic.
    """

    id_column: str
    text_column: str
    synonym_columns: list[str] = field(default_factory=list)
    metadata_columns: list[str] = field(default_factory=list)
    source_name: str = "unknown"
    year_column: str | None = None
    response_column: str | None = None
    id_prefix: str | None = None
    id_generator: Callable[[dict[str, str]], str] | None = None


# Pre-defined mappings for common questionnaire sources
NHANES_MAPPING = ColumnMapping(
    id_column="Variable Name",
    text_column="SAS Label",
    synonym_columns=[],
    metadata_columns=["Component", "Data File Name", "Data File Description"],
    source_name="NHANES",
    year_column="Years",
    id_prefix="nhanes",
)

UK_BIOBANK_MAPPING = ColumnMapping(
    id_column="FieldID",
    text_column="Field",
    synonym_columns=[],
    metadata_columns=["Category", "ValueType", "Units"],
    source_name="UK_Biobank",
    id_prefix="ukbb",
)


class GenericQuestionParser:
    """Parser for generic questionnaire CSV/TSV files.

    Supports configurable column mappings to handle different
    questionnaire source formats.

    Attributes:
        file_path: Path to the questionnaire file.
        mapping: Column mapping configuration.
        delimiter: CSV delimiter (auto-detected if None).
    """

    def __init__(
        self,
        file_path: Path | str,
        mapping: ColumnMapping,
        delimiter: str | None = None,
    ):
        """Initialize questionnaire parser.

        Args:
            file_path: Path to the questionnaire CSV/TSV file.
            mapping: Column mapping configuration.
            delimiter: CSV delimiter. If None, auto-detects from extension.
        """
        self.file_path = Path(file_path)
        self.mapping = mapping

        if delimiter is None:
            delimiter = "\t" if self.file_path.suffix.lower() == ".tsv" else ","
        self.delimiter = delimiter

    def parse(self) -> list[BaseEntity]:
        """Parse questionnaire file into BaseEntity objects.

        Returns:
            List of BaseEntity objects representing questions.
        """
        entities: list[BaseEntity] = []

        with open(self.file_path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)

            if reader.fieldnames is None:
                raise ValueError("File has no headers")

            # Validate required columns
            required = {self.mapping.id_column, self.mapping.text_column}
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            for row in reader:
                entity = self._parse_row(row)
                if entity is not None:
                    entities.append(entity)

        return entities

    def _parse_row(self, row: dict[str, str]) -> BaseEntity | None:
        """Parse a single row into a BaseEntity.

        Args:
            row: Dictionary of column values.

        Returns:
            BaseEntity or None if row should be skipped.
        """
        # Use custom ID generator if provided
        if self.mapping.id_generator is not None:
            entity_id = self.mapping.id_generator(row)
            if not entity_id:
                return None
            # Store raw_id from id_column for metadata if available
            raw_id = row.get(self.mapping.id_column, "").strip() or entity_id
        else:
            # Extract ID from column
            raw_id = row.get(self.mapping.id_column, "").strip()
            if not raw_id:
                return None

            # Build unique ID with optional prefix and year
            id_parts = []
            if self.mapping.id_prefix:
                id_parts.append(self.mapping.id_prefix)
            if self.mapping.year_column:
                year = row.get(self.mapping.year_column, "").strip()
                if year:
                    # Handle year ranges like "2021-2022" -> "2021"
                    year = year.split("-")[0].strip()
                    id_parts.append(year)
            id_parts.append(raw_id.lower().replace(" ", "_"))
            entity_id = "_".join(id_parts)

        # Extract question text
        name = row.get(self.mapping.text_column, "").strip()
        if not name:
            return None

        # Extract synonyms
        synonyms: list[str] = []
        for col in self.mapping.synonym_columns:
            value = row.get(col, "").strip()
            if value and value != name:
                synonyms.append(value)

        # Build metadata
        metadata: dict = {
            "source_questionnaire": self.mapping.source_name,
            "variable_name": raw_id,
        }

        if self.mapping.year_column:
            year = row.get(self.mapping.year_column, "").strip()
            if year:
                metadata["year"] = year

        if self.mapping.response_column:
            responses = row.get(self.mapping.response_column, "").strip()
            if responses:
                metadata["response_options"] = responses

        for col in self.mapping.metadata_columns:
            value = row.get(col, "").strip()
            if value:
                # Convert column name to snake_case for metadata key
                key = col.lower().replace(" ", "_")
                metadata[key] = value

        return BaseEntity(
            id=entity_id,
            name=name,
            synonyms=synonyms,
            metadata=metadata,
        )

    def save_entities(
        self,
        entities: list[BaseEntity],
        output_path: Path | str,
    ) -> None:
        """Save parsed entities to JSON file.

        Args:
            entities: List of BaseEntity objects to save.
            output_path: Path to output JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [e.to_dict() for e in entities]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def parse_questionnaire_csv(
    file_path: Path | str,
    mapping: ColumnMapping,
    delimiter: str | None = None,
) -> list[BaseEntity]:
    """Convenience function to parse a questionnaire file.

    Args:
        file_path: Path to questionnaire CSV/TSV file.
        mapping: Column mapping configuration.
        delimiter: CSV delimiter (auto-detected if None).

    Returns:
        List of BaseEntity objects.
    """
    parser = GenericQuestionParser(file_path, mapping, delimiter)
    return parser.parse()


def consolidate_entities(
    entity_lists: list[list[BaseEntity]],
    output_path: Path | str | None = None,
    deduplicate: bool = True,
) -> list[BaseEntity]:
    """Consolidate entities from multiple sources.

    Combines entity lists. By default, keeps the first occurrence of
    duplicate IDs (within same source) and raises errors for cross-source
    duplicates. Set deduplicate=False to raise errors for all duplicates.

    Args:
        entity_lists: List of entity lists to combine.
        output_path: Optional path to save consolidated JSON.
        deduplicate: If True, silently skip duplicate IDs from the same
            source (true duplicates in raw data). Default True.

    Returns:
        Combined list of entities.

    Raises:
        ValueError: If duplicate IDs are found across different sources.
    """
    import logging

    logger = logging.getLogger(__name__)
    seen_ids: dict[str, str] = {}  # id -> source
    consolidated: list[BaseEntity] = []
    duplicate_count = 0

    for entities in entity_lists:
        for entity in entities:
            source = entity.metadata.get("source_questionnaire", "unknown")

            if entity.id in seen_ids:
                first_source = seen_ids[entity.id]

                if first_source == source and deduplicate:
                    # Same source duplicate - likely a true duplicate row, skip it
                    duplicate_count += 1
                    continue
                else:
                    # Cross-source duplicate - this is a mapping error
                    raise ValueError(
                        f"Duplicate ID '{entity.id}' found. "
                        f"First seen in {first_source}, "
                        f"also in {source}"
                    )

            seen_ids[entity.id] = source
            consolidated.append(entity)

    if duplicate_count > 0:
        logger.info(f"Deduplicated {duplicate_count} true duplicate rows")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in consolidated]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    return consolidated
