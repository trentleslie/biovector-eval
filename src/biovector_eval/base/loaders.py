"""Data loaders for different file formats.

Provides unified loading interface for JSON and TSV entity files,
enabling support for different data sources across domains.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from biovector_eval.base.entity import BaseEntity


def load_entities_json(path: Path | str) -> list[BaseEntity]:
    """Load entities from JSON file.

    Supports both standardized format and legacy metabolite format.

    Args:
        path: Path to JSON file.

    Returns:
        List of BaseEntity objects.

    Raises:
        FileNotFoundError: If file doesn't exist.
        json.JSONDecodeError: If file isn't valid JSON.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    entities = []
    for item in data:
        entities.append(BaseEntity.from_dict(item))

    return entities


def load_entities_tsv(
    path: Path | str,
    id_col: str,
    name_col: str,
    synonym_cols: list[str] | None = None,
    delimiter: str = "\t",
) -> list[BaseEntity]:
    """Load entities from TSV file.

    Converts tabular data to BaseEntity objects, extracting synonyms
    from specified columns (which may contain semicolon-separated values).

    Args:
        path: Path to TSV file.
        id_col: Column name for entity ID.
        name_col: Column name for entity primary name.
        synonym_cols: Column names containing synonyms (optional).
        delimiter: Field delimiter (default: tab).

    Returns:
        List of BaseEntity objects.

    Raises:
        FileNotFoundError: If file doesn't exist.
        KeyError: If required columns are missing.
    """
    path = Path(path)
    synonym_cols = synonym_cols or []

    entities = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        for row in reader:
            # Extract ID and name
            entity_id = row[id_col].strip()
            name = row[name_col].strip()

            if not entity_id or not name:
                continue

            # Extract synonyms from specified columns
            synonyms = []
            for col in synonym_cols:
                if col in row and row[col]:
                    # Split on semicolon for multi-value fields
                    values = row[col].split(";")
                    synonyms.extend(v.strip() for v in values if v.strip())

            # Collect remaining columns as metadata
            metadata = {
                k: v
                for k, v in row.items()
                if k not in [id_col, name_col] + synonym_cols and v
            }

            entities.append(
                BaseEntity(
                    id=entity_id,
                    name=name,
                    synonyms=synonyms,
                    metadata=metadata,
                )
            )

    return entities


def load_entities(
    path: Path | str,
    id_col: str = "id",
    name_col: str = "name",
    synonym_cols: list[str] | None = None,
) -> list[BaseEntity]:
    """Load entities with auto-detection of file format.

    Automatically selects JSON or TSV loader based on file extension.

    Args:
        path: Path to entities file.
        id_col: Column name for entity ID (TSV only).
        name_col: Column name for entity name (TSV only).
        synonym_cols: Column names containing synonyms (TSV only).

    Returns:
        List of BaseEntity objects.

    Raises:
        ValueError: If file format is not supported.
        FileNotFoundError: If file doesn't exist.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        return load_entities_json(path)
    elif suffix in (".tsv", ".txt"):
        return load_entities_tsv(
            path,
            id_col=id_col,
            name_col=name_col,
            synonym_cols=synonym_cols,
        )
    elif suffix == ".csv":
        return load_entities_tsv(
            path,
            id_col=id_col,
            name_col=name_col,
            synonym_cols=synonym_cols,
            delimiter=",",
        )
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: .json, .tsv, .txt, .csv"
        )


def save_entities_json(entities: list[BaseEntity], path: Path | str) -> None:
    """Save entities to JSON file.

    Args:
        entities: List of BaseEntity objects.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [e.to_dict() for e in entities]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
