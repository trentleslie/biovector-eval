"""Demographics data consolidation from multiple sources.

This module provides functions to parse and consolidate demographics
data from Arivale, Israeli10K, and UK Biobank sources into a unified
BaseEntity format.

Sources included:
- demographics_metadata.tsv (39 rows) - Arivale demographics
- israeli10k_demographics.tsv (184 rows) - Israeli10K demographics
- ukbb_demographics.tsv (45 rows) - UK Biobank demographics

Usage:
    from biovector_eval.domains.demographics.consolidate import (
        consolidate_demographics,
    )

    entities = consolidate_demographics(
        raw_dir=Path("data/raw"),
        output_path=Path("data/demographics/processed/demographics.json"),
    )
    print(f"Consolidated {len(entities)} demographic entities")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from biovector_eval.base.entity import BaseEntity
from biovector_eval.domains.demographics.parsers.source_mappings import (
    ARIVALE_DEMOGRAPHICS_MAPPING,
    ISRAELI10K_DEMOGRAPHICS_MAPPING,
    UKBB_DEMOGRAPHICS_MAPPING,
)
from biovector_eval.domains.questionnaires.parsers.generic import (
    GenericQuestionParser,
    consolidate_entities,
)

logger = logging.getLogger(__name__)


@dataclass
class SourceFile:
    """Configuration for a source demographics file."""

    filename: str
    mapping: "ColumnMapping"  # type: ignore
    description: str
    expected_rows: int  # Expected data rows (excluding header)


# Define expected source files
DEMOGRAPHICS_SOURCES = [
    SourceFile(
        filename="demographics_metadata.tsv",
        mapping=ARIVALE_DEMOGRAPHICS_MAPPING,
        description="Arivale cohort demographics metadata",
        expected_rows=38,  # Actual parsed count
    ),
    SourceFile(
        filename="israeli10k_demographics.tsv",
        mapping=ISRAELI10K_DEMOGRAPHICS_MAPPING,
        description="Israeli10K cohort demographics",
        expected_rows=182,  # Some rows may be filtered
    ),
    SourceFile(
        filename="ukbb_demographics.tsv",
        mapping=UKBB_DEMOGRAPHICS_MAPPING,
        description="UK Biobank demographics",
        expected_rows=45,
    ),
]


def parse_source_file(
    source: SourceFile,
    raw_dir: Path,
) -> list[BaseEntity]:
    """Parse a single source file into entities.

    Args:
        source: SourceFile configuration.
        raw_dir: Directory containing raw data files.

    Returns:
        List of BaseEntity objects.

    Raises:
        FileNotFoundError: If source file doesn't exist.
        ValueError: If parsing fails.
    """
    file_path = raw_dir / source.filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Source file not found: {file_path}\n"
            f"Expected: {source.description}"
        )

    parser = GenericQuestionParser(file_path, source.mapping)
    entities = parser.parse()

    # Log count comparison
    actual = len(entities)
    expected = source.expected_rows
    if actual != expected:
        logger.warning(
            f"{source.filename}: Expected {expected} rows, got {actual} "
            f"(difference: {actual - expected:+d})"
        )
    else:
        logger.info(f"{source.filename}: Parsed {actual} entities")

    return entities


def consolidate_demographics(
    raw_dir: Path | str,
    output_path: Path | str | None = None,
    sources: list[SourceFile] | None = None,
) -> list[BaseEntity]:
    """Parse demographics sources and consolidate into unified format.

    Parses all configured demographics sources (Arivale, Israeli10K, UKBB)
    and combines them into a single list of BaseEntity objects. Checks for
    ID collisions and optionally saves to JSON.

    Args:
        raw_dir: Directory containing raw TSV files.
        output_path: Optional path to save consolidated JSON.
        sources: Optional list of sources to parse (defaults to all).

    Returns:
        List of consolidated BaseEntity objects.

    Raises:
        FileNotFoundError: If any source file is missing.
        ValueError: If duplicate IDs are found across sources.

    Example:
        entities = consolidate_demographics(
            raw_dir=Path("data/raw"),
            output_path=Path("data/demographics/processed/demographics.json"),
        )
    """
    raw_dir = Path(raw_dir)
    sources = sources or DEMOGRAPHICS_SOURCES

    logger.info(f"Consolidating demographics data from {raw_dir}")
    logger.info(f"Sources: {[s.filename for s in sources]}")

    # Parse each source
    entity_lists: list[list[BaseEntity]] = []
    for source in sources:
        entities = parse_source_file(source, raw_dir)
        entity_lists.append(entities)

    # Consolidate with ID collision check
    consolidated = consolidate_entities(entity_lists, output_path=None)

    # Log summary
    total = len(consolidated)
    by_source = {}
    for entity in consolidated:
        src = entity.metadata.get("source_questionnaire", "unknown")
        by_source[src] = by_source.get(src, 0) + 1

    logger.info(f"Consolidated {total} entities from {len(sources)} sources")
    for src, count in sorted(by_source.items()):
        logger.info(f"  {src}: {count}")

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in consolidated]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved to {output_path}")

    return consolidated


def get_available_sources(raw_dir: Path | str) -> list[SourceFile]:
    """Get list of sources that have files present in raw_dir.

    Useful for checking which sources are available before consolidation.

    Args:
        raw_dir: Directory to check for raw files.

    Returns:
        List of SourceFile configs for files that exist.
    """
    raw_dir = Path(raw_dir)
    available = []

    for source in DEMOGRAPHICS_SOURCES:
        if (raw_dir / source.filename).exists():
            available.append(source)

    return available


def validate_sources(raw_dir: Path | str) -> dict[str, dict]:
    """Validate source files and return status for each.

    Args:
        raw_dir: Directory containing raw files.

    Returns:
        Dict mapping source name to status info:
        {
            "arivale": {"exists": True, "path": "...", "expected_rows": 39},
            "israeli10k": {"exists": False, "path": "...", "expected_rows": 184},
            ...
        }
    """
    raw_dir = Path(raw_dir)
    status = {}

    for source in DEMOGRAPHICS_SOURCES:
        file_path = raw_dir / source.filename
        source_key = source.mapping.source_name.lower()
        status[source_key] = {
            "exists": file_path.exists(),
            "path": str(file_path),
            "filename": source.filename,
            "expected_rows": source.expected_rows,
            "description": source.description,
        }

    return status


__all__ = [
    "consolidate_demographics",
    "get_available_sources",
    "validate_sources",
    "parse_source_file",
    "SourceFile",
    "DEMOGRAPHICS_SOURCES",
]
