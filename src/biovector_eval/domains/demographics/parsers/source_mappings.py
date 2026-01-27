"""Source-specific column mappings for demographics data.

Provides pre-configured ColumnMapping instances for each data source:
- Arivale: Demographics metadata with Column Name/Description columns
- Israeli10K: 17-column format with dataset metadata (same as questionnaires)
- UKBB (UK Biobank): 10-column format with field IDs

Each mapping includes appropriate ID prefixes to ensure uniqueness
when consolidating across sources.
"""

from __future__ import annotations

import hashlib

from biovector_eval.domains.questionnaires.parsers.generic import ColumnMapping


def generate_arivale_demographics_id(row: dict[str, str]) -> str:
    """Generate deterministic ID from Column Name + Source Table.

    Arivale demographics may have same Column Name in different tables,
    so we include Source Table to ensure uniqueness.

    Args:
        row: Dictionary of column values from CSV row.

    Returns:
        Unique ID in format "arivale_demo_{column_name}_{hash8}".
    """
    column_name = row.get("Column Name", "").strip().lower().replace(" ", "_")
    source_table = row.get("Source Table", "").strip()
    # Include source table hash to differentiate same-named columns from different tables
    content = f"{column_name}|{source_table}"
    hash_val = hashlib.sha256(content.encode()).hexdigest()[:8]
    return f"arivale_demo_{column_name}_{hash_val}"


# =============================================================================
# Arivale Demographics Mapping
# =============================================================================
# Source: "demographics_metadata.tsv"
# Columns: Column Name, Description, Variable Type, Units, Category, Sex, Source Table
# Rows: 39 demographic variables
#
# Uses custom ID generator to handle same-named columns from different tables.

ARIVALE_DEMOGRAPHICS_MAPPING = ColumnMapping(
    id_column="Column Name",
    text_column="Description",
    synonym_columns=[],
    metadata_columns=[
        "Variable Type",
        "Units",
        "Category",
        "Sex",
        "Source Table",
    ],
    source_name="Arivale",
    year_column=None,
    response_column=None,
    id_prefix=None,  # Handled by id_generator
    id_generator=generate_arivale_demographics_id,
)


# =============================================================================
# Israeli10K Demographics Mapping
# =============================================================================
# Source: "israeli10k_demographics.tsv"
# Columns: dataset_id, dataset_name, table_name, field_name, field_string,
#          field_description, field_type, data_type_pandas, units, data_coding,
#          strata, sexed, stability, debut, array, relative_location, category
# Rows: 184 demographic variables
#
# Uses same format as Israeli10K questionnaires.

ISRAELI10K_DEMOGRAPHICS_MAPPING = ColumnMapping(
    id_column="field_name",
    text_column="field_string",
    synonym_columns=["field_description"],
    metadata_columns=[
        "dataset_name",
        "table_name",
        "field_type",
        "data_type_pandas",
        "units",
        "data_coding",
        "category",
        "strata",
        "sexed",
    ],
    source_name="Israeli10K",
    year_column=None,
    response_column=None,
    id_prefix="il10k_demo",
)


# =============================================================================
# UK Biobank (UKBB) Demographics Mapping
# =============================================================================
# Source: "ukbb_demographics.tsv"
# Columns: field_id, field_name, description, category, subcategory,
#          data_type, units, value_encoding, notes, participant_count
# Rows: 45 demographic variables
#
# Uses field_id as ID with field_name as synonym.

UKBB_DEMOGRAPHICS_MAPPING = ColumnMapping(
    id_column="field_id",
    text_column="description",
    synonym_columns=["field_name"],
    metadata_columns=[
        "category",
        "subcategory",
        "data_type",
        "units",
        "value_encoding",
        "participant_count",
    ],
    source_name="UKBB",
    year_column=None,
    response_column=None,
    id_prefix="ukbb_demo",
)


# =============================================================================
# Mapping Registry
# =============================================================================

DEMOGRAPHICS_MAPPINGS = {
    "arivale": ARIVALE_DEMOGRAPHICS_MAPPING,
    "israeli10k": ISRAELI10K_DEMOGRAPHICS_MAPPING,
    "ukbb": UKBB_DEMOGRAPHICS_MAPPING,
}


__all__ = [
    "ARIVALE_DEMOGRAPHICS_MAPPING",
    "ISRAELI10K_DEMOGRAPHICS_MAPPING",
    "UKBB_DEMOGRAPHICS_MAPPING",
    "DEMOGRAPHICS_MAPPINGS",
]
