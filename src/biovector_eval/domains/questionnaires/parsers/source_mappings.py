"""Source-specific column mappings for questionnaire data.

Provides pre-configured ColumnMapping instances for each data source:
- Arivale: Survey questions with Assessment/Question Text/Options columns
- Israeli10K: 17-column format with dataset metadata
- UKBB (UK Biobank): 12-column format with field IDs

Each mapping includes appropriate ID prefixes to ensure uniqueness
when consolidating across sources.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from biovector_eval.domains.questionnaires.parsers.generic import ColumnMapping

if TYPE_CHECKING:
    pass


def generate_arivale_id(row: dict[str, str]) -> str:
    """Generate deterministic ID from assessment + question text + options.

    Arivale questionnaire data lacks unique identifiers, so we create
    stable IDs by hashing the Assessment, Question Text, and Options fields.
    Including Options helps differentiate questions that might have the
    same text but different response options.

    Args:
        row: Dictionary of column values from CSV row.

    Returns:
        Unique ID in format "arivale_{hash12}".

    Note:
        IDs are deterministic but will change if question text/options are edited.
        True duplicate rows (identical content) will still produce the same ID;
        the consolidation step will handle deduplication.
    """
    assessment = row.get("Assessment", "").strip()
    question_text = row.get("Question Text", "").strip()
    options = row.get("Options", "").strip()
    content = f"{assessment}|{question_text}|{options}"
    hash_val = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"arivale_{hash_val}"


# =============================================================================
# Arivale Questionnaire Mapping
# =============================================================================
# Source: "Arivale Questionnaire - Sheet1.tsv"
# Columns: Assessment, Question Text, Options
# Rows: 562 questions
#
# Uses custom hash-based ID generator since file has no unique identifiers.

ARIVALE_QUESTIONNAIRE_MAPPING = ColumnMapping(
    id_column="Question Text",  # Placeholder - overridden by id_generator
    text_column="Question Text",
    synonym_columns=[],
    metadata_columns=["Assessment"],
    source_name="Arivale",
    year_column=None,
    response_column="Options",
    id_prefix=None,  # ID prefix handled by generate_arivale_id
    id_generator=generate_arivale_id,
)


# =============================================================================
# Israeli10K Questionnaire Mapping
# =============================================================================
# Source: "israeli10k_questionnaires.tsv"
# Columns: dataset_id, dataset_name, table_name, field_name, field_string,
#          field_description, field_type, data_type_pandas, units, data_coding,
#          strata, sexed, stability, debut, array, relative_location, category
# Rows: 569 questions
#
# Uses field_name as ID with field_description as synonym.

ISRAELI10K_QUESTIONNAIRE_MAPPING = ColumnMapping(
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
    ],
    source_name="Israeli10K",
    year_column=None,
    response_column=None,
    id_prefix="il10k",
)


# =============================================================================
# UK Biobank (UKBB) Questionnaire Mapping
# =============================================================================
# Source: "ukbb_questionnaires.tsv"
# Columns: field_id, field_name, description, parent_category, parent_category_id,
#          subcategory, subcategory_id, data_type, units, value_encoding,
#          notes, participant_count
# Rows: 499 questions
#
# Uses field_id as ID with field_name as synonym.

UKBB_QUESTIONNAIRE_MAPPING = ColumnMapping(
    id_column="field_id",
    text_column="description",
    synonym_columns=["field_name"],
    metadata_columns=[
        "parent_category",
        "subcategory",
        "data_type",
        "units",
        "value_encoding",
        "participant_count",
    ],
    source_name="UKBB",
    year_column=None,
    response_column=None,
    id_prefix="ukbb",
)


# =============================================================================
# Mapping Registry
# =============================================================================

QUESTIONNAIRE_MAPPINGS = {
    "arivale": ARIVALE_QUESTIONNAIRE_MAPPING,
    "israeli10k": ISRAELI10K_QUESTIONNAIRE_MAPPING,
    "ukbb": UKBB_QUESTIONNAIRE_MAPPING,
}


__all__ = [
    "ARIVALE_QUESTIONNAIRE_MAPPING",
    "ISRAELI10K_QUESTIONNAIRE_MAPPING",
    "UKBB_QUESTIONNAIRE_MAPPING",
    "QUESTIONNAIRE_MAPPINGS",
    "generate_arivale_id",
]
