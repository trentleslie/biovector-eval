"""Questionnaire data parsers.

This module provides parsers for various questionnaire data formats,
converting them into the standardized BaseEntity format for embedding
and harmonization.
"""

from biovector_eval.domains.questionnaires.parsers.generic import (
    ColumnMapping,
    GenericQuestionParser,
    consolidate_entities,
    parse_questionnaire_csv,
)
from biovector_eval.domains.questionnaires.parsers.source_mappings import (
    ARIVALE_QUESTIONNAIRE_MAPPING,
    ISRAELI10K_QUESTIONNAIRE_MAPPING,
    QUESTIONNAIRE_MAPPINGS,
    UKBB_QUESTIONNAIRE_MAPPING,
    generate_arivale_id,
)

__all__ = [
    # Generic parser
    "GenericQuestionParser",
    "ColumnMapping",
    "parse_questionnaire_csv",
    "consolidate_entities",
    # Source-specific mappings
    "ARIVALE_QUESTIONNAIRE_MAPPING",
    "ISRAELI10K_QUESTIONNAIRE_MAPPING",
    "UKBB_QUESTIONNAIRE_MAPPING",
    "QUESTIONNAIRE_MAPPINGS",
    "generate_arivale_id",
]
