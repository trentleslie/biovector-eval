"""Demographics data parsers.

This module provides parsers and mappings for demographics data formats,
reusing the GenericQuestionParser from the questionnaires domain.
"""

from biovector_eval.domains.demographics.parsers.source_mappings import (
    ARIVALE_DEMOGRAPHICS_MAPPING,
    DEMOGRAPHICS_MAPPINGS,
    ISRAELI10K_DEMOGRAPHICS_MAPPING,
    UKBB_DEMOGRAPHICS_MAPPING,
)

__all__ = [
    "ARIVALE_DEMOGRAPHICS_MAPPING",
    "ISRAELI10K_DEMOGRAPHICS_MAPPING",
    "UKBB_DEMOGRAPHICS_MAPPING",
    "DEMOGRAPHICS_MAPPINGS",
]
