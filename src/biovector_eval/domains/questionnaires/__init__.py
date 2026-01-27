"""Questionnaires domain for survey item evaluation and harmonization.

This module provides support for:
- Questionnaire item parsing from multiple sources (NHANES, UK Biobank, etc.)
- LOINC code mapping for standardized clinical terminology
- Cross-questionnaire harmonization via vector similarity
- Ground truth generation for evaluation

Domain Registration:
- "questionnaires": Main domain for consolidated question entities
- "loinc": Sub-domain for LOINC codes (registered separately)

Data Format:
Questions are stored as BaseEntity objects with metadata tracking
source questionnaire, year, variable names, and response options.

Example usage:
    from biovector_eval.domains import get_domain

    # Load questionnaire items
    questionnaires = get_domain("questionnaires")
    questions = questionnaires.load_entities()

    # Load LOINC codes
    loinc = get_domain("loinc")
    loinc_codes = loinc.load_entities()
"""

from biovector_eval.domains import register_domain
from biovector_eval.domains.questionnaires.domain import QuestionnairesDomain
from biovector_eval.domains.questionnaires.loinc.domain import LOINCDomain
from biovector_eval.domains.questionnaires.models import (
    LOINC_MODELS,
    QUESTIONNAIRE_MODELS,
    ModelConfig,
    get_model_config,
    list_models,
)

# Register both domains
register_domain("questionnaires", QuestionnairesDomain)
register_domain("loinc", LOINCDomain)

__all__ = [
    # Domain classes
    "QuestionnairesDomain",
    "LOINCDomain",
    # Model configuration
    "QUESTIONNAIRE_MODELS",
    "LOINC_MODELS",
    "ModelConfig",
    "get_model_config",
    "list_models",
]
