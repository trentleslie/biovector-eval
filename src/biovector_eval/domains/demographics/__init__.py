"""Demographics domain for cross-cohort variable harmonization.

This module provides support for demographic variable evaluation
across multiple cohort sources (Arivale, Israeli10K, UK Biobank).

Example:
    from biovector_eval.domains import get_domain

    demographics = get_domain("demographics")
    entities = demographics.load_entities()
    ground_truth = demographics.generate_ground_truth(entities)
"""

from biovector_eval.domains import register_domain
from biovector_eval.domains.demographics.domain import DemographicsDomain

# Register the domain
register_domain("demographics", DemographicsDomain)

__all__ = [
    "DemographicsDomain",
]
