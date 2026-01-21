"""Metabolite domain for HMDB-based entity evaluation.

This module provides the metabolite domain implementation, which supports
evaluation of embedding models for metabolite/biochemical entity search
using HMDB (Human Metabolome Database) data.
"""

from biovector_eval.domains import register_domain
from biovector_eval.domains.metabolites.domain import MetaboliteDomain
from biovector_eval.domains.metabolites.ground_truth import HMDBGroundTruthGenerator
from biovector_eval.domains.metabolites.models import METABOLITE_MODELS

# Register the metabolite domain
register_domain("metabolites", MetaboliteDomain)

__all__ = [
    "MetaboliteDomain",
    "HMDBGroundTruthGenerator",
    "METABOLITE_MODELS",
]
