"""Base infrastructure for multi-domain biological entity evaluation.

This module provides domain-agnostic base classes and protocols that enable
the biovector-eval framework to support multiple biological domains
(metabolites, demographics, questionnaires, etc.) while sharing common
evaluation infrastructure.
"""

from biovector_eval.base.domain import Domain, DomainConfig
from biovector_eval.base.entity import BaseEntity
from biovector_eval.base.ground_truth import GroundTruthDataset, GroundTruthQuery
from biovector_eval.base.loaders import (
    load_entities,
    load_entities_json,
    load_entities_tsv,
)

__all__ = [
    # Entity
    "BaseEntity",
    # Ground truth
    "GroundTruthQuery",
    "GroundTruthDataset",
    # Domain protocol
    "Domain",
    "DomainConfig",
    # Loaders
    "load_entities",
    "load_entities_json",
    "load_entities_tsv",
]
