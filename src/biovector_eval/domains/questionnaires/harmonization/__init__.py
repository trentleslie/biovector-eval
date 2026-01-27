"""Questionnaire harmonization framework.

This module provides tools for:
- Generating candidate pairs between questionnaires
- Question-to-Question (Q2Q) direct matching
- Question-to-LOINC (Q2LOINC) code mapping
- Question-to-HPO (Q2HPO) mapping via GenOMA LLM agent
- Question-to-MONDO (Q2MONDO) disease ontology mapping
- Exporting results for human review

The harmonization workflow produces CandidatePair objects that can be
exported for review in external applications.

Ontology Adapters:
    The framework supports multiple ontology adapters through a common protocol:

    - GenOMAAdapter: Maps questions to HPO (phenotypes) using LLM
    - MONDOAdapter: Maps questions to MONDO (diseases) using vector similarity

    Both adapters implement the OntologyAdapter protocol and return results
    that can be converted to CandidatePair format.

    Usage:
        from biovector_eval.domains.questionnaires.harmonization import (
            GenOMAAdapter,
            MONDOAdapter,
        )

        # HPO mapping via LLM
        hpo_adapter = GenOMAAdapter()
        hpo_result = hpo_adapter.map_entity(question_entity)

        # MONDO mapping via vector search
        mondo_adapter = MONDOAdapter(
            mondo_index_path="data/mondo/mondo_sapbert.faiss",
            mondo_entities_path="data/mondo/mondo_entities.json",
        )
        mondo_result = mondo_adapter.map_entity(question_entity)
"""

from biovector_eval.domains.questionnaires.harmonization.cache import GenOMACache
from biovector_eval.domains.questionnaires.harmonization.candidate_pair import (
    CandidatePair,
    CandidatePairDataset,
)
from biovector_eval.domains.questionnaires.harmonization.genoma_adapter import (
    GenOMAAdapter,
    GenOMAResult,
)
from biovector_eval.domains.questionnaires.harmonization.mondo_adapter import (
    MONDOAdapter,
    MONDOResult,
)
from biovector_eval.domains.questionnaires.harmonization.ontology_adapter import (
    OntologyAdapter,
    OntologyResult,
    confidence_to_level,
    determine_match_quality,
)

__all__ = [
    # Candidate pairs
    "CandidatePair",
    "CandidatePairDataset",
    # GenOMA (HPO) adapter
    "GenOMAAdapter",
    "GenOMAResult",
    "GenOMACache",
    # MONDO adapter
    "MONDOAdapter",
    "MONDOResult",
    # Base ontology protocol
    "OntologyAdapter",
    "OntologyResult",
    "confidence_to_level",
    "determine_match_quality",
]
