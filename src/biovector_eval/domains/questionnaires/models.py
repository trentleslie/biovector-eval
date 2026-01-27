"""Embedding model configurations for questionnaire domain.

This module defines recommended embedding models for questionnaire/survey
item search and LOINC code mapping. Models are selected for their
performance on clinical and biomedical text understanding.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for an embedding model."""

    name: str
    model_id: str
    device: str = "cpu"
    batch_size: int = 512


# Model configurations for questionnaire/LOINC evaluation
QUESTIONNAIRE_MODELS: dict[str, ModelConfig] = {
    "sapbert": ModelConfig(
        name="SapBERT",
        model_id="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    ),
    "pubmedbert": ModelConfig(
        name="PubMedBERT",
        model_id="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    ),
    "bge-m3": ModelConfig(
        name="BGE-M3",
        model_id="BAAI/bge-m3",
    ),
    "minilm": ModelConfig(
        name="MiniLM-L6-v2",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
    ),
    "baseline": ModelConfig(
        name="BGE-Small (Baseline)",
        model_id="BAAI/bge-small-en-v1.5",
    ),
}

# LOINC-specific models (same as questionnaire for now, but allows
# future specialization for the 6-part LOINC structure)
LOINC_MODELS: dict[str, ModelConfig] = QUESTIONNAIRE_MODELS.copy()


def get_model_config(name: str) -> ModelConfig:
    """Get model configuration by short name.

    Args:
        name: Model short name (sapbert, pubmedbert, bge-m3, minilm, baseline).

    Returns:
        ModelConfig for the requested model.

    Raises:
        KeyError: If model name is not found.
    """
    if name not in QUESTIONNAIRE_MODELS:
        raise KeyError(
            f"Unknown model: {name}. " f"Available: {list(QUESTIONNAIRE_MODELS.keys())}"
        )
    return QUESTIONNAIRE_MODELS[name]


def list_models() -> list[str]:
    """List available model short names.

    Returns:
        List of model short names.
    """
    return list(QUESTIONNAIRE_MODELS.keys())
