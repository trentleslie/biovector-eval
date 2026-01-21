"""Embedding model configurations for metabolite domain.

This module defines the recommended embedding models for metabolite/biochemical
entity search. Models are chosen based on their performance on metabolite
name matching tasks.
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


# Model configurations for metabolite evaluation
METABOLITE_MODELS: dict[str, ModelConfig] = {
    "baseline": ModelConfig(
        name="BGE-Small (Baseline)",
        model_id="BAAI/bge-small-en-v1.5",
    ),
    "bge-m3": ModelConfig(
        name="BGE-M3",
        model_id="BAAI/bge-m3",
    ),
    "chemberta-mtr": ModelConfig(
        name="ChemBERTa-MTR",
        model_id="DeepChem/ChemBERTa-77M-MTR",
    ),
    "minilm": ModelConfig(
        name="MiniLM-L6-v2",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
    ),
}


def get_model_config(name: str) -> ModelConfig:
    """Get model configuration by short name.

    Args:
        name: Model short name (baseline, bge-m3, chemberta-mtr, minilm).

    Returns:
        ModelConfig for the requested model.

    Raises:
        KeyError: If model name is not found.
    """
    if name not in METABOLITE_MODELS:
        raise KeyError(
            f"Unknown model: {name}. " f"Available: {list(METABOLITE_MODELS.keys())}"
        )
    return METABOLITE_MODELS[name]


def list_models() -> list[str]:
    """List available model short names.

    Returns:
        List of model short names.
    """
    return list(METABOLITE_MODELS.keys())
