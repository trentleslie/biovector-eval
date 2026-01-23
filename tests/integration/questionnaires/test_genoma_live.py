"""Live integration tests for GenOMA (requires OPENAI_API_KEY and GenOMA deps).

These tests make actual API calls to OpenAI and should only be run:
1. Locally when you have OPENAI_API_KEY set
2. With GenOMA dependencies installed (uv add langgraph langchain langchain-openai openai)
3. Never in CI (excluded by integration marker)

Usage:
    # Install GenOMA dependencies first
    uv add langgraph langchain langchain-openai openai

    # Run integration tests locally
    OPENAI_API_KEY=sk-xxx uv run pytest -m integration -v

    # Run only unit tests (CI default)
    uv run pytest tests/unit
"""

from __future__ import annotations

import os

import pytest

from biovector_eval.base.entity import BaseEntity
from biovector_eval.domains.questionnaires.harmonization import GenOMAAdapter

# Check if GenOMA dependencies are installed
try:
    import langgraph  # noqa: F401

    GENOMA_DEPS_AVAILABLE = True
except ImportError:
    GENOMA_DEPS_AVAILABLE = False

# Apply integration marker to all tests in this module
pytestmark = pytest.mark.integration

# Common skip conditions
requires_genoma = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or not GENOMA_DEPS_AVAILABLE,
    reason="Requires OPENAI_API_KEY and GenOMA dependencies (langgraph)",
)


@requires_genoma
def test_live_genoma_mapping() -> None:
    """Test actual GenOMA mapping with live API."""
    adapter = GenOMAAdapter(use_cache=False)
    entity = BaseEntity(
        id="test_q1",
        name="How often do you experience feelings of anxiety?",
        synonyms=[],
        metadata={"source": "integration_test"},
    )

    result = adapter.map_entity(entity)

    # Either we get a match or a no_match (both are valid responses)
    assert result.success or result.error_type == "no_match"
    if result.success and result.hpo_code:
        assert result.hpo_code.startswith("HP:")
        assert result.confidence > 0


@requires_genoma
def test_genoma_validates_mappings_schema() -> None:
    """Verify GenOMA returns expected schema structure."""
    adapter = GenOMAAdapter(use_cache=False)
    entity = BaseEntity(
        id="test_q2",
        name="Do you have difficulty sleeping at night?",
        synonyms=["insomnia"],
        metadata={"source": "integration_test"},
    )

    result = adapter.map_entity(entity)

    # Verify result structure regardless of match success
    assert hasattr(result, "hpo_code")
    assert hasattr(result, "hpo_term")
    assert hasattr(result, "confidence")
    assert hasattr(result, "extracted_terms")
    assert isinstance(result.extracted_terms, list)
