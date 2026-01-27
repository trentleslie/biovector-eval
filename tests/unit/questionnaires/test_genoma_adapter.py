"""Unit tests for GenOMA adapter.

These tests use mocks to avoid actual GenOMA API calls. The tests verify:
- Entity to GenOMA input conversion
- Result parsing
- Cache behavior
- Error handling
- CandidatePair conversion
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from biovector_eval.base.entity import BaseEntity
from biovector_eval.domains.questionnaires.harmonization.cache import GenOMACache
from biovector_eval.domains.questionnaires.harmonization.genoma_adapter import (
    GenOMAAdapter,
    GenOMAResult,
)


# Sample GenOMA response fixtures
MOCK_GENOMA_RESPONSE_SUCCESS = {
    "text": "How often do you feel anxious?",
    "is_mappable": True,
    "extracted_terms": ["anxiety", "frequency"],
    "validated_mappings": [
        {
            "original": "anxiety",
            "best_match_code": "HP:0000739",
            "best_match_term": "Anxiety",
            "confidence": 0.92,
        }
    ],
    "best_match_code": "HP:0000739",
    "best_match_term": "Anxiety",
    "confidence": 0.92,
}

MOCK_GENOMA_RESPONSE_NO_MATCH = {
    "text": "What is your favorite color?",
    "is_mappable": False,
    "extracted_terms": [],
    "validated_mappings": [],
}

MOCK_GENOMA_RESPONSE_LOW_CONFIDENCE = {
    "text": "Do you exercise regularly?",
    "is_mappable": True,
    "extracted_terms": ["exercise"],
    "validated_mappings": [
        {
            "original": "exercise",
            "best_match_code": "HP:0001288",
            "best_match_term": "Gait disturbance",
            "confidence": 0.35,
        }
    ],
    "best_match_code": "HP:0001288",
    "best_match_term": "Gait disturbance",
    "confidence": 0.35,
}


@pytest.fixture
def sample_entity() -> BaseEntity:
    """Create a sample BaseEntity for testing."""
    return BaseEntity(
        id="q1",
        name="How often do you feel anxious?",
        synonyms=["anxiety frequency", "nervousness"],
        metadata={"source_questionnaire": "Test Survey"},
    )


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "genoma_cache"
    cache_dir.mkdir()
    return cache_dir


class TestGenOMACache:
    """Tests for GenOMACache."""

    def test_cache_miss(self, temp_cache_dir: Path):
        """Test cache returns None for missing entries."""
        cache = GenOMACache(cache_dir=temp_cache_dir)
        result = cache.get("test text", "radio")
        assert result is None
        assert cache.stats["misses"] == 1

    def test_cache_set_and_get(self, temp_cache_dir: Path):
        """Test setting and retrieving cache entries."""
        cache = GenOMACache(cache_dir=temp_cache_dir)

        # Set a value
        test_result = {"hpo_code": "HP:0000001", "confidence": 0.9}
        cache.set("test text", "radio", test_result)

        # Retrieve it
        result = cache.get("test text", "radio")
        assert result == test_result
        assert cache.stats["hits"] == 1

    def test_cache_key_includes_model_version(self, temp_cache_dir: Path):
        """Test that different model versions have different cache keys."""
        cache_v1 = GenOMACache(cache_dir=temp_cache_dir, model_version="gpt-4")
        cache_v2 = GenOMACache(cache_dir=temp_cache_dir, model_version="gpt-4-turbo")

        # Set with v1
        cache_v1.set("test", "radio", {"version": "v1"})

        # Should miss with v2 (different model version)
        result = cache_v2.get("test", "radio")
        assert result is None

    def test_cache_has(self, temp_cache_dir: Path):
        """Test checking cache existence without loading."""
        cache = GenOMACache(cache_dir=temp_cache_dir)

        assert not cache.has("test", "radio")
        cache.set("test", "radio", {"data": 1})
        assert cache.has("test", "radio")

    def test_cache_delete(self, temp_cache_dir: Path):
        """Test deleting cache entries."""
        cache = GenOMACache(cache_dir=temp_cache_dir)

        cache.set("test", "radio", {"data": 1})
        assert cache.has("test", "radio")

        deleted = cache.delete("test", "radio")
        assert deleted
        assert not cache.has("test", "radio")

        # Delete non-existent returns False
        deleted = cache.delete("nonexistent", "radio")
        assert not deleted

    def test_cache_clear(self, temp_cache_dir: Path):
        """Test clearing all cache entries."""
        cache = GenOMACache(cache_dir=temp_cache_dir)

        cache.set("test1", "radio", {"data": 1})
        cache.set("test2", "checkbox", {"data": 2})
        assert cache.stats["cached_entries"] == 2

        count = cache.clear()
        assert count == 2
        assert cache.stats["cached_entries"] == 0


class TestGenOMAAdapter:
    """Tests for GenOMAAdapter."""

    def test_entity_to_genoma_input(self, sample_entity: BaseEntity):
        """Test converting BaseEntity to GenOMA input format."""
        adapter = GenOMAAdapter(use_cache=False)
        genoma_input = adapter.entity_to_genoma_input(sample_entity)

        assert genoma_input["text"] == "How often do you feel anxious?"
        assert genoma_input["field_type"] == "radio"  # Default

    def test_infer_field_type_checkbox(self):
        """Test field type inference for checkbox questions."""
        adapter = GenOMAAdapter(use_cache=False)
        entity = BaseEntity(
            id="q1",
            name="Which of the following symptoms have you experienced?",
            synonyms=[],
            metadata={"response_options": ["Headache", "Nausea", "Fatigue", "Dizziness"]},
        )

        field_type = adapter._infer_field_type(entity)
        assert field_type == "checkbox"

    def test_infer_field_type_short(self):
        """Test field type inference for short-answer questions."""
        adapter = GenOMAAdapter(use_cache=False)
        entity = BaseEntity(
            id="q1",
            name="Age:",  # Very short
            synonyms=[],
            metadata={},
        )

        field_type = adapter._infer_field_type(entity)
        assert field_type == "short"

    def test_parse_result_success(self):
        """Test parsing successful GenOMA response."""
        adapter = GenOMAAdapter(use_cache=False)
        result = adapter._parse_result(MOCK_GENOMA_RESPONSE_SUCCESS)

        assert result.hpo_code == "HP:0000739"
        assert result.hpo_term == "Anxiety"
        assert result.confidence == 0.92
        assert "anxiety" in result.extracted_terms
        assert result.success

    def test_parse_result_not_mappable(self):
        """Test parsing non-mappable question response."""
        adapter = GenOMAAdapter(use_cache=False)
        result = adapter._parse_result(MOCK_GENOMA_RESPONSE_NO_MATCH)

        assert result.hpo_code == ""
        assert result.confidence == 0.0
        assert result.error_type == "no_match"
        assert result.success  # Not an error, just not mappable

    def test_confidence_to_level(self):
        """Test confidence score to level conversion."""
        adapter = GenOMAAdapter(use_cache=False)

        assert adapter._confidence_to_level(0.95) == "high"
        assert adapter._confidence_to_level(0.85) == "high"
        assert adapter._confidence_to_level(0.70) == "medium"
        assert adapter._confidence_to_level(0.50) == "medium"
        assert adapter._confidence_to_level(0.30) == "low"

    def test_determine_quality(self):
        """Test confidence to match quality mapping."""
        adapter = GenOMAAdapter(use_cache=False)

        assert adapter._determine_quality(0.90) == "auto_accept"
        assert adapter._determine_quality(0.70) == "needs_review"
        assert adapter._determine_quality(0.30) == "rejected"

    def test_should_invoke_genoma_hybrid_mode(self, sample_entity: BaseEntity):
        """Test hybrid mode skips LLM for high-confidence vector matches."""
        adapter = GenOMAAdapter(
            use_cache=False,
            hybrid_mode=True,
            vector_confidence_threshold=0.85,
        )

        # High LOINC score - should NOT invoke GenOMA
        assert not adapter.should_invoke_genoma(sample_entity, loinc_similarity=0.90)

        # Low LOINC score - should invoke GenOMA
        assert adapter.should_invoke_genoma(sample_entity, loinc_similarity=0.60)

        # No LOINC score - should invoke GenOMA
        assert adapter.should_invoke_genoma(sample_entity, loinc_similarity=None)

    def test_dry_run(self, sample_entity: BaseEntity, temp_cache_dir: Path):
        """Test dry run returns preview without API call."""
        adapter = GenOMAAdapter(use_cache=True, cache_dir=temp_cache_dir)

        dry_result = adapter.dry_run(sample_entity)

        assert dry_result["input"]["text"] == sample_entity.name
        assert "field_type" in dry_result["input"]
        assert dry_result["entity_id"] == "q1"
        assert not dry_result["cached_exists"]

    @patch(
        "biovector_eval.domains.questionnaires.harmonization.genoma_adapter.GenOMAAdapter._get_genoma_graph"
    )
    def test_map_entity_success(
        self,
        mock_get_graph: MagicMock,
        sample_entity: BaseEntity,
        temp_cache_dir: Path,
    ):
        """Test successful mapping with mocked GenOMA graph."""
        # Set up mock
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = MOCK_GENOMA_RESPONSE_SUCCESS
        mock_get_graph.return_value = mock_graph

        adapter = GenOMAAdapter(use_cache=False)
        result = adapter.map_entity(sample_entity)

        assert result.success
        assert result.hpo_code == "HP:0000739"
        assert result.hpo_term == "Anxiety"
        assert result.confidence == 0.92
        mock_graph.invoke.assert_called_once()

    @patch(
        "biovector_eval.domains.questionnaires.harmonization.genoma_adapter.GenOMAAdapter._get_genoma_graph"
    )
    def test_map_entity_uses_cache(
        self,
        mock_get_graph: MagicMock,
        sample_entity: BaseEntity,
        temp_cache_dir: Path,
    ):
        """Test that cached results are returned without API call."""
        # Set up mock
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = MOCK_GENOMA_RESPONSE_SUCCESS
        mock_get_graph.return_value = mock_graph

        adapter = GenOMAAdapter(use_cache=True, cache_dir=temp_cache_dir)

        # First call - should hit API
        result1 = adapter.map_entity(sample_entity)
        assert mock_graph.invoke.call_count == 1

        # Second call - should use cache
        result2 = adapter.map_entity(sample_entity)
        assert mock_graph.invoke.call_count == 1  # No additional call
        assert result2.cached

    @patch(
        "biovector_eval.domains.questionnaires.harmonization.genoma_adapter.GenOMAAdapter._get_genoma_graph"
    )
    def test_map_entity_retry_on_error(
        self,
        mock_get_graph: MagicMock,
        sample_entity: BaseEntity,
    ):
        """Test retry behavior on transient errors."""
        mock_graph = MagicMock()
        # Fail twice, then succeed
        mock_graph.invoke.side_effect = [
            Exception("Network error"),
            Exception("Timeout"),
            MOCK_GENOMA_RESPONSE_SUCCESS,
        ]
        mock_get_graph.return_value = mock_graph

        adapter = GenOMAAdapter(use_cache=False)
        result = adapter.map_entity(sample_entity, max_retries=3)

        assert result.success
        assert mock_graph.invoke.call_count == 3

    @patch(
        "biovector_eval.domains.questionnaires.harmonization.genoma_adapter.GenOMAAdapter._get_genoma_graph"
    )
    def test_map_entity_all_retries_failed(
        self,
        mock_get_graph: MagicMock,
        sample_entity: BaseEntity,
    ):
        """Test error result when all retries exhausted."""
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = Exception("Persistent error")
        mock_get_graph.return_value = mock_graph

        adapter = GenOMAAdapter(use_cache=False)
        result = adapter.map_entity(sample_entity, max_retries=2)

        assert not result.success
        assert result.error_type == "api_error"
        assert "Persistent error" in result.error_message

    def test_to_candidate_pair(self, sample_entity: BaseEntity):
        """Test conversion of GenOMAResult to CandidatePair."""
        adapter = GenOMAAdapter(use_cache=False)

        genoma_result = GenOMAResult(
            hpo_code="HP:0000739",
            hpo_term="Anxiety",
            confidence=0.92,
            extracted_terms=["anxiety", "frequency"],
            raw_state=MOCK_GENOMA_RESPONSE_SUCCESS,
        )

        pair = adapter.to_candidate_pair(sample_entity, genoma_result)

        assert pair.source_id == "q1"
        assert pair.source_text == sample_entity.name
        assert pair.target_id == "HP:0000739"
        assert pair.target_text == "Anxiety"
        assert pair.harmonization_pathway == "q2hpo"
        assert pair.hpo_code == "HP:0000739"
        assert pair.llm_processed
        assert pair.confidence_level == "high"
        assert pair.match_quality == "auto_accept"


class TestGenOMAResult:
    """Tests for GenOMAResult dataclass."""

    def test_has_match_true(self):
        """Test has_match returns True when HPO code is present."""
        result = GenOMAResult(
            hpo_code="HP:0000739",
            hpo_term="Anxiety",
            confidence=0.9,
            extracted_terms=[],
            raw_state={},
        )
        assert result.has_match

    def test_has_match_false_no_code(self):
        """Test has_match returns False when no HPO code."""
        result = GenOMAResult(
            hpo_code="",
            hpo_term="",
            confidence=0.0,
            extracted_terms=[],
            raw_state={},
        )
        assert not result.has_match

    def test_has_match_false_on_error(self):
        """Test has_match returns False on error."""
        result = GenOMAResult(
            hpo_code="HP:0000739",
            hpo_term="Anxiety",
            confidence=0.9,
            extracted_terms=[],
            raw_state={},
            success=False,
            error_type="api_error",
        )
        assert not result.has_match
