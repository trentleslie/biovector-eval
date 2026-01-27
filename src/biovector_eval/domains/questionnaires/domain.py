"""Questionnaires domain implementation.

Implements the Domain protocol for questionnaire items, supporting
harmonization across multiple questionnaire sources, LOINC mapping,
and HPO mapping via GenOMA LLM integration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from biovector_eval.base.domain import DomainConfig
from biovector_eval.base.entity import BaseEntity
from biovector_eval.base.ground_truth import GroundTruthDataset, GroundTruthQuery
from biovector_eval.base.loaders import load_entities
from biovector_eval.domains.questionnaires.models import QUESTIONNAIRE_MODELS

if TYPE_CHECKING:
    from biovector_eval.domains.questionnaires.harmonization.candidate_pair import (
        CandidatePairDataset,
    )

logger = logging.getLogger(__name__)


class QuestionnairesDomain:
    """Domain implementation for questionnaire items.

    Handles questionnaire data from multiple sources (NHANES, UK Biobank,
    custom surveys) for cross-questionnaire harmonization and LOINC mapping.

    The domain supports:
    - Loading consolidated questions from JSON
    - Multi-source provenance tracking via metadata
    - Ground truth generation for harmonization evaluation
    - Biomedical embedding models for clinical survey text
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        entities_file: str = "questions.json",
    ):
        """Initialize questionnaires domain.

        Args:
            data_dir: Base directory for questionnaire data.
                     Defaults to data/questionnaires/ in project root.
            entities_file: Name of consolidated questions file.
        """
        if data_dir is None:
            # Default to data/questionnaires/ relative to project root
            current = Path.cwd()
            while current != current.parent:
                if (current / "pyproject.toml").exists():
                    data_dir = current / "data" / "questionnaires"
                    break
                current = current.parent
            else:
                data_dir = Path("data/questionnaires")

        self._data_dir = Path(data_dir)
        self._config = DomainConfig(
            name="questionnaires",
            data_dir=self._data_dir,
            entities_file=entities_file,
            ground_truth_file="ground_truth.json",
            id_column="id",
            name_column="name",
            synonym_columns=[],
            file_format="json",
        )

    @property
    def name(self) -> str:
        """Domain identifier."""
        return "questionnaires"

    @property
    def config(self) -> DomainConfig:
        """Domain configuration."""
        return self._config

    @property
    def processed_dir(self) -> Path:
        """Directory for processed data files."""
        return self._data_dir / "processed"

    @property
    def raw_dir(self) -> Path:
        """Directory for raw data files."""
        return self._data_dir / "raw"

    def load_entities(self, path: Path | str | None = None) -> list[BaseEntity]:
        """Load questionnaire entities from file.

        Args:
            path: Optional override path. Uses config default if not provided.

        Returns:
            List of BaseEntity objects representing questions.
        """
        if path is None:
            # Try processed directory first
            processed_path = self.processed_dir / self._config.entities_file
            if processed_path.exists():
                path = processed_path
            else:
                # Fall back to default path
                path = self._config.entities_path

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Questions file not found at {path}. "
                "Please parse questionnaire data first using the parsers module."
            )

        return load_entities(path)

    def load_entities_by_source(
        self,
        source: str,
        path: Path | str | None = None,
    ) -> list[BaseEntity]:
        """Load entities filtered by source questionnaire.

        Args:
            source: Source questionnaire name (e.g., "NHANES").
            path: Optional override path for entities file.

        Returns:
            List of BaseEntity objects from the specified source.
        """
        all_entities = self.load_entities(path)
        return [
            e for e in all_entities if e.metadata.get("source_questionnaire") == source
        ]

    def get_sources(self, path: Path | str | None = None) -> list[str]:
        """Get list of unique source questionnaires in the dataset.

        Args:
            path: Optional override path for entities file.

        Returns:
            Sorted list of source questionnaire names.
        """
        all_entities = self.load_entities(path)
        sources = {
            e.metadata.get("source_questionnaire", "unknown") for e in all_entities
        }
        return sorted(sources)

    def generate_ground_truth(
        self,
        entities: list,
        **kwargs,
    ) -> GroundTruthDataset:
        """Generate ground truth dataset for questionnaires.

        For questionnaire harmonization, ground truth can be generated from:
        - Exact text matches (same question in different sources)
        - Near-duplicate detection within sources
        - Known harmonization mappings (if available)

        Args:
            entities: List of questionnaire entities.
            **kwargs: Generation parameters:
                - seed: Random seed (default: 42)
                - exact: Number of exact match queries
                - within_source: Number of within-source similarity queries

        Returns:
            GroundTruthDataset with queries for evaluation.
        """
        import random

        seed = kwargs.get("seed", 42)
        exact_count = kwargs.get("exact", 100)
        within_source_count = kwargs.get("within_source", 50)

        random.seed(seed)

        # Convert to BaseEntity if needed
        if entities and not isinstance(entities[0], BaseEntity):
            entities = [BaseEntity.from_dict(e) for e in entities]

        queries: list[GroundTruthQuery] = []

        # Generate exact match queries (using question text)
        entities_with_names = [e for e in entities if e.name]
        if entities_with_names:
            sampled = random.sample(
                entities_with_names, min(exact_count, len(entities_with_names))
            )
            for entity in sampled:
                queries.append(
                    GroundTruthQuery(
                        query=entity.name,
                        expected=[entity.id],
                        category="exact_match",
                        notes=f"Source: {entity.metadata.get('source_questionnaire', 'unknown')}",
                    )
                )

        # Generate within-source queries (questions from same source should
        # potentially match if they're related)
        sources: dict[str, list[BaseEntity]] = {}
        for entity in entities:
            source = entity.metadata.get("source_questionnaire", "unknown")
            if source not in sources:
                sources[source] = []
            sources[source].append(entity)

        # Sample entities that have synonyms for synonym-based queries
        entities_with_synonyms = [e for e in entities if e.synonyms]
        if entities_with_synonyms:
            sampled = random.sample(
                entities_with_synonyms,
                min(within_source_count, len(entities_with_synonyms)),
            )
            for entity in sampled:
                synonym = random.choice(entity.synonyms)
                queries.append(
                    GroundTruthQuery(
                        query=synonym,
                        expected=[entity.id],
                        category="synonym_match",
                        notes=f"Original: {entity.name}",
                    )
                )

        return GroundTruthDataset(queries=queries)

    def get_embedding_models(self) -> dict[str, str]:
        """Get recommended embedding models for questionnaires.

        Returns:
            Dict mapping model short names to HuggingFace model IDs.
        """
        return {name: config.model_id for name, config in QUESTIONNAIRE_MODELS.items()}

    def harmonize_questions(
        self,
        questions: list[BaseEntity],
        loinc_entities: list[BaseEntity] | None = None,
        search_fn: Callable[[str, int], list[tuple[str, float]]] | None = None,
        use_genoma: bool = True,
        use_vector_search: bool = True,
        genoma_kwargs: dict | None = None,
        loinc_top_k: int = 5,
        loinc_min_score: float = 0.5,
        reconciliation_mode: str = "keep_both",
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> "CandidatePairDataset":
        """Full harmonization pipeline combining vector search and GenOMA.

        This method runs questions through both LOINC vector similarity
        search and GenOMA's LLM-based HPO mapping to produce a comprehensive
        set of candidate pairs for review.

        Args:
            questions: Survey questions to harmonize (as BaseEntity objects).
            loinc_entities: LOINC codes for vector search (optional).
                           Required if use_vector_search=True.
            search_fn: FAISS search function with signature:
                      search_fn(query_text, top_k) -> list[(entity_id, score)]
                      Required if use_vector_search=True.
            use_genoma: Enable GenOMA HPO mapping (requires OpenAI API key).
            use_vector_search: Enable LOINC vector similarity search.
            genoma_kwargs: Optional kwargs for GenOMAAdapter initialization.
            loinc_top_k: Number of LOINC candidates per question.
            loinc_min_score: Minimum similarity score for LOINC matches.
            reconciliation_mode: How to handle dual LOINC/HPO mappings:
                - "keep_both": Include both LOINC and HPO mappings
                - "prefer_loinc": Only add HPO if LOINC score < threshold
                - "prefer_hpo": Only add LOINC if HPO confidence < threshold
                - "highest_confidence": Keep only the most confident mapping
            progress_callback: Optional callback(current, total, stage) for progress.

        Returns:
            CandidatePairDataset with combined LOINC and HPO mappings.

        Example:
            questionnaires = get_domain("questionnaires")
            questions = questionnaires.load_entities()[:100]

            # Run with both pipelines
            results = questionnaires.harmonize_questions(
                questions,
                loinc_entities=loinc.load_entities(),
                search_fn=faiss_search,
                use_genoma=True,
                use_vector_search=True,
            )

            # Check mappings by pathway
            loinc_pairs = results.filter_by_pathway("q2loinc")
            hpo_pairs = results.filter_by_pathway("q2hpo")
        """
        from biovector_eval.domains.questionnaires.harmonization.candidate_pair import (
            CandidatePair,
            CandidatePairDataset,
        )

        pairs: list[CandidatePair] = []
        total = len(questions)
        loinc_scores: dict[str, float] = {}  # Track best LOINC score per question

        # Phase 1: Vector similarity → LOINC
        if use_vector_search:
            if search_fn is None:
                logger.warning(
                    "Vector search enabled but no search_fn provided. Skipping LOINC mapping."
                )
            else:
                loinc_lookup = {e.id: e for e in loinc_entities or []}

                for i, question in enumerate(questions):
                    if progress_callback:
                        progress_callback(i + 1, total, "loinc")

                    try:
                        results = search_fn(question.name, loinc_top_k)
                        best_score = 0.0

                        for entity_id, score in results:
                            if score < loinc_min_score:
                                continue

                            best_score = max(best_score, score)
                            target = loinc_lookup.get(entity_id)
                            if target is None:
                                continue

                            pair = CandidatePair(
                                source_id=question.id,
                                source_text=question.name,
                                source_metadata=question.metadata,
                                target_id=target.id,
                                target_text=target.name,
                                target_metadata=target.metadata,
                                similarity_score=score,
                                match_tier="primary",
                                harmonization_pathway="q2loinc",
                                loinc_code=target.id,
                                confidence_level=self._score_to_confidence(score),
                            )
                            pairs.append(pair)

                        loinc_scores[question.id] = best_score

                    except Exception as e:
                        logger.error(f"LOINC search failed for {question.id}: {e}")

        # Phase 2: GenOMA → HPO
        if use_genoma:
            from biovector_eval.domains.questionnaires.harmonization.genoma_adapter import (
                GenOMAAdapter,
            )

            adapter_kwargs = genoma_kwargs or {}

            # In hybrid mode, use LOINC scores to skip high-confidence matches
            if reconciliation_mode == "prefer_loinc" and loinc_scores:
                adapter_kwargs.setdefault("hybrid_mode", True)
                adapter_kwargs.setdefault("vector_confidence_threshold", 0.85)

            adapter = GenOMAAdapter(**adapter_kwargs)

            for i, question in enumerate(questions):
                if progress_callback:
                    progress_callback(i + 1, total, "hpo")

                # Skip if LOINC already high confidence in prefer_loinc mode
                loinc_score = loinc_scores.get(question.id, 0.0)
                if reconciliation_mode == "prefer_loinc":
                    if not adapter.should_invoke_genoma(question, loinc_score):
                        continue

                try:
                    result = adapter.map_entity(question)
                    if result.has_match:
                        pair = adapter.to_candidate_pair(question, result)

                        # Apply reconciliation logic
                        if reconciliation_mode == "highest_confidence":
                            if result.confidence > loinc_score:
                                pairs.append(pair)
                        elif reconciliation_mode == "prefer_hpo":
                            if result.confidence >= 0.5:  # HPO threshold
                                pairs.append(pair)
                        else:  # keep_both (default)
                            pairs.append(pair)

                except Exception as e:
                    logger.error(f"GenOMA mapping failed for {question.id}: {e}")

        return CandidatePairDataset(
            pairs=pairs,
            metadata={
                "total_questions": total,
                "use_vector_search": use_vector_search,
                "use_genoma": use_genoma,
                "reconciliation_mode": reconciliation_mode,
            },
        )

    def _score_to_confidence(self, score: float) -> str:
        """Convert similarity score to confidence level.

        Args:
            score: Cosine similarity score (0-1)

        Returns:
            Confidence level string
        """
        if score >= 0.85:
            return "high"
        elif score >= 0.65:
            return "medium"
        return "low"
