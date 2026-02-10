#!/usr/bin/env python3
"""Generate Q2HPO review dataset using GenOMA LLM-based ontology mapping.

This script generates a candidate pair dataset for human review by:
1. Loading the SAME 100 questions as Q2LOINC (seed=42 stratified sample)
2. Using GenOMA to map each question to HPO codes
3. Caching results to avoid repeated API costs
4. Exporting in a format compatible with the campaign pair import system

Prerequisites:
    export OPENAI_API_KEY="sk-..."
    uv add langgraph langchain langchain-openai openai  # if not installed

Usage:
    # Smoke test (single question, no API cost if cached)
    uv run python scripts/generate_q2hpo_review.py --smoke-test

    # Full run
    uv run python scripts/generate_q2hpo_review.py

    # Dry run (shows what would be processed without calling GenOMA)
    uv run python scripts/generate_q2hpo_review.py --dry-run

Output:
    data/review/q2hpo_review_100.json - Final review dataset
    data/review/generation_log_hpo.json - Timing, cache stats, errors
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/questionnaires")
QUESTIONS_PATH = DATA_DIR / "processed" / "questions.json"
OUTPUT_DIR = Path("data/review")
CACHE_DIR = Path("data/cache/genoma")

# Sampling configuration (SAME as Q2LOINC for consistency)
SAMPLE_DISTRIBUTION = {
    "Arivale": 34,
    "Israeli10K": 33,
    "UKBB": 33,
}
TOTAL_SAMPLE = sum(SAMPLE_DISTRIBUTION.values())  # 100


def check_openai_api_key() -> bool:
    """Check if OPENAI_API_KEY is set."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def load_questions() -> list[dict[str, Any]]:
    """Load all questions from processed JSON."""
    with open(QUESTIONS_PATH) as f:
        return json.load(f)


def stratified_sample(
    questions: list[dict[str, Any]],
    distribution: dict[str, int],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample questions stratified by source questionnaire.

    Uses SAME seed as Q2LOINC to get identical sample.
    """
    random.seed(seed)

    by_source: dict[str, list[dict[str, Any]]] = {}
    for q in questions:
        source = q.get("metadata", {}).get("source_questionnaire", "Unknown")
        by_source.setdefault(source, []).append(q)

    sampled: list[dict[str, Any]] = []
    for source, target_count in distribution.items():
        available = by_source.get(source, [])
        if len(available) < target_count:
            logger.warning(
                f"Source {source} has only {len(available)} questions, requested {target_count}"
            )
            sampled.extend(available)
        else:
            sampled.extend(random.sample(available, target_count))

    logger.info(f"Sampled {len(sampled)} questions from {len(distribution)} sources")
    return sampled


def question_to_entity(question: dict[str, Any]) -> Any:
    """Convert question dict to BaseEntity for GenOMA."""
    from biovector_eval.base.entity import BaseEntity

    return BaseEntity(
        id=question.get("id", ""),
        name=question.get("name", ""),
        synonyms=question.get("synonyms", []),
        metadata=question.get("metadata", {}),
    )


def build_pair_record(
    question: dict[str, Any],
    genoma_result: Any,
) -> dict[str, Any]:
    """Build a pair record matching the campaign import schema.

    Uses same field names as Q2LOINC for consistency:
    - source_text, source_dataset, source_id
    - target_text, target_dataset, target_id
    - llm_confidence, llm_model, llm_reasoning
    - target_metadata (HPO-specific fields)
    """
    # Handle unmappable/failed cases
    if not genoma_result.success or not genoma_result.has_match:
        target_id = "NO_MATCH"
        target_text = genoma_result.error_message or "Not mappable to HPO"
    else:
        target_id = genoma_result.hpo_code
        target_text = genoma_result.hpo_term

    # Build reasoning string from extracted terms
    reasoning_parts = []
    if genoma_result.extracted_terms:
        reasoning_parts.append(f"Extracted terms: {', '.join(genoma_result.extracted_terms)}")
    if genoma_result.hpo_term:
        reasoning_parts.append(f"Mapped to HPO: {genoma_result.hpo_term}")
    if genoma_result.error_message:
        reasoning_parts.append(f"Note: {genoma_result.error_message}")

    return {
        "source_text": question.get("name", ""),
        "source_dataset": question.get("metadata", {}).get("source_questionnaire", "Unknown"),
        "source_id": question.get("id", ""),
        "target_text": target_text,
        "target_dataset": "HPO",
        "target_id": target_id,
        "llm_confidence": genoma_result.confidence,
        "llm_model": genoma_result.model_version,
        "llm_reasoning": " | ".join(reasoning_parts) if reasoning_parts else "",
        "target_metadata": {
            "extracted_terms": genoma_result.extracted_terms,
            "is_mappable": genoma_result.success and not genoma_result.error_type,
            "cached": genoma_result.cached,
            "error_type": genoma_result.error_type or None,
        },
        "source_metadata": question.get("metadata", {}),
    }


def run_smoke_test() -> bool:
    """Run single-question smoke test to verify GenOMA adapter interface.

    Returns True if adapter works as expected.
    """
    logger.info("=" * 60)
    logger.info("SMOKE TEST: Verifying GenOMA adapter interface")
    logger.info("=" * 60)

    # Check API key
    if not check_openai_api_key():
        logger.error("OPENAI_API_KEY not set. Export it before running:")
        logger.error('  export OPENAI_API_KEY="sk-..."')
        return False

    try:
        from biovector_eval.base.entity import BaseEntity
        from biovector_eval.domains.questionnaires.harmonization.genoma_adapter import (
            GenOMAAdapter,
        )

        logger.info("✓ GenOMA adapter imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import GenOMA adapter: {e}")
        logger.error("Install dependencies: uv add langgraph langchain langchain-openai openai")
        return False

    # Create adapter
    adapter = GenOMAAdapter(
        use_cache=True,
        cache_dir=str(CACHE_DIR),
        model_version="gpt-4",
    )
    logger.info(f"✓ GenOMAAdapter initialized (cache_dir={CACHE_DIR})")

    # Test entity
    test_entity = BaseEntity(
        id="smoke_test_001",
        name="Do you experience difficulty sleeping?",
        synonyms=[],
        metadata={"source_questionnaire": "SmokeTest"},
    )

    logger.info(f'Test question: "{test_entity.name}"')

    # Dry run first
    dry_result = adapter.dry_run(test_entity)
    logger.info("Dry run result:")
    logger.info(f"  - Would invoke GenOMA: {dry_result['would_invoke']}")
    logger.info(f"  - Cached exists: {dry_result['cached_exists']}")
    logger.info(f"  - Inferred field_type: {dry_result['inferred_field_type']}")

    # Actual mapping
    logger.info("Invoking GenOMA...")
    try:
        result = adapter.map_entity(test_entity, timeout=60.0)
    except Exception as e:
        logger.error(f"GenOMA invocation failed: {e}")
        return False

    # Print full result
    logger.info("")
    logger.info("GenOMAResult fields:")
    logger.info(f"  - hpo_code: {result.hpo_code!r}")
    logger.info(f"  - hpo_term: {result.hpo_term!r}")
    logger.info(f"  - confidence: {result.confidence}")
    logger.info(f"  - extracted_terms: {result.extracted_terms}")
    logger.info(f"  - success: {result.success}")
    logger.info(f"  - error_type: {result.error_type!r}")
    logger.info(f"  - error_message: {result.error_message!r}")
    logger.info(f"  - cached: {result.cached}")
    logger.info(f"  - model_version: {result.model_version}")
    logger.info(f"  - has_match: {result.has_match}")

    if result.success:
        logger.info("")
        logger.info("✓ SMOKE TEST PASSED")
        logger.info(f"  HPO: {result.hpo_code} - {result.hpo_term}")
        logger.info(f"  Confidence: {result.confidence:.2f}")
        return True
    else:
        logger.warning("")
        logger.warning("⚠ SMOKE TEST: No match found (may be expected for some questions)")
        logger.warning(f"  Error type: {result.error_type}")
        logger.warning(f"  Message: {result.error_message}")
        return True  # Still consider it passed if adapter didn't crash


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate Q2HPO review dataset")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42, same as Q2LOINC)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without calling GenOMA",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run single-question test to verify GenOMA adapter",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit processing to N questions (0 = all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between GenOMA calls in seconds (rate limiting)",
    )
    args = parser.parse_args()

    # Smoke test mode
    if args.smoke_test:
        success = run_smoke_test()
        exit(0 if success else 1)

    # Check API key for non-dry-run
    if not args.dry_run and not check_openai_api_key():
        logger.error("OPENAI_API_KEY not set. Export it before running:")
        logger.error('  export OPENAI_API_KEY="sk-..."')
        logger.error("")
        logger.error("Or run with --dry-run to preview without API calls.")
        exit(1)

    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading questions...")
    questions = load_questions()
    logger.info(f"Loaded {len(questions)} questions")

    # Stratified sampling (SAME as Q2LOINC)
    logger.info("Sampling questions (seed=42, same as Q2LOINC)...")
    sampled = stratified_sample(questions, SAMPLE_DISTRIBUTION, seed=args.seed)

    # Initialize GenOMA adapter
    if not args.dry_run:
        from biovector_eval.domains.questionnaires.harmonization.genoma_adapter import (
            GenOMAAdapter,
        )

        adapter = GenOMAAdapter(
            use_cache=True,
            cache_dir=str(CACHE_DIR),
            model_version="gpt-4",
        )
        logger.info(f"GenOMA adapter initialized (cache_dir={CACHE_DIR})")

    # Process questions
    pairs = []
    errors = []
    cache_hits = 0
    cache_misses = 0
    processing_limit = args.limit if args.limit > 0 else len(sampled)

    for i, question in enumerate(sampled[:processing_limit]):
        q_text = question.get("name", "")
        q_id = question.get("id", "")
        source = question.get("metadata", {}).get("source_questionnaire", "Unknown")

        logger.info(f"\n[{i + 1}/{processing_limit}] Processing: {q_text[:60]}...")
        logger.info(f"  Source: {source}, ID: {q_id}")

        if args.dry_run:
            entity = question_to_entity(question)
            dry_result = adapter.dry_run(entity) if not args.dry_run else {}
            logger.info(
                f"  [DRY RUN] Would invoke: True, Cached: {dry_result.get('cached_exists', '?')}"
            )
            continue

        # Convert to entity and map
        entity = question_to_entity(question)

        try:
            result = adapter.map_entity(entity, timeout=60.0, max_retries=2)

            # Track cache stats
            if result.cached:
                cache_hits += 1
                logger.info("  [CACHE HIT]")
            else:
                cache_misses += 1
                logger.info("  [API CALL]")
                # Rate limiting delay only for uncached calls
                if i < processing_limit - 1:  # Don't delay after last question
                    time.sleep(args.delay)

            if result.has_match:
                logger.info(f"  HPO: {result.hpo_code} - {result.hpo_term}")
                logger.info(f"  Confidence: {result.confidence:.2f}")
            else:
                logger.info(f"  No match: {result.error_type or 'unmappable'}")

            # Build pair record
            pair = build_pair_record(question, result)
            pairs.append(pair)

        except Exception as e:
            logger.error(f"  Error processing question: {e}")
            errors.append(
                {
                    "question_id": q_id,
                    "question_text": q_text,
                    "error": str(e),
                }
            )
            # Create error pair record
            from biovector_eval.domains.questionnaires.harmonization.genoma_adapter import (
                GenOMAResult,
            )

            error_result = GenOMAResult(
                hpo_code="",
                hpo_term="",
                confidence=0.0,
                extracted_terms=[],
                raw_state={},
                success=False,
                error_type="exception",
                error_message=str(e),
            )
            pair = build_pair_record(question, error_result)
            pairs.append(pair)

    if args.dry_run:
        logger.info("\n[DRY RUN] No output files generated.")
        return

    # Save results
    output_path = OUTPUT_DIR / "q2hpo_review_100.json"
    output_data = {"pairs": pairs}
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"\nSaved {len(pairs)} pairs: {output_path}")

    # Save generation log
    elapsed = time.time() - start_time
    log_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "elapsed_human": f"{elapsed / 60:.1f} minutes",
        "total_questions": len(sampled),
        "processed": len(sampled[:processing_limit]),
        "successful_pairs": len([p for p in pairs if p["target_id"] != "NO_MATCH"]),
        "no_match_pairs": len([p for p in pairs if p["target_id"] == "NO_MATCH"]),
        "errors": len(errors),
        "error_details": errors,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": cache_hits / max(1, cache_hits + cache_misses),
        "sample_distribution": SAMPLE_DISTRIBUTION,
        "model_version": "gpt-4",
        "seed": args.seed,
        "delay_seconds": args.delay,
    }
    log_path = OUTPUT_DIR / "generation_log_hpo.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    logger.info(f"Saved generation log: {log_path}")

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total time: {elapsed / 60:.1f} minutes")
    logger.info(f"Successful HPO mappings: {log_data['successful_pairs']}")
    logger.info(f"No match: {log_data['no_match_pairs']}")
    logger.info(f"Errors: {len(errors)}")
    logger.info(
        f"Cache: {cache_hits} hits, {cache_misses} misses ({log_data['cache_hit_rate']:.0%} hit rate)"
    )


if __name__ == "__main__":
    main()
