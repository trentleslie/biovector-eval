#!/usr/bin/env python3
"""Generate Q2LOINC review dataset using vector search + Claude Opus reasoning.

This script generates a candidate pair dataset for human review by:
1. Sampling 100 questions stratified by source (34 Arivale, 33 Israeli10K, 33 UKBB)
2. Using vector search to find top-5 LOINC candidates per question
3. Using Claude Opus to select best match and provide reasoning
4. Exporting in a format compatible with the campaign pair import system

Usage:
    uv run python scripts/generate_q2loinc_review.py

Output:
    data/review/q2loinc_review_100.json - Final review dataset
    data/review/sample_questions.json - The 100 sampled source questions
    data/review/generation_log.json - Timing, costs, any errors
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/questionnaires")
QUESTIONS_PATH = DATA_DIR / "processed" / "questions.json"
LOINC_ENTITIES_PATH = DATA_DIR / "loinc_entities.json"
LOINC_EMBEDDINGS_PATH = DATA_DIR / "embeddings" / "loinc" / "all-minilm-l6-v2_single-primary.npy"
LOINC_ID_MAPPING_PATH = DATA_DIR / "embeddings" / "loinc" / "id_mapping.json"
OUTPUT_DIR = Path("data/review")

# Sampling configuration
SAMPLE_DISTRIBUTION = {
    "Arivale": 34,
    "Israeli10K": 33,
    "UKBB": 33,
}
TOTAL_SAMPLE = sum(SAMPLE_DISTRIBUTION.values())  # 100

# Model configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


@dataclass
class LOINCCandidate:
    """A LOINC code candidate from vector search."""

    code: str
    name: str
    vector_similarity: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OpusResult:
    """Result from Claude Opus reasoning."""

    best_match_code: str
    confidence: float
    reasoning: str
    candidate_scores: list[dict[str, Any]]


def load_questions() -> list[dict[str, Any]]:
    """Load all questions from processed JSON."""
    with open(QUESTIONS_PATH) as f:
        return json.load(f)


def load_loinc_data() -> tuple[list[dict[str, Any]], np.ndarray, list[str]]:
    """Load LOINC entities, embeddings, and ID mapping.

    Returns:
        Tuple of (entities, embeddings, id_list)
    """
    with open(LOINC_ENTITIES_PATH) as f:
        entities = json.load(f)

    embeddings = np.load(LOINC_EMBEDDINGS_PATH)

    with open(LOINC_ID_MAPPING_PATH) as f:
        id_mapping = json.load(f)
        id_list = id_mapping["ids"]

    # Create lookup dict
    entity_lookup = {e["id"]: e for e in entities}

    return entity_lookup, embeddings, id_list


def stratified_sample(
    questions: list[dict[str, Any]],
    distribution: dict[str, int],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample questions stratified by source questionnaire.

    Args:
        questions: All questions.
        distribution: Target count per source.
        seed: Random seed for reproducibility.

    Returns:
        Stratified sample of questions.
    """
    random.seed(seed)

    # Group by source
    by_source: dict[str, list[dict[str, Any]]] = {}
    for q in questions:
        source = q.get("metadata", {}).get("source_questionnaire", "Unknown")
        by_source.setdefault(source, []).append(q)

    # Sample from each source
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


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build FAISS index for inner product (cosine) search.

    The embeddings should already be L2-normalized.
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))
    return index


def search_loinc_candidates(
    question_text: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    id_list: list[str],
    entity_lookup: dict[str, dict[str, Any]],
    top_k: int = 5,
) -> list[LOINCCandidate]:
    """Find top-k LOINC candidates for a question via vector search.

    Args:
        question_text: The question to match.
        model: Sentence transformer for encoding.
        index: FAISS index of LOINC embeddings.
        id_list: List of LOINC codes in index order.
        entity_lookup: Dict mapping LOINC code to entity.
        top_k: Number of candidates to return.

    Returns:
        List of LOINC candidates with similarity scores.
    """
    # Encode question
    q_emb = model.encode(
        [question_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Search
    scores, indices = index.search(q_emb.astype(np.float32), top_k)

    # Build candidates
    candidates = []
    for score, idx in zip(scores[0], indices[0], strict=True):
        loinc_code = id_list[idx]
        entity = entity_lookup.get(loinc_code, {})
        candidates.append(
            LOINCCandidate(
                code=loinc_code,
                name=entity.get("name", "Unknown"),
                vector_similarity=float(score),
                metadata=entity.get("metadata", {}),
            )
        )

    return candidates


def format_candidates_for_prompt(candidates: list[LOINCCandidate]) -> str:
    """Format candidates for the LLM prompt."""
    lines = []
    for i, c in enumerate(candidates, 1):
        lines.append(f'{i}. LOINC {c.code}: "{c.name}" (similarity: {c.vector_similarity:.3f})')
    return "\n".join(lines)


def extract_json_from_text(text: str) -> dict[str, Any]:
    """Extract JSON from text that may contain markdown code blocks.

    Args:
        text: Raw text that may contain JSON wrapped in markdown.

    Returns:
        Parsed JSON dict.

    Raises:
        json.JSONDecodeError: If no valid JSON found.
    """
    # Try to extract from markdown code block
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1).strip()
        return json.loads(json_str)

    # Try direct parse
    return json.loads(text)


def get_opus_reasoning(
    question: str,
    candidates: list[LOINCCandidate],
    timeout: int = 120,
) -> OpusResult:
    """Get Claude Opus to evaluate candidates via Claude Code CLI.

    Args:
        question: The survey question.
        candidates: Top-k LOINC candidates from vector search.
        timeout: CLI timeout in seconds.

    Returns:
        OpusResult with best match, confidence, and reasoning.
    """
    candidates_text = format_candidates_for_prompt(candidates)

    prompt = f"""You are evaluating LOINC code matches for a survey question. LOINC codes are standard identifiers for clinical observations and questionnaire items.

Survey Question: "{question}"

Top-5 LOINC candidates (by vector similarity):
{candidates_text}

Task:
1. Evaluate each candidate for semantic match to the question
2. Select the BEST matching LOINC code (or indicate "NO_MATCH" if none are appropriate)
3. Provide a confidence score (0.0-1.0) for your selection
4. Explain your reasoning briefly

Important: Consider both the question intent and the LOINC code's intended clinical use.

Respond ONLY with valid JSON (no markdown, no code blocks, no extra text):
{{"best_match_code": "LOINC-CODE or NO_MATCH", "confidence": 0.XX, "reasoning": "Your explanation here", "candidate_scores": [{{"code": "CODE", "confidence": 0.XX}}, ...]}}"""

    try:
        result = subprocess.run(
            [
                "claude",
                "-p",
                prompt,
                "--output-format",
                "json",
                "--model",
                "claude-opus-4-5-20251101",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            logger.error(f"Claude CLI failed: {result.stderr}")
            raise RuntimeError(f"Claude CLI failed: {result.stderr}")

        # Parse the CLI wrapper JSON
        cli_response = json.loads(result.stdout)

        # The CLI returns {"result": "...", ...} where result contains the LLM output
        llm_output = cli_response.get("result", "")

        # Extract the actual JSON from the LLM output (may have markdown)
        content = extract_json_from_text(llm_output)

        return OpusResult(
            best_match_code=content.get("best_match_code", "NO_MATCH"),
            confidence=float(content.get("confidence", 0.0)),
            reasoning=content.get("reasoning", ""),
            candidate_scores=content.get("candidate_scores", []),
        )

    except subprocess.TimeoutExpired:
        logger.error(f"Claude CLI timed out after {timeout}s")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response: {e}")
        logger.error(f"Raw output: {result.stdout[:500]}")
        raise


def build_pair_record(
    question: dict[str, Any],
    candidates: list[LOINCCandidate],
    opus_result: OpusResult,
    entity_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a pair record in the campaign import format.

    Args:
        question: Source question dict.
        candidates: Top-k LOINC candidates.
        opus_result: Claude Opus evaluation result.
        entity_lookup: LOINC entity lookup dict.

    Returns:
        Pair record matching the review app schema.
    """
    # Find the best match entity
    best_code = opus_result.best_match_code
    best_entity = entity_lookup.get(best_code, {})

    # Build top-5 alternatives for metadata
    top_5_loinc = []
    for c in candidates:
        # Find LLM confidence for this candidate
        llm_conf = 0.0
        for cs in opus_result.candidate_scores:
            if cs.get("code") == c.code:
                llm_conf = cs.get("confidence", 0.0)
                break

        top_5_loinc.append(
            {
                "code": c.code,
                "name": c.name,
                "confidence": llm_conf,
                "vector_similarity": c.vector_similarity,
            }
        )

    return {
        "source_text": question.get("name", ""),
        "source_dataset": question.get("metadata", {}).get("source_questionnaire", "Unknown"),
        "source_id": question.get("id", ""),
        "target_text": best_entity.get("name", best_code),
        "target_dataset": "LOINC",
        "target_id": best_code,
        "llm_confidence": opus_result.confidence,
        "llm_model": "claude-opus-4",
        "llm_reasoning": opus_result.reasoning,
        "target_metadata": {
            "top_5_loinc": top_5_loinc,
        },
        "source_metadata": question.get("metadata", {}),
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate Q2LOINC review dataset")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample questions and show candidates without calling Opus",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit processing to N questions (0 = all)",
    )
    args = parser.parse_args()

    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading questions...")
    questions = load_questions()
    logger.info(f"Loaded {len(questions)} questions")

    logger.info("Loading LOINC data...")
    entity_lookup, embeddings, id_list = load_loinc_data()
    logger.info(f"Loaded {len(id_list)} LOINC codes with embeddings")

    # Stratified sampling
    logger.info("Sampling questions...")
    sampled = stratified_sample(questions, SAMPLE_DISTRIBUTION, seed=args.seed)

    # Save sampled questions
    sample_path = OUTPUT_DIR / "sample_questions.json"
    with open(sample_path, "w") as f:
        json.dump(sampled, f, indent=2)
    logger.info(f"Saved sampled questions: {sample_path}")

    # Build FAISS index
    logger.info("Building FAISS index...")
    index = build_faiss_index(embeddings)

    # Load embedding model
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Process each question
    pairs = []
    errors = []
    processing_limit = args.limit if args.limit > 0 else len(sampled)

    for i, question in enumerate(sampled[:processing_limit]):
        q_text = question.get("name", "")
        q_id = question.get("id", "")
        source = question.get("metadata", {}).get("source_questionnaire", "Unknown")

        logger.info(f"\n[{i + 1}/{processing_limit}] Processing: {q_text[:60]}...")
        logger.info(f"  Source: {source}, ID: {q_id}")

        # Vector search
        candidates = search_loinc_candidates(
            q_text, model, index, id_list, entity_lookup, top_k=TOP_K
        )

        logger.info("  Top candidates:")
        for c in candidates[:3]:
            logger.info(f"    {c.code}: {c.name[:50]}... ({c.vector_similarity:.3f})")

        if args.dry_run:
            logger.info("  [DRY RUN - skipping Opus call]")
            continue

        # Get Opus reasoning
        try:
            opus_result = get_opus_reasoning(q_text, candidates)
            logger.info(
                f"  Opus selected: {opus_result.best_match_code} "
                f"(confidence: {opus_result.confidence:.2f})"
            )
            logger.info(f"  Reasoning: {opus_result.reasoning[:100]}...")

            # Build pair record
            pair = build_pair_record(question, candidates, opus_result, entity_lookup)
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

    if args.dry_run:
        logger.info("\n[DRY RUN] No output files generated.")
        return

    # Save results
    output_path = OUTPUT_DIR / "q2loinc_review_100.json"
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
        "successful_pairs": len(pairs),
        "errors": len(errors),
        "error_details": errors,
        "sample_distribution": SAMPLE_DISTRIBUTION,
        "embedding_model": EMBEDDING_MODEL,
        "top_k": TOP_K,
        "seed": args.seed,
    }
    log_path = OUTPUT_DIR / "generation_log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    logger.info(f"Saved generation log: {log_path}")

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total time: {elapsed / 60:.1f} minutes")
    logger.info(f"Successful pairs: {len(pairs)}")
    logger.info(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()
