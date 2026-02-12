#!/usr/bin/env python3
"""Generate Q2Q cross-source question matching review dataset.

This script generates a candidate pair dataset for human review by:
1. Embedding ALL questions from all sources
2. For each question, finding its best cross-source match
3. Ranking ALL pairs globally by similarity
4. Taking the top N most similar pairs for review

This approach surfaces the actual best semantic matches across the entire
dataset, rather than finding matches for randomly sampled questions.

Usage:
    uv run python scripts/generate_q2q_review.py
    uv run python scripts/generate_q2q_review.py --dry-run
    uv run python scripts/generate_q2q_review.py --top-n 200

Output:
    data/review/q2q_review_100.json - Final review dataset (top 100 pairs)
    data/review/q2q_generation_log.json - Timing and similarity statistics
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
OUTPUT_DIR = Path("data/review")

# Model configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_CANDIDATES = 5  # Candidates to store per pair
DEFAULT_TOP_N = 100  # Number of top pairs to output

# Cross-source matching targets (each source matches against OTHER sources)
MATCH_TARGETS: dict[str, list[str]] = {
    "Arivale": ["Israeli10K", "UKBB"],
    "Israeli10K": ["Arivale", "UKBB"],
    "UKBB": ["Arivale", "Israeli10K"],
}

# Confidence thresholds (may need adjustment based on results)
HIGH_CONFIDENCE_THRESHOLD = 0.85
MEDIUM_CONFIDENCE_THRESHOLD = 0.50


@dataclass
class QuestionCandidate:
    """A question candidate from vector search."""

    id: str
    text: str
    source: str
    similarity: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceIndex:
    """FAISS index and metadata for a single source."""

    source: str
    index: faiss.IndexFlatIP
    questions: list[dict[str, Any]]
    id_list: list[str]
    embeddings: np.ndarray


@dataclass
class CrossSourcePair:
    """A cross-source question pair with similarity score."""

    source_question: dict[str, Any]
    target_question: dict[str, Any]
    similarity: float
    candidates: list[QuestionCandidate]

    def __lt__(self, other: CrossSourcePair) -> bool:
        """Enable sorting by similarity (descending)."""
        return self.similarity > other.similarity


def load_questions() -> list[dict[str, Any]]:
    """Load all questions from processed JSON."""
    with open(QUESTIONS_PATH) as f:
        return json.load(f)


def group_questions_by_source(
    questions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group questions by source questionnaire."""
    by_source: dict[str, list[dict[str, Any]]] = {}
    for q in questions:
        source = q.get("metadata", {}).get("source_questionnaire", "Unknown")
        by_source.setdefault(source, []).append(q)
    return by_source


def build_source_indices(
    questions_by_source: dict[str, list[dict[str, Any]]],
    model: SentenceTransformer,
) -> dict[str, SourceIndex]:
    """Build FAISS index for each source.

    Args:
        questions_by_source: Questions grouped by source.
        model: Sentence transformer for encoding.

    Returns:
        Dict mapping source name to SourceIndex.
    """
    indices: dict[str, SourceIndex] = {}

    for source, questions in questions_by_source.items():
        logger.info(f"Building index for {source} ({len(questions)} questions)...")

        # Extract texts and IDs
        texts = [q.get("name", "") for q in questions]
        ids = [q.get("id", "") for q in questions]

        # Encode
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Build FAISS index
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings.astype(np.float32))

        indices[source] = SourceIndex(
            source=source,
            index=index,
            questions=questions,
            id_list=ids,
            embeddings=embeddings,
        )

        logger.info(f"  Built index with {len(ids)} vectors")

    return indices


def find_cross_source_matches(
    source_indices: dict[str, SourceIndex],
    top_k: int = TOP_K_CANDIDATES,
) -> list[CrossSourcePair]:
    """Find best cross-source match for every question.

    For each question, searches OTHER sources and finds the best match.
    Returns all pairs sorted by similarity.

    Args:
        source_indices: Dict of per-source FAISS indices.
        top_k: Number of candidates to store per pair.

    Returns:
        List of all cross-source pairs, sorted by similarity descending.
    """
    all_pairs: list[CrossSourcePair] = []

    for source_name, source_idx in source_indices.items():
        target_sources = MATCH_TARGETS.get(source_name, [])
        if not target_sources:
            continue

        logger.info(f"\nMatching {source_name} against {target_sources}...")

        # Build combined index of target sources
        target_questions: list[dict[str, Any]] = []
        target_embeddings_list: list[np.ndarray] = []
        target_source_labels: list[str] = []

        for target_name in target_sources:
            if target_name not in source_indices:
                continue
            target_idx = source_indices[target_name]
            target_questions.extend(target_idx.questions)
            target_embeddings_list.append(target_idx.embeddings)
            target_source_labels.extend([target_name] * len(target_idx.questions))

        if not target_questions:
            continue

        # Combine embeddings
        target_embeddings = np.vstack(target_embeddings_list)

        # Build combined FAISS index
        d = target_embeddings.shape[1]
        combined_index = faiss.IndexFlatIP(d)
        combined_index.add(target_embeddings.astype(np.float32))

        # Search all source questions at once (batch search)
        scores, indices = combined_index.search(source_idx.embeddings.astype(np.float32), top_k)

        # Build pairs
        for i, source_q in enumerate(source_idx.questions):
            candidates: list[QuestionCandidate] = []

            for j in range(top_k):
                idx = indices[i, j]
                if idx < 0:
                    continue

                target_q = target_questions[idx]
                target_source = target_source_labels[idx]
                sim = float(scores[i, j])

                candidates.append(
                    QuestionCandidate(
                        id=target_q.get("id", ""),
                        text=target_q.get("name", ""),
                        source=target_source,
                        similarity=sim,
                        metadata=target_q.get("metadata", {}),
                    )
                )

            if candidates:
                all_pairs.append(
                    CrossSourcePair(
                        source_question=source_q,
                        target_question=target_questions[indices[i, 0]],
                        similarity=float(scores[i, 0]),
                        candidates=candidates,
                    )
                )

        logger.info(f"  Found {len(source_idx.questions)} pairs from {source_name}")

    # Sort by similarity (descending)
    all_pairs.sort()

    logger.info(f"\nTotal cross-source pairs: {len(all_pairs)}")

    # Build lookup of all pair relationships
    pair_lookup: dict[tuple[str, str], CrossSourcePair] = {}
    for pair in all_pairs:
        src_id = pair.source_question.get("id", "")
        tgt_id = pair.target_question.get("id", "")
        pair_lookup[(src_id, tgt_id)] = pair

    # Find mutual pairs (A→B exists AND B→A exists)
    mutual_pairs: list[CrossSourcePair] = []
    seen_pairs: set[tuple[str, str]] = set()

    for pair in all_pairs:
        src_id = pair.source_question.get("id", "")
        tgt_id = pair.target_question.get("id", "")

        # Skip if already processed
        if (src_id, tgt_id) in seen_pairs or (tgt_id, src_id) in seen_pairs:
            continue

        # Check if reverse pair exists
        if (tgt_id, src_id) in pair_lookup:
            # This is a mutual match - keep the higher scoring direction
            mutual_pairs.append(pair)
            seen_pairs.add((src_id, tgt_id))

    logger.info(f"Mutual pairs (A↔B): {len(mutual_pairs)}")
    return mutual_pairs


def classify_confidence(similarity: float) -> str:
    """Classify similarity score into confidence level."""
    if similarity >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    elif similarity >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    else:
        return "low"


def classify_match_quality(similarity: float) -> str:
    """Classify match quality for review workflow."""
    if similarity >= HIGH_CONFIDENCE_THRESHOLD:
        return "auto_accept"
    else:
        return "needs_review"


def pair_to_dict(pair: CrossSourcePair) -> dict[str, Any]:
    """Convert CrossSourcePair to output dict format."""
    source_q = pair.source_question
    target_q = pair.target_question

    # Build top-5 candidates list
    top_5 = [
        {
            "id": c.id,
            "text": c.text,
            "source": c.source,
            "similarity": round(c.similarity, 4),
        }
        for c in pair.candidates
    ]

    return {
        "source_id": source_q.get("id", ""),
        "source_text": source_q.get("name", ""),
        "source_metadata": source_q.get("metadata", {}),
        "target_id": target_q.get("id", ""),
        "target_text": target_q.get("name", ""),
        "target_metadata": target_q.get("metadata", {}),
        "similarity_score": round(pair.similarity, 4),
        "harmonization_pathway": "direct_q2q",
        "confidence_level": classify_confidence(pair.similarity),
        "match_quality": classify_match_quality(pair.similarity),
        "match_tier": "primary",
        "top_5_candidates": top_5,
    }


def compute_similarity_stats(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute descriptive statistics for similarity scores."""
    if not pairs:
        return {}

    scores = [p["similarity_score"] for p in pairs]
    scores_arr = np.array(scores)

    # Histogram bins
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(scores_arr, bins=bins)
    histogram = {f"{bins[i]:.1f}-{bins[i + 1]:.1f}": int(hist[i]) for i in range(len(hist))}

    return {
        "min": round(float(np.min(scores_arr)), 4),
        "max": round(float(np.max(scores_arr)), 4),
        "mean": round(float(np.mean(scores_arr)), 4),
        "median": round(float(np.median(scores_arr)), 4),
        "std": round(float(np.std(scores_arr)), 4),
        "histogram": histogram,
    }


def compute_source_distribution(pairs: list[dict[str, Any]]) -> dict[str, int]:
    """Count pairs by source questionnaire combination."""
    distribution: dict[str, int] = {}
    for p in pairs:
        src = p["source_metadata"].get("source_questionnaire", "Unknown")
        tgt = p["target_metadata"].get("source_questionnaire", "Unknown")
        key = f"{src}→{tgt}"
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Q2Q cross-source question matching review dataset"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Find matches but don't generate output files",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of top pairs to output (default: {DEFAULT_TOP_N})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_CANDIDATES,
        help=f"Number of candidates per pair (default: {TOP_K_CANDIDATES})",
    )
    args = parser.parse_args()

    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading questions...")
    questions = load_questions()
    logger.info(f"Loaded {len(questions)} total questions")

    # Group by source
    questions_by_source = group_questions_by_source(questions)
    for source, qs in questions_by_source.items():
        logger.info(f"  {source}: {len(qs)} questions")

    # Load embedding model
    logger.info(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Build per-source indices
    logger.info("\nBuilding source indices...")
    source_indices = build_source_indices(questions_by_source, model)

    # Find all cross-source matches
    logger.info("\nFinding cross-source matches...")
    all_pairs = find_cross_source_matches(source_indices, top_k=args.top_k)

    # Take top N
    top_pairs = all_pairs[: args.top_n]
    logger.info(f"\nSelected top {len(top_pairs)} pairs by similarity")

    # Preview top matches
    logger.info("\nTop 10 matches:")
    for i, pair in enumerate(top_pairs[:10]):
        src = pair.source_question.get("metadata", {}).get("source_questionnaire", "?")
        tgt = pair.candidates[0].source if pair.candidates else "?"
        logger.info(f"  {i + 1}. [{src}→{tgt}] {pair.similarity:.3f}")
        logger.info(f"      Source: {pair.source_question.get('name', '')[:60]}")
        logger.info(f"      Target: {pair.target_question.get('name', '')[:60]}")

    if args.dry_run:
        logger.info("\n[DRY RUN] No output files generated.")
        return

    # Convert to output format
    pairs_dicts = [pair_to_dict(p) for p in top_pairs]

    # Compute statistics
    stats = compute_similarity_stats(pairs_dicts)
    source_dist = compute_source_distribution(pairs_dicts)

    # Build output data
    output_data = {
        "pairs": pairs_dicts,
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "embedding_model": EMBEDDING_MODEL,
            "total_questions": len(questions),
            "total_cross_source_pairs": len(all_pairs),
            "top_n": len(pairs_dicts),
            "top_k_candidates": args.top_k,
            "confidence_thresholds": {
                "high": HIGH_CONFIDENCE_THRESHOLD,
                "medium": MEDIUM_CONFIDENCE_THRESHOLD,
            },
            "similarity_stats": stats,
            "source_distribution": source_dist,
        },
    }

    # Save results
    output_path = OUTPUT_DIR / f"q2q_review_{len(pairs_dicts)}.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"\nSaved {len(pairs_dicts)} pairs: {output_path}")

    # Save generation log
    elapsed = time.time() - start_time
    log_data = {
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "elapsed_human": f"{elapsed:.1f} seconds",
        "total_questions": len(questions),
        "total_cross_source_pairs": len(all_pairs),
        "pairs_output": len(pairs_dicts),
        "embedding_model": EMBEDDING_MODEL,
        "top_n": args.top_n,
        "top_k_candidates": args.top_k,
        "similarity_stats": stats,
        "source_distribution": source_dist,
        "confidence_distribution": {
            level: sum(1 for p in pairs_dicts if p["confidence_level"] == level)
            for level in ["high", "medium", "low"]
        },
    }
    log_path = OUTPUT_DIR / "q2q_generation_log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    logger.info(f"Saved generation log: {log_path}")

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total time: {elapsed:.1f} seconds")
    logger.info(f"Total cross-source pairs evaluated: {len(all_pairs)}")
    logger.info(f"Top pairs output: {len(pairs_dicts)}")
    logger.info("\nSimilarity distribution (top pairs):")
    if stats:
        logger.info(f"  Min: {stats['min']:.3f}")
        logger.info(f"  Max: {stats['max']:.3f}")
        logger.info(f"  Mean: {stats['mean']:.3f}")
        logger.info(f"  Median: {stats['median']:.3f}")
        logger.info("\nHistogram:")
        for bucket, count in stats.get("histogram", {}).items():
            logger.info(f"  {bucket}: {count}")
    logger.info("\nSource pair distribution:")
    for combo, count in sorted(source_dist.items()):
        logger.info(f"  {combo}: {count}")
    logger.info("\nConfidence levels:")
    for level in ["high", "medium", "low"]:
        count = sum(1 for p in pairs_dicts if p["confidence_level"] == level)
        logger.info(f"  {level}: {count}")


if __name__ == "__main__":
    main()
