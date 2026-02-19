#!/usr/bin/env python3
"""Generate Q2Q Expert-in-the-Loop review dataset for human validation.

This script creates a paired questions dataset from Q2Q transitive clusters,
formatted for the expertintheloop.io review platform. It extracts clusters of
size 2-3 and generates all pairwise combinations for human validation of
semantic matching quality.

Output: data/review/q2q_expertintheloop.tsv

Usage:
    uv run python scripts/generate_q2q_expertintheloop.py
"""

import csv
from itertools import combinations
from pathlib import Path

# Paths
INPUT_PATH = Path("data/review/q2q_clusters_all.tsv")
OUTPUT_PATH = Path("data/review/q2q_expertintheloop.tsv")

# Output columns (matches q2hpo_expertintheloop.tsv + enhanced metadata)
OUTPUT_COLUMNS = [
    "source_text",
    "source_dataset",
    "source_id",
    "target_text",
    "target_dataset",
    "target_id",
    "pair_type",
    "llm_confidence",
    "llm_model",
    "llm_reasoning",
    "cluster_id",
    "cluster_size",
    "is_cross_source",
]


def load_clusters(path: Path) -> dict[str, list[dict]]:
    """Load TSV and group rows by cluster_id.

    Args:
        path: Path to q2q_clusters_all.tsv

    Returns:
        Dictionary mapping cluster_id to list of member rows
    """
    clusters: dict[str, list[dict]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cid = row["cluster_id"]
            clusters.setdefault(cid, []).append(row)
    return clusters


def generate_pairs(
    clusters: dict[str, list[dict]],
) -> list[tuple[dict, dict, str, int, str]]:
    """Generate all pairs from clusters of size 2-3.

    For size-2 clusters: 1 pair (A↔B)
    For size-3 clusters: 3 pairs (A↔B, A↔C, B↔C)

    Args:
        clusters: Dictionary mapping cluster_id to member rows

    Returns:
        List of tuples: (q1, q2, cluster_id, cluster_size, similarity)
    """
    pairs = []
    for cid, members in clusters.items():
        size = int(members[0]["cluster_size"])
        if size not in (2, 3):
            continue
        similarity = members[0]["avg_cluster_similarity"]
        # All combinations of 2 from cluster members
        for q1, q2 in combinations(members, 2):
            pairs.append((q1, q2, cid, size, similarity))
    return pairs


def format_tsv_row(
    q1: dict,
    q2: dict,
    cluster_id: str,
    cluster_size: int,
    similarity: str,
) -> dict:
    """Format pair as TSV row with enhanced metadata for post-review analysis.

    Args:
        q1: First question row from cluster
        q2: Second question row from cluster
        cluster_id: ID of the source cluster
        cluster_size: Size of the cluster (2 or 3)
        similarity: Average cluster similarity score

    Returns:
        Dictionary matching OUTPUT_COLUMNS schema
    """
    src_survey = q1["source_survey"]
    tgt_survey = q2["source_survey"]
    is_cross_source = src_survey != tgt_survey

    return {
        "source_text": q1["question_text"],
        "source_dataset": src_survey,
        "source_id": q1["question_id"],
        "target_text": q2["question_text"],
        "target_dataset": tgt_survey,
        "target_id": q2["question_id"],
        "pair_type": "q2q_cluster_validation",
        "llm_confidence": similarity,
        "llm_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_reasoning": (
            f"Cluster {cluster_id} (size {cluster_size}, sim={similarity}): "
            f"{src_survey} ↔ {tgt_survey} semantic match"
        ),
        "cluster_id": cluster_id,
        "cluster_size": cluster_size,
        "is_cross_source": is_cross_source,
    }


def main() -> None:
    """Main entry point."""
    print(f"Loading clusters from {INPUT_PATH}...")
    clusters = load_clusters(INPUT_PATH)
    print(f"  Found {len(clusters)} total clusters")

    # Count size-2 and size-3 clusters
    size_2_clusters = sum(
        1 for members in clusters.values() if int(members[0]["cluster_size"]) == 2
    )
    size_3_clusters = sum(
        1 for members in clusters.values() if int(members[0]["cluster_size"]) == 3
    )
    print(f"  Size-2 clusters: {size_2_clusters}")
    print(f"  Size-3 clusters: {size_3_clusters}")

    print("\nGenerating pairs...")
    pairs = generate_pairs(clusters)
    print(f"  Generated {len(pairs)} pairs")

    # Sort by cluster_id (numeric) for logical grouping during review
    pairs.sort(key=lambda x: int(x[2]))

    print(f"\nWriting output to {OUTPUT_PATH}...")
    rows = [format_tsv_row(*pair) for pair in pairs]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    # Summary statistics
    cross_source_count = sum(1 for row in rows if row["is_cross_source"])
    same_source_count = len(rows) - cross_source_count
    print("\nSummary:")
    print(f"  Total pairs: {len(rows)}")
    print(f"  Cross-source pairs: {cross_source_count}")
    print(f"  Same-source pairs: {same_source_count}")
    print(f"\nDone! Output written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
