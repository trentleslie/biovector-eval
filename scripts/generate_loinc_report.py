#!/usr/bin/env python
"""Generate LOINC coverage report for questionnaire embedding evaluation.

This script analyzes the LOINC dataset and produces a detailed JSON report
showing which classes are included/excluded and category coverage.

Usage:
    uv run python scripts/generate_loinc_report.py \
        --output data/questionnaires/loinc_coverage_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Category mappings: which classes contribute to which thematic categories
CATEGORY_MAPPINGS: dict[str, list[str]] = {
    "diet_nutrition": [
        "SURVEY.SDOH",
        "SURVEY.SAMHSA",
        "SURVEY.NURSE.OMAHA",
        "PHENX",
        "SURVEY.PHQ",  # appetite items
    ],
    "stress_anxiety": [
        "SURVEY.PROMIS",
        "SURVEY.MTLHLTH",
        "SURVEY.NIH.EMO",
        "SURVEY.NEUROQ",
        "SURVEY.PHQ",
        "SURVEY.NMMDS",
        "SURVEY.BPI",
    ],
    "happiness_wellbeing": [
        "SURVEY.PROMIS",
        "SURVEY.NIH.EMO",
        "SURVEY.NEUROQ",
        "SURVEY.R-OUTCOMES",
    ],
    "lifestyle": [
        "SURVEY.IPAQ",
        "SURVEY.PROMIS",
        "SURVEY.NURSE.OMAHA",
        "SURVEY.NIDA",
        "SURVEY.COVID",
        "PHENX",
    ],
    "health_history": [
        "SURVEY.MDS",
        "SURVEY.USSGFHT",
        "SURVEY.OASIS",
        "SURVEY.CMS",
    ],
    "digestion": [
        "SURVEY.NURSE.OMAHA",
        "SURVEY.PHQ",
        "SURVEY.GNHLTH",
    ],
    "mental_health": [
        "SURVEY.GDS",
        "SURVEY.EPDS",
        "SURVEY.MTLHLTH",
        "SURVEY.PHQ",
    ],
    "personality": [
        "PHENX",
    ],
}

# Classes to include (from the plan)
INCLUDED_CLASSES = [
    "SURVEY.PROMIS",
    "SURVEY.GNHLTH",
    "SURVEY.MTLHLTH",
    "SURVEY.CMS",
    "SURVEY.CDC",
    "SURVEY.AHRQ",
    "SURVEY.IPAQ",
    "SURVEY.SAMHSA",
    "SURVEY.HAQ",
    "SURVEY.GDS",
    "SURVEY.EPDS",
    "SURVEY.PNDS",
    "SURVEY.MISC",
    "SURVEY.USSGFHT",
    "SURVEY.NIH.EMO",
    "SURVEY.NEUROQ",
    "SURVEY.R-OUTCOMES",
    "SURVEY.OASIS",
    "SURVEY.MDS",
    "SURVEY.SDOH",
    "SURVEY.PHQ",
    "SURVEY.NIDA",
    "SURVEY.NURSE.OMAHA",
    "SURVEY.NURSE.HHCC",
    "SURVEY.NMMDS",
    "SURVEY.COVID",
    "SURVEY.BPI",
    "PHENX",
]

# Excluded classes with reasons
EXCLUDED_CLASSES: dict[str, str] = {
    "SURVEY.OPTIMAL": "Rehab equipment assessment - not questionnaire content",
    "SURVEY.RFC": "Residual Functional Capacity (disability) - specialized",
    "SURVEY.ESRD": "End-stage renal staffing metrics - not survey items",
    "SURVEY.FBT": "Family-based treatment - specialized clinical",
    "SURVEY.SEEK": "Child safety screening - specialized",
    "SURVEY.CAM": "Confusion Assessment Method - specialized clinical",
    "SURVEY.MFS": "Morse Fall Scale - fall risk only",
    "SURVEY.TAPS": "Substance screening tool - already covered by NIDA",
    "SURVEY.WHOOLEY": "Depression screening (2 items only) - minimal",
    "SURVEY.MDC": "Medical device classification - not surveys",
    "SURVEY.SF": "SF surveys often panels, not individual items",
    "PANEL.SURVEY.*": "Panel definitions, not individual items",
    "H&P.*": "Clinical history documentation, not surveys",
    "DOC.*": "Document types, not survey items",
}


def get_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


def count_all_classes(loinc_csv_path: Path) -> dict[str, int]:
    """Count codes per CLASS in LOINC CSV.

    Args:
        loinc_csv_path: Path to LoincTableCore.csv

    Returns:
        Dict mapping CLASS to count of ACTIVE codes.
    """
    class_counts: Counter = Counter()

    with open(loinc_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("STATUS", "").strip() != "ACTIVE":
                continue
            loinc_class = row.get("CLASS", "").strip()
            if loinc_class:
                class_counts[loinc_class] += 1

    return dict(class_counts)


def categorize_classes(class_counts: dict[str, int]) -> dict[str, dict]:
    """Categorize classes into families.

    Args:
        class_counts: Dict mapping CLASS to code count.

    Returns:
        Dict with family statistics.
    """
    families: dict[str, dict] = defaultdict(
        lambda: {"classes": [], "total_codes": 0, "included_codes": 0}
    )

    for loinc_class, count in class_counts.items():
        # Determine family
        if loinc_class.startswith("SURVEY."):
            family = "SURVEY.*"
        elif loinc_class.startswith("PANEL."):
            family = "PANEL.*"
        elif loinc_class.startswith("H&P."):
            family = "H&P.*"
        elif loinc_class == "PHENX":
            family = "PHENX"
        else:
            family = "OTHER"

        families[family]["classes"].append(loinc_class)
        families[family]["total_codes"] += count

        if loinc_class in INCLUDED_CLASSES:
            families[family]["included_codes"] += count

    return dict(families)


def generate_category_coverage(
    class_counts: dict[str, int],
) -> dict[str, int]:
    """Estimate code coverage per thematic category.

    Note: This is approximate as codes can belong to multiple categories.

    Args:
        class_counts: Dict mapping CLASS to code count.

    Returns:
        Dict mapping category to estimated code count.
    """
    category_coverage: dict[str, int] = {}

    for category, classes in CATEGORY_MAPPINGS.items():
        total = 0
        for cls in classes:
            if cls in class_counts and cls in INCLUDED_CLASSES:
                total += class_counts[cls]
        category_coverage[category] = total

    return category_coverage


def generate_report(loinc_csv_path: Path) -> dict:
    """Generate comprehensive LOINC coverage report.

    Args:
        loinc_csv_path: Path to LoincTableCore.csv

    Returns:
        Report dict suitable for JSON serialization.
    """
    logger.info(f"Reading LOINC CSV: {loinc_csv_path}")
    class_counts = count_all_classes(loinc_csv_path)
    logger.info(f"Found {len(class_counts)} unique LOINC classes")

    # Calculate totals
    total_active_codes = sum(class_counts.values())
    included_codes = sum(
        count for cls, count in class_counts.items() if cls in INCLUDED_CLASSES
    )

    # Family statistics
    family_stats = categorize_classes(class_counts)

    # Category coverage
    category_coverage = generate_category_coverage(class_counts)

    # Build included classes detail
    included_classes_detail = []
    for cls in INCLUDED_CLASSES:
        count = class_counts.get(cls, 0)
        categories = [cat for cat, classes in CATEGORY_MAPPINGS.items() if cls in classes]
        included_classes_detail.append(
            {
                "class": cls,
                "codes": count,
                "categories": categories,
            }
        )

    # Sort by code count descending
    included_classes_detail.sort(key=lambda x: x["codes"], reverse=True)

    # Build excluded classes detail (from SURVEY.* family)
    excluded_classes_detail = []
    survey_classes = [cls for cls in class_counts if cls.startswith("SURVEY.")]
    for cls in survey_classes:
        if cls not in INCLUDED_CLASSES:
            reason = EXCLUDED_CLASSES.get(
                cls, "Specialized clinical or low relevance to target categories"
            )
            excluded_classes_detail.append(
                {
                    "class": cls,
                    "codes": class_counts[cls],
                    "reason": reason,
                }
            )

    # Sort by code count descending
    excluded_classes_detail.sort(key=lambda x: x["codes"], reverse=True)

    # Build family summary
    coverage_by_family = {}
    for family, stats in family_stats.items():
        coverage_by_family[family] = {
            "included_classes": len(
                [c for c in stats["classes"] if c in INCLUDED_CLASSES]
            ),
            "total_classes": len(stats["classes"]),
            "included_codes": stats["included_codes"],
            "total_codes": stats["total_codes"],
            "coverage_ratio": (
                round(stats["included_codes"] / stats["total_codes"], 3)
                if stats["total_codes"] > 0
                else 0
            ),
        }

    report = {
        "generated_at": datetime.now().isoformat(),
        "source_file": str(loinc_csv_path),
        "summary": {
            "total_loinc_classes": len(class_counts),
            "total_active_codes": total_active_codes,
            "included_classes": len(INCLUDED_CLASSES),
            "included_codes": included_codes,
            "coverage_percentage": round(included_codes / total_active_codes * 100, 2)
            if total_active_codes > 0
            else 0,
        },
        "coverage_by_family": coverage_by_family,
        "category_coverage": category_coverage,
        "included_classes": included_classes_detail,
        "excluded_survey_classes": excluded_classes_detail,
        "selection_criteria": {
            "target_domains": [
                "diet_nutrition",
                "stress_anxiety",
                "happiness_wellbeing",
                "lifestyle",
                "health_history",
                "digestion",
                "personality",
            ],
            "exclusion_criteria": [
                "Specialized clinical instruments (rehab, disability assessment)",
                "Panel definitions (not individual items)",
                "Laboratory and radiology codes",
                "Equipment/device classifications",
            ],
        },
    }

    return report


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate LOINC coverage report for questionnaire embeddings"
    )
    parser.add_argument(
        "--loinc-csv",
        help="Path to LoincTableCore.csv (defaults to data/questionnaires/raw/loinc/LoincTableCore/LoincTableCore.csv)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for JSON report",
    )
    args = parser.parse_args()

    # Find project root and default paths
    project_root = get_project_root()

    if args.loinc_csv:
        loinc_csv_path = Path(args.loinc_csv)
    else:
        loinc_csv_path = (
            project_root
            / "data"
            / "questionnaires"
            / "raw"
            / "loinc"
            / "LoincTableCore"
            / "LoincTableCore.csv"
        )

    if not loinc_csv_path.exists():
        logger.error(f"LOINC CSV not found at: {loinc_csv_path}")
        logger.error(
            "Please extract LOINC data to data/questionnaires/raw/loinc/ directory"
        )
        raise FileNotFoundError(f"LOINC CSV not found: {loinc_csv_path}")

    # Generate report
    report = generate_report(loinc_csv_path)

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved: {output_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("LOINC COVERAGE REPORT SUMMARY")
    logger.info("=" * 60)
    summary = report["summary"]
    logger.info(f"Total LOINC classes: {summary['total_loinc_classes']}")
    logger.info(f"Total active codes: {summary['total_active_codes']}")
    logger.info(f"Included classes: {summary['included_classes']}")
    logger.info(f"Included codes: {summary['included_codes']}")
    logger.info(f"Coverage: {summary['coverage_percentage']}%")

    logger.info("\nCategory coverage:")
    for category, count in report["category_coverage"].items():
        logger.info(f"  {category}: {count} codes")

    logger.info("\nTop 5 included classes:")
    for cls_info in report["included_classes"][:5]:
        logger.info(f"  {cls_info['class']}: {cls_info['codes']} codes")


if __name__ == "__main__":
    main()
