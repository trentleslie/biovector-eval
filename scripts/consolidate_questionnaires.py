#!/usr/bin/env python
"""Consolidate questionnaire data from multiple sources.

Parses and consolidates questionnaire data from Arivale, Israeli10K,
and UK Biobank sources into a unified JSON format.

Usage:
    # Default: Parse all sources, save to data/questionnaires/processed/questions.json
    uv run python scripts/consolidate_questionnaires.py

    # Custom output path
    uv run python scripts/consolidate_questionnaires.py --output data/questions.json

    # Custom raw data directory
    uv run python scripts/consolidate_questionnaires.py --raw-dir /path/to/raw

    # Validate sources without processing
    uv run python scripts/consolidate_questionnaires.py --validate

    # Verbose output
    uv run python scripts/consolidate_questionnaires.py -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from biovector_eval.domains.questionnaires.consolidate import (
    consolidate_questionnaires,
    validate_sources,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Consolidate questionnaire data from multiple sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sources:
  - Arivale Questionnaire - Sheet1.tsv (562 questions)
  - israeli10k_questionnaires.tsv (569 questions)
  - ukbb_questionnaires.tsv (499 questions)

Expected output: ~1,630 unified questionnaire entities
        """,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=project_root / "data" / "raw",
        help="Directory containing raw TSV files (default: data/raw)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=project_root / "data" / "questionnaires" / "processed" / "questions.json",
        help="Output JSON file path (default: data/questionnaires/processed/questions.json)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate source files without processing",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Validate mode
    if args.validate:
        logger.info(f"Validating sources in {args.raw_dir}")
        status = validate_sources(args.raw_dir)

        all_present = True
        for source, info in status.items():
            exists = info["exists"]
            symbol = "[OK]" if exists else "[MISSING]"
            if not exists:
                all_present = False
            print(f"{symbol} {source}: {info['filename']}")
            if exists:
                print(f"       Expected rows: {info['expected_rows']}")
            print(f"       Description: {info['description']}")

        return 0 if all_present else 1

    # Consolidation mode
    try:
        logger.info(f"Raw directory: {args.raw_dir}")
        logger.info(f"Output path: {args.output}")

        entities = consolidate_questionnaires(
            raw_dir=args.raw_dir,
            output_path=args.output,
        )

        print(f"\nSuccess! Consolidated {len(entities)} questionnaire entities")
        print(f"Output: {args.output}")

        return 0

    except FileNotFoundError as e:
        logger.error(str(e))
        logger.info("Run with --validate to check source file availability")
        return 1

    except ValueError as e:
        logger.error(f"Consolidation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
