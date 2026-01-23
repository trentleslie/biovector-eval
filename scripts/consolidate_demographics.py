#!/usr/bin/env python
"""Consolidate demographics data from multiple sources.

Parses and consolidates demographics data from Arivale, Israeli10K,
and UK Biobank sources into a unified JSON format.

Usage:
    # Default: Parse all sources, save to data/demographics/processed/demographics.json
    uv run python scripts/consolidate_demographics.py

    # Custom output path
    uv run python scripts/consolidate_demographics.py --output data/demographics.json

    # Custom raw data directory
    uv run python scripts/consolidate_demographics.py --raw-dir /path/to/raw

    # Validate sources without processing
    uv run python scripts/consolidate_demographics.py --validate

    # Verbose output
    uv run python scripts/consolidate_demographics.py -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from biovector_eval.domains.demographics.consolidate import (
    consolidate_demographics,
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
        description="Consolidate demographics data from multiple sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sources:
  - demographics_metadata.tsv (39 demographics) - Arivale
  - israeli10k_demographics.tsv (184 demographics) - Israeli10K
  - ukbb_demographics.tsv (45 demographics) - UK Biobank

Expected output: ~268 unified demographic entities
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
        default=project_root / "data" / "demographics" / "processed" / "demographics.json",
        help="Output JSON file path (default: data/demographics/processed/demographics.json)",
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

        entities = consolidate_demographics(
            raw_dir=args.raw_dir,
            output_path=args.output,
        )

        print(f"\nSuccess! Consolidated {len(entities)} demographic entities")
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
