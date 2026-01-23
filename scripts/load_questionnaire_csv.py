#!/usr/bin/env python
"""Load questionnaire data from CSV/Excel into BaseEntity format.

This script converts custom questionnaire data files into the standardized
BaseEntity JSON format used by biovector-eval for harmonization.

Usage:
    uv run python scripts/load_questionnaire_csv.py \
        data/questionnaires/raw/your_questions.csv \
        --output data/questionnaires/processed/custom_questions.json \
        --question-col "Question Text" \
        --id-col "Variable Name" \
        --source-col "Source"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from biovector_eval.base.entity import BaseEntity


def load_questionnaires(
    input_path: Path,
    question_col: str = "question_text",
    id_col: str | None = "question_id",
    source_col: str | None = "source",
    response_col: str | None = "response_options",
    synonym_col: str | None = None,
) -> list[BaseEntity]:
    """Load questionnaires from CSV/Excel.

    Args:
        input_path: Path to CSV or Excel file
        question_col: Column containing question text
        id_col: Column for unique ID (auto-generated if None or missing)
        source_col: Column for data source (e.g., "Arivale", "UK Biobank")
        response_col: Column for response options
        synonym_col: Column for alternate phrasings (semicolon-separated)

    Returns:
        List of BaseEntity objects
    """
    # Read file based on extension
    if input_path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    # Validate required column exists
    if question_col not in df.columns:
        available = ", ".join(df.columns.tolist())
        raise ValueError(
            f"Question column '{question_col}' not found. Available columns: {available}"
        )

    entities = []
    for idx, row in df.iterrows():
        # Generate ID if column not specified or not present
        if id_col and id_col in df.columns:
            entity_id = str(row[id_col])
        else:
            entity_id = f"q_{idx}"

        # Skip rows with empty question text
        question_text = row[question_col]
        if pd.isna(question_text) or str(question_text).strip() == "":
            continue

        # Build metadata from various columns
        metadata: dict = {}
        if source_col and source_col in df.columns and pd.notna(row[source_col]):
            metadata["source_questionnaire"] = row[source_col]
        if response_col and response_col in df.columns and pd.notna(row[response_col]):
            # Parse response options - could be semicolon-separated or already a list
            resp = row[response_col]
            if isinstance(resp, str) and ";" in resp:
                metadata["response_options"] = [r.strip() for r in resp.split(";")]
            else:
                metadata["response_options"] = resp

        # Add all other columns as metadata (excluding primary columns)
        primary_cols = {question_col, id_col, source_col, response_col, synonym_col}
        for col in df.columns:
            if col not in primary_cols:
                val = row[col]
                if pd.notna(val):
                    metadata[col] = val

        # Parse synonyms if column specified
        synonyms: list[str] = []
        if synonym_col and synonym_col in df.columns and pd.notna(row[synonym_col]):
            syn_val = row[synonym_col]
            if isinstance(syn_val, str):
                synonyms = [s.strip() for s in syn_val.split(";") if s.strip()]

        entity = BaseEntity(
            id=entity_id,
            name=str(question_text).strip(),
            synonyms=synonyms,
            metadata=metadata,
        )
        entities.append(entity)

    return entities


def main():
    parser = argparse.ArgumentParser(
        description="Load questionnaires from CSV/Excel into BaseEntity JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default column names
  python scripts/load_questionnaire_csv.py data/raw/questions.csv -o data/processed/questions.json

  # Specify custom column names
  python scripts/load_questionnaire_csv.py data/raw/survey.xlsx \\
      -o output.json \\
      --question-col "Question Text" \\
      --id-col "Variable" \\
      --source-col "Survey Source"

  # Include alternate phrasings as synonyms
  python scripts/load_questionnaire_csv.py data/raw/questions.csv \\
      -o output.json \\
      --synonym-col "Alternate Phrasings"
""",
    )
    parser.add_argument("input", help="Input CSV or Excel file")
    parser.add_argument("--output", "-o", help="Output JSON file", required=True)
    parser.add_argument(
        "--question-col",
        default="question_text",
        help="Column containing question text (default: question_text)",
    )
    parser.add_argument(
        "--id-col",
        default=None,
        help="Column for unique ID (auto-generated if not specified)",
    )
    parser.add_argument(
        "--source-col",
        default=None,
        help="Column for data source (e.g., 'Arivale', 'UK Biobank')",
    )
    parser.add_argument(
        "--response-col",
        default=None,
        help="Column for response options (semicolon-separated)",
    )
    parser.add_argument(
        "--synonym-col",
        default=None,
        help="Column for alternate phrasings (semicolon-separated)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    try:
        entities = load_questionnaires(
            input_path,
            question_col=args.question_col,
            id_col=args.id_col,
            source_col=args.source_col,
            response_col=args.response_col,
            synonym_col=args.synonym_col,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in entities], f, indent=2)

    print(f"Loaded {len(entities)} questions to {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
