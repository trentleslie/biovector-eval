# Q2Q Cross-Source Question Matching Results

Generated: 2026-02-10

## Overview

This dataset contains the top 100 most similar question pairs across three health survey questionnaires, identified using vector similarity search.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total questions | 1,574 |
| Unique cross-source pairs | 1,348 |
| Top pairs selected | 100 |
| Generation time | 2.1 seconds |
| Embedding model | all-MiniLM-L6-v2 |

## Similarity Distribution (Top 100 Pairs)

| Statistic | Value |
|-----------|-------|
| Minimum | 0.707 |
| Maximum | 1.000 |
| Mean | 0.769 |
| Median | 0.758 |
| Std Dev | 0.059 |

### Histogram

| Score Range | Count |
|-------------|-------|
| 0.8 - 1.0 | 18 |
| 0.6 - 0.8 | 82 |
| 0.4 - 0.6 | 0 |
| 0.2 - 0.4 | 0 |
| 0.0 - 0.2 | 0 |

## Source Pair Distribution

| Source → Target | Count |
|-----------------|-------|
| Arivale → Israeli10K | 52 |
| Israeli10K → UKBB | 22 |
| Israeli10K → Arivale | 15 |
| UKBB → Arivale | 5 |
| UKBB → Israeli10K | 4 |
| Arivale → UKBB | 2 |

## Confidence Levels

| Level | Threshold | Count |
|-------|-----------|-------|
| High | ≥ 0.85 | 8 |
| Medium | 0.50 - 0.85 | 92 |
| Low | < 0.50 | 0 |

## Top 10 Matching Pairs

| Rank | Score | Source Survey | Source Question | Target Survey | Target Question |
|------|-------|---------------|-----------------|---------------|-----------------|
| 1 | 1.000 | Arivale | Cardiac catheterization | Israeli10K | Cardiac Catheterization |
| 2 | 1.000 | Arivale | Cardiac Catheterization | Israeli10K | Cardiac Catheterization |
| 3 | 0.967 | Arivale | Pneumonia vaccine | Israeli10K | Vaccine pneumonia |
| 4 | 0.897 | Israeli10K | Cardiac catheterization results | Arivale | Cardiac Catheterization |
| 5 | 0.882 | Arivale | Macular degeneration | Israeli10K | Age macular degeneration diagnosed |
| 6 | 0.881 | Arivale | Flu vaccine | Israeli10K | Vaccine influenza |
| 7 | 0.878 | Israeli10K | Which eye(s) affected by macular degeneration | Arivale | Macular degeneration |
| 8 | 0.875 | Arivale | Periodontal disease | Israeli10K | Periodontal diseases diagnosis |
| 9 | 0.849 | Arivale | Colonoscopy | Israeli10K | Normal colonoscopy |
| 10 | 0.843 | Arivale | Mammogram | Israeli10K | Normal mammogram |

## Key Findings

1. **Arivale and Israeli10K share the most vocabulary** - 67 of the top 100 pairs are between these two surveys
2. **UKBB uses distinct phrasing** - UK English conventions and "ACE touchscreen question" prefixes reduce similarity scores
3. **Medical procedures and conditions match well** - cardiac catheterization, vaccines, and diagnostic tests show high similarity
4. **Word order variations are handled** - "Pneumonia vaccine" ↔ "Vaccine pneumonia" scores 0.967

## Files

- `q2q_review_100.json` - Full dataset with 100 pairs and top-5 candidates each
- `q2q_generation_log.json` - Generation metadata and statistics
