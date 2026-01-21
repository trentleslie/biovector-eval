# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

biovector-eval is a multi-domain framework for evaluating and optimizing vector databases for biological/biomedical domains. It benchmarks embedding models, quantization strategies, and index configurations across metabolites, demographics, questionnaires, and future domains.

## Commands

```bash
# Run all checks (ruff, black, pyright, pytest)
./scripts/check.sh

# Auto-fix formatting/linting issues
./scripts/fix.sh

# Run tests
uv run pytest -v

# Run single test file
uv run pytest tests/unit/test_metrics.py -v

# Run specific test
uv run pytest tests/unit/test_metrics.py::test_recall_at_k -v

# Type checking
uv run pyright src/

# CLI commands
uv run biovector-eval check-gpu
uv run biovector-eval generate-ground-truth --domain metabolites --entities <path> --output <path>
uv run biovector-eval evaluate-models --metabolites <path> --ground-truth <path> --output <dir>
uv run biovector-eval evaluate-quantization --metabolites <path> --ground-truth <path> --output <path>
```

## Architecture

### Multi-Domain Design

The framework uses a **domain-agnostic base** with **domain-specific implementations**:

```
src/biovector_eval/
├── base/           # Domain-agnostic infrastructure
│   ├── entity.py      # BaseEntity dataclass
│   ├── ground_truth.py # GroundTruthQuery/Dataset
│   ├── domain.py      # Domain protocol + DomainConfig
│   ├── evaluator.py   # BaseEvaluator
│   └── loaders.py     # JSON/TSV/CSV loaders
├── domains/        # Domain registry + implementations
│   ├── __init__.py    # get_domain(), list_domains(), register_domain()
│   ├── metabolites/   # Full implementation
│   ├── demographics/  # Scaffold
│   └── questionnaires/ # Scaffold
├── core/           # Shared evaluation logic
│   ├── embedding_strategy.py
│   ├── metrics.py
│   ├── quantization.py
│   └── performance.py
└── metabolites/    # Legacy module (re-exports from domains/)
```

### Domain Protocol

New domains implement the `Domain` protocol (`base/domain.py`):
- `name`: Domain identifier
- `config`: DomainConfig with paths and column mappings
- `load_entities()`: Load domain data
- `generate_ground_truth()`: Create evaluation queries
- `get_embedding_models()`: Recommended models for domain

Register new domains in `domains/__init__.py`.

### Embedding Strategies (core/embedding_strategy.py)

Three approaches for converting entities to vectors:
- `SinglePrimaryStrategy`: One vector from primary name (minimal storage)
- `SinglePooledStrategy`: One vector averaged from name + synonyms (balance)
- `MultiVectorStrategy`: Multiple vectors per entity with tiered weights (highest recall)

### Multi-Vector Search

When using `MultiVectorStrategy`, the `MultiVectorSearcher`:
1. Oversamples results (k × 50) to account for duplicate entities
2. Deduplicates by entity ID, keeping best weighted score
3. Weights: primary (1.0), synonym (0.9), variation (0.7)

### Index Types

Uses FAISS:
- `IndexFlatIP`: Exact search baseline
- `IndexHNSWFlat`: Fast approximate search (default)
- IVF variants for scaling beyond 10M vectors

### Data Layout

```
data/
├── metabolites/     # Primary metabolite data location
│   ├── embeddings/
│   └── indices/
├── demographics/    # Demographics domain data
├── questionnaires/  # Questionnaires domain data
├── hmdb -> metabolites        # Backward-compatible symlink
├── embeddings -> metabolites/embeddings
└── indices -> metabolites/indices
```

### Notebook Organization

```
notebooks/
├── metabolites/     # Metabolite-specific analysis
│   ├── mean_field/  # Educational transformer theory
│   └── similarity_*.ipynb
├── shared/          # Cross-domain analysis tools
├── demographics/    # Scaffold
├── questionnaires/  # Scaffold
└── cross_domain/    # Multi-domain comparisons
```

## Key Evaluation Flow

1. Parse raw data (e.g., HMDB XML → JSON via `scripts/parse_hmdb_xml.py`)
2. Generate ground truth: `generate-ground-truth --domain <name> --entities <path>`
3. Generate embeddings: `scripts/generate_embeddings.py`
4. Build FAISS indices: `scripts/build_indices.py`
5. Run evaluation comparing models/strategies/quantization

## Code Style

- Use type hints
- Follow Conventional Commits (feat:, fix:, docs:, test:, refactor:, chore:)
- Add dependencies with `uv add <package>` and commit both pyproject.toml and uv.lock
