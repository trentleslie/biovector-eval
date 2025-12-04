# BioVector Eval - Implementation Progress

> **Last Updated**: 2024-12-03 (Session 4)
> **Current Phase**: Phase 5 - Documentation
> **Plan Reference**: `.claude/plans/mutable-noodling-pretzel.md`
> **Decisions Document**: `docs/decisions.md`

---

## Project Overview

Extend the biovector-eval framework to evaluate **four dimensions**:
1. **Embedding Models**: BGE-M3, SapBERT, BioLORD-2023, GTE-Qwen2-1.5B (4)
2. **Indexing**: Flat, HNSW, IVFFlat, IVFPQ (4 base types)
3. **Quantization**: None, SQ8, SQ4, PQ (for HNSW)
4. **Embedding Strategy**: single-primary, single-pooled, multi-vector (3)

**Total configurations**: 4 models x 7 index types x 3 strategies = **84 configurations**

### Embedding Models

| Model | HuggingFace ID | Dim | Purpose |
|-------|----------------|-----|---------|
| BGE-M3 | `BAAI/bge-m3` | 1024 | General multilingual baseline |
| SapBERT | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | 768 | Biomedical entity linking (SOTA) |
| BioLORD-2023 | `FremyCompany/BioLORD-2023` | 768 | Clinical/biomedical concepts |
| GTE-Qwen2-1.5B | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | 1536 | Top MTEB performance |

> **Note**: ChemBERTa was removed - it's trained on SMILES molecular structures, not text names. See `docs/decisions.md` for details.

---

## Current Status

- [x] **Phase 0**: Progress Tracking Setup
- [x] **Phase 1**: Core Framework (TDD) - Embedding strategies + Index builders
- [x] **Phase 2**: Embedding Generation - Add strategy support to generate_embeddings.py
- [x] **Phase 3**: Index Building - Create build_indices.py for all index types
- [x] **Phase 4**: Evaluation - Multi-vector search, 4D evaluation loop
- [ ] **Phase 5**: Documentation - Update metabolite_roadmap.md with results

---

## Test-Driven Development Tasks

> **TDD Rule**: Write tests BEFORE implementation. Tests should fail initially, then pass after implementation.

---

### Phase 1: Core Framework (COMPLETE)

#### 1.1 Embedding Strategy Module

**Implementation File**: `src/biovector_eval/core/embedding_strategy.py`
**Test File**: `tests/unit/test_embedding_strategy.py`

| Task | Test Function | Implementation | Status |
|------|---------------|----------------|--------|
| StrategyType enum | `test_strategy_type_values` | Define enum with 3 values | [x] |
| SinglePrimaryStrategy | `test_single_primary_one_vector` | Returns 1 vector per metabolite | [x] |
| SinglePrimaryStrategy | `test_single_primary_uses_name` | Uses metabolite["name"] only | [x] |
| SinglePooledStrategy | `test_single_pooled_averages_vectors` | Mean of name + synonym embeddings | [x] |
| SinglePooledStrategy | `test_single_pooled_normalizes` | Output is L2 normalized | [x] |
| SinglePooledStrategy | `test_single_pooled_handles_no_synonyms` | Works when synonyms empty | [x] |
| MultiVectorStrategy | `test_multi_vector_count` | Returns ~36 vectors per metabolite | [x] |
| MultiVectorStrategy | `test_multi_vector_primary_weight` | Primary name has weight 1.0 | [x] |
| MultiVectorStrategy | `test_multi_vector_synonym_weight` | Synonyms have weight 0.9 | [x] |
| MultiVectorStrategy | `test_multi_vector_variation_weight` | Variations have weight 0.7 | [x] |
| MultiVectorStrategy | `test_multi_vector_generates_variations` | Greek letters, prefix removal | [x] |

#### 1.2 Index Builders

**Implementation File**: `src/biovector_eval/utils/persistence.py`
**Test File**: `tests/unit/test_persistence.py`

| Task | Test Function | Implementation | Status |
|------|---------------|----------------|--------|
| build_flat_index | `test_flat_index_exact` | IndexFlatIP for cosine | [x] |
| build_flat_index | `test_flat_index_returns_all` | Exhaustive search | [x] |
| build_hnsw_index | `test_hnsw_index_params` | IndexHNSWFlat with M=32 | [x] |
| build_hnsw_index | `test_hnsw_ef_construction` | efConstruction=200 | [x] |
| build_hnsw_sq_index | `test_hnsw_sq8_compression` | IndexHNSWSQ with QT_8bit | [x] |
| build_hnsw_sq_index | `test_hnsw_sq4_compression` | IndexHNSWSQ with QT_4bit | [x] |
| build_hnsw_pq_index | `test_hnsw_pq_compression` | IndexHNSWPQ | [x] |
| build_ivf_flat_index | `test_ivf_flat_requires_train` | Needs training data | [x] |
| build_ivf_flat_index | `test_ivf_flat_nlist_param` | nlist=sqrt(N) default | [x] |
| build_ivf_flat_index | `test_ivf_flat_nprobe_default` | nprobe=32 default | [x] |
| build_ivf_pq_index | `test_ivf_pq_compression` | IndexIVFPQ | [x] |
| build_ivf_pq_index | `test_ivf_pq_requires_train` | Needs training data | [x] |

---

### Phase 2: Embedding Generation (COMPLETE)

#### 2.1 Strategy-Aware Embedding Script

**Implementation File**: `scripts/generate_embeddings.py`
**Test File**: `tests/unit/test_generate_embeddings.py`

| Task | Test Function | Implementation | Status |
|------|---------------|----------------|--------|
| Parse --strategy argument | `test_strategy_argument_parsing_*` | Accept single-primary, single-pooled, multi-vector | [x] |
| Load strategy class | `test_load_strategy_by_name_*` | Use get_strategy() factory | [x] |
| Single-primary output | `test_single_primary_output_shape` | (N, dim) embeddings | [x] |
| Single-pooled output | `test_single_pooled_output_shape` | (N, dim) embeddings | [x] |
| Multi-vector output | `test_multi_vector_output_structure` | embeddings.npy + metadata.json | [x] |
| Multi-vector metadata | `test_multi_vector_metadata_schema` | {idx: {hmdb_id, tier, weight, text}} | [x] |
| Output file naming | `test_output_file_naming_*` | {model}_{strategy}.npy or {model}_{strategy}/ | [x] |
| Save multi-vector | `test_save_multi_vector_output` | Save embeddings.npy + metadata.json | [x] |

---

### Phase 3: Index Building (COMPLETE)

#### 3.1 Batch Index Builder Script

**Implementation File**: `scripts/build_indices.py`
**Test File**: `tests/unit/test_build_indices.py`

| Task | Test Function | Implementation | Status |
|------|---------------|----------------|--------|
| INDEX_TYPES config | `test_index_types_*` | All 7 types with builders | [x] |
| Parse arguments | `test_parse_*` | --embeddings, --output-dir, --index-types, --skip-existing | [x] |
| Output path generation | `test_*_path` | {model}_{strategy}_{index_type}.faiss | [x] |
| Skip existing logic | `test_skip_*` | Don't rebuild if exists and flag set | [x] |
| Build single index | `test_builds_*_index` | Flat, HNSW, IVF Flat, IVF PQ | [x] |
| Build all indices | `test_builds_all_*` | Batch building for all types | [x] |
| Multi-vector metadata | `test_*_metadata_*` | Copy metadata.json alongside index | [x] |
| Search quality | `test_*_searches_correctly` | Verify indices can search | [x] |

---

### Phase 4: Evaluation (COMPLETE)

#### 4.1 Multi-Vector Search Support (COMPLETE)

**Implementation File**: `src/biovector_eval/metabolites/evaluator.py`
**Test File**: `tests/unit/test_multi_vector_search.py`

| Task | Test Function | Implementation | Status |
|------|---------------|----------------|--------|
| SearchResult dataclass | `test_search_result_*` | Store hmdb_id, score, tier, weight | [x] |
| Weighted score property | `test_*_weighted_score` | score * weight calculation | [x] |
| Load index with metadata | `test_load_index_*` | Load .faiss + .metadata.json | [x] |
| MultiVectorSearcher class | `test_searcher_*` | Handle single/multi-vector indices | [x] |
| Deduplication | `test_search_deduplicates_*` | Keep best weighted score per HMDB ID | [x] |
| Result ranking | `test_search_keeps_best_*` | Sort by weighted score descending | [x] |
| Oversample factor | `test_search_oversamples_*` | Request k*50 for multi-vector | [x] |
| ID mapping support | `test_search_with_id_mapping` | Support explicit ID mapping | [x] |

#### 4.2 Evaluation Pipeline (COMPLETE)

**Implementation File**: `scripts/run_evaluation.py`
**Test File**: `tests/integration/test_evaluation_pipeline.py`

| Task | Test Function | Implementation | Status |
|------|---------------|----------------|--------|
| EVALUATION_DIMENSIONS | `test_dimensions_includes_*` | Models, index_types, strategies | [x] |
| EvaluationConfig | `test_config_*` | Dataclass with path generation | [x] |
| get_all_configurations | `test_*_all_*_included` | Generate 84 configurations | [x] |
| save_evaluation_results | `test_saves_json_file` | Save results with metadata | [x] |
| Results summary | `test_includes_summary_stats` | Best recall, MRR, latency, size | [x] |
| Strategy identification | `test_*_uses_*_search` | Identify single vs multi-vector | [x] |

---

## File Modification Tracker

### Phase 1 (Complete)
| File | Status | Description |
|------|--------|-------------|
| `docs/decisions.md` | [x] NEW | Design decisions and citations |
| `src/biovector_eval/core/embedding_strategy.py` | [x] NEW | Strategy classes (3 strategies) |
| `tests/unit/test_embedding_strategy.py` | [x] NEW | Strategy tests (20 tests) |
| `src/biovector_eval/utils/persistence.py` | [x] MODIFY | Add IVF builders |
| `tests/unit/test_persistence.py` | [x] NEW | Index builder tests (38 tests) |

### Phase 2 (Complete)
| File | Status | Description |
|------|--------|-------------|
| `tests/unit/test_generate_embeddings.py` | [x] NEW | Embedding generation tests (20 tests) |
| `scripts/generate_embeddings.py` | [x] MODIFY | Add strategy functions (parse, generate, output paths, save) |

### Phase 3 (Complete)
| File | Status | Description |
|------|--------|-------------|
| `tests/unit/test_build_indices.py` | [x] NEW | Index building tests (27 tests) |
| `scripts/build_indices.py` | [x] NEW | Unified index builder for all 7 index types |

### Phase 4 (Complete)
| File | Status | Description |
|------|--------|-------------|
| `tests/unit/test_multi_vector_search.py` | [x] NEW | Multi-vector search tests (18 tests) |
| `src/biovector_eval/metabolites/evaluator.py` | [x] MODIFY | Added SearchResult, MultiVectorSearcher, load_index_with_metadata |
| `tests/integration/test_evaluation_pipeline.py` | [x] NEW | 4D evaluation pipeline tests (21 tests) |
| `scripts/run_evaluation.py` | [x] REWRITE | Complete 4D evaluation loop with 84 configurations |

### Phase 5
| File | Status | Description |
|------|--------|-------------|
| `docs/metabolite_roadmap.md` | [ ] UPDATE | Document 4D framework results |

---

## Session Log

### Session 1: 2024-12-03
**Focus**: Initial planning, setup, embedding strategies, and model selection

**Completed**:
- Analyzed existing codebase structure
- Identified 4 evaluation dimensions
- Designed 3 embedding strategies (single-primary, single-pooled, multi-vector)
- Added IVF-based indexes (IVFFlat, IVFPQ) for future scale
- Created TDD task structure
- Created progress.md for cross-session tracking
- Wrote 20 tests for embedding strategies (`tests/unit/test_embedding_strategy.py`)
- Implemented `embedding_strategy.py` with 3 strategy classes (all tests passing)
- Researched and selected new embedding models (web search)
- Created `docs/decisions.md` with rationale and citations
- Updated model configuration:
  - **Added**: SapBERT, BioLORD-2023, GTE-Qwen2-1.5B
  - **Removed**: ChemBERTa (wrong tool - SMILES vs text), BGE-Small (superseded)
  - **Kept**: BGE-M3 (strong baseline)

**Key Decision**: ChemBERTa underperformed because it's trained on SMILES molecular structures, not metabolite text names. Replaced with biomedical-specialized models (SapBERT, BioLORD).

**Next Session**:
- Write tests for index builders (IVFFlat, IVFPQ)
- Update `persistence.py` with IVF index builders

**Blockers**: None

---

### Session 2: 2024-12-03
**Focus**: Index builders (TDD for IVF indexes)

**Completed**:
- Created comprehensive test suite for all index builders (`tests/unit/test_persistence.py`)
  - 38 tests covering Flat, HNSW, SQ8, SQ4, PQ, IVFFlat, IVFPQ
  - Includes quality comparison tests (HNSW vs Flat recall, IVF vs Flat recall)
- Implemented `build_ivf_flat_index()` in persistence.py
  - Clustering-based index for large-scale datasets
  - Default nlist=sqrt(N), nprobe=32
  - Uses inner product metric for normalized vectors
- Implemented `build_ivf_pq_index()` in persistence.py
  - Clustering + product quantization for maximum compression
  - Auto-adjusts pq_m to divide dimension evenly
- All 88 project tests passing

**Next Session** (Phase 2 - TDD):
1. Write tests for `generate_embeddings.py` strategy support (`tests/unit/test_generate_embeddings.py`)
2. Implement `--strategy` parameter in `generate_embeddings.py`
3. Run tests to verify implementation

**Blockers**: None

---

### Session 3: 2024-12-03
**Focus**: Phase 2 (Embedding Generation) + Phase 3 (Index Building) - TDD

**Completed**:

**Phase 2 - Embedding Generation:**
- Set up code quality infrastructure:
  - Created/updated `scripts/check.sh` (ruff, black, pyright, pytest)
  - Created `scripts/fix.sh` (auto-fix formatting/linting)
  - Added black and pyright to dev dependencies
  - Configured pyright in pyproject.toml for FAISS compatibility
- Wrote TDD tests for embedding generation (`tests/unit/test_generate_embeddings.py`)
  - 20 tests covering strategy argument parsing, strategy loading, output shapes, metadata schema, file naming
- Implemented strategy support in `scripts/generate_embeddings.py`:
  - `parse_strategy_argument()`: Parse CLI --strategy argument to StrategyType enum
  - `generate_embeddings_with_strategy()`: Generate embeddings using a strategy class
  - `get_output_paths()`: Get output paths based on strategy (flat file vs directory)
  - `save_multi_vector_output()`: Save embeddings.npy + metadata.json for multi-vector

**Phase 3 - Index Building:**
- Wrote TDD tests for batch index builder (`tests/unit/test_build_indices.py`)
  - 27 tests covering INDEX_TYPES config, argument parsing, output paths, skip logic, building indices, metadata copy
- Created `scripts/build_indices.py`:
  - `INDEX_TYPES`: Configuration dict for all 7 index types with builders
  - `parse_arguments()`: Parse CLI arguments
  - `get_index_output_path()`: Generate output path for index files
  - `should_skip_index()`: Check if index should be skipped (--skip-existing flag)
  - `build_index_for_embedding_file()`: Build a single index from embedding file
  - `build_all_indices()`: Batch build all indices for all embedding files
  - Copies metadata.json alongside multi-vector indices

**Phase 4.1 - Multi-Vector Search:**
- Wrote TDD tests for multi-vector search (`tests/unit/test_multi_vector_search.py`)
  - 18 tests covering SearchResult, load_index_with_metadata, MultiVectorSearcher
- Added to `src/biovector_eval/metabolites/evaluator.py`:
  - `SearchResult`: Dataclass with hmdb_id, score, tier, weight, weighted_score property
  - `load_index_with_metadata()`: Load .faiss index + optional .metadata.json
  - `MultiVectorSearcher`: Class handling both single-vector and multi-vector search
    - Oversamples to account for multiple vectors per metabolite
    - Deduplicates by HMDB ID, keeping best weighted score
    - Returns results sorted by weighted score descending

**Test Summary**: All 153 tests passing
**Code Quality**: All checks passing (ruff, black, pyright)

**Next Session** (Phase 4.2 - TDD):
1. Write tests for evaluation pipeline (`tests/integration/test_evaluation.py`)
2. Create/update `scripts/run_evaluation.py` for 4D evaluation loop
3. Run tests to verify implementation

**Blockers**: None

---

### Session 4: 2024-12-03
**Focus**: Phase 4.2 - 4D Evaluation Pipeline (TDD)

**Completed**:

**Phase 4.2 - 4D Evaluation Pipeline:**
- Wrote TDD tests for evaluation pipeline (`tests/integration/test_evaluation_pipeline.py`)
  - 21 tests covering EVALUATION_DIMENSIONS, EvaluationConfig, get_all_configurations, save_evaluation_results
- Completely rewrote `scripts/run_evaluation.py` for 4D evaluation:
  - `EVALUATION_DIMENSIONS`: dict with models (4), index_types (7), strategies (3)
  - `MODEL_SLUGS` and `MODEL_IDS`: Mappings for file paths and HuggingFace model IDs
  - `EvaluationConfig`: Dataclass with path generation methods (embedding_path, index_path, metadata_path)
  - `get_all_configurations()`: Generate all 84 configurations (4×7×3)
  - `evaluate_configuration()`: Run evaluation for single config using MultiVectorSearcher
  - `save_evaluation_results()`: Save JSON with metadata, summary (best recall, MRR, latency, size), and results
  - `run_4d_evaluation()`: Main loop grouping by model for efficient model loading
  - CLI with `--models`, `--index-types`, `--strategies`, `--device` arguments
- Fixed robust handling of optional metrics (p95_ms, index_size_mb) in summary calculation
- All linting/formatting issues resolved with fix.sh

**Test Summary**: All 174 tests passing
**Code Quality**: All checks passing (ruff, black, pyright)

**Next Session** (Phase 5):
1. Update `docs/metabolite_roadmap.md` with 4D framework description
2. Generate embeddings for all model/strategy combinations
3. Build indices for all configurations
4. Run full evaluation and document results

**Blockers**: None

---

## Quick Reference

### Embedding Models

```python
MODELS = {
    "bge-m3": "BAAI/bge-m3",                                      # 1024d
    "sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",   # 768d
    "biolord": "FremyCompany/BioLORD-2023",                       # 768d
    "gte-qwen2": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",           # 1536d
}
```

### Embedding Strategies

```python
# single-primary: Current approach
embedding = model.encode(metabolite["name"])

# single-pooled: Average name + synonyms
texts = [metabolite["name"]] + metabolite["synonyms"][:15]
embeddings = model.encode(texts)
embedding = normalize(np.mean(embeddings, axis=0))

# multi-vector: Separate vectors with weights
# Primary (1.0) + Synonyms (0.9) + Variations (0.7)
```

### Index Types

```python
# Flat (exact)
faiss.IndexFlatIP(dim)

# HNSW (graph)
faiss.IndexHNSWFlat(dim, M=32)

# HNSW + SQ8
faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit, M=32)

# IVFFlat (clustering)
faiss.IndexIVFFlat(quantizer, dim, nlist=512)

# IVFPQ (clustering + PQ)
faiss.IndexIVFPQ(quantizer, dim, nlist=512, pq_m=32, nbits=8)
```

### Run Tests

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/unit/test_embedding_strategy.py -v

# Run with coverage
uv run pytest tests/ --cov=src/biovector_eval
```
