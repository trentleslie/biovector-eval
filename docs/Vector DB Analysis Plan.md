# Vector Database Optimization Roadmap

**Systematic evaluation of embedding configurations for biological entity collections**

Updated: 2025-12-03

---

## Executive Summary

**Goal**: Determine optimal embedding model + strategy + index configuration for each entity type through the evaluation framework defined in [`docs/embedding_strategy_plan.md`](docs/embedding_strategy_plan.md).

**Entity Collections**:
| Collection | Entities | Primary Challenge |
|------------|----------|-------------------|
| HMDB (Metabolites) | 217K | Synonyms, Greek letters, chemical names |
| LOINC (Clinical Tests) | ~100K | Units, qualifiers, panel variants |
| MONDO (Diseases) | ~50K | Abbreviations, hierarchical relationships |
| LipidMaps | ~48K | Structured nomenclature (simplest) |

**Evaluation Matrix**: 4 models × 3 strategies × 7 index types = 84 configurations per entity type

**Current Status**: Metabolite evaluation in progress (biovector-eval repository)

---

## Model Candidates by Entity Type

| Entity Type | Priority Models | Rationale |
|-------------|-----------------|-----------|
| **Metabolites** | SapBERT, BioLORD, BGE-M3 | UMLS coverage, chemical terminology |
| **Clinical** | SapBERT, BioLORD, BGE-M3 | Medical terminology, LOINC mappings |
| **Diseases** | SapBERT (primary) | Validated +25-40% on entity linking |
| **Lipids** | BGE-M3 | Structured naming, domain models likely unnecessary |

**General-purpose baseline**: BGE-M3 (1024d) — strong retrieval, multilingual
**Domain specialists**: SapBERT (768d, UMLS-trained), BioLORD (768d, ontology-aware)

---

## Evaluation Framework

### Ground Truth Structure
```
fixtures/{entity}_ground_truth.json
├── exact_match: 40%      # Primary name lookups
├── synonym_match: 40%    # Alternate names, abbreviations
└── edge_cases: 20%       # Greek letters, special chars, ambiguous
```

### Metrics (per configuration)
| Metric | Target | Purpose |
|--------|--------|---------|
| Recall@1 | >0.80 | Exact match rate |
| Recall@5 | >0.90 | Candidate generation |
| MRR | >0.85 | Ranking quality |
| P95 Latency | <10ms | Production viability |
| Index Size | — | Storage cost |

### Decision Logic
```
1. Run all 84 configurations
2. Filter: Recall@1 > 0.75 AND P95 < 20ms
3. Rank by: Recall@1 (primary), then Index Size (secondary)
4. Select top performer per strategy (single-primary, single-pooled, multi-vector)
5. Final pick based on recall vs. storage tradeoff
```

---

## Execution Phases

### Phase 1: Metabolites (HMDB) — IN PROGRESS
- Ground truth: 217K metabolites, ~500 evaluation queries
- Models: BGE-M3, SapBERT, BioLORD, GTE-Qwen2
- Strategies: single-primary, single-pooled, multi-vector
- Index types: flat, hnsw, sq8, sq4, pq, ivf_flat, ivf_pq
- **Output**: Optimal configuration + reusable evaluation framework

### Phase 2: Clinical Tests (LOINC)
- Apply framework from Phase 1
- Ground truth from Nightingale panel mappings
- Expect similar model rankings; validate with clinical terminology

### Phase 3: Diseases (MONDO)
- SapBERT expected to dominate (literature-validated)
- Key preprocessing: truncate to 25 tokens (SapBERT limit)
- Test synonym/hierarchy handling

### Phase 4: Lipids (LipidMaps)
- Structured nomenclature — simplest case
- Quick validation run; BGE-M3 likely sufficient
- Confirm quantization settings from earlier phases

---

## Resource Estimates

| Phase | GPU Hours | Primary Deliverable |
|-------|-----------|---------------------|
| Metabolites | ~8-12h | Full 84-config evaluation, framework |
| Clinical | ~4h | Validated config |
| Diseases | ~3h | SapBERT confirmation |
| Lipids | ~2h | Final validation |

**Total**: ~20 GPU hours (local RTX 4060)

---

## Key Insights from Research

| Finding | Implication |
|---------|-------------|
| SapBERT +25-40% on entity linking | High priority for medical entities |
| SQ8 <2% recall loss with 4x compression | Default quantization choice |
| BGE-M3 competitive with domain models | Strong baseline, simpler deployment |
| Multi-vector needs quantization | Essential for 7M+ vector indices |
| HNSW outperforms IVF at <10M scale | Default index choice for MVP |

---

## Outputs

Per entity type:
1. **Optimal configuration**: model + strategy + index + quantization
2. **Benchmark results**: full metrics for all 84 configurations
3. **Deployment config**: ready-to-use FAISS index parameters

Cross-entity:
- Framework code in `biovector_eval/` (reusable for future entities)
- Architecture recommendations for KRAKEN integration

---

*See [`docs/embedding_strategy_plan.md`](docs/embedding_strategy_plan.md) for general methodology and MVP recommendations.*
