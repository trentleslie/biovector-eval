# Embedding/Vector Search Strategy Plan

*Summary for biomapper MVP planning (Issue #15)*

---

## Executive Summary

Vector search enables semantic matching of biomedical entities (metabolites, proteins, assays, etc.) by converting text representations into numerical embeddings and finding nearest neighbors. The strategy involves five key decisions:

- **Embedding Approach**: How to represent each entity - ultimately single-vector vs. multi-vector. For metabolites, three approaches are being evaluated: single vector from primary name, single vector of pooled average of primary name and  synonyms, and multiple vectors per entity. Trade-off between storage and recall.

- **Model Selection**: Domain-specific models (SapBERT, BioLORD) for biomedical terminology vs. general models (BGE-M3, GTE-Qwen2). Choose based on entity type and vocabulary coverage.

- **Quantization**: Compress vectors to reduce storage and speed up search. Options: none (baseline), SQ8 (4x, minimal accuracy loss), SQ4 (8x), PQ (10-25x). Essential for large indices (>1M vectors).

- **Index Structure**: HNSW provides fast approximate search with high recall. Flat indices for exact baseline. IVF variants for scaling beyond 10M vectors.

- **Evaluation**: Measure recall@k (accuracy), MRR (ranking quality), latency (speed), and index size (cost). Ground truth from known entity mappings.

**Not Addressed Here**:
- Unified embedding architecture across entity types—if optimal models differ per entity (e.g., 768d for metabolites, 1024d for proteins), handling dimension mismatches in a shared KRAKEN index or maintaining separate indices per entity type
- LLM integration for the annotation engine (retrieval-augmented generation, re-ranking, or hybrid semantic/LLM pipelines)
- Index parameter tuning (HNSW: M, efConstruction, efSearch; IVF: nlist, nprobe; PQ: segment count)—optimization details determined during deployment

---

## 1. Embedding Approach

| Strategy | Description | Vectors/Entity | Trade-off |
|----------|-------------|----------------|-----------|
| **Single-primary** | Embed primary name only | 1 | Fast, simple baseline |
| **Single-pooled** | Average of name + synonym embeddings | 1 | Better synonym coverage, same storage |
| **Multi-vector** | Separate vectors for name, synonyms, variations | ~36 | Highest recall, largest storage |

**Recommendation**: Evaluate all three. Single-pooled may be the sweet spot (better recall than primary, same index size). Multi-vector best for high-recall requirements but needs quantization.

## 2. Embedding Models

| Model | Dimension | Type | Notes |
|-------|-----------|------|-------|
| **BGE-M3** | 1024 | General | Strong multilingual, good baseline |
| **SapBERT** | 768 | Biomedical | Trained on UMLS, best for medical terminology |
| **BioLORD** | 768 | Biomedical | Clinical focus, ontology-aware |
| **GTE-Qwen2** | 1536 | General | Latest architecture, highest dimension |

**Recommendation**: Start with SapBERT or BioLORD for biomedical entities (HMDB, LOINC). BGE-M3 as general fallback.

## 3. Quantization

| Method | Compression | Accuracy Impact | Use Case |
|--------|-------------|-----------------|----------|
| **None** | 1x | Baseline | Small indices (<1M vectors) |
| **SQ8** | 4x | Minimal | Default choice for most cases |
| **SQ4** | 8x | Slight | Memory-constrained deployment |
| **PQ** | 10-25x | Moderate | Large multi-vector indices |

**Recommendation**: SQ8 for single-vector, PQ for multi-vector (essential for 7M+ vector indices).

## 4. Index Types

| Index | Type | Best For |
|-------|------|----------|
| **Flat** | Exact | Baseline, <500K vectors |
| **HNSW** | Graph ANN | Default choice, fast queries |
| **IVF-Flat** | Clustering | Tunable accuracy/speed |
| **IVF-PQ** | Clustering + PQ | Large-scale, memory-limited |

**Recommendation**: HNSW for MVP (fast, good recall). IVF variants for future scaling.

## 5. Evaluation Metrics

| Metric | Measures | Target |
|--------|----------|--------|
| **Recall@1** | Exact match rate | >0.80 |
| **Recall@5** | Top-5 hit rate | >0.90 |
| **Recall@10** | Top-10 hit rate | >0.95 |
| **MRR** | Ranking quality | >0.85 |
| **P95 Latency** | Query speed | <10ms |
| **Index Size** | Storage cost | Minimize |

## MVP Recommendation

For HMDB/LOINC metabolite search:

```
Model:        SapBERT or BioLORD
Strategy:     Single-pooled (evaluate multi-vector if recall insufficient)
Index:        HNSW + SQ8
Expected:     ~200K vectors, <500MB index, <5ms queries
```

---

*Evaluation in progress at biovector-eval (4 models × 3 strategies × 7 index types = 84 configurations)*
