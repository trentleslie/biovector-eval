# Design Decisions & Rationale

> This document captures key technical decisions, their rationale, and supporting references.

---

## Table of Contents
1. [Embedding Models](#1-embedding-models)
2. [Embedding Strategies](#2-embedding-strategies)
3. [Index Types](#3-index-types)
4. [Evaluation Metrics](#4-evaluation-metrics)

---

## 1. Embedding Models

### Decision: Model Selection (2024-12-03)

**Selected Models:**

| Model | HuggingFace ID | Dimension | Purpose |
|-------|----------------|-----------|---------|
| BGE-M3 | `BAAI/bge-m3` | 1024 | General multilingual baseline |
| SapBERT | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | 768 | Biomedical entity linking |
| BioLORD-2023 | `FremyCompany/BioLORD-2023` | 768 | Clinical/biomedical concepts |
| GTE-Qwen2-1.5B | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | 1536 | Top MTEB performance |

**Rejected Models:**

| Model | Reason for Rejection |
|-------|---------------------|
| ChemBERTa (`DeepChem/ChemBERTa-77M-MTR`) | Trained on SMILES molecular structures, not text names. Underperformed by 22% on metabolite name matching. |
| BGE-Small (`BAAI/bge-small-en-v1.5`) | Replaced by larger, more capable models. |

### Rationale

1. **BGE-M3**: Strong general-purpose multilingual model. Serves as our baseline for comparing domain-specific models. High scores on MTEB retrieval tasks.

2. **SapBERT**: Specifically designed for biomedical entity linking using self-alignment pretraining on UMLS synonyms. Achieves SOTA on 6 medical entity linking benchmarks. Critical insight: treats synonyms as equivalent representations of the same concept - exactly what we need for metabolite matching.

3. **BioLORD-2023**: Extends SapBERT's approach by integrating knowledge graph information. Particularly strong at distinguishing between similar but distinct biomedical concepts. Uses full definitions to extract fine-grained information.

4. **GTE-Qwen2-1.5B**: Top performer on MTEB leaderboard (2024-2025). Provides comparison point between domain-specific and general SOTA models. Larger dimension (1536) may capture more nuance.

### Why ChemBERTa Failed

ChemBERTa is trained on **SMILES strings** (e.g., `C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O` for glucose), not text names like "D-Glucose" or "Dextrose". This is a fundamental mismatch:

- **Our task**: Match text queries ("glucose") to text names ("D-Glucose")
- **ChemBERTa's training**: Molecular structure representation

For future work with molecular structures (SMILES, InChI), ChemBERTa or similar models would be appropriate.

### References

- SapBERT: Liu et al., "Self-Alignment Pretraining for Biomedical Entity Representations" (NAACL 2021)
  - GitHub: https://github.com/cambridgeltl/sapbert
  - HuggingFace: https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext

- BioLORD-2023: Remy et al., "BioLORD-2023: semantic textual representations fusing LLMs and clinical knowledge graph insights" (JAMIA 2024)
  - Paper: https://academic.oup.com/jamia/article/31/9/1844/7614965

- Biomedical Entity Linking Evaluation: https://pmc.ncbi.nlm.nih.gov/articles/PMC11097978/

- MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard

- NVIDIA NV-Embed (context): https://developer.nvidia.com/blog/nvidia-text-embedding-model-tops-mteb-leaderboard/

---

## 2. Embedding Strategies

### Decision: Three-Strategy Approach (2024-12-03)

**Strategies:**

| Strategy | Description | Vectors/Metabolite |
|----------|-------------|-------------------|
| `single-primary` | Embed primary name only | 1 |
| `single-pooled` | Average embeddings of name + synonyms | 1 |
| `multi-vector` | Separate vectors with tiered weights | ~36 |

### Rationale

1. **single-primary**: Baseline approach. Simple, fast, minimal storage. Tests raw model capability without any data augmentation.

2. **single-pooled**: Hypothesis that averaging synonym embeddings creates a "semantic centroid" that better represents the metabolite concept. Same storage as single-primary but potentially better recall.

3. **multi-vector**: Maximum recall approach from production system (`create_hmdb_vector_database.py`). Each synonym and variation gets its own vector with tiered weights:
   - Primary name: 1.0
   - Synonyms: 0.9
   - Variations (Greek letters, prefix removal): 0.7

### Trade-offs

| Strategy | Storage | Recall | Precision | Complexity |
|----------|---------|--------|-----------|------------|
| single-primary | Lowest | Lowest | Highest | Lowest |
| single-pooled | Low | Medium | Medium | Low |
| multi-vector | Highest (~36x) | Highest | Medium | Highest |

### Mathematical Note

`single-pooled` ≠ concatenating text then embedding:

```python
# Pooled: Average of separate embeddings
mean(embed("D-Glucose"), embed("Dextrose"), embed("Grape sugar"))

# Concatenated: Single embedding of combined text
embed("D-Glucose | Dextrose | Grape sugar")
```

Pooling gives equal semantic weight to each name; concatenation lets transformer attention decide importance.

---

## 3. Index Types

### Decision: HNSW + IVF Comparison (2024-12-03)

**Selected Index Types:**

| Index | Type | Quantization | Purpose |
|-------|------|--------------|---------|
| Flat | Exact | N/A | Ground truth baseline |
| HNSW | Graph | None, SQ8, SQ4, PQ | Primary ANN approach |
| IVFFlat | Clustering | N/A | Clustering baseline |
| IVFPQ | Clustering+PQ | Built-in | Large-scale preparation |

### Rationale

1. **HNSW (Hierarchical Navigable Small World)**: State-of-the-art for high-dimensional semantic search. Graph-based navigation mirrors semantic similarity structure. Best for our 217K metabolite dataset.

2. **IVF (Inverted File Index)**: Clustering-based approach. Included for future scalability - when we add proteins (250M+), genes, diseases, IVF may become competitive or necessary.

3. **Quantization variants** (SQ8, SQ4, PQ): Test accuracy/storage trade-offs. SQ8 often provides near-lossless compression at 4x reduction.

### IVF Parameters

For 217K vectors:
- `nlist = 512` (√217920 ≈ 467, rounded up)
- `nprobe = 32` (search ~6% of clusters)

### References

- FAISS documentation: https://github.com/facebookresearch/faiss
- HNSW paper: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using HNSW graphs" (2018)

---

## 4. Evaluation Metrics

### Decision: Recall-Focused Evaluation (2024-12-03)

**Primary Metrics:**
- Recall@1, Recall@5, Recall@10
- MRR (Mean Reciprocal Rank)
- Latency (P50, P95, P99)
- Index size (MB)

### Rationale

For entity matching, **recall is paramount** - missing a correct match is worse than including extra candidates. Downstream processes can filter false positives, but cannot recover false negatives.

MRR provides insight into ranking quality - how often is the correct answer ranked first?

### Target

From Vector DB Analysis Plan: **85% Recall@1** target for metabolites.

Current best (BGE-M3 + SQ8 + single-primary): **52.6% Recall@1**

Gap suggests significant room for improvement via:
1. Better models (SapBERT, BioLORD)
2. Better strategies (single-pooled, multi-vector)
3. Combination of above

---

## Changelog

| Date | Decision | Author |
|------|----------|--------|
| 2024-12-03 | Initial model selection | Claude |
| 2024-12-03 | Replaced ChemBERTa with biomedical models | Claude |
| 2024-12-03 | Added IVF indexes for future scale | Claude |
