# Technical Report: Question-to-Ontology Mapping Comparison

## Q2HPO (GenOMA) vs Q2LOINC (Vector+LLM) Analysis

**Report Date:** February 10, 2026  
**Dataset:** 100 stratified-sampled survey questions (seed=42)  
**Sources:** Arivale (34), Israeli10K (33), UK Biobank (33)

---

## Executive Summary

This report compares two distinct approaches for mapping biomedical survey questions to standardized ontologies:

| Approach | Target Ontology | Method | Match Rate |
|----------|-----------------|--------|------------|
| **Q2HPO** | Human Phenotype Ontology | GenOMA LangGraph agent (GPT-4) | **50%** |
| **Q2LOINC** | LOINC | Vector similarity + Claude Opus | **34%** |

**Key Finding:** The two approaches show only 60% agreement on mappability, but they exhibit complementary strengths—HPO excels at clinical phenotypes while LOINC excels at clinical observations and administrative data.

---

## Methodology

### Q2HPO Pipeline (GenOMA)

```
Question Text → Term Extraction (GPT-4) → HPO API Lookup → 
Candidate Ranking (GPT-4) → Validation → Iterative Refinement
```

- **Model:** OpenAI GPT-4
- **Ontology API:** ontology.jax.org (HPO)
- **Retry Strategy:** Up to 6 term reformulations per question
- **Caching:** SHA-256 content-addressed cache
- **Runtime:** 119.1 minutes (97 API calls, 3 cache hits)

### Q2LOINC Pipeline (Vector+LLM)

```
Question Text → Embedding (all-MiniLM-L6-v2) → FAISS IndexFlatIP → 
Top-5 Candidates → Claude Opus Reasoning → Best Match Selection
```

- **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2
- **Vector Index:** FAISS IndexFlatIP (cosine similarity)
- **Reasoning Model:** Claude Opus via CLI
- **Candidate Selection:** Top-5 by vector similarity
- **Runtime:** ~45 minutes

---

## Results Overview

### Match Rate Comparison

| Metric | Q2HPO | Q2LOINC |
|--------|-------|---------|
| Successful mappings | 50 | 34 |
| No match | 50 | 66 |
| Match rate | 50.0% | 34.0% |
| Avg confidence (matched) | 0.80 | 0.71 |
| High confidence (≥0.9) | 28 | 12 |

### Mappability Agreement Matrix

|  | LOINC Match | LOINC No Match |
|--|-------------|----------------|
| **HPO Match** | 22 | 28 |
| **HPO No Match** | 12 | 38 |

- **Agreement Rate:** 60% (22 + 38 = 60 questions)
- **Disagreement Rate:** 40% (28 + 12 = 40 questions)

---

## Agreement Analysis by Source

| Source | N | Both Match | Both No Match | HPO Only | LOINC Only |
|--------|---|------------|---------------|----------|------------|
| Arivale | 34 | 9 (26%) | 9 (26%) | 13 (38%) | 3 (9%) |
| Israeli10K | 33 | 6 (18%) | 11 (33%) | 11 (33%) | 5 (15%) |
| UK Biobank | 33 | 7 (21%) | 18 (55%) | 4 (12%) | 4 (12%) |

**Observations:**
- **Arivale** questions showed highest HPO-only match rate (38%), reflecting their focus on health conditions and symptoms
- **UK Biobank** had highest mutual no-match rate (55%), reflecting many administrative and lifestyle questions
- **Israeli10K** showed balanced disagreement between HPO and LOINC

---

## Qualitative Analysis

### Questions Where HPO Outperformed LOINC

These represent clinical phenotypes well-captured by HPO but absent from LOINC:

| Question | HPO Code | HPO Term | Confidence |
|----------|----------|----------|------------|
| Anorexia | HP:0002039 | Anorexia | 1.00 |
| Chronic bronchitis | HP:0004469 | Chronic bronchitis | 1.00 |
| High Blood Pressure | HP:0000822 | Hypertension | 1.00 |
| Vitiligo | HP:0001045 | Vitiligo | 1.00 |
| Malignant hyperthermia | HP:0002047 | Malignant hyperthermia | 1.00 |
| Myasthenia Gravis | HP:0003398 | Abnormal synaptic transmission at NMJ | 0.90 |
| Seizures or Epilepsy | HP:0033349 | Seizure cluster | 0.90 |

### Questions Where LOINC Outperformed HPO

These represent clinical observations and administrative data well-captured by LOINC:

| Question | LOINC Code | LOINC Term | Confidence |
|----------|------------|------------|------------|
| Are you currently lactating? | 63895-7 | Breastfeeding status | 0.95 |
| Current prescription medications | 66423-5 | Medications Current medication | 0.90 |
| Family history | 65947-4 | Family history notes | 0.85 |
| Mother's age at death | 54113-6 | Age range at death Family member | 0.80 |
| Overall health rating | 71901-3 | I am as healthy as anyone I know | 0.75 |

### High-Confidence Agreement (Both ≥0.9)

Only **1 question** achieved high-confidence matches in both ontologies:

| Question | HPO | LOINC |
|----------|-----|-------|
| "Do you get short of breath walking with people of your own age?" | HP:0002875 Exertional dyspnea (0.90) | 89439-4 Shortness of breath walking (0.98) |

---

## Confidence Distribution Analysis

### Q2HPO Confidence Bands

| Confidence | Count | % of Matches |
|------------|-------|--------------|
| ≥ 0.90 | 28 | 56% |
| 0.70–0.89 | 14 | 28% |
| 0.50–0.69 | 8 | 16% |
| < 0.50 | 0 | 0% |

### Q2LOINC Confidence Bands

| Confidence | Count | % of Matches |
|------------|-------|--------------|
| ≥ 0.90 | 12 | 35% |
| 0.70–0.89 | 8 | 24% |
| 0.50–0.69 | 10 | 29% |
| < 0.50 | 4 | 12% |

**Observation:** GenOMA's iterative refinement strategy produces higher average confidence, rejecting low-confidence matches as "unmappable" rather than returning weak candidates.

---

## Ontology Coverage Analysis

### HPO Coverage Gaps

HPO lacks terms for:
- Lifestyle behaviors (diet, exercise habits)
- Administrative/demographic data (family member ages)
- Non-phenotypic clinical observations (medication lists)
- Environmental exposures (secondhand smoke)

### LOINC Coverage Gaps

LOINC lacks specific terms for:
- Rare disease phenotypes
- Detailed symptom descriptions
- Genetic/hereditary conditions by name
- Psychological trait assessments

---

## Recommendations

### 1. Hybrid Approach for Maximum Coverage

A combined Q2HPO + Q2LOINC pipeline would achieve:
- **Unique coverage:** 62 questions (50 HPO + 34 LOINC - 22 overlap)
- **Coverage rate:** 62% vs 50% (HPO alone) or 34% (LOINC alone)

### 2. Ontology Selection by Question Type

| Question Type | Recommended Ontology |
|---------------|---------------------|
| Medical conditions/diagnoses | HPO |
| Symptoms and phenotypes | HPO |
| Clinical measurements | LOINC |
| Lifestyle/behavioral | LOINC |
| Administrative/demographic | LOINC |
| Medications | LOINC |

### 3. Confidence Threshold Tuning

- **HPO:** 0.70 threshold appropriate (56% matches ≥0.90)
- **LOINC:** Consider 0.60 threshold to include more candidates for human review

### 4. Human Review Prioritization

For the expertintheloop.io campaign, prioritize review of:
1. **Disagreement cases** (40 questions) - require domain expert adjudication
2. **Low-confidence matches** (conf < 0.70) - may need manual mapping
3. **High-value questions** - clinical phenotypes relevant to downstream analysis

---

## Technical Artifacts

| File | Description |
|------|-------------|
| `data/review/q2hpo_review_100.json` | Q2HPO results (GenOMA) |
| `data/review/q2loinc_review_100.json` | Q2LOINC results (Vector+LLM) |
| `data/review/q2hpo_vs_q2loinc_comparison.tsv` | Side-by-side comparison |
| `data/review/generation_log_hpo.json` | Q2HPO execution metrics |
| `data/review/generation_log.json` | Q2LOINC execution metrics |

---

## Appendix: Method Details

### GenOMA Term Extraction Strategy

GenOMA uses an iterative refinement loop:
1. Extract initial medical terms from question text
2. Query HPO API for candidates
3. Rank candidates by semantic relevance (GPT-4)
4. If confidence < 0.9, reformulate terms and retry (up to 6 iterations)
5. Return best match or "unmappable" if all attempts fail

Example refinement chain for "Back pain or back problems":
```
Abnormality of the back → Back pain → Musculoskeletal pain → 
Spinal pain → Abnormality of nervous system → Recurrent seizures
```

### Vector Similarity Search Strategy

Q2LOINC uses cosine similarity in embedding space:
1. Generate 384-dim embedding for question text
2. Search FAISS index for top-5 nearest LOINC terms
3. Pass candidates to Claude Opus for semantic reasoning
4. Select best match or return NO_MATCH if none appropriate

---

*Report generated by biovector-eval Q2HPO/Q2LOINC comparison pipeline*
