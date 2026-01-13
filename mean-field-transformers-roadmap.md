# Learning Roadmap: Mean-Field Dynamics of Transformers

## Overview

This roadmap guides development of a Jupyter notebook series for understanding Philippe Rigollet's paper "The Mean-Field Dynamics of Transformers" (arXiv:2512.01868v3). The series bridges from foundational transformer mechanics to the paper's mathematical insights, with specific applications to vector database design for biological entity mapping.

**Target Learner**: Bioinformatics engineer with rudimentary transformer understanding, strong practical experience with embedding models (ChemBERTa, SapBERT, MoLFormer), FAISS indexing, and quantization.

**Core Question**: How do the mathematical dynamics of self-attention explain embedding behavior, clustering phenomena, and representation collapse—and what does this mean for vector database design in biological entity mapping?

---

## Model Architecture Reference

The following models have been discussed in prior work and are relevant to connecting theory to practice:

| Model | Layers | Attention Heads | Hidden Size | Embedding Dim | Normalization | Notes |
|-------|--------|-----------------|-------------|---------------|---------------|-------|
| **ChemBERTa** | 6 | 12 | 768 | 384 | Post-LN (RoBERTa-based) | SMILES tokenization, chemistry-agnostic |
| **ChemBERTa-2 (77M-MTR)** | 6 | 12 | 768 | 768 | Post-LN | Multi-task regression pretraining |
| **SapBERT** | 12 | 12 | 768 | 768 | Post-LN | Based on PubMedBERT, UMLS self-alignment |
| **PubMedBERT** | 12 | 12 | 768 | 768 | Post-LN | Domain-specific vocabulary |
| **MoLFormer-XL** | 12 | 12 | 768 | 768 | Pre-LN (RoBERTa-based) | **Linear attention**, rotary embeddings |
| **Clinical-Longformer** | 12 | 12 | 768 | 768 | Pre-LN | Sliding window + global attention, 4096 tokens |
| **BioClinical ModernBERT** | 12/24 | 12/16 | 768/1024 | 768/1024 | Pre-LN | 8192 token context |

### Key Architectural Observations (Paper-Relevant)

1. **Normalization schemes differ**: MoLFormer-XL and Clinical-Longformer use Pre-LN; ChemBERTa/SapBERT use Post-LN
   - Paper Section 6.2: Pre-LN gives polynomial contraction ($1-\rho(t) \sim 1/t^2$), Post-LN gives exponential ($1-\rho(t) \sim e^{-2t}$)
   - **Caveat**: This describes the rate of *final collapse*, not the metastable phase where useful semantic structure lives
   - The metastable lifetime ($\log T_2 \sim \beta$) is independent of normalization scheme

2. **Layer depth varies**: 6 layers (ChemBERTa) vs 12 layers (SapBERT, MoLFormer)
   - Paper: More layers = more time for dynamics to evolve
   - **Caveat**: Whether this means "more clustering" depends on whether the model operates in the metastable regime or has already collapsed

3. **MoLFormer uses linear attention**: Different from standard softmax attention
   - Paper focuses on softmax attention dynamics; linear attention has different properties
   - Worth noting but outside paper's direct scope

### Critical Role of Initialization

The paper's theorems hold for "almost every initialization," but the *specific behavior* depends heavily on the starting geometry:

1. **Initialization happens BEFORE transformer dynamics**
   - Vocabulary embeddings are learned during pretraining
   - These set the initial geometry on the sphere
   - Transformer layers then evolve this geometry according to SA/USA dynamics

2. **Number of metastable clusters depends on initialization**
   - Section 5.1: Results assume "well-separated initial configuration"
   - The equiangular model (Section 6) assumes very specific initialization ($\rho_0$ equal for all pairs)
   - Real embeddings have complex, non-uniform initialization geometry

3. **What we can and cannot claim**
   
   | Claim | Status | Why |
   |-------|--------|-----|
   | Attention dynamics lead to clustering | ✅ Supported | Theorem 1 |
   | Metastable multi-cluster states exist | ✅ Supported | Section 5 |
   | Post-LN contracts faster than Pre-LN | ✅ Supported | Section 6.2 (equiangular model) |
   | Pre-LN "preserves semantic structure better" | ⚠️ Speculative | Depends on initialization, β, and whether we're in metastable regime |
   | Deeper models have more collapsed embeddings | ⚠️ Speculative | Depends on initialization and metastable lifetime |

4. **Open questions for connecting theory to practice**
   - What is the effective β in trained models?
   - Are production embeddings in the metastable regime or past it?
   - How does pretraining objective affect initial embedding geometry?

### The Synonym Problem: What "Semantic Similarity" Actually Means

A critical insight for applying this theory to biological entity mapping: **the transformer dynamics cannot create similarity that wasn't present at initialization**.

#### Distributional Similarity ≠ Synonym Equivalence

These models learn **distributional similarity**—words that appear in similar contexts get similar embeddings:
- "L-ascorbic acid" appears near: "antioxidant", "vitamin C deficiency", "collagen synthesis"
- "ascorbate" appears near: similar contexts → embeddings cluster reasonably well
- "D-isoascorbic acid" (stereoisomer) appears in different contexts (food preservation) → distant embedding

**Key insight**: Scientific publications are terminologically consistent within a paper. Authors pick one canonical name for clarity. So the training signal for "these two strings mean the same thing" is actually quite weak.

The models learn:
- ✅ Contextual co-occurrence: "glucose" appears near "metabolism", "insulin", "blood"
- ❌ NOT explicit equivalence: "D-glucose" = "dextrose" = "grape sugar"

Synonym pairs rarely appear in the same context (that would be redundant writing), so the transformer never learns they're equivalent.

#### How This Connects to the Dynamics

The Rigollet paper tells us:
1. Attention dynamics cluster tokens that **start nearby** (metastability preserves local structure)
2. Eventually everything collapses to one cluster (global convergence)

But: **the dynamics can't create similarity that wasn't there at initialization**.

If "sphinganine" and "dihydrosphingosine" start far apart on the sphere (different publication contexts), the attention dynamics will:
- Keep them in separate metastable clusters initially
- Eventually merge them with *everything else*, not specifically with each other

The transformer doesn't learn "these are synonyms"—it learns "these have similar contextual distributions." For scientific terminology, those aren't the same thing.

#### Implications for Your Work

| Approach | What it does | Theory connection |
|----------|--------------|-------------------|
| Domain-specific model (SapBERT) | Better initialization via UMLS contrastive learning | Changes starting geometry, but limited by UMLS coverage |
| Enrich queries with metadata | Shift embedding toward shared functional context | Effectively changes initialization at query time |
| Multi-vector approach | Index multiple representations per entity | Hedges against bad initialization for any single representation |
| Explicit synonym injection | Add known synonyms to index | Bypasses the distributional limitation entirely |

**The theory clarifies why the synonym problem is fundamentally an initialization/pretraining issue, not a model architecture issue.** Better attention dynamics won't help if the initial embeddings don't place synonyms nearby.

#### The Metadata Enrichment Hypothesis

Your idea about enriching queries with pathway/class information is theoretically motivated:

```
Bare query: "sphinganine"
→ Embedding determined solely by pretraining distributional context
→ If distant from "dihydrosphingosine", dynamics won't help

Enriched query: "sphinganine | sphingoid base | ceramide biosynthesis | HMDB0000269"
→ Projects into region defined by multiple contextual terms
→ If "dihydrosphingosine" shares pathway associations, enriched embeddings may be closer
→ You're providing the contextual information the model needed but didn't see in training
```

This is essentially **changing the effective initialization at query time**—a workaround for the fundamental limitation that you can't fix bad initialization with better dynamics.

---

## Notebook Series Structure

### Notebook 0: Foundations Refresher (Optional)
**Duration**: 1-2 hours  
**Prerequisites**: Basic linear algebra (vectors, matrices, inner products)

#### Learning Objectives
- [ ] Explain what token embeddings represent as vectors in high-dimensional space
- [ ] Implement basic attention computation from scratch
- [ ] Understand softmax temperature parameter β and its effect on attention "sharpness"
- [ ] Describe what layer normalization does geometrically (projection to unit sphere)

#### Content Outline
1. **Token Embeddings as Vectors**
   - Vectors in $\mathbb{R}^d$, cosine similarity
   - Why "similar things should be close"
   - Connection to your FAISS work: this IS what you're searching

2. **The Attention Mechanism**
   - Query, Key, Value matrices: $Q, K, V \in \mathbb{R}^{d \times d}$
   - Attention scores: $\text{softmax}(\beta \langle Qx_i, Kx_j \rangle)$
   - Weighted averaging of values
   
3. **Softmax Temperature**
   - Low β → uniform attention (all tokens equal weight)
   - High β → peaked attention (winner-take-all)
   - Paper's Equation (1): $\text{Attention}(X)_i = \sum_j \frac{\exp(\beta \langle Qx_i, Kx_j \rangle)}{\sum_k \exp(\beta \langle Qx_i, Kx_k \rangle)} Vx_j$

4. **Layer Normalization**
   - Projects tokens to unit sphere: $\|x\| = 1$
   - Why this matters: keeps representations bounded
   - Connection to paper: dynamics on $\mathbb{S}^{d-1}$

#### Deliverable
- `attention_basics.py`: Implement single-head attention with configurable β
- Visualization: Attention weights as heatmap for varying β

#### Connection to Your Work
This is the machinery inside ChemBERTa/SapBERT that produces your embeddings. When you call `model.encode(smiles)`, this is what's happening inside.

---

### Notebook 1: Attention as Particle Dynamics
**Duration**: 2-3 hours  
**Prerequisites**: Notebook 0 (or equivalent understanding)

#### Learning Objectives
- [ ] Interpret tokens as particles on a sphere
- [ ] Understand attention as an interaction force between particles
- [ ] Convert discrete layer updates to continuous-time dynamics
- [ ] Write down the SA and USA differential equations

#### Content Outline
1. **The Particle Metaphor**
   - Each token $x_i(t) \in \mathbb{S}^{d-1}$ is a particle on the unit sphere
   - Layer index becomes continuous time $t$
   - Residual connection: $x_{k+1} = x_k + F(x_k)$ → $\dot{x} = F(x)$

2. **Attention as Interaction**
   - Tokens "attract" each other based on similarity
   - High similarity → strong attraction
   - Softmax weights determine interaction strength

3. **The Core Equations**
   ```
   Self-Attention (SA):
   ẋᵢ(t) = P⊥_{xᵢ(t)} [ (1/Z_{β,i}) Σⱼ exp(β⟨xᵢ,xⱼ⟩) xⱼ ]
   
   Unnormalized Self-Attention (USA):
   ẋᵢ(t) = P⊥_{xᵢ(t)} [ (1/n) Σⱼ exp(β⟨xᵢ,xⱼ⟩) xⱼ ]
   ```
   - $P^\perp_x y = y - \langle x, y \rangle x$ is orthogonal projection (keeps particles on sphere)
   - $Z_{\beta,i}$ is the softmax normalization

4. **Interpreting the Dynamics**
   - Each particle moves toward the weighted average of all others
   - Projection keeps everything on the sphere
   - This IS what happens to your metabolite embeddings through transformer layers

#### Deliverable
- `particle_dynamics.py`: Simulate SA dynamics for n=10 particles on $\mathbb{S}^2$
- 3D visualization: Particle trajectories on unit sphere
- Animation showing evolution toward clustering

#### Key Insight
Your ChemBERTa embeddings undergo ~6 iterations of this process. SapBERT undergoes ~12 iterations. The paper tells you what happens mathematically.

---

### Notebook 2: The Clustering Theorem
**Duration**: 2-3 hours  
**Prerequisites**: Notebook 1

#### Learning Objectives
- [ ] State Theorem 1 (global clustering) in plain language
- [ ] Understand why clustering is almost sure (saddle point analysis)
- [ ] Derive local exponential rates from Theorem 3
- [ ] Connect to representation collapse in deep networks

#### Content Outline
1. **Theorem 1: Global Clustering**
   > For almost every initial condition, all tokens converge to a single cluster:
   > $\lim_{t \to \infty} \|x_i(t) - x_j(t)\| = 0$
   
   - "Almost every" = except a measure-zero set of initializations
   - This holds for ANY temperature β ≥ 0
   - This holds in dimension d ≥ 3

2. **Why Clustering Happens**
   - Energy functional: $E_\beta(\mu) = \frac{1}{2\beta} \int\int e^{\beta\langle x,y \rangle} d\mu(x) d\mu(y)$
   - SA/USA are gradient flows of this energy
   - Only stable equilibrium is all particles at one point
   - Other equilibria are saddles (unstable)

3. **Local Rates (Theorem 3)**
   - If tokens start in same hemisphere: exponential convergence
   - $\|x_i(t) - x^*\| \leq C e^{-\lambda t}$
   - Rate depends on β and initialization

4. **Reproducing Figure 1**
   - Paper shows ALBERT XLarge v2 pairwise inner products
   - Concentration near 1 = clustering
   - Your simulation should show same pattern

#### Deliverable
- Reproduce Figure 1 pattern with simulated SA dynamics
- Plot: Distribution of pairwise inner products over time
- Calculate convergence rates for different β

#### Connection to Your Work
**This explains representation collapse**. If you use a model with too many layers or too high β, your metabolite embeddings become indistinguishable. The theory predicts this will happen eventually for ANY model—the question is whether it happens before or after useful semantic structure is captured.

---

### Notebook 3: Metastability and Multiple Clusters
**Duration**: 3-4 hours  
**Prerequisites**: Notebook 2

#### Learning Objectives
- [ ] Define metastability in the context of attention dynamics
- [ ] Understand the timescale separation: fast intra-cluster, slow inter-cluster
- [ ] Interpret Figure 2 (energy staircase)
- [ ] Relate metastable cluster count to β

#### Content Outline
1. **The Metastability Phenomenon**
   - Theory says: eventual single cluster
   - Practice shows: multiple clusters persist for long times
   - This is the USEFUL regime for embeddings!

2. **Timescale Separation**
   - Fast dynamics: tokens within a cluster merge quickly ($t \sim \beta$)
   - Slow dynamics: clusters merge slowly ($\log T \sim \beta$)
   - Metastable window: $[T_1, T_2]$ where $T_1 \sim \beta$, $\log T_2 \sim \beta$

3. **The Energy Staircase (Figure 2)**
   - Energy is constant during metastable phases (plateaus)
   - Sharp jumps when clusters merge
   - Higher β → longer plateaus → more metastability

4. **Saddle-to-Saddle Dynamics (Theorem 6)**
   - As β → ∞, only closest pair merges at each step
   - Other clusters remain frozen
   - Deterministic ordering of merge events

5. **Number of Metastable Clusters**
   - Expected number ~ $\sqrt{\beta} \log \beta$ (Section 5.3)
   - Higher temperature → more initial clusters
   - Trade-off: more clusters but faster eventual collapse

#### Deliverable
- Simulate multi-cluster metastable dynamics
- Reproduce Figure 2 energy staircase for varying β
- Interactive: Show how β controls cluster formation

#### Connection to Your Work
**This explains why semantic clustering works**. Your metabolites with similar structures form stable subclusters (lipids together, amino acids together, etc.) that persist long enough to be useful, even though the theory says they'll eventually merge. The metastable regime IS your operating regime.

---

### Notebook 4: The Kuramoto Connection
**Duration**: 2 hours  
**Prerequisites**: Notebook 1

#### Learning Objectives
- [ ] Recognize the Kuramoto model as the β=0 limit of USA
- [ ] Understand synchronization on the circle (d=2)
- [ ] See how higher dimensions change the dynamics
- [ ] Connect to synchronization literature for analytical tools

#### Content Outline
1. **The Kuramoto Model**
   - Classical model for coupled oscillators
   - β=0 in USA gives: $\dot{\theta}_i = -\frac{1}{n} \sum_j \sin(\theta_i - \theta_j)$
   - Angles θ on the circle (d=2 case)

2. **Known Results for Kuramoto**
   - Synchronization proven: all oscillators align
   - This is Theorem 1 specialized to d=2, β=0
   - Rich literature with analytical tools

3. **Extension to General β**
   - Equation (4): $\dot{\theta}_i = -\frac{1}{n} \sum_j e^{\beta \cos(\theta_i - \theta_j)} \sin(\theta_i - \theta_j)$
   - β > 0 changes interaction strength based on alignment
   - Higher β → sharper interactions

4. **Higher Dimensions**
   - d > 2: particles on $\mathbb{S}^{d-1}$ instead of circle
   - Same qualitative behavior, more complex geometry
   - Theorems 1-3 hold for d ≥ 3

#### Deliverable
- Implement Kuramoto model (β=0)
- Compare dynamics for varying β on the circle
- Visualization: Phase portraits on the torus

#### Connection to Your Work
Provides intuition and connects to well-studied mathematical field. The synchronization literature gives you tools to understand attention dynamics.

---

### Notebook 5: Normalization Schemes and Contraction Rates
**Duration**: 3-4 hours  
**Prerequisites**: Notebooks 1-3

#### Learning Objectives
- [ ] Distinguish Pre-LN vs Post-LN normalization
- [ ] Derive the equiangular model reduction
- [ ] Calculate exponential vs polynomial contraction rates
- [ ] Explain why Pre-LN "makes better use of depth"

#### Content Outline
1. **Normalization Variants**
   | Type | Description | Where Used |
   |------|-------------|------------|
   | Post-LN | Normalize AFTER attention | ChemBERTa, SapBERT |
   | Pre-LN | Normalize BEFORE attention | MoLFormer, GPT, LLaMA |
   | Peri-LN | Normalize inputs AND outputs | Hybrid approaches |

2. **The Equiangular Model**
   - Special initialization: $\langle x_i(0), x_j(0) \rangle = \rho_0$ for all $i \neq j$
   - Symmetry preserved: remains equiangular for all time
   - Reduces to 1D ODE for correlation $\rho(t)$

3. **Contraction Rates**
   ```
   Post-LN: 1 - ρ(t) ~ exp(-2t)     [EXPONENTIAL]
   Pre-LN:  1 - ρ(t) ~ 1/t²         [POLYNOMIAL]
   ```
   - Post-LN clusters FAST (bad for deep networks)
   - Pre-LN clusters SLOW (preserves structure longer)

4. **Reproducing Figure 4**
   - Cosine similarity evolution with confidence intervals
   - Different curves for different normalization schemes
   - Clear separation between exponential and polynomial

5. **Practical Implications**
   - Why GPT/LLaMA use Pre-LN
   - Why shallow models (ChemBERTa, 6 layers) can get away with Post-LN
   - Trade-off: faster training vs representation collapse

#### Deliverable
- Implement equiangular model for both normalizations
- Reproduce Figure 4 comparison
- Calculate: how many layers before embeddings become indistinguishable?

#### Connection to Your Work
**What the theory tells us with confidence:**
- Post-LN contracts faster than Pre-LN in the equiangular model
- This is about the rate of *final collapse*, not the metastable phase

**What remains speculative:**
- Whether this translates to "better semantic preservation" in real models
- The connection depends on:
  - Initial embedding geometry (set by pretraining, before transformer dynamics)
  - Whether embeddings are in the metastable regime or past it
  - The effective β in trained models

Your observation that "SapBERT relies MORE on string matching than MiniLM" *could* be explained by faster clustering, but other explanations exist (different training data, different tokenization, different effective β). The theory provides a *possible* mechanism, not a confirmed explanation.

---

### Notebook 6: Long-Context Phase Transition
**Duration**: 2-3 hours  
**Prerequisites**: Notebook 5

#### Learning Objectives
- [ ] Understand why long sequences flatten attention
- [ ] Derive the critical scaling β ~ log(n)
- [ ] Interpret Theorem 7's three regimes
- [ ] Connect to practical long-context models

#### Content Outline
1. **The Long-Context Problem**
   - n tokens → softmax denominator grows ~ n
   - Fixed β → weights approach 1/n (uniform)
   - Uniform attention = no information, fast collapse

2. **Logarithmic Scaling**
   - Fix $\beta_n = \gamma \log n$
   - Attention weights: $A_{ij} = \frac{n^{\gamma \rho}}{n^\gamma + (n-1)n^{\gamma \rho}}$
   - Critical boundary at $\gamma = \frac{1}{1-\rho}$

3. **Three Regimes (Theorem 7)**
   | Regime | Condition | Behavior |
   |--------|-----------|----------|
   | Subcritical | $\gamma < \frac{1}{1-\rho}$ | Uniform contraction → collapse |
   | Critical | $\gamma = \frac{1}{1-\rho}$ | Sparse mixing → preserved structure |
   | Supercritical | $\gamma > \frac{1}{1-\rho}$ | Identity-like → no interaction |

4. **Practical Systems**
   - Qwen, SSMax, SWAN-GPT use $\beta \sim \log n$
   - This maintains useful attention even for long contexts
   - Clinical-Longformer (4096 tokens) faces these issues

#### Deliverable
- Implement attention with varying β scaling
- Visualize phase transition boundary
- Plot: Output correlation vs γ for different n

#### Connection to Your Work
Relevant when building vector databases over large biological knowledge graphs. If you're using context windows with many entities, the scaling of attention matters for maintaining semantic structure.

---

### Notebook 7: Implications for Vector Databases
**Duration**: 3-4 hours  
**Prerequisites**: Notebooks 1-6

#### Learning Objectives
- [ ] Articulate what "semantic similarity" means in distributional models
- [ ] Explain why transformer dynamics can't solve the synonym problem
- [ ] Connect initialization geometry to practical embedding behavior
- [ ] Evaluate the metadata enrichment hypothesis

#### Content Outline
1. **Model Selection Through Theory Lens**
   
   | Model | Layers | Norm | What Theory Predicts | Confidence |
   |-------|--------|------|---------------------|------------|
   | ChemBERTa | 6 | Post-LN | Faster final collapse rate, but fewer iterations | Low—depends on initialization |
   | SapBERT | 12 | Post-LN | Faster final collapse rate, more iterations | Low—depends on initialization |
   | MoLFormer-XL | 12 | Pre-LN | Slower final collapse rate | Low—linear attention not covered by theory |
   | Clinical-Longformer | 12 | Pre-LN | Slower final collapse rate | Low—sparse attention not covered |

   **Important caveat**: These predictions are about the *collapse phase*, not the *metastable phase* where useful semantic structure exists. Whether a model is "in" the metastable regime depends on initialization geometry from pretraining.

2. **The Synonym Problem: Why Dynamics Can't Help**
   
   Core insight: **Transformer dynamics preserve and eventually collapse local structure, but can't create similarity that wasn't there at initialization.**
   
   | What models learn | What they don't learn |
   |-------------------|----------------------|
   | Distributional context ("glucose" near "metabolism") | Explicit equivalence ("D-glucose" = "dextrose") |
   | Terms in similar publication contexts cluster | Synonyms used in different contexts stay apart |
   
   Scientific writing is terminologically consistent—authors don't use synonyms interchangeably for clarity. So the training signal for synonym equivalence is weak or absent.

3. **Connecting Theory to the Synonym Problem**
   
   Paper says: Tokens that start nearby stay in same metastable cluster
   
   Implication: If "sphinganine" and "dihydrosphingosine" have different distributional contexts in PubMed, they start far apart and will:
   - Remain in separate clusters during metastable phase
   - Eventually merge with *everything*, not specifically each other
   
   **The dynamics amplify or preserve initial structure—they don't create new semantic relationships.**

4. **Revisiting Your Observations (With Appropriate Uncertainty)**
   
   | Your Finding | Possible Theory Connection | Confidence |
   |--------------|---------------------------|------------|
   | String-cosine correlation r=0.788 | Clustering dynamics create correlated structure | Medium—mechanism plausible |
   | SapBERT relies more on string matching | *Could* be faster collapse, but many other explanations | Low—speculative |
   | Biomedical models show larger semantic gaps | Different training → different initialization geometry | Medium—plausible |
   | Need different models for different entity types | Different data distributions → different dynamics | Medium—plausible |
   | Synonym queries underperform | Synonyms don't co-occur in training → distant initialization | High—directly explained by theory |

   **The honest answer**: The theory provides *possible mechanisms* for these observations, not confirmed explanations. Connecting the idealized math to real model behavior requires empirical validation.

5. **The Metadata Enrichment Hypothesis**
   
   Your insight: Enriching queries with pathway/class information may help because it **changes the effective initialization at query time**.
   
   ```python
   # Bare query - limited to pretraining distributional context
   embed("sphinganine")  
   
   # Enriched query - provides contextual signal missing from training
   embed("sphinganine | sphingoid base | ceramide biosynthesis | HMDB0000269")
   ```
   
   Theory framing:
   - Bare embedding = wherever pretraining placed this term
   - Enriched embedding = projection into region defined by multiple terms
   - If synonyms share functional context (pathways, classes), enriched embeddings cluster better
   
   **This is a workaround for bad initialization, not a fix for the dynamics.**

6. **Experimental Predictions from Theory**
   
   | Hypothesis | Prediction | How to test |
   |------------|------------|-------------|
   | Enrichment helps synonyms | Synonym recall improves more than non-synonym recall with enrichment | Compare Δrecall for synonym vs non-synonym query sets |
   | Initialization dominates | Models with similar pretraining show similar behavior despite architectural differences | Compare ChemBERTa vs other RoBERTa-based models |
   | Metastable structure matters | Layer-wise embeddings show distinct cluster structure before final layer | Extract intermediate layer embeddings, measure cluster counts |

7. **What the Theory Tells Us vs What It Doesn't**
   
   **High confidence (from paper)**:
   - Attention dynamics lead to clustering
   - Metastable multi-cluster states exist
   - Normalization affects collapse rate (in equiangular model)
   
   **Medium confidence (reasonable extrapolation)**:
   - Initial geometry determines which things cluster together
   - Enrichment may help by shifting effective initialization
   
   **Low confidence (speculative)**:
   - Specific predictions about real model behavior
   - Quantitative predictions about synonym performance

#### Deliverable
- Analysis script: Compute pairwise similarity distributions for your embeddings
- Experiment design: Test enrichment hypothesis on synonym vs non-synonym queries
- Decision framework: When to use enrichment, multi-vector, or explicit synonym injection

---

## Synthesis: Key Takeaways

### What the Paper Establishes (High Confidence)

1. **Why embeddings cluster**: Attention dynamics are gradient flows that converge to single clusters (Theorem 1)
2. **Why clustering is slow enough to be useful**: Metastability creates long-lived multi-cluster states (Section 5)
3. **How normalization affects collapse rate**: Pre-LN gives polynomial, Post-LN gives exponential—in the equiangular model (Section 6.2)
4. **Why long contexts need scaling**: Phase transition at β ~ log(n) (Theorem 7)

### The Critical Initialization Insight

**Transformer dynamics preserve and amplify initial structure—they cannot create semantic relationships that weren't present at initialization.**

This has direct implications for the synonym problem in biological entity mapping:
- Synonyms that don't co-occur in training corpora start far apart on the embedding sphere
- Attention dynamics will keep them in separate metastable clusters
- Eventually they merge with *everything*, not specifically with each other
- **The dynamics can't fix what pretraining didn't provide**

Scientific writing is terminologically consistent (authors pick one name for clarity), so the training signal for synonym equivalence is weak. This is a fundamental limitation of distributional semantics, not a model architecture issue.

### What Remains Speculative (Requires Empirical Validation)

1. **Whether real models are in metastable regime**: Depends on initialization from pretraining
2. **Whether normalization differences matter in practice**: The equiangular model is highly idealized
3. **Direct predictions about specific models**: Linear attention (MoLFormer), sparse attention (Longformer) not covered
4. **Causal explanations for empirical observations**: Theory provides mechanisms, not confirmed explanations

### Practical Implications

| Problem | Theory-informed approach | Why it might help |
|---------|-------------------------|-------------------|
| Synonym recall is poor | Enrich queries with metadata (pathways, classes) | Changes effective initialization at query time |
| Need to compare models | Look at initialization (pretraining), not just architecture | Initialization dominates over dynamics |
| Some entity types work better | Different training corpora → different initial geometry | Match model pretraining to your domain |
| String similarity correlates with cosine | Expected—both reflect surface-level distributional patterns | Not a bug, a feature of how these models work |

### What This Means for Your Work

1. **The theory provides a conceptual framework**: Useful for understanding *why* things behave as they do
2. **Initialization matters more than architecture details**: Pretraining sets the geometry that dynamics evolve
3. **Metastability is the key concept**: The useful semantic structure lives in metastable states
4. **The synonym problem is fundamental**: Can't be solved by better dynamics—requires better initialization or workarounds (enrichment, explicit synonym injection)
5. **Empirical validation needed**: Don't over-interpret theoretical predictions for real models

### Limitations of the Theory

1. **Simplified model**: No MLP, no multi-head, no positional encoding
2. **Continuous time**: Real transformers are discrete layers
3. **Identity Q/K/V**: Real models have learned projections
4. **Linear/sparse attention not covered**: MoLFormer, Longformer outside scope
5. **Initialization assumed generic**: Real pretraining creates specific, non-generic geometry
6. **β treated as fixed**: Real models may have layer-dependent effective temperatures
7. **Doesn't address pretraining**: The paper is about dynamics given an initialization, not how initialization is created

---

## Learning Path Diagram

```
                      [Notebook 0: Foundations]
                              ↓
                      [Notebook 1: Particle Dynamics]
                              ↓
            ┌─────────────────┼─────────────────┐
            ↓                 ↓                 ↓
  [NB2: Clustering]  [NB3: Metastability]  [NB4: Kuramoto]
            └─────────────────┼─────────────────┘
                              ↓
                      [Notebook 5: Normalization]
                              ↓
                      [Notebook 6: Long-Context]
                              ↓
                      [Notebook 7: Vector DB Implications]
```

**Minimum path for practical insights**: 1 → 2 → 3 → 5 → 7  
**Full mathematical understanding**: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7

---

## Mathematical Prerequisites Checklist

| Concept | Where Used | Quick Refresh |
|---------|------------|---------------|
| Inner products, cosine similarity | Everywhere | $\langle x, y \rangle = \sum x_i y_i$; $\cos\theta = \frac{\langle x,y \rangle}{\|x\|\|y\|}$ |
| Unit sphere $\mathbb{S}^{d-1}$ | Particle dynamics | Points with $\|x\| = 1$ |
| Orthogonal projection | SA equation | $P^\perp_x y = y - \langle x,y \rangle x$ |
| Gradient flow | Energy perspective | $\dot{x} = -\nabla E(x)$ |
| Exponential convergence | Rate analysis | $f(t) \leq C e^{-\lambda t}$ |
| Softmax | Attention weights | $\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ |

---

## NotebookLM Scope Prompt (for your workflow)

```markdown
# Scope: Mean-Field Transformer Dynamics

## Core Question
How do the mathematical dynamics of self-attention explain embedding behavior, clustering phenomena, and representation collapse—and what does this mean for vector database design in biological entity mapping?

## In Scope
- Rigollet paper (2512.01868v3) and referenced works
- SA/USA dynamics equations and their properties
- Clustering theorems and convergence rates
- Metastability and multi-cluster formation
- Normalization schemes (Pre-LN vs Post-LN)
- Long-context phase transitions
- The role of initialization in determining clustering behavior
- Why distributional similarity ≠ synonym equivalence
- Applications to embedding model evaluation (not selection—too speculative)

## Out of Scope
- Full transformer training dynamics
- MLP layers and their role
- Multi-head attention decomposition
- Positional encodings
- Specific dataset benchmarks
- Implementation details of production models
- How to design better pretraining (paper doesn't address this)

## Success Criteria
- Can explain why embeddings cluster in plain language (high confidence)
- Can articulate what the theory does and doesn't tell us about real models
- Can explain why the synonym problem is fundamentally about initialization, not dynamics
- Can identify which claims are well-supported vs speculative
- Can use the theory as a conceptual framework while acknowledging limitations

## Atomic Ideas
1. Attention = particle interaction on a sphere
2. All particles eventually cluster (Theorem 1)
3. Metastability preserves useful structure temporarily
4. Pre-LN delays collapse (polynomial vs exponential) in equiangular model
5. Long contexts need β ~ log(n) scaling
6. **Dynamics can't create similarity that wasn't there at initialization**
7. **The synonym problem is a pretraining/initialization issue, not architecture**

## Connection to Work
BioMapper vector database for biological entity harmonization. Phase 1: metabolite evaluation using FAISS. Key insight: synonym queries underperform because synonyms don't co-occur in scientific training corpora, so they start far apart in embedding space. The theory explains why better architectures won't fix this—need to address initialization (enrichment, explicit synonyms, better pretraining).
```

---

## References from Paper

Key papers to have on hand for deeper dives:

- [GLPR25] Geshkovski et al., "A mathematical perspective on transformers" - Bulletin of AMS
- [GKPR24] Geshkovski et al., "Dynamic metastability in the self-attention model"
- [KGPR25] Karagodin et al., "Normalization in attention dynamics"
- [CLPR25] Chen et al., "Quantitative clustering in mean-field transformer models"
- [MTG17] Markdahl et al., "Almost global consensus on the n-sphere"
- [Tay12] Taylor, "There is no non-zero stable fixed point for dense networks in the homogeneous Kuramoto model"

---

## Implementation Notes

### Recommended Libraries
```python
numpy          # Core numerics
scipy          # ODE solvers (solve_ivp)
matplotlib     # Visualization
plotly         # Interactive 3D plots
torch          # Optional: compare with real models
transformers   # Optional: extract embeddings from models
```

### Key Functions to Implement
```python
def project_to_sphere(x):
    """Orthogonal projection keeping points on S^{d-1}"""
    
def attention_weights(X, beta):
    """Compute softmax attention weights"""
    
def sa_velocity(X, beta):
    """Right-hand side of SA dynamics"""
    
def simulate_sa(X0, beta, t_span, n_steps):
    """Solve SA ODE with scipy.integrate.solve_ivp"""
    
def pairwise_similarity_dist(X):
    """Compute distribution of pairwise inner products"""
    
def equiangular_ode(rho, t, beta, n, norm_type):
    """1D ODE for equiangular model"""
```

---

*Document version: 1.0*  
*Created: 2026-01-08*  
*Source paper: arXiv:2512.01868v3*
