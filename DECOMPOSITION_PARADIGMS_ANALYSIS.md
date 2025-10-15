# Decomposition Paradigms: Deep Design Analysis

## Critical Questions

1. **Can we test generalization without collecting new activations?**
2. **What exactly are spectral methods and how do they differ from PCA/LDA?**
3. **What are the competing decomposition paradigms?**
4. **How do we integrate layer choice (generalization vs specificity tradeoff)?**
5. **What design-level tradeoffs cannot be solved with math alone?**

---

## Q1: Can We Test Generalization Without New Activations?

### Short Answer: NO (not exactly) - We're Using a Proxy

**What we have:**
```python
# Probe vectors (mean-diffing extracts these)
v_A = mean(activations_A_true) - mean(activations_A_false)  # Shape: (D,)
v_B = mean(activations_B_true) - mean(activations_B_false)  # Shape: (D,)
```

**What we want to measure:**
```python
# True generalization: How well does v_A classify dataset B?
scores_B_true = [v_A @ act for act in activations_B_true]
scores_B_false = [v_A @ act for act in activations_B_false]
accuracy_A_on_B = measure_separation(scores_B_true, scores_B_false)
```

**What we actually measured:**
```python
# Proxy: Cosine similarity between probe vectors
similarity(v_A, v_B) = |v_A @ v_B| / (||v_A|| ||v_B||)
```

### The Mathematical Relationship

**When does similarity predict generalization?**

Assume activations for dataset B are sampled from:
- True: `act ~ N(μ_B_true, Σ_B)`
- False: `act ~ N(μ_B_false, Σ_B)`

Then:
```
v_B = μ_B_true - μ_B_false  (population mean difference)

Accuracy of v_A on B ≈ Φ(v_A @ (μ_B_true - μ_B_false) / (2σ_B))
                      = Φ(v_A @ v_B / (2σ_B))
                      ∝ v_A @ v_B  (under constant σ_B assumption)
```

**So similarity IS a proxy for generalization, BUT:**

**Assumption 1:** σ_B (spread of activations) is similar across datasets
- **Likely violated:** Some datasets have tighter clusters than others
- **Impact:** Similarity overpredicts transfer to tight clusters, underpredicts to loose clusters

**Assumption 2:** Distributions are Gaussian
- **Likely violated:** Activations may have multiple modes, heavy tails
- **Impact:** Similarity may not capture distribution overlap

**Assumption 3:** We're using the same layer
- **Violated in our case:** We extract different layers per model
- **Impact:** Similarity across layers is even less predictive

### What We Should Have Done

**Option 1: Full cross-dataset evaluation (EXPENSIVE)**
```python
for train_ds in datasets:
    for test_ds in datasets:
        if train_ds != test_ds:
            # Extract probe on train_ds
            probe = extract_probe(train_ds)

            # Collect NEW activations on test_ds
            test_acts = get_activations(test_ds)

            # Measure actual accuracy
            accuracy = test_probe(probe, test_acts)
```

**Cost:** 18×17 = 306 train/test pairs × activation extraction time

**Option 2: Use pre-computed activations (what we should do)**
```python
# If we already cached activations for all datasets
for train_ds in datasets:
    probe = extract_probe(train_ds, cached_activations)
    for test_ds in datasets:
        accuracy = test_probe(probe, cached_activations[test_ds])
```

**Cost:** Storage (cache activations) + 306 probe tests (cheap)

**Option 3: Similarity proxy (what we did)**
```python
# Fast but approximate
similarity_matrix[i,j] = cosine_similarity(probe_i, probe_j)
```

**Cost:** Minimal, but less accurate

### The Verdict

**What our similarity analysis tells us:**
- ✓ Relative ordering of which datasets are related
- ✓ Clustering structure (which probes form groups)
- ✓ Orthogonality structure (which domains don't interfere)
- ✗ Absolute generalization performance (needs activations)
- ✗ Tail behavior (needs actual distribution testing)

**For safety probes, we NEED actual cross-dataset evaluation:**
- Similarity proxy is insufficient for tail coverage
- Must test on adversarial variants (not just similar datasets)
- Must measure @ extreme FPR thresholds (10^-4 to 10^-5)

**Recommendation:**
1. Use similarity for initial clustering (fast, good enough)
2. Validate with actual cross-dataset tests on representative pairs
3. Final evaluation must use adversarial test set with real activations

---

## Q2: Spectral Methods vs PCA/LDA - What's the Difference?

### The Matrix Zoo

All these methods operate on matrices, but DIFFERENT matrices:

```python
# Data matrix
X = (n_samples, n_features)  # In our case: (18 datasets, D dimensions)

# Key matrices:
Covariance:     C = X^T X / n          # (D, D) - feature covariances
Gram:           G = X X^T / n          # (n, n) - sample similarities
Within-class:   S_W = Σ_k (X_k - μ_k)^T (X_k - μ_k)
Between-class:  S_B = Σ_k n_k (μ_k - μ)^T (μ_k - μ)
Laplacian:      L = D - W              # Graph Laplacian
```

### PCA (Principal Component Analysis)

**Operates on:** Covariance matrix C = X^T X

**Objective:** Maximize variance
```
max_v  v^T C v
s.t.   ||v|| = 1
```

**Solution:** Eigenvectors of C (principal components)

**Finds:** Directions of maximum variance in feature space

**Example:**
```
PC1: [0.3, 0.5, 0.2, ...]  (D-dimensional)
Explains 25% of variance across all probes
```

**Use case:** Dimensionality reduction, finding dominant patterns

**In our setting:**
- PC1 = "general truth" direction (25% variance)
- PC2 = "language structure" (17% variance)
- PC3 = "probabilistic" (8% variance)

### LDA (Linear Discriminant Analysis)

**Operates on:** Within-class scatter S_W and between-class scatter S_B

**Objective:** Maximize between-class variance, minimize within-class variance
```
max_v  v^T S_B v / v^T S_W v
```

**Solution:** Eigenvectors of S_W^(-1) S_B

**Finds:** Directions that best separate classes

**Example:**
```
If classes = {reasoning, factual, probabilistic}
LDA finds direction that separates these 3 groups maximally
```

**Use case:** Supervised classification, class separation

**In our setting:**
- Would need to define classes (e.g., our 7 groups)
- Then finds directions that separate groups
- Only works if we KNOW the grouping (supervised)

### Spectral Clustering

**Operates on:** Gram matrix G = X X^T (or Laplacian L)

**Objective:** Partition into subspace-aligned clusters
```
min_{C_1,...,C_k}  Σ_i Σ_{x,y ∈ C_i} ||x - y||^2
(with relaxation via eigenvectors)
```

**Solution:**
1. Compute top-k eigenvectors of G
2. Embed samples in k-dimensional spectral space
3. Run k-means on embeddings

**Finds:** Clusters that form coherent subspaces

**Example:**
```
Eigenvector 1: [0.3, 0.3, 0.3, -0.4, -0.4, ...]  (n-dimensional = 18)
Separates samples into two groups
```

**Use case:** Clustering data that lies on manifolds/subspaces

**In our setting:**
- Finds 7 groups that form coherent subspaces
- Unsupervised (doesn't need labels)
- Better than hierarchical clustering for subspace structure

### Key Differences

| Method | Matrix | Space | Supervised | Output | Optimizes |
|--------|--------|-------|------------|--------|-----------|
| **PCA** | C (D×D) | Feature | No | Components (D-dim) | Variance |
| **LDA** | S_B, S_W | Feature | Yes | Discriminants (D-dim) | Separation |
| **Spectral** | G (n×n) | Sample | No | Clusters (groups) | Subspace coherence |

**Why spectral wins for clustering:**
- Hierarchical clustering: Uses pairwise distances (local)
- Spectral clustering: Uses global subspace structure (global)
- Hierarchical: Merges nearest points iteratively
- Spectral: Finds coherent subspaces in one shot

**When to use each:**
- **PCA:** Find dominant patterns in features (e.g., "what is PC1?")
- **LDA:** Supervised classification with known classes
- **Spectral:** Unsupervised clustering with subspace structure

**In our case:**
- Used PCA to understand what components exist (PC1 = general truth)
- Used spectral clustering to group probes into specialists
- Could use LDA to validate if groups are well-separated (with group labels from spectral)

---

## Q3: Competing Decomposition Paradigms

### Paradigm 1: Hierarchical (Shared + Specific)

**Architecture:**
```
                [Universal Component - PC1]
                           |
                           v
              +------------+------------+
              |            |            |
         [Residual 1] [Residual 2] [Residual 3]
              |            |            |
         Specialist 1  Specialist 2  Specialist 3
```

**Mathematical formulation:**
```python
# Step 1: Extract shared component
w_shared = PC1(X)  # (D,) - universal truth direction

# Step 2: Project out shared component
X_residual = X - (X @ w_shared) @ w_shared^T

# Step 3: Cluster residuals
for each cluster k:
    w_k = mean(X_residual[cluster == k])
```

**Decision rule:**
```python
def predict(activation):
    # Stage 1: Shared gate
    score_shared = w_shared @ activation
    if score_shared < threshold_gate:
        return "benign"

    # Stage 2: Specialist routing
    for k, w_k in enumerate(specialists):
        score_k = w_k @ activation
        if score_k > threshold_k:
            return f"threat_type_{k}"

    return "benign"
```

**Pros:**
- ✓ Computational efficiency (shared gate filters 99%)
- ✓ Clear hierarchy (universal → specific)
- ✓ Interpretable (PC1 has meaning)
- ✓ Reduces redundancy (share computation)

**Cons:**
- ✗ Assumes PC1 is meaningful (only explains 25%)
- ✗ Forces orthogonality (residuals by construction)
- ✗ Hard to add new specialists (must recompute shared)
- ✗ What if some threats are NOT orthogonal to PC1?

**When to use:**
- PC1 explains significant variance (>50%)
- Computational budget is tight
- Want interpretable hierarchy

### Paradigm 2: Orthogonal Bases (Subspace Partitioning)

**Architecture:**
```
    [Specialist 1] ⊥ [Specialist 2] ⊥ [Specialist 3]
         ∈ S_1          ∈ S_2          ∈ S_3

    S_1 ⊥ S_2,  S_2 ⊥ S_3,  S_1 ⊥ S_3  (orthogonal subspaces)
```

**Mathematical formulation:**
```python
# Step 1: Cluster into K groups (e.g., spectral clustering)
labels = spectral_clustering(X @ X^T, n_clusters=K)

# Step 2: Extract centroid for each cluster
for k in range(K):
    w_k = mean(X[labels == k])
    w_k = w_k / ||w_k||  # normalize

# No hierarchy - all specialists are independent
```

**Decision rule:**
```python
def predict(activation):
    # Test all specialists independently
    triggered = []
    for k, w_k in enumerate(specialists):
        score_k = w_k @ activation
        if score_k > threshold_k:
            triggered.append(k)

    if len(triggered) == 0:
        return "benign"

    # Apply FDR control if multiple fire
    if len(triggered) > 1:
        significant = benjamini_hochberg(triggered, alpha=0.05)
        return significant

    return triggered[0]
```

**Pros:**
- ✓ No forced hierarchy (treats all specialists equally)
- ✓ Easy to add/remove specialists
- ✓ Minimal cross-contamination (orthogonality)
- ✓ Matches our finding (some pairs are 0.5% similar!)

**Cons:**
- ✗ No shared gate (must evaluate all specialists)
- ✗ May miss true hierarchical structure
- ✗ Requires K threshold settings (one per specialist)

**When to use:**
- Subspaces are truly orthogonal (<10% overlap)
- Need flexibility to add specialists
- Don't care about computational efficiency

### Paradigm 3: Dictionary Learning (Sparse Decomposition)

**Architecture:**
```
probe_i = α_i1 * atom_1 + α_i2 * atom_2 + ... + α_iK * atom_K
where most α_ij = 0 (sparse)
```

**Mathematical formulation:**
```python
# Learn dictionary atoms D and sparse coefficients α
min_{D, α}  ||X - α D||^2 + λ ||α||_1

# Each probe is sparse combination of atoms
```

**Decision rule:**
```python
def predict(activation):
    # Sparse coding: represent activation as sparse combination of atoms
    coeffs = sparse_encode(activation, dictionary)

    # Check which atoms are active
    active_atoms = [i for i, c in enumerate(coeffs) if abs(c) > threshold]

    # Map atoms to threat types
    return map_atoms_to_threats(active_atoms)
```

**Pros:**
- ✓ Captures compositional structure
- ✓ More expressive than centroids
- ✓ Handles overlapping concepts

**Cons:**
- ✗ More complex (requires sparse coding)
- ✗ Harder to interpret (atoms may not align with semantics)
- ✗ Slower inference

**When to use:**
- Concepts are compositional (threat = atom_1 + atom_2)
- Have enough data to learn overcomplete dictionary
- Interpretability is secondary

### Which Paradigm for Safety Probes?

**My recommendation: HYBRID (Paradigm 1 + 2)**

Looking at our data:
- PC1 explains 25% variance (meaningful shared component)
- But Groups 3 & 5 are nearly orthogonal to everything (0.5%)
- This suggests: SOME structure is shared, SOME is independent

**Hybrid Architecture:**
```python
Stage 1: Shared gate (PC1)
         - Fast filtering (10^-3 FPR)
         - Filters 99.9% of benign examples
         ↓ (only 0.1% proceed)

Stage 2: Independent specialists (5 orthogonal subspaces)
         - Precise detection (10^-5 FPR per specialist)
         - Each specialist is independent
         ↓

Stage 3: FDR control (Benjamini-Hochberg)
         - Correct for multiple testing
         - Control false discovery rate
```

**Why hybrid:**
- Leverages PC1 for efficiency (don't evaluate specialists on obvious benign)
- Allows orthogonal specialists (matches data: 0.5% - 60% overlap)
- Best of both worlds

---

## Q4: Layer Choice - Generalization vs Specificity Tradeoff

### The Core Tension

**Our findings:**
```
Layer Depth    Generalization    Specificity    Magnitude
─────────────────────────────────────────────────────────
20-40%         Poor              None           ~0.02
60-65%         ★★★ BEST         Moderate       ~20-80
75-85%         Poor              ★★★ BEST       ~170-525
```

**The tradeoff:**
- **Middle layers (60-65%):** Truth representation is general, transfers well
- **Late layers (75-85%):** Truth representation specializes, doesn't transfer

### Key Question: Should We Use DIFFERENT LAYERS for Different Components?

**Option A: Single layer for everything (current approach)**
```python
layer = 0.625  # Middle layer (best generalization)

# Extract both shared and specialists from same layer
w_shared = PC1(extract_probes(layer=0.625))
specialists = cluster(extract_probes(layer=0.625))
```

**Pros:**
- ✓ Simple, consistent
- ✓ All probes in same space (easy to compare)

**Cons:**
- ✗ Compromise between generalization and specificity
- ✗ Doesn't leverage full model capacity

**Option B: Layer-wise hybrid (NOVEL APPROACH)**
```python
# Shared component from middle layer (generalization)
w_shared = PC1(extract_probes(layer=0.625))

# Specialists from late layer (specificity)
specialists = {
    'factual': extract_specialist(layer=0.875, domain='factual'),
    'reasoning': extract_specialist(layer=0.875, domain='reasoning'),
    'probabilistic': extract_specialist(layer=0.875, domain='probabilistic'),
}
```

**Pros:**
- ✓ Shared gate has high generalization (catches broad threats)
- ✓ Specialists have high specificity (precise detection)
- ✓ Leverages both: middle for recall, late for precision

**Cons:**
- ✗ More complex (mixing layers)
- ✗ Probes in different spaces (harder to compare)
- ✗ Requires validating that this actually works

### Mathematical Consideration

**Can we mix layers?**

If probes are from different layers:
- `w_shared` is in space R^D (activations at layer L1)
- `w_specialist` is in space R^D (activations at layer L2)

**Problem:** Cannot directly compare or combine!

**Solution 1:** Apply sequentially
```python
# Step 1: Extract activations at L1, apply shared gate
act_L1 = model(text, return_layer=L1)
if w_shared @ act_L1 < threshold:
    return "benign"

# Step 2: Extract activations at L2, apply specialists
act_L2 = model(text, return_layer=L2)
for specialist in specialists:
    if specialist @ act_L2 > threshold:
        return "threat"
```

**Solution 2:** Map between layers
```python
# Learn linear map M: L1 → L2
M = learn_linear_map(layer1_acts, layer2_acts)

# Apply shared gate at L1
act_L1 = model(text, return_layer=L1)
if w_shared @ act_L1 < threshold:
    return "benign"

# Project to L2 space, apply specialists
act_L2_predicted = M @ act_L1
for specialist in specialists:
    if specialist @ act_L2_predicted > threshold:
        return "threat"
```

### Empirical Question

**Does extracting specialists from late layers actually improve precision?**

We found:
- Late layers have worse generalization (don't transfer to other datasets)
- BUT: Do they have BETTER in-distribution performance on their own task?

**Hypothesis:**
- Specialist from layer 0.875 has >95% in-dist on factual hallucination
- But transfers poorly (<60%) to reasoning tasks
- This is GOOD for safety (high precision, low cross-contamination)

**We should test:**
```python
for dataset in datasets:
    for layer_depth in [0.375, 0.50, 0.625, 0.75, 0.875]:
        probe = extract_probe(dataset, layer_depth)

        # In-distribution accuracy
        in_dist_acc[dataset, layer_depth] = test_on_same_dataset(probe)

        # Cross-dataset transfer
        cross_acc[dataset, layer_depth] = avg_transfer_to_others(probe)

        print(f"{dataset:30} L={layer_depth:.3f}: "
              f"in-dist={in_dist_acc:.1%}, cross={cross_acc:.1%}, "
              f"gap={in_dist_acc - cross_acc:.1%}")
```

**Prediction:**
- In-dist should INCREASE at late layers (specialization improves own-task)
- Cross-dataset should DECREASE at late layers (specialization hurts transfer)
- Gap should INCREASE at late layers

**If this holds:**
- Use late layers for specialists (high in-dist = good for safety)
- Use middle layers for shared gate (high cross-dataset = catches broad threats)

---

## Q5: Design-Level Tradeoffs Math Cannot Solve

### The Meta-Problem

**Mathematics can:**
- ✓ Measure separation, variance, reconstruction error
- ✓ Optimize within a given objective
- ✓ Compare different decompositions quantitatively

**Mathematics CANNOT:**
- ✗ Tell you which objective to optimize
- ✗ Tell you what granularity is useful
- ✗ Tell you what architecture to deploy

These require **design judgment** based on:
1. Application goals
2. Deployment constraints
3. Interpretability needs
4. Risk tolerance

### Tradeoff 1: Generalization vs Precision

**The tension:**
- High generalization = good cross-dataset transfer = "mushy" probe
- High precision = good in-dist accuracy = specialist probe

**Math says:**
- Correlation between in-dist and gap: r = 0.98
- Can measure both metrics

**Design question:**
- **For research:** Optimize generalization (understand universal truth)
- **For safety:** Optimize precision (catch specific threats at 10^-5 FPR)

**No math can tell you which to choose!** It depends on your goal.

### Tradeoff 2: Compression vs Expressiveness

**The tension:**
- More compression (7 specialists) = better separation, less expressive
- Less compression (18 probes) = full expressiveness, more redundancy

**Math says:**
- 7 specialists: 79% reconstruction, 9× better conditioning
- 18 probes: 100% reconstruction, worse conditioning

**Design question:**
- **For production:** Use 7 specialists (cleaner, faster, interpretable)
- **For research:** Keep 18 probes (study all phenomena)
- **For resource-constrained:** Use 5 specialists (maximum compression)

**No math can tell you where to stop compressing!** It depends on your constraints.

### Tradeoff 3: Shared vs Independent Architecture

**The tension:**
- Shared gate = computational efficiency, but forces hierarchy
- Independent specialists = flexibility, but more expensive

**Math says:**
- PC1 explains 25% variance (moderate shared component)
- Groups 3 & 5 are 0.5% similar (nearly independent)

**Design question:**
- **If compute is tight:** Use shared gate (filters 99% before specialists)
- **If need flexibility:** Use independent specialists (easy to add/remove)
- **If both:** Use hybrid (shared gate + independent specialists)

**No math can tell you which architecture to deploy!** It depends on your constraints.

### Tradeoff 4: Same Layer vs Layer-Wise Hybrid

**The tension:**
- Same layer = simple, consistent, easy to validate
- Layer-wise = leverages model capacity, but complex

**Math says:**
- Middle layers (60-65%) have best generalization
- Late layers (75-85%) have best specificity (hypothesis)

**Design question:**
- **For simplicity:** Extract everything from middle layer
- **For max performance:** Extract shared from middle, specialists from late
- **For robustness:** Extract from multiple layers, ensemble across layers

**No math can tell you whether complexity is worth it!** It depends on your engineering capacity.

### Tradeoff 5: Number of Specialists

**The tension:**
- Fewer specialists (3-5) = coarse-grained, high compression, interpretable
- More specialists (7-10) = fine-grained, lower compression, complex

**Math says:**
- Spectral clustering: k=7 optimizes separation metric
- But separation keeps improving up to k=18 (no compression)

**Design question:**
- **For interpretability:** Use 5 specialists (clear taxonomy)
- **For flexibility:** Use 7 specialists (finer granularity)
- **For coverage:** Use 10+ specialists (very specific threats)

**No math can tell you the right granularity!** It depends on your threat model.

### Tradeoff 6: Orthogonality vs Coverage

**The tension:**
- High orthogonality = low cross-contamination, but may miss overlapping threats
- Low orthogonality = better coverage, but more false positives

**Math says:**
- Groups 3 & 5: 0.5% overlap (highly orthogonal)
- Groups 0 & 6: 60% overlap (not orthogonal)

**Design question:**
- **For precision:** Keep highly orthogonal specialists (Groups 3, 4, 5)
- **For coverage:** Allow some overlap (merge Groups 0 & 6)
- **For robustness:** Use overlapping specialists, apply FDR control

**No math can tell you how orthogonal is orthogonal enough!** It depends on your false positive tolerance.

### Tradeoff 7: Unsupervised vs Supervised Grouping

**The tension:**
- Unsupervised (spectral) = optimizes math metric, may not match semantics
- Supervised (manual) = matches human understanding, may be suboptimal

**Math says:**
- Spectral clustering: 4.2× better separation than manual
- But groups don't perfectly align with semantic categories

**Design question:**
- **For pure performance:** Trust spectral clustering
- **For interpretability:** Use manual grouping (DLK, REPE, GOT, etc.)
- **For hybrid:** Start with spectral, adjust based on semantics

**No math can tell you whether to trust the algorithm or the human!** It depends on your interpretability needs.

---

## The Integrated Design Framework

### Step 1: Define Your Goal

**Research Goal:**
- Understand universal truth geometry
- Find what generalizes across tasks
- Extract from middle layers (60-65%)
- Use PCA for interpretability
- Optimize for variance explained

**Safety Goal:**
- Build high-precision threat detectors
- Catch tail cases at 10^-5 FPR
- Extract specialists from late layers (75-85%)
- Use spectral clustering for separation
- Optimize for in-distribution recall

**Generation Goal:**
- Auto-generate high-quality probe datasets
- Use PC1 for quality metric
- Extract from middle layers (generalization)
- Iteratively improve weak spots

### Step 2: Choose Your Paradigm

**If computational budget is tight:**
- → Hierarchical (shared + specific)
- → Shared gate filters 99%
- → 5 specialists for precision

**If need flexibility:**
- → Orthogonal bases (independent specialists)
- → Easy to add/remove
- → 7-10 specialists for coverage

**If need both:**
- → Hybrid architecture
- → Shared gate (middle layer, 10^-3 FPR)
- → Independent specialists (late layer, 10^-5 FPR)

### Step 3: Select Your Compression Level

**High compression (5 specialists):**
- Merge high-confounding groups (>40% overlap)
- Clear taxonomy, easy to interpret
- Sacrifice some granularity for simplicity

**Medium compression (7 specialists):**
- Optimal separation metric
- Good balance of granularity and interpretability
- Our current recommendation

**Low compression (10+ specialists):**
- Very specific threat types
- Maximum flexibility
- More complex to manage

### Step 4: Decide on Layer Strategy

**Simple approach:**
- Extract everything from middle layer (60-65%)
- Consistent, easy to validate
- Good enough for most cases

**Advanced approach:**
- Shared gate from middle layer (generalization)
- Specialists from late layer (specificity)
- Requires validation, but potentially better performance

**Ensemble approach:**
- Extract from multiple layers
- Ensemble across layers
- Maximum robustness, highest complexity

### Step 5: Validate Your Choices

**Critical validation experiments:**

1. **Test generalization proxy:**
   ```python
   # Does similarity predict actual transfer?
   for train_ds, test_ds in pairs:
       similarity = cosine_similarity(probe[train_ds], probe[test_ds])
       actual_transfer = test_on_activations(probe[train_ds], acts[test_ds])
       correlation = pearsonr(similarity, actual_transfer)

   # If correlation < 0.5, similarity is poor proxy!
   ```

2. **Test layer hypothesis:**
   ```python
   # Do late layers have better in-dist?
   for dataset in datasets:
       for layer in [0.625, 0.75, 0.875]:
           in_dist = test_on_same(dataset, layer)
           cross = test_on_others(dataset, layer)
           print(f"{dataset} @ {layer}: in={in_dist}, cross={cross}")

   # If in_dist doesn't increase, layer hypothesis fails!
   ```

3. **Test ensemble improvement:**
   ```python
   # Does averaging within groups improve quality?
   for group in groups:
       individual_perfs = [test_probe(p) for p in group]
       ensemble_perf = test_probe(average(group))
       improvement = (ensemble_perf - mean(individual_perfs)) / mean(individual_perfs)
       print(f"Group {group}: {improvement:+.1%} improvement")

   # If improvement < 0, ensembling hurts!
   ```

---

## Recommendations for Your Use Case

Based on your questions, I think you want:

### For Safety Probes (Production)

**Architecture:** Hybrid (shared + independent specialists)

**Configuration:**
```python
# Shared gate
w_shared = PC1(extract_probes(layer=0.625))  # Middle layer for generalization
threshold_gate = calibrate_to_fpr(w_shared, benign_corpus, fpr=1e-3)

# Independent specialists (test late layer hypothesis first!)
specialists = {
    'factual_logic': avg(Group0 + Group6),      # 7 probes
    'alignment_class': avg(Group1 + Group2),    # 5 probes
    'probabilistic': Group3,                     # 1 probe
    'complex_reasoning': avg(Group4),           # 4 probes
    'complex_truth': Group5                     # 1 probe
}

# Extract from late layer (IF validation shows better in-dist)
for name in specialists:
    specialists[name] = extract_specialist(layer=0.875, probes=specialists[name])
    thresholds[name] = calibrate_to_fpr(specialists[name], benign_corpus, fpr=1e-5)
```

**Validation needed:**
1. Test layer hypothesis (do late layers improve in-dist?)
2. Test on adversarial variants (not just cross-dataset)
3. Measure tail coverage @ 10^-5 FPR (not average accuracy)

### For Research (Understanding Truth Geometry)

**Architecture:** PCA decomposition

**Configuration:**
```python
# Extract from middle layer (best generalization)
probes = extract_all_probes(layer=0.625)

# PCA decomposition
pca = PCA(n_components=10)
components = pca.fit_transform(probes)

# Interpret each component
for i in range(5):
    print(f"PC{i+1}: {pca.explained_variance_ratio_[i]:.1%}")
    print(f"  Top loading datasets: {get_top_loadings(components[:, i])}")
    print(f"  Interpretation: {interpret_component(i)}")
```

**Focus on:**
- What does PC1 represent? (universal truth)
- Why is PC3 dominated by "likely"? (probabilistic is orthogonal)
- How do components relate to task types?

### For Auto-Probe Generation

**Architecture:** Shared-plus-residual with iterative refinement

**Configuration:**
```python
# Use PC1 for quality metric
w_shared = PC1(extract_probes(layer=0.625))

# Quality function
def quality(new_probe):
    similarity_to_pc1 = abs(w_shared @ new_probe)
    return quality_score(similarity_to_pc1)  # 0.4-0.6 is optimal

# Generate examples targeting weak spots
for iteration in range(max_iterations):
    weak_spots = find_weak_cells(cross_dataset_matrix)
    new_examples = generate_for_weak_spots(weak_spots)
    filtered = [ex for ex in new_examples if quality(ex) > threshold]
    probes.extend(filtered)
```

**Focus on:**
- Iterative improvement of cross-dataset matrix
- Targeting specific weak cells (train on A, test on B is poor)
- Maintaining quality (PC1 similarity in optimal range)

---

## The Bottom Line

**You're absolutely right to push back on my earlier analysis.**

The math gives us:
- Separation metrics
- Variance explained
- Reconstruction error
- Similarity as a proxy for transfer

But math CANNOT tell us:
- Whether to optimize for generalization or precision (depends on goal)
- How much compression is enough (depends on constraints)
- Whether to use shared or independent architecture (depends on tradeoffs)
- What layer depth to use (depends on generalization vs specificity needs)
- What granularity of specialists (depends on threat taxonomy)

**These are DESIGN choices** that require:
1. Understanding your application (safety vs research vs generation)
2. Defining your constraints (compute, interpretability, flexibility)
3. Making explicit tradeoffs (precision vs recall, simplicity vs performance)
4. Validating your assumptions (does late layer improve in-dist? does similarity predict transfer?)

The mathematical analysis is a TOOL for making informed design choices, not the answer itself.
