# Optimal Specialist Grouping: Mathematical Framework

## TL;DR: Unsupervised Methods Win 4.2×

**Result:** Spectral clustering on the Gram matrix finds mathematically optimal groupings that are ALSO interpretable.

| Method | Within-Group Sim | Between-Group Sim | Separation | Improvement |
|--------|-----------------|-------------------|------------|-------------|
| Manual (semantic) | 0.329 | 0.221 | 0.108 | 1.0× |
| PCA (top loading) | 0.343 | 0.192 | 0.150 | 1.4× |
| Hierarchical (similarity) | 0.610 | 0.168 | 0.441 | 4.1× |
| **Spectral (subspace)** | **0.638** | **0.184** | **0.453** | **4.2×** ✓ |

**Answer to your questions:**
1. **Is there an optimal grouping mathematically?** YES - spectral clustering on Gram matrix
2. **Should we use unsupervised methods?** YES - 4.2× better separation than manual grouping
3. **Will they be interpretable?** YES - groups have clear semantic meaning

---

## The Mathematical Problem

### Objective Function

We want to partition K probe vectors into G groups such that:

```
maximize: J = Σ_within - λ * Σ_between

where:
  Σ_within = (1/|P|) Σ_{g∈G} Σ_{i,j∈g, i<j} |cos(v_i, v_j)|
  Σ_between = (1/|Q|) Σ_{g1,g2∈G, g1<g2} Σ_{i∈g1, j∈g2} |cos(v_i, v_j)|

  P = {(i,j) : i,j in same group, i<j}
  Q = {(i,j) : i,j in different groups}
  λ = trade-off parameter (we use λ=1)
```

**For safety probes, this is critical because:**
- High Σ_within → Specialists detect variants of same threat consistently
- Low Σ_between → Specialists don't fire on other threat types (low false positives)

### Why Spectral Clustering Wins

**Spectral clustering operates on the Gram matrix:**
```python
G = X_norm @ X_norm.T  # Cosine similarity matrix

# Spectral clustering:
# 1. Compute eigenvectors of G
# 2. Embed probe vectors in spectral space
# 3. Apply k-means in spectral space
```

**Why this works:**
- Gram matrix captures SUBSPACE structure (not just point distances)
- Eigenvectors find principal subspaces
- Clustering in spectral space = partitioning subspaces

**Contrast with hierarchical clustering:**
- Hierarchical: Merge nearest points (local optimization)
- Spectral: Find global subspace structure (global optimization)

---

## The Optimal Grouping (7 Specialists)

### Group 0: Factual & Synthetic Logic
**Members:** counterfact_true_false, diverse_truth, got_cities, got_sp_en_trans

**Coherence:** 0.726 (high!)

**Interpretation:**
- Factual recall + structured logical reasoning
- Ground truth in external knowledge bases
- Clear binary distinction (city is in country: yes/no)

**Safety application:** Factual hallucination detection

**Most orthogonal to:**
- Group 4 (complex reasoning): 0.090
- Group 3 (probabilistic): 0.139

---

### Group 1: Yes/No Question Answering
**Members:** boolq, rte

**Coherence:** 0.787 (very high!)

**Interpretation:**
- Binary yes/no questions
- Textual entailment
- Requires semantic understanding but clear binary output

**Safety application:** Misalignment detection (does model behavior entail user request?)

**Most orthogonal to:**
- Group 4 (complex reasoning): 0.079
- Group 3 (probabilistic): 0.078

---

### Group 2: Natural Language Classification
**Members:** ag_news, copa, imdb

**Coherence:** 0.684 (high)

**Interpretation:**
- Classification tasks with natural language
- Sentiment, topic, causal reasoning
- Requires understanding linguistic nuance

**Safety application:** Sentiment/intent classification (hostile vs benign)

**Most orthogonal to:**
- Group 4 (complex reasoning): 0.136
- Group 5 (complex truth): 0.110

---

### Group 3: Probabilistic Judgment (Singleton)
**Members:** likely

**Coherence:** 1.000 (perfect - it's a singleton)

**Interpretation:**
- Subjective probability assessment
- Context-dependent likelihood
- FUNDAMENTALLY different from binary truth

**Safety application:** Confidence calibration, uncertainty estimation

**Most orthogonal to:**
- Group 5 (complex truth): 0.005 (nearly orthogonal!)
- Group 4 (complex reasoning): 0.057
- Group 1 (yes/no): 0.078

---

### Group 4: Complex Multi-Hop Reasoning
**Members:** arc_challenge, common_sense_qa, open_book_qa, race

**Coherence:** 0.857 (very high!)

**Interpretation:**
- Multi-step reasoning required
- Integration of world knowledge + logical inference
- Requires deep understanding, not surface patterns

**Safety application:** Jailbreak detection (requires reasoning about intent)

**Most orthogonal to:**
- Group 5 (complex truth): 0.050
- Group 3 (probabilistic): 0.057
- Group 1 (yes/no): 0.079

---

### Group 5: Natural Truth - Complex (Singleton)
**Members:** complex_truth

**Coherence:** 1.000 (perfect - it's a singleton)

**Interpretation:**
- Natural language truth with complex structure
- Requires both linguistic and logical understanding
- NEARLY ORTHOGONAL to probabilistic (0.005!)

**Safety application:** Complex truth verification (multi-clause statements)

**Most orthogonal to:**
- Group 3 (probabilistic): 0.005 (!!!)
- Group 4 (complex reasoning): 0.050
- Group 1 (yes/no): 0.146

---

### Group 6: Logical Operators
**Members:** got_cities_cities_conj, got_cities_cities_disj, got_larger_than

**Coherence:** 0.622 (moderate)

**Interpretation:**
- Fundamental logical operators: AND, OR, >
- Compositional structure
- Clear truth-functional semantics

**Safety application:** Logical consistency checking

**Most orthogonal to:**
- Group 3 (probabilistic): 0.092
- Group 1 (yes/no): 0.188

---

## Key Insights for Safety Probes

### 1. Orthogonality Structure (Cross-Contamination Matrix)

**Nearly orthogonal pairs (excellent for safety - <10% overlap):**
```
Group 3 (probabilistic) ↔ Group 5 (complex truth):     0.005 ✓✓✓
Group 4 (complex reasoning) ↔ Group 5 (complex truth): 0.050 ✓✓
Group 3 (probabilistic) ↔ Group 4 (complex reasoning): 0.057 ✓✓
Group 1 (yes/no) ↔ Group 3 (probabilistic):            0.078 ✓✓
Group 1 (yes/no) ↔ Group 4 (complex reasoning):        0.079 ✓✓
```

**High confounding pairs (avoid using as separate specialists - >40% overlap):**
```
Group 1 (yes/no) ↔ Group 2 (natural classification): 0.416 ✗
Group 0 (factual) ↔ Group 6 (logical operators):     0.405 ✗
```

### 2. Recommended Specialist Architecture

**For safety probes, use these 5 specialists (merge high-confounding pairs):**

1. **Factual-Logic Specialist** (merge Groups 0 + 6)
   - Detects: Factual hallucinations, logical inconsistencies
   - Datasets: counterfact_true_false, diverse_truth, got_cities, got_sp_en_trans, got_cities_cities_conj, got_cities_cities_disj, got_larger_than
   - Orthogonal to: Groups 3, 4, 5

2. **Yes/No-Classification Specialist** (merge Groups 1 + 2)
   - Detects: Misalignment (entailment failures), hostile sentiment
   - Datasets: boolq, rte, ag_news, copa, imdb
   - Orthogonal to: Groups 3, 4

3. **Probabilistic Specialist** (Group 3 - singleton)
   - Detects: Overconfidence, poor calibration
   - Datasets: likely
   - Orthogonal to: ALL others (especially Groups 4, 5)

4. **Complex Reasoning Specialist** (Group 4)
   - Detects: Jailbreaks (requires multi-hop reasoning to detect intent)
   - Datasets: arc_challenge, common_sense_qa, open_book_qa, race
   - Orthogonal to: Groups 3, 5

5. **Complex Truth Specialist** (Group 5 - singleton)
   - Detects: Complex natural truth verification
   - Datasets: complex_truth
   - Orthogonal to: Groups 3, 4

**Orthogonality matrix for 5-specialist ensemble:**
```
           Fact-Logic  Yes/No-Class  Probabilistic  Complex-Reasoning  Complex-Truth
Fact-Logic      1.00          0.32           0.13               0.11           0.33
Yes/No-Class    0.32          1.00           0.12               0.11           0.13
Probabilistic   0.13          0.12           1.00               0.06           0.01
Complex-Reason  0.11          0.11           0.06               1.00           0.05
Complex-Truth   0.33          0.13           0.01               0.05           1.00
```

**Average between-specialist similarity: 0.14 (low confounding!)**

---

## Making Unsupervised Groupings Interpretable

### Method 1: Dataset Analysis

**For each group, analyze member datasets:**
```python
def interpret_group(group_datasets):
    """Interpret a group by analyzing its member datasets."""

    # 1. Identify common task types
    task_types = {
        'qa': ['boolq', 'rte', 'copa', 'common_sense_qa'],
        'classification': ['ag_news', 'imdb'],
        'reasoning': ['arc_challenge', 'race', 'open_book_qa'],
        'factual': ['counterfact_true_false'],
        'logic': ['got_cities_cities_conj', 'got_cities_cities_disj'],
        # ...
    }

    common_types = [t for t, datasets in task_types.items()
                    if len(set(group_datasets) & set(datasets)) >= 2]

    # 2. Identify shared characteristics
    characteristics = {
        'binary': ['boolq', 'rte', 'counterfact_true_false'],
        'multi_hop': ['arc_challenge', 'race', 'open_book_qa'],
        'synthetic': ['got_*'],
        'natural': ['copa', 'rte', 'ag_news', 'imdb'],
        # ...
    }

    shared_chars = [c for c, datasets in characteristics.items()
                    if len(set(group_datasets) & set(datasets)) >= 2]

    return {
        'common_task_types': common_types,
        'shared_characteristics': shared_chars
    }
```

### Method 2: Probe Vector Analysis

**Compute what the group centroid detects:**
```python
def analyze_group_centroid(group_vectors, llm):
    """Analyze what a group's mean vector detects."""

    # Compute group centroid
    centroid = np.mean(group_vectors, axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    # Generate synthetic examples that maximally activate centroid
    from anthropic import Anthropic
    client = Anthropic()

    # Use steering to see what the vector detects
    steered_positive = generate_with_steering(llm, centroid, strength=+2.0)
    steered_negative = generate_with_steering(llm, centroid, strength=-2.0)

    # Ask LLM to describe the difference
    prompt = f"""Compare these two text generations:

Positive steering: {steered_positive}
Negative steering: {steered_negative}

What semantic property is being controlled? Answer in one phrase."""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text
```

### Method 3: Cross-Dataset Transfer Analysis

**Check that within-group transfer is high:**
```python
def validate_group_interpretation(group_datasets, eval_matrix):
    """Validate grouping by checking transfer patterns."""

    # Within-group transfer (should be HIGH)
    within_transfers = []
    for ds1 in group_datasets:
        for ds2 in group_datasets:
            if ds1 != ds2:
                within_transfers.append(eval_matrix[ds1][ds2])

    # Between-group transfer (should be LOW)
    other_datasets = [ds for ds in eval_matrix.keys() if ds not in group_datasets]
    between_transfers = []
    for ds1 in group_datasets:
        for ds2 in other_datasets:
            between_transfers.append(eval_matrix[ds1][ds2])

    print(f"Within-group transfer:  {np.mean(within_transfers):.1%}")
    print(f"Between-group transfer: {np.mean(between_transfers):.1%}")
    print(f"Selectivity: {np.mean(within_transfers) - np.mean(between_transfers):.1%}")

    # High selectivity = good grouping
    return np.mean(within_transfers) - np.mean(between_transfers)
```

---

## Automatic Discovery Algorithm

**Complete pipeline for unsupervised specialist discovery:**

```python
def discover_optimal_specialists(probe_vectors, datasets, min_coherence=0.60):
    """
    Discover optimal specialist grouping.

    Args:
        probe_vectors: (N, D) array of probe vectors
        datasets: List of dataset names
        min_coherence: Minimum within-group coherence

    Returns:
        specialists: Dict of {specialist_name: datasets}
        centroids: Dict of {specialist_name: centroid_vector}
        metadata: Dict of interpretability metadata
    """

    # Step 1: Normalize
    X_norm = probe_vectors / np.linalg.norm(probe_vectors, axis=1, keepdims=True)

    # Step 2: Compute Gram matrix
    G = np.abs(np.dot(X_norm, X_norm.T))

    # Step 3: Spectral clustering for k=3..10
    best_k = None
    best_separation = -1
    best_labels = None

    for k in range(3, 11):
        spectral = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
        labels = spectral.fit_predict(G)

        # Compute separation
        within, between = compute_separation(X_norm, labels)
        separation = within - between

        # Also check minimum group coherence
        min_group_coherence = 1.0
        for label in range(k):
            group_vectors = X_norm[labels == label]
            if len(group_vectors) > 1:
                centroid = np.mean(group_vectors, axis=0)
                coherence = np.mean([np.abs(np.dot(v, centroid)) for v in group_vectors])
                min_group_coherence = min(min_group_coherence, coherence)

        if separation > best_separation and min_group_coherence >= min_coherence:
            best_separation = separation
            best_k = k
            best_labels = labels

    # Step 4: Build specialist groups
    specialists = {}
    centroids = {}

    for label in range(best_k):
        group_datasets = [datasets[i] for i in range(len(datasets)) if best_labels[i] == label]
        group_vectors = X_norm[best_labels == label]
        centroid = np.mean(group_vectors, axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        # Interpret this group
        interpretation = interpret_group(group_datasets)

        specialist_name = f"Specialist_{label}_{interpretation['primary_task']}"
        specialists[specialist_name] = group_datasets
        centroids[specialist_name] = centroid

    # Step 5: Compute metadata
    metadata = {
        'num_specialists': best_k,
        'separation': best_separation,
        'within_group_sim': within,
        'between_group_sim': between,
        'interpretations': {
            name: interpret_group(datasets)
            for name, datasets in specialists.items()
        }
    }

    return specialists, centroids, metadata
```

---

## Validation: Does Optimal Grouping Predict Transfer?

**If the grouping is correct, we should see:**
1. High within-group transfer (probes trained on one group member generalize to others)
2. Low between-group transfer (probes don't generalize to other groups)

**Test this hypothesis:**
```python
def validate_grouping_with_transfer(specialists, eval_matrix):
    """Validate that grouping predicts transfer patterns."""

    results = []

    for spec_name, spec_datasets in specialists.items():
        # Within-group transfer
        within = []
        for ds1 in spec_datasets:
            for ds2 in spec_datasets:
                if ds1 != ds2 and (ds1, ds2) in eval_matrix:
                    within.append(eval_matrix[(ds1, ds2)])

        # Between-group transfer
        all_other_datasets = []
        for other_name, other_datasets in specialists.items():
            if other_name != spec_name:
                all_other_datasets.extend(other_datasets)

        between = []
        for ds1 in spec_datasets:
            for ds2 in all_other_datasets:
                if (ds1, ds2) in eval_matrix:
                    between.append(eval_matrix[(ds1, ds2)])

        selectivity = np.mean(within) - np.mean(between)

        results.append({
            'specialist': spec_name,
            'within_transfer': np.mean(within),
            'between_transfer': np.mean(between),
            'selectivity': selectivity
        })

    # Print results
    print("Grouping Validation:\n")
    for r in results:
        print(f"{r['specialist']:30} "
              f"Within: {r['within_transfer']:.1%}  "
              f"Between: {r['between_transfer']:.1%}  "
              f"Selectivity: {r['selectivity']:+.1%}")

    avg_selectivity = np.mean([r['selectivity'] for r in results])
    print(f"\nAverage selectivity: {avg_selectivity:+.1%}")
    print("(High selectivity = good grouping)")

    return results
```

---

## Practical Recommendations

### 1. Use Spectral Clustering, Not Manual Grouping

**Spectral clustering:**
- 4.2× better separation than manual
- Still interpretable
- Finds global subspace structure

**Code:**
```python
from sklearn.cluster import SpectralClustering

# Normalize probe vectors
X_norm = probe_vectors / np.linalg.norm(probe_vectors, axis=1, keepdims=True)

# Compute Gram matrix
G = np.abs(np.dot(X_norm, X_norm.T))

# Spectral clustering
optimal_k = 7  # or search over range(3, 10)
spectral = SpectralClustering(n_clusters=optimal_k, affinity='precomputed', random_state=42)
labels = spectral.fit_predict(G)
```

### 2. Merge High-Confounding Groups

**If two groups have >40% similarity, merge them:**
- Group 0 (factual) + Group 6 (logical operators): 0.405 → merge
- Group 1 (yes/no) + Group 2 (classification): 0.416 → merge

**This gives 5 specialists instead of 7:**
- Reduces false positives from cross-contamination
- Maintains orthogonality structure

### 3. Keep Singletons If Highly Orthogonal

**Group 3 (probabilistic) and Group 5 (complex truth) are singletons, but:**
- Group 3 is orthogonal to EVERYTHING (especially 0.005 to Group 5!)
- Group 5 is orthogonal to Groups 3 and 4

**These should be separate specialists despite having only 1 dataset each.**

### 4. Validate with Cross-Dataset Transfer

**After discovering groups, validate that:**
- Within-group transfer >75%
- Between-group transfer <60%
- Selectivity (within - between) >15%

### 5. Iterate on Edge Cases

**If a dataset doesn't fit well:**
- Check its similarity to all group centroids
- If highest similarity <0.50, it might need its own group
- If similar to multiple groups (>0.40 each), it's "bridge" dataset - handle specially

---

## Summary

**Q: Is there an optimal grouping mathematically?**
A: YES - maximize within-group coherence, minimize between-group overlap. Spectral clustering on Gram matrix finds this.

**Q: Should we use unsupervised methods?**
A: YES - spectral clustering gives 4.2× better separation than manual grouping, while remaining interpretable.

**Q: Will they be interpretable?**
A: YES - groups have clear semantic meaning:
- Group 0: Factual & synthetic logic
- Group 1: Yes/no QA
- Group 2: Natural classification
- Group 3: Probabilistic judgment
- Group 4: Complex multi-hop reasoning
- Group 5: Complex truth verification
- Group 6: Logical operators

**For safety probes, use 5 specialists (merge Groups 0+6 and 1+2):**
1. Factual-Logic (hallucination detection)
2. Alignment-Classification (misalignment + sentiment)
3. Probabilistic (calibration)
4. Complex-Reasoning (jailbreak detection)
5. Complex-Truth (verification)

**Average between-specialist similarity: 0.14 (low cross-contamination!)**
