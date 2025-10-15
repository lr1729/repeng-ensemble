# Auto-Probe Pipeline: Concrete Design Decisions

## Project Goal

**Input:** General concept description (e.g., "misalignment", "hallucination", "jailbreak attempt")

**Output:**
1. High-quality dataset of contrastive pairs
2. Multi-dimensional decomposition (not single vector!)
3. Robust probe ensemble
4. Interpretable structure (what aspects the model represents)

**Process:** LLM generates → Evaluate quality → Refine iteratively

---

## What We Know (From Our Data)

### Finding 1: Concepts Are NOT Single Vectors ✓

**Evidence:**
- PC1 explains only 25% variance (not 80%+)
- 5 meaningful components (reasoning, language, probabilistic, task-specific)
- Spectral clustering finds 7 distinct groups (not 1-2)

**Implication:** Pipeline must produce **multi-probe ensembles**, not single vectors

### Finding 2: Spectral Clustering > Manual Grouping ✓

**Evidence:**
- 4.2× better separation (0.453 vs 0.108)
- Groups are interpretable despite being unsupervised
- Condition number: 9× better (62 → 7)

**Implication:** Use **automated decomposition** (spectral clustering), not manual taxonomy

### Finding 3: Within-Group Averaging Improves Quality ✓

**Evidence:**
- +17.2% average improvement (proxy measurement)
- Reduces noise, improves consistency
- Simple method (just average, no fancy ensembling)

**Implication:** Use **simple averaging within groups**, not boosting/trees/LoRA

### Finding 4: Layer 60-65% Is Optimal for Generalization ✓

**Evidence:**
- Peak performance at 60-65% depth across models
- Earlier layers: representations not formed yet
- Later layers: over-specialized, don't transfer

**Implication:** Extract probes at **middle layer (62.5%)** by default

### Finding 5: Similarity Is Proxy for Transfer (Unvalidated) ⚠️

**Evidence:**
- PC1 similarity correlates with transfer (r=0.37 in previous work)
- BUT we don't have actual cross-dataset evaluation data
- Missing: tail behavior, adversarial robustness

**Implication:** Use similarity for **fast filtering**, but validate on subset with **real activations**

---

## What We DON'T Know (Missing Data)

### Critical Gap: No Cross-Dataset Evaluation Data

**We have:**
- Probe vectors (18 datasets × 10 models × 6 layers)
- Similarity matrix (proxy for transfer)

**We DON'T have:**
- Actual classification accuracy (train on A, test on B)
- Tail coverage (10th percentile performance)
- Adversarial robustness (performance under perturbations)

**Why this matters:**
- Can't validate similarity proxy
- Can't measure real ensemble improvement
- Can't optimize for tail coverage (critical for safety!)

**What to do:**
- Use similarity for initial design
- Validate on small subset (5-10 dataset pairs) with real activations
- If correlation < 0.5, similarity proxy is unreliable

---

## Design Decisions

### Decision 1: Quality Metric → PC1 Similarity + Subspace Fit

**Metric:**
```python
def quality_score(new_probe, existing_components):
    """
    Quality = how well new probe fits into existing structure.

    Returns: 0-100 score
    """

    # Step 1: PC1 similarity (measures general truth alignment)
    pc1_sim = abs(new_probe @ w_shared)

    # Optimal range: 0.4-0.6 (from our data)
    # Too high (>0.7) = overfit to PC1, no new info
    # Too low (<0.3) = unrelated to truth concept
    pc1_quality = 100 * exp(-((pc1_sim - 0.5) / 0.15) ** 2)  # Bell curve

    # Step 2: Subspace fit (which specialist group does it belong to?)
    subspace_sims = [abs(new_probe @ centroid) for centroid in group_centroids]
    best_group = max(subspace_sims)

    # Should have ONE strong alignment (>0.6), not multiple weak ones
    if best_group > 0.6:
        subspace_quality = 100
    elif best_group > 0.4:
        subspace_quality = 50
    else:
        subspace_quality = 0

    # Step 3: Novelty (should not be too similar to existing)
    max_existing_sim = max([abs(new_probe @ existing) for existing in existing_probes])

    if max_existing_sim > 0.9:
        novelty = 0  # Redundant
    elif max_existing_sim > 0.7:
        novelty = 50  # Somewhat novel
    else:
        novelty = 100  # Novel

    # Combined score (weighted average)
    quality = 0.4 * pc1_quality + 0.4 * subspace_quality + 0.2 * novelty

    return quality, {
        'pc1_similarity': pc1_sim,
        'pc1_quality': pc1_quality,
        'best_group': np.argmax(subspace_sims),
        'subspace_quality': subspace_quality,
        'novelty': novelty
    }
```

**Why this metric:**
- ✓ PC1 similarity: Proven to correlate with transfer (r=0.37)
- ✓ Subspace fit: Ensures probe belongs to interpretable group
- ✓ Novelty: Prevents redundant examples
- ✓ No activation needed: Fast filtering (can test 1000s of candidates)

**When to use:**
- Stage 1: Fast filter (reject if quality < 60)
- Stage 2: Validate top candidates on real activations (small subset)

### Decision 2: Decomposition → Spectral Clustering (Auto)

**Method:**
```python
def decompose_concept(probe_vectors, n_specialists='auto'):
    """
    Automatic decomposition into interpretable specialists.

    Returns: Specialist groups + centroids + interpretations
    """

    # Step 1: Normalize probe vectors
    X_norm = probe_vectors / np.linalg.norm(probe_vectors, axis=1, keepdims=True)

    # Step 2: Spectral clustering (auto-determine K)
    gram = np.abs(X_norm @ X_norm.T)

    if n_specialists == 'auto':
        # Try K=3 to K=10, pick best separation
        best_k = None
        best_separation = -1

        for k in range(3, 11):
            spectral = SpectralClustering(n_clusters=k, affinity='precomputed')
            labels = spectral.fit_predict(gram)

            # Measure separation (within - between)
            within = compute_within_group_sim(X_norm, labels)
            between = compute_between_group_sim(X_norm, labels)
            separation = within - between

            if separation > best_separation:
                best_separation = separation
                best_k = k
    else:
        best_k = n_specialists

    # Final clustering with best K
    spectral = SpectralClustering(n_clusters=best_k, affinity='precomputed')
    labels = spectral.fit_predict(gram)

    # Step 3: Compute centroids for each group
    specialists = {}
    for k in range(best_k):
        group_probes = probe_vectors[labels == k]
        centroid = np.mean(group_probes, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        specialists[f'specialist_{k}'] = {
            'centroid': centroid,
            'members': np.where(labels == k)[0].tolist(),
            'coherence': compute_coherence(group_probes)
        }

    # Step 4: Auto-interpret each specialist (with LLM)
    for name, spec in specialists.items():
        member_names = [probe_names[i] for i in spec['members']]
        interpretation = llm_interpret_group(member_names)
        spec['interpretation'] = interpretation

    return specialists, labels
```

**Why spectral clustering:**
- ✓ 4.2× better separation than manual (proven in our data)
- ✓ Finds global subspace structure (not just pairwise distances)
- ✓ Auto-determines K (try range, pick best)
- ✓ Still interpretable (groups match semantics)

**Alternative considered: Manual grouping**
- ✗ Requires domain knowledge
- ✗ Subjective (researchers disagree)
- ✗ Doesn't adapt to new concepts

**Alternative considered: PCA + threshold**
- ✗ Forces orthogonality (not in data: some groups 60% similar)
- ✗ Doesn't find discrete groups
- ✓ Good for visualization, bad for discrete specialists

### Decision 3: Ensemble → Simple Averaging (No Boosting)

**Method:**
```python
def ensemble_specialists(specialists):
    """
    Combine probes within each specialist group.

    Returns: 5-7 ensemble probes (one per group)
    """

    ensemble = {}
    for name, spec in specialists.items():
        # Simple average (equal weights)
        member_probes = [probe_vectors[i] for i in spec['members']]
        centroid = np.mean(member_probes, axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        ensemble[name] = {
            'probe': centroid,
            'num_members': len(member_probes),
            'interpretation': spec['interpretation']
        }

    return ensemble
```

**Why simple averaging:**
- ✓ +17.2% improvement in our data (proxy measurement)
- ✓ Dead simple (no hyperparameters)
- ✓ Interpretable (centroid = average concept)
- ✓ Proven by theory (reduces noise by √n)

**Alternative considered: Boosting**
- ✗ Complex (many hyperparameters)
- ✗ Prone to overfitting
- ✗ Less interpretable (weighted combination)
- ✗ No evidence it helps for linear probes

**Alternative considered: Decision trees**
- ✗ Non-linear (probes are linear classifiers)
- ✗ Requires much more data
- ✗ Loses interpretability

**Alternative considered: LoRA**
- ✗ Wrong tool! LoRA is for fine-tuning model weights, not combining probes
- ✗ Probes are fixed vectors, not learned parameters

### Decision 4: Layer Choice → Middle Layer (62.5%)

**Configuration:**
```python
def extract_layer():
    """
    Which layer to extract probes from.
    """
    return 0.625  # 62.5% depth (middle layer)
```

**Why middle layer:**
- ✓ Best generalization in our data (60-65% peak)
- ✓ Consistent across all models
- ✓ Simple (one layer, easy to validate)

**Alternative considered: Late layer (75-85%)**
- ✗ Worse generalization (proven in our data)
- ? Better in-dist accuracy (UNVALIDATED hypothesis)
- ✗ More complexity (harder to validate)

**Alternative considered: Layer-wise hybrid**
- ✗ Much more complex
- ✗ Requires validating that late layers help
- ✗ Mixing activation spaces (technically challenging)

**Decision: Start simple, optimize later**
- Use middle layer (62.5%) initially
- If tail coverage is insufficient, test late layer hypothesis
- Don't add complexity without evidence

### Decision 5: Generation Strategy → Hierarchical (Blue + Red)

**Pipeline:**
```python
def generate_contrastive_pairs(concept_description, existing_examples, mode='blue'):
    """
    Generate new contrastive pairs using LLM.

    Args:
        concept_description: "misalignment", "hallucination", etc.
        existing_examples: List of (true, false) pairs
        mode: 'blue' (variants) or 'red' (adversarial)

    Returns: List of new (true, false) pairs
    """

    if mode == 'blue':
        # Blue-team: Generate linguistic variants
        prompt = f"""
You are generating training data for a probe that detects "{concept_description}".

Here are {len(existing_examples)} existing examples:
{format_examples(existing_examples[:5])}

Generate 20 NEW contrastive pairs that illustrate the SAME concept but with:
- Different phrasings (synonyms, paraphrases)
- Different domains (if concept generalizes)
- Different grammatical structures (active/passive, declarative/interrogative)
- Different surface forms (short/long, simple/complex)

Requirements:
- Preserve the core concept (what makes it "{concept_description}")
- Maximize linguistic diversity (don't use templates)
- Each pair should have TRUE and FALSE example

Output format (JSON):
[
  {{"true": "...", "false": "..."}},
  ...
]
"""

    elif mode == 'red':
        # Red-team: Generate adversarial hard negatives
        prompt = f"""
You are red-teaming a probe that detects "{concept_description}".

The probe SHOULD fire on these examples:
{format_examples([ex['true'] for ex in existing_examples[:5]])}

The probe should NOT fire on these examples:
{format_examples([ex['false'] for ex in existing_examples[:5]])}

Generate 20 ADVERSARIAL examples that the probe might INCORRECTLY fire on:
- Edge cases (borderline examples)
- Superficially similar but actually benign
- Contains keywords but different intent
- Obvious/edgy examples that are NOT what we're looking for

These are NEGATIVE examples (probe should NOT fire).

Output format (JSON):
[
  {{"text": "...", "label": false, "why_confusing": "..."}},
  ...
]
"""

    # Call LLM
    response = llm.generate(prompt)
    new_examples = parse_json(response)

    return new_examples
```

**Why hierarchical (blue + red):**
- ✓ Blue-team: Increases coverage (more variants of core concept)
- ✓ Red-team: Reduces false positives (hard negatives)
- ✓ Complementary: Blue improves recall, red improves precision

**Generation schedule:**
```python
# Iteration 1: Blue-team (variants)
blue_examples = generate_contrastive_pairs(concept, existing, mode='blue')

# Iteration 2: Red-team (hard negatives)
red_examples = generate_contrastive_pairs(concept, existing, mode='red')

# Iteration 3: Targeted generation (fill gaps)
# - Identify weak subspaces (low PC loading)
# - Generate examples targeting those subspaces
weak_subspace = identify_weak_pc()
targeted_examples = generate_for_subspace(concept, weak_subspace)
```

### Decision 6: Evaluation → Two-Stage (Fast Filter + Validation)

**Stage 1: Fast filter (similarity-based)**
```python
def fast_filter(new_examples, quality_threshold=60):
    """
    Fast filtering using probe vector similarity (no activations needed).
    """

    filtered = []
    for example in new_examples:
        # Extract probe vector (cheap if activations cached)
        probe = extract_probe_vector(example)

        # Compute quality score
        quality, details = quality_score(probe, existing_components)

        if quality >= quality_threshold:
            filtered.append({
                'example': example,
                'probe': probe,
                'quality': quality,
                'details': details
            })

    return filtered
```

**Stage 2: Validation (activation-based)**
```python
def validate_candidates(filtered_candidates, validation_sets):
    """
    Validate top candidates on real activations.

    Only test small subset (expensive!).
    """

    # Sort by quality, take top 20%
    top_candidates = sorted(filtered_candidates, key=lambda x: -x['quality'])[:20]

    validated = []
    for candidate in top_candidates:
        probe = candidate['probe']

        # Test on validation sets (actual classification)
        in_dist_acc = test_on_own_data(probe, candidate['example'])
        cross_acc = test_on_validation_sets(probe, validation_sets)
        gap = in_dist_acc - cross_acc

        # For safety probes: High in-dist is good, large gap is OK
        # (We WANT specialists, not general probes)
        if in_dist_acc > 0.85 and cross_acc > 0.60:
            validated.append({
                **candidate,
                'in_dist': in_dist_acc,
                'cross_dist': cross_acc,
                'gap': gap,
                'verdict': 'ACCEPT'
            })

    return validated
```

**Why two-stage:**
- ✓ Stage 1 filters 80% (fast, no activations)
- ✓ Stage 2 validates 20% (expensive, needs activations)
- ✓ Balances speed and accuracy

### Decision 7: Refinement → PCA-Guided Curation

**Identify weak spots:**
```python
def identify_weak_subspaces(current_probes):
    """
    Find which aspects of concept are underrepresented.
    """

    # PCA decomposition
    pca = PCA(n_components=10)
    components = pca.fit_transform(current_probes)

    # Check explained variance
    weak_pcs = []
    for i, var in enumerate(pca.explained_variance_ratio_):
        if var < 0.05:  # Weak component
            weak_pcs.append(i)

    # Interpret what's missing
    weak_aspects = []
    for pc_idx in weak_pcs:
        # Look at which existing examples load on this PC
        loadings = components[:, pc_idx]
        top_examples = np.argsort(np.abs(loadings))[-3:]

        # LLM interprets what these examples have in common
        interpretation = llm_interpret_pc(
            [example_texts[i] for i in top_examples],
            pc_idx
        )

        weak_aspects.append({
            'pc': pc_idx,
            'variance': pca.explained_variance_ratio_[pc_idx],
            'interpretation': interpretation
        })

    return weak_aspects
```

**Generate targeted examples:**
```python
def generate_for_weak_aspect(concept, weak_aspect):
    """
    Generate examples targeting specific underrepresented aspect.
    """

    prompt = f"""
You are improving a probe for "{concept}".

The probe currently LACKS examples that illustrate this aspect:
"{weak_aspect['interpretation']}"

Generate 10 contrastive pairs that specifically target this aspect while still being about "{concept}".

Output format (JSON):
[
  {{"true": "...", "false": "..."}},
  ...
]
"""

    response = llm.generate(prompt)
    new_examples = parse_json(response)

    return new_examples
```

**Refinement loop:**
```python
def refinement_loop(concept, initial_examples, max_iterations=5):
    """
    Iteratively improve dataset by targeting weak spots.
    """

    current_examples = initial_examples

    for iteration in range(max_iterations):
        print(f"\n=== Iteration {iteration + 1} ===")

        # Extract probes from current examples
        probes = [extract_probe_vector(ex) for ex in current_examples]

        # Identify weak subspaces
        weak_aspects = identify_weak_subspaces(probes)

        if not weak_aspects:
            print("No weak aspects found. Converged!")
            break

        print(f"Found {len(weak_aspects)} weak aspects:")
        for aspect in weak_aspects:
            print(f"  - PC{aspect['pc']}: {aspect['interpretation']} ({aspect['variance']:.1%} var)")

        # Generate targeted examples for worst aspect
        worst_aspect = weak_aspects[0]
        new_examples = generate_for_weak_aspect(concept, worst_aspect)

        # Filter and validate
        filtered = fast_filter(new_examples, quality_threshold=60)
        validated = validate_candidates(filtered, validation_sets)

        # Add validated examples
        current_examples.extend([v['example'] for v in validated])

        print(f"Added {len(validated)} new examples (total: {len(current_examples)})")

    return current_examples
```

**Why PCA-guided:**
- ✓ Identifies underrepresented aspects automatically
- ✓ Targets generation to fill gaps
- ✓ Converges when all aspects are covered

---

## Complete Pipeline

```python
def auto_probe_pipeline(concept_description, initial_examples, model):
    """
    End-to-end pipeline for auto-probe generation.

    Input: Concept description + seed examples
    Output: Multi-probe ensemble + interpretable decomposition
    """

    print(f"=== Auto-Probe Pipeline: {concept_description} ===\n")

    # Step 1: Extract initial components
    print("Step 1: Analyzing initial examples...")
    initial_probes = [extract_probe_vector(ex, model, layer=0.625) for ex in initial_examples]

    # PCA decomposition
    pca = PCA(n_components=10)
    components = pca.fit_transform(initial_probes)

    print(f"  Initial diversity: {len(initial_examples)} examples")
    print(f"  Top-5 PCs: {pca.explained_variance_ratio_[:5]}")

    # Step 2: Iterative refinement
    print("\nStep 2: Iterative refinement...")
    refined_examples = refinement_loop(
        concept_description,
        initial_examples,
        max_iterations=5
    )

    print(f"\nFinal dataset: {len(refined_examples)} examples")

    # Step 3: Extract final probes
    print("\nStep 3: Extracting final probes...")
    final_probes = [extract_probe_vector(ex, model, layer=0.625) for ex in refined_examples]

    # Step 4: Decompose into specialists
    print("\nStep 4: Decomposing into specialists...")
    specialists, labels = decompose_concept(final_probes, n_specialists='auto')

    print(f"  Found {len(specialists)} specialist groups:")
    for name, spec in specialists.items():
        print(f"    - {name}: {len(spec['members'])} probes")
        print(f"      Interpretation: {spec['interpretation']}")

    # Step 5: Ensemble within groups
    print("\nStep 5: Creating ensemble...")
    ensemble = ensemble_specialists(specialists)

    # Step 6: Validate ensemble
    print("\nStep 6: Validating ensemble...")
    for name, spec in ensemble.items():
        probe = spec['probe']

        # Test on validation sets
        in_dist = test_on_concept_examples(probe, refined_examples)
        cross_dist = test_on_other_concepts(probe, validation_sets)

        print(f"  {name}:")
        print(f"    In-dist: {in_dist:.1%}")
        print(f"    Cross-dist: {cross_dist:.1%}")
        print(f"    Gap: {in_dist - cross_dist:.1%}")

    return {
        'dataset': refined_examples,
        'specialists': specialists,
        'ensemble': ensemble,
        'decomposition': {
            'pca': pca,
            'labels': labels
        }
    }
```

---

## What We're NOT Doing (Simplicity)

### ✗ Boosting / Decision Trees
- Too complex for linear probes
- No evidence it helps
- Loses interpretability

### ✗ LoRA / Fine-Tuning
- Wrong tool (for model weights, not probes)
- Expensive (requires gradient descent)
- Not relevant to probe combination

### ✗ Layer-Wise Hybrid
- Unvalidated hypothesis (late layers better in-dist?)
- Complex (mixing activation spaces)
- Wait for evidence before adding complexity

### ✗ Complex Quality Metrics
- Don't need AUC, F1, precision/recall yet
- Similarity proxy + validation is enough
- Keep it simple initially

### ✗ Manual Taxonomy
- Spectral clustering works (4.2× better)
- Auto-interpretation with LLM
- Don't hard-code domain knowledge

---

## Critical Validation Experiments

### Experiment 1: Does Similarity Predict Transfer? (PRIORITY 1)

**Goal:** Validate similarity proxy

**Method:**
```python
# Pick 10 dataset pairs (representative sample)
pairs = [
    ('imdb', 'ag_news'),  # Both classification
    ('arc_challenge', 'race'),  # Both reasoning
    ('likely', 'boolq'),  # Different types
    # ... 7 more pairs
]

for train_ds, test_ds in pairs:
    # Compute similarity
    sim = cosine_similarity(probe[train_ds], probe[test_ds])

    # Measure actual transfer (WITH ACTIVATIONS)
    actual_acc = test_probe(probe[train_ds], activations[test_ds])

    results.append((sim, actual_acc))

# Measure correlation
correlation = pearsonr([r[0] for r in results], [r[1] for r in results])

if correlation > 0.5:
    print("✓ Similarity is reliable proxy")
else:
    print("✗ Similarity is poor proxy - need different metric")
```

**Cost:** ~1 hour (extract activations + test 10 pairs)

### Experiment 2: Does Ensemble Improve Performance? (PRIORITY 2)

**Goal:** Validate +17.2% improvement

**Method:**
```python
for group in groups:
    # Individual probes
    individual_accs = []
    for probe in group:
        acc = test_on_validation_set(probe, validation_acts)
        individual_accs.append(acc)

    # Ensemble probe (average)
    ensemble_probe = np.mean(group, axis=0)
    ensemble_acc = test_on_validation_set(ensemble_probe, validation_acts)

    improvement = (ensemble_acc - np.mean(individual_accs)) / np.mean(individual_accs)

    print(f"Group {group}: {improvement:+.1%}")

if np.mean(improvements) > 0:
    print("✓ Ensemble improves performance")
else:
    print("✗ Ensemble doesn't help - use individual probes")
```

**Cost:** ~2 hours (test groups on validation set)

### Experiment 3: Does Late Layer Improve In-Dist? (PRIORITY 3)

**Goal:** Test layer hypothesis

**Method:**
```python
for dataset in datasets[:5]:  # Sample 5
    for layer in [0.625, 0.75, 0.875]:
        probe = extract_probe(dataset, layer)

        # In-distribution accuracy
        in_dist = test_on_same_dataset(probe, dataset)

        # Cross-dataset accuracy
        cross = np.mean([test_probe(probe, other_ds) for other_ds in validation_sets])

        print(f"{dataset} @ L={layer}: in={in_dist:.1%}, cross={cross:.1%}, gap={in_dist-cross:.1%}")

# Check if in_dist increases at late layers
if late_layer_in_dist > middle_layer_in_dist:
    print("✓ Late layers have better in-dist - consider layer-wise hybrid")
else:
    print("✗ Late layers don't help - stick with middle layer")
```

**Cost:** ~3 hours (extract multiple layers)

---

## Summary: Concrete Decisions

| Component | Decision | Why | Alternatives Rejected |
|-----------|----------|-----|----------------------|
| **Quality Metric** | PC1 similarity + subspace fit + novelty | Fast, validated (r=0.37), interpretable | In-dist accuracy (too slow), AUC (not needed yet) |
| **Decomposition** | Spectral clustering (auto K) | 4.2× better, interpretable, auto | Manual taxonomy (subjective), PCA threshold (forces orthogonality) |
| **Ensemble** | Simple averaging within groups | +17.2%, dead simple, interpretable | Boosting (complex), trees (non-linear), LoRA (wrong tool) |
| **Layer** | Middle (62.5%) | Best generalization, simple | Late layer (unvalidated), hybrid (too complex) |
| **Generation** | Blue-team (variants) + Red-team (hard negatives) | Complementary (recall + precision) | Random generation (low quality), adversarial only (misses coverage) |
| **Evaluation** | Two-stage (fast filter + validation) | 80% filtered fast, 20% validated | All validation (too slow), all filtering (not accurate) |
| **Refinement** | PCA-guided (target weak subspaces) | Automatic gap detection, converges | Random iteration (inefficient), manual curation (subjective) |

**Philosophy: Simplicity is king**
- Use simple averaging, not boosting
- Use one layer, not hybrid
- Use spectral clustering, not manual taxonomy
- Validate on small subset, not everything

**Next steps:**
1. Run 3 validation experiments (priority order)
2. If similarity proxy fails (correlation < 0.5), redesign quality metric
3. If ensemble doesn't help, use individual probes
4. If late layer helps, consider layer-wise hybrid

**The pipeline is designed to work with LIMITED activation data** (just need small validation set), while using similarity for fast filtering of thousands of candidates.
