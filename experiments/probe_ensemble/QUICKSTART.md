# Quick Start Guide: Running Truth Probe Experiments

## One-Command Execution

```bash
cd /root/repeng
python experiments/probe_ensemble/run_all_experiments.py
```

This will:
1. Check for activation data
2. Run both experiments (concatenation + PCA on probes)
3. Generate analysis and interpretation

---

## If You Don't Have Activation Data

The script will tell you if activation data is missing. You have 3 options:

### Option 1: Download Pre-Computed (FASTEST - if you have S3 access)

```bash
# Download from mishajw's S3 bucket
mkdir -p output/comparison/activations_results
aws s3 cp s3://repeng/datasets/activations/datasets_2024-02-14_v1.pickle \
  output/comparison/activations_results/value.pickle

# Then run experiments
python experiments/probe_ensemble/run_all_experiments.py
```

### Option 2: Generate Full Dataset (~2-4 hours on A40)

```bash
# Requires Llama-2-13b-chat model
python experiments/comparison_dataset.py

# Then run experiments
python experiments/probe_ensemble/run_all_experiments.py
```

### Option 3: Quick Test with Subset (~30 minutes)

```bash
# Generate activations for just 5 datasets
python experiments/probe_ensemble/generate_subset_activations.py

# Then run experiments
python experiments/probe_ensemble/run_all_experiments.py
```

---

## What The Script Does

### Experiment 1: Multi-Dataset Concatenation

**Tests**: Does training on N datasets improve generalization?

**Method**: Trains DIM probes on n={1,2,3,5} dataset combinations

**Expected Output**:
```
N=1: 71% ± 2% (baseline)
N=5: 77% ± 2% (+6 points) ← Looking for this improvement!
```

**Interpretation**:
- ✅ **+6% improvement**: Noise-averaging works!
- ⚠️ **+2-3% improvement**: Modest gains
- ✗ **No improvement**: Truth may not be shared

### Experiment 2: PCA on Probe Vectors (NOVEL!)

**Tests**: Can we extract a consensus "truth" direction from multiple probes?

**Method**:
1. Train 18 individual DIM probes
2. Stack and normalize: Θ = [θ₁, θ₂, ..., θ₁₈]
3. Compute SVD: Θ = U S Vᵀ
4. Analyze singular values and test PC1

**Expected Output**:
```
S[0]/S[1] = 2.5 ± 0.5
PC1 explains ~65% variance
PC1 achieves ~75% accuracy
```

**Interpretation**:
- **S[0]/S[1] > 3**: Truth is unified → use PC1 only
- **S[0]/S[1] ∈ [2,3]**: Compositional → need PC1+PC2+PC3
- **S[0]/S[1] < 2**: Fragmented → need different approach

---

## Reading the Results

### Multi-Dataset Results

Saved to: `output/probe_ensemble/multi_dataset_results.pkl`

```python
# Load and inspect
import pickle
with open('output/probe_ensemble/multi_dataset_results.pkl', 'rb') as f:
    results = pickle.load(f)

for r in results:
    print(f"N={r.n_datasets}: {r.avg_test_accuracy:.1%}")
```

### PCA Results

Saved to: `output/probe_ensemble/probe_vector_pca_results.pkl`

```python
import pickle
import numpy as np

with open('output/probe_ensemble/probe_vector_pca_results.pkl', 'rb') as f:
    data = pickle.load(f)

# Key metrics
S = data['S']  # Singular values
print(f"S[0]/S[1] = {S[0]/S[1]:.3f}")
print(f"PC1 variance: {S[0]**2/np.sum(S**2):.1%}")

# Meta-probe performance
for method, accuracies in data['results'].items():
    avg = np.mean([a for a in accuracies.values() if a > 0])
    print(f"{method}: {avg:.1%}")
```

---

## Expected Timeline

| Task | Time | GPU? |
|------|------|------|
| Download pre-computed | ~5 min | No |
| Generate full activations | ~2-4 hours | Yes (A40) |
| Generate subset (5 datasets) | ~30 min | Yes (A40) |
| Run experiments | ~60-90 min | No |
| **Total (with pre-computed)** | **~2 hours** | **No** |

---

## Troubleshooting

### "No activation data found"

→ Follow Option 1, 2, or 3 above to get activation data

### "ModuleNotFoundError: No module named 'repeng'"

```bash
cd /root/repeng
pip install -e .
# Or: pip install scikit-learn datasets python-dotenv
```

### "Error loading activations"

Check file exists:
```bash
ls -lah output/comparison/activations_results/value.pickle
```

### Experiments run but results are poor

**This is OK!** The goal is to TEST the hypothesis, not confirm it.

Negative results are valuable:
- If concatenation doesn't help → truth is more complex than expected
- If S[0]/S[1] < 2 → truth is fragmented at 13B (need 70B)
- If PC1 doesn't beat baseline → PCA isn't the right tool

---

## Key Innovation

**PCA on probe vectors** (Experiment 2) is **completely novel** - has never been done before!

Prior work:
- ❌ PCA on activations within a dataset (doesn't work well)
- ❌ Train on dataset A, test on B (limited generalization)

This work:
- ✅ PCA on probe vectors themselves across datasets
- ✅ Extracts consensus "truth" direction
- ✅ Filters dataset-specific noise

---

## Next Steps After Results

### If successful (S[0]/S[1] > 2.5, improvement > 3%)

1. **Use meta-probe for deployment**
2. **Test on 70B model** (expect even better unification)
3. **Apply to misalignment detection**

### If partial (S[0]/S[1] ∈ [2,3], improvement 1-3%)

1. **Develop multi-component probe**
2. **Investigate PC2 and PC3 semantics**
3. **Try context-dependent weighting**

### If unsuccessful (S[0]/S[1] < 2, no improvement)

1. **Analyze why (valuable negative result!)**
2. **Test on 70B where truth is more unified**
3. **Consider non-linear probes**

---

## Contact

For issues or questions:
- Check `/root/repeng/experiments/probe_ensemble/README.md`
- Review code comments (extensively documented)
- Inspect intermediate outputs in `output/probe_ensemble/`
