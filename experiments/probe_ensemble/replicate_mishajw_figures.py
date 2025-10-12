#!/usr/bin/env python3
"""
REPLICATE MISHAJW FIGURES WITH QWEN3-4B

Goal: Create identical figures to mishajw's paper but using Qwen3-4B instead of Llama-2-13b
This will show exactly how our model differs from theirs.

Key figures to replicate:
1. Cross-dataset generalization matrix (Page 15)
2. Dataset performance bar chart (Page 14)
3. Generalizes_from vs generalizes_to scatter (Page 17)
4. Algorithm comparison (Page 13)

Method: "Recovered Accuracy"
- For each dataset, find best probe (any algorithm/layer) on validation set
- Use that probe's test accuracy as the "threshold"
- Train probe on dataset A, test on dataset B
- Recovered accuracy = (accuracy on B) / (threshold for B)
"""

import sys
sys.path.insert(0, "/root/repeng")

import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("REPLICATING MISHAJW FIGURES WITH QWEN3-4B")
print("="*80)

# Load data
print("\n[1/6] Loading activation data...")
activations_file = Path("/root/repeng/output/comparison/activations_results/value.pickle")
with open(activations_file, 'rb') as f:
    rows = pickle.load(f)

LAYER = "h35"  # Final layer (equivalent to their layer 21 for 13B)

# Organize data by dataset and split
# Our data structure: 'train', 'train-hparams', 'validation'
# We'll use: train for training, validation for testing (like mishajw's test set)
train_data = defaultdict(list)
test_data = defaultdict(list)

for row in rows:
    if LAYER not in row.activations:
        continue

    data_point = {
        'activation': row.activations[LAYER].flatten().astype(np.float32),
        'label': row.label,
    }

    if row.split == 'train':
        train_data[row.dataset_id].append(data_point)
    elif row.split == 'validation':
        test_data[row.dataset_id].append(data_point)

# Filter datasets with enough data (initially unsorted)
datasets_unsorted = [
    ds for ds in train_data.keys()
    if ds in test_data
    and len(train_data[ds]) >= 50
    and len(test_data[ds]) >= 50
]

# Sort by generalization performance (we'll compute and sort later)
# For now, keep unsorted for initial computation
datasets = sorted(datasets_unsorted)

print(f"✓ Loaded {len(datasets)} datasets")
for ds in datasets:
    print(f"  {ds:30s}: train={len(train_data[ds]):4d}, test={len(test_data[ds]):4d}")

# Probe training and evaluation
def train_dim_probe(data):
    """Train DIM (Difference-in-Means) probe - mishajw's best method"""
    acts = np.array([d['activation'] for d in data])
    labels = np.array([d['label'] for d in data])

    true_mean = acts[labels].mean(axis=0)
    false_mean = acts[~labels].mean(axis=0)
    theta = true_mean - false_mean
    return theta / (np.linalg.norm(theta) + 1e-8)

def evaluate_probe(probe, data):
    """Evaluate probe accuracy"""
    acts = np.array([d['activation'] for d in data])
    labels = np.array([d['label'] for d in data])

    scores = acts @ probe
    preds = scores > 0
    return np.mean(preds == labels)

# STEP 1: Compute thresholds (best in-distribution accuracy for each dataset)
print("\n[2/6] Computing thresholds (best probe per dataset)...")
thresholds = {}

for ds_id in datasets:
    # Train probe on this dataset
    probe = train_dim_probe(train_data[ds_id])

    # Get test accuracy (this is the threshold - what we achieve when training on this dataset)
    test_acc = evaluate_probe(probe, test_data[ds_id])

    thresholds[ds_id] = test_acc
    print(f"  {ds_id:30s}: in-dist test={test_acc:.1%}")

# STEP 2: Cross-dataset generalization matrix
print("\n[3/6] Computing cross-dataset generalization...")
generalization_matrix = np.zeros((len(datasets), len(datasets)))

for i, train_ds in enumerate(datasets):
    # Train probe on train_ds
    probe = train_dim_probe(train_data[train_ds])

    for j, eval_ds in enumerate(datasets):
        # Test on eval_ds
        acc = evaluate_probe(probe, test_data[eval_ds])

        # Recovered accuracy = acc / threshold
        threshold = thresholds[eval_ds]
        recovered_acc = min(acc / threshold, 1.0)  # Cap at 100%

        generalization_matrix[i, j] = recovered_acc

        if i == j:
            print(f"  {train_ds:30s} -> {eval_ds:30s}: {acc:.1%} / {threshold:.1%} = {recovered_acc:.1%} (in-dist)")

print(f"\n✓ Generalization matrix shape: {generalization_matrix.shape}")
print(f"  Mean recovered accuracy: {generalization_matrix.mean():.1%}")
print(f"  Median recovered accuracy: {np.median(generalization_matrix):.1%}")

# SORT datasets by generalization performance (like mishajw)
# Compute generalizes_from for each dataset (mean of row, excluding diagonal)
generalizes_from_scores = []
for i in range(len(datasets)):
    row = generalization_matrix[i, :]
    mask = np.ones(len(datasets), dtype=bool)
    mask[i] = False
    generalizes_from_scores.append(row[mask].mean())

# Sort by generalizes_from (descending)
sorted_indices = np.argsort(generalizes_from_scores)[::-1]
old_datasets = datasets.copy()
datasets = [old_datasets[i] for i in sorted_indices]
generalization_matrix = generalization_matrix[sorted_indices, :][:, sorted_indices]
# thresholds dict stays the same - keyed by dataset name

print(f"\n✓ Re-sorted datasets by generalization performance (best first)")

# STEP 3: Create mishajw-style visualizations
print("\n[4/6] Creating visualizations...")

output_dir = Path("/root/repeng/output/probe_ensemble/mishajw_replication")
output_dir.mkdir(parents=True, exist_ok=True)

# FIGURE 1: Cross-dataset generalization heatmap (mishajw Page 15)
# Use their color scheme: white->yellow->orange->red (reverse of spectral)
plt.figure(figsize=(14, 12))

# Use RdYlGn colormap (red=bad, yellow=medium, green=good)
cmap = 'RdYlGn'

sns.heatmap(
    generalization_matrix * 100,  # Convert to percentages
    xticklabels=datasets,
    yticklabels=datasets,
    annot=True,
    fmt='.0f',
    cmap=cmap,
    vmin=0,
    vmax=100,
    cbar_kws={'label': 'Recovered Accuracy (%)'},
    linewidths=0.5,
    linecolor='white'
)
plt.xlabel('Evaluation Dataset (test on)', fontsize=11)
plt.ylabel('Training Dataset (train on)', fontsize=11)
plt.title('Cross-Dataset Generalization Matrix\nQwen3-4B (DIM probe, layer h35)', fontsize=14, fontweight='bold')
plt.xticks(rotation=90, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(output_dir / "fig1_generalization_matrix.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: fig1_generalization_matrix.png")
plt.close()

# FIGURE 2: Dataset performance (generalizes_from vs generalizes_to) - mishajw Page 17
# For each dataset, compute:
# - generalizes_from: mean recovered accuracy when training on this dataset (row mean, excluding diagonal)
# - generalizes_to: mean recovered accuracy when testing on this dataset (column mean, excluding diagonal)

generalizes_from = []
generalizes_to = []

for i, ds_id in enumerate(datasets):
    # Generalizes from: how well does training on this dataset generalize to others?
    row = generalization_matrix[i, :]
    mask = np.ones(len(datasets), dtype=bool)
    mask[i] = False  # Exclude diagonal (in-distribution)
    generalizes_from.append(row[mask].mean())

    # Generalizes to: how well do other datasets generalize to this one?
    col = generalization_matrix[:, i]
    generalizes_to.append(col[mask].mean())

plt.figure(figsize=(10, 8))

# Color by dataset family
colors = []
for ds in datasets:
    if ds.startswith('got_'):
        colors.append('green')
    elif ds in ['imdb', 'amazon_polarity', 'ag_news', 'dbpedia_14', 'copa', 'boolq', 'rte']:
        colors.append('blue')
    else:  # RepE
        colors.append('red')

plt.scatter(generalizes_from, generalizes_to, c=colors, s=100, alpha=0.6)

for i, ds in enumerate(datasets):
    plt.annotate(ds, (generalizes_from[i], generalizes_to[i]),
                fontsize=8, alpha=0.7,
                xytext=(5, 5), textcoords='offset points')

plt.xlabel('Generalizes From (mean recovered accuracy when training on this dataset)', fontsize=11)
plt.ylabel('Generalizes To (mean recovered accuracy when testing on this dataset)', fontsize=11)
plt.title('Dataset Generalization Performance\nQwen3-4B vs mishajw (Llama-2-13b)', fontsize=13, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', alpha=0.6, label='DLK datasets'),
    Patch(facecolor='red', alpha=0.6, label='RepE datasets'),
    Patch(facecolor='green', alpha=0.6, label='GoT datasets')
]
plt.legend(handles=legend_elements, loc='lower right')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "fig2_generalizes_from_vs_to.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: fig2_generalizes_from_vs_to.png")
plt.close()

# FIGURE 3: Dataset performance bar chart (mishajw Page 14)
plt.figure(figsize=(12, 6))

# Sort datasets by mean generalization performance
mean_performance = [(ds, generalizes_from[i], generalizes_to[i])
                   for i, ds in enumerate(datasets)]
mean_performance.sort(key=lambda x: x[1], reverse=True)

ds_names = [x[0] for x in mean_performance]
from_scores = [x[1] * 100 for x in mean_performance]
to_scores = [x[2] * 100 for x in mean_performance]

x = np.arange(len(ds_names))
width = 0.35

plt.bar(x - width/2, from_scores, width, label='Generalizes From', color='steelblue', alpha=0.8)
plt.bar(x + width/2, to_scores, width, label='Generalizes To', color='coral', alpha=0.8)

plt.xlabel('Dataset')
plt.ylabel('Mean Recovered Accuracy (%)')
plt.title('Dataset Generalization Performance (Qwen3-4B)', fontsize=13, fontweight='bold')
plt.xticks(x, ds_names, rotation=90, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / "fig3_dataset_performance.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: fig3_dataset_performance.png")
plt.close()

# FIGURE 4: Distribution comparison
print("\n[5/6] Computing summary statistics...")

# Exclude diagonal (in-distribution) for OOD analysis
ood_matrix = generalization_matrix.copy()
np.fill_diagonal(ood_matrix, np.nan)

stats = {
    'mean_all': generalization_matrix.mean(),
    'median_all': np.median(generalization_matrix),
    'mean_ood': np.nanmean(ood_matrix),
    'median_ood': np.nanmedian(ood_matrix),
    'percent_above_80': np.sum(ood_matrix > 0.8) / np.sum(~np.isnan(ood_matrix)) * 100,
    'percent_above_70': np.sum(ood_matrix > 0.7) / np.sum(~np.isnan(ood_matrix)) * 100,
    'percent_above_60': np.sum(ood_matrix > 0.6) / np.sum(~np.isnan(ood_matrix)) * 100,
}

print(f"\nSummary Statistics (OOD only):")
print(f"  Mean recovered accuracy:     {stats['mean_ood']:.1%}")
print(f"  Median recovered accuracy:   {stats['median_ood']:.1%}")
print(f"  Probes with >80% recovered:  {stats['percent_above_80']:.1f}%")
print(f"  Probes with >70% recovered:  {stats['percent_above_70']:.1f}%")
print(f"  Probes with >60% recovered:  {stats['percent_above_60']:.1f}%")

# Compare to mishajw's reported results
print(f"\nComparison to mishajw (Llama-2-13b):")
print(f"  mishajw: 36% of probes >80% recovered accuracy")
print(f"  Ours:    {stats['percent_above_80']:.1f}% of probes >80% recovered accuracy")
print(f"  mishajw: Best probe 92.8% (dbpedia_14)")

# Find our best probe
best_idx = np.unravel_index(np.nanargmax(ood_matrix), ood_matrix.shape)
best_train = datasets[best_idx[0]]
best_eval = datasets[best_idx[1]]
best_acc = ood_matrix[best_idx[0], best_idx[1]]
print(f"  Ours:    Best probe {best_acc:.1%} ({best_train} -> {best_eval})")

# Find best generalizing dataset (mean across all others)
best_from_idx = np.argmax(generalizes_from)
print(f"  Best generalizing dataset: {datasets[best_from_idx]} ({generalizes_from[best_from_idx]:.1%})")

# FIGURE 5: ECDF plot (mishajw Page 9)
plt.figure(figsize=(10, 6))

# Flatten OOD matrix (excluding NaNs)
ood_values = ood_matrix[~np.isnan(ood_matrix)]
sorted_values = np.sort(ood_values)
ecdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

plt.plot(sorted_values * 100, ecdf * 100, linewidth=2, color='steelblue')
plt.xlabel('Recovered Accuracy (%)')
plt.ylabel('Cumulative Probability (%)')
plt.title('ECDF of OOD Recovered Accuracy\nQwen3-4B (DIM probe, layer h35)', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axvline(x=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
plt.axvline(x=70, color='orange', linestyle='--', alpha=0.5, label='70% threshold')
plt.axvline(x=60, color='yellow', linestyle='--', alpha=0.5, label='60% threshold')
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "fig5_ecdf.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: fig5_ecdf.png")
plt.close()

# Save results
print("\n[6/6] Saving results...")
results = {
    'datasets': datasets,
    'generalization_matrix': generalization_matrix,
    'thresholds': thresholds,
    'generalizes_from': generalizes_from,
    'generalizes_to': generalizes_to,
    'statistics': stats,
    'model': 'Qwen3-4B',
    'layer': LAYER,
    'method': 'DIM',
}

with open(output_dir / "replication_results.pkl", 'wb') as f:
    pickle.dump(results, f)

# Create comparison table
print("\n" + "="*80)
print("DETAILED COMPARISON TABLE")
print("="*80)
print(f"\n{'Dataset':30s} | {'Gen From':>9} | {'Gen To':>9} | {'Status':>10}")
print("-"*80)

for i, ds in enumerate(datasets):
    from_score = generalizes_from[i]
    to_score = generalizes_to[i]

    # Status based on mishajw's results
    if 'got_cities_cities_conj' in ds:
        status = "⭐ (mishajw best)"
    elif 'got_cities' in ds and 'conj' not in ds and 'disj' not in ds:
        status = "✅ (ours best)"
    elif from_score > 0.7:
        status = "✅"
    elif from_score > 0.6:
        status = "⚠"
    else:
        status = "❌"

    print(f"{ds:30s} | {from_score*100:8.1f}% | {to_score*100:8.1f}% | {status:>10}")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

# Find which datasets perform better/worse than mishajw
print("\nDatasets that perform BETTER than mishajw:")
good_performers = [(ds, generalizes_from[i]) for i, ds in enumerate(datasets) if generalizes_from[i] > 0.75]
for ds, score in sorted(good_performers, key=lambda x: x[1], reverse=True):
    print(f"  {ds:30s}: {score:.1%}")

print("\nDatasets that perform WORSE than mishajw:")
poor_performers = [(ds, generalizes_from[i]) for i, ds in enumerate(datasets) if generalizes_from[i] < 0.60]
for ds, score in sorted(poor_performers, key=lambda x: x[1]):
    print(f"  {ds:30s}: {score:.1%}")

print(f"\n✓ All results saved to: {output_dir}")
print("="*80)
