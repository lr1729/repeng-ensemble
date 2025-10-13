#!/usr/bin/env python3
"""
Generate Cross-Dataset Generalization Matrix

This script loads cached probe results and generates the key cross-dataset
generalization matrix showing how probes trained on one dataset perform on others.

Methodology (from mishajw paper):
- Threshold: Best in-distribution accuracy for each dataset (train on X, test on X)
- Recovered Accuracy: (probe accuracy) / (threshold) for that eval dataset
- Matrix[i,j]: Recovered accuracy when training on dataset i, testing on dataset j
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Parse arguments
parser = argparse.ArgumentParser(description='Generate cross-dataset generalization matrix')
parser.add_argument('--model', type=str, required=True,
                    choices=['qwen3-4b', 'qwen3-8b', 'qwen3-14b', 'llama2-7b', 'llama2-13b', 'llama2-70b'],
                    help='Model to analyze')
parser.add_argument('--probe', type=str, default='dim',
                    help='Probe algorithm (default: dim)')
parser.add_argument('--layer', type=int, default=35,
                    help='Layer to use (default: 35, equivalent to final layer for Qwen3-8B)')
args = parser.parse_args()

# Setup paths
script_dir = Path(__file__).parent
repo_root = script_dir.parent if script_dir.name == "experiments" else script_dir
output_dir_name = args.model
output_path = repo_root / f"output/comparison/{output_dir_name}"
results_file = output_path / "probe_evaluate-v2.jsonl"

if not results_file.exists():
    print(f"Error: Results file not found: {results_file}")
    print(f"Please run: python experiments/comparison_dataset.py --model {args.model}")
    sys.exit(1)

print("=" * 80)
print(f"CROSS-DATASET GENERALIZATION MATRIX: {args.model.upper()} - {args.probe.upper()}")
print("=" * 80)

# Load cached results
print(f"\n[1/4] Loading cached probe results...")
import jsonlines
results = []
with jsonlines.open(results_file) as reader:
    for item in reader:
        results.append(item["value"])

df = pd.DataFrame(results)
print(f"  Loaded {len(df)} cached probe evaluations")

# Filter to selected probe
df = df[df["probe_method"] == args.probe].copy()
if len(df) == 0:
    print(f"Error: No results found for probe '{args.probe}'")
    sys.exit(1)

print(f"  Filtered to {len(df)} evaluations for {args.probe.upper()} probe")

# Rename columns to match paper terminology
df = df.rename(columns={
    "train_dataset": "train",
    "eval_dataset": "eval",
    "probe_method": "algorithm",
})

# Extract layer number from point_name (e.g., "h21" -> 21)
df["layer"] = df["point_name"].apply(lambda p: int(p.lstrip("h")))

# Print dataset info
datasets = sorted(df["train"].unique())
print(f"  Found {len(datasets)} datasets:")
for ds in datasets:
    print(f"    - {ds}")

# STEP 1: Calculate thresholds (best in-distribution accuracy)
print(f"\n[2/4] Computing thresholds (best probe per dataset)...")
# Use accuracy_hparams (validation accuracy) as the main metric
# For each dataset, find the best probe configuration (any layer) when train==eval
thresholds_idx = df.query("train == eval").groupby("eval")["accuracy_hparams"].idxmax()
thresholds = (
    df.loc[thresholds_idx][["eval", "accuracy_hparams"]]
    .rename(columns={"accuracy_hparams": "threshold"})
    .set_index("eval")
)

print("  Thresholds (best in-distribution accuracy):")
for ds, row in thresholds.iterrows():
    print(f"    {ds:30s}: {row['threshold']:.1%}")

# STEP 2: Calculate recovered accuracy
print(f"\n[3/4] Computing recovered accuracies...")
# Join thresholds and calculate recovered_accuracy = accuracy_hparams / threshold
df = df.join(thresholds, on="eval")
df["recovered_accuracy"] = (df["accuracy_hparams"] / df["threshold"]).clip(0, 1)

# Filter to specific layer
df_filtered = df.query(f"layer == {args.layer}").copy()
print(f"  Filtered to layer {args.layer}: {len(df_filtered)} evaluations")

# Get train/eval pairs (no aggregation needed since single layer)
df_matrix = (
    df_filtered
    .groupby(["train", "eval"])["recovered_accuracy"]
    .mean()
    .reset_index()
)
print(f"  {len(df_matrix)} train/eval pairs")

# STEP 3: Create and order matrix
print(f"\n[4/4] Generating matrix visualization...")
# Pivot to matrix format
matrix = df_matrix.pivot(index="train", columns="eval", values="recovered_accuracy")

# Order datasets by generalization performance (mean recovered accuracy when training on that dataset)
# This matches the paper's visualization approach
train_performance = df_filtered.groupby("train")["recovered_accuracy"].mean().sort_values(ascending=False)
dataset_order = train_performance.index.tolist()

# Reorder matrix
matrix = matrix.reindex(dataset_order, axis=0).reindex(dataset_order, axis=1)

# Print statistics
print(f"\n  Matrix statistics:")
print(f"    Shape: {matrix.shape[0]}x{matrix.shape[1]}")
print(f"    Range: {matrix.min().min():.1%} to {matrix.max().max():.1%}")
print(f"    Mean (all): {matrix.mean().mean():.1%}")

# Calculate OOD statistics (excluding diagonal)
ood_matrix = matrix.copy()
np.fill_diagonal(ood_matrix.values, np.nan)
print(f"    Mean (OOD only): {np.nanmean(ood_matrix.values):.1%}")
print(f"    Probes >80% OOD: {(ood_matrix.values > 0.8).sum() / (~np.isnan(ood_matrix.values)).sum():.1%}")

# Create visualization
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(
    matrix * 100,  # Convert to percentages for display
    annot=True,
    fmt=".0f",
    cmap="RdYlGn",
    vmin=0,
    vmax=100,
    cbar=True,
    ax=ax,
    linewidths=0.5,
    annot_kws={"size": 7},
    cbar_kws={"label": "Recovered Accuracy (%)"}
)

ax.set_title(
    f"Cross-Dataset Generalization Matrix (Layer h{args.layer})\n{args.model.upper()} - {args.probe.upper()} Probe",
    fontsize=16,
    pad=20,
    weight='bold'
)
ax.set_xlabel("Evaluation Dataset (test on)", fontsize=12, weight='bold')
ax.set_ylabel("Training Dataset (train on)", fontsize=12, weight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

# Save
output_file = output_path / "cross_dataset_matrix.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"\n✓ Saved: {output_file}")
print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")
plt.close()

# Print top generalizing datasets
print(f"\n" + "=" * 80)
print("TOP GENERALIZING DATASETS")
print("=" * 80)
print(f"\nBest datasets for generalization (train on these):")
for i, ds in enumerate(dataset_order[:5], 1):
    score = train_performance[ds]
    print(f"  {i}. {ds:30s}: {score:.1%} mean recovered accuracy")

print(f"\nWorst datasets for generalization:")
for i, ds in enumerate(dataset_order[-5:], 1):
    score = train_performance[ds]
    print(f"  {i}. {ds:30s}: {score:.1%} mean recovered accuracy")

print("\n" + "=" * 80)
