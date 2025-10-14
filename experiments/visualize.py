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
                    choices=[
                        'qwen3-4b', 'qwen3-8b', 'qwen3-14b',
                        'qwen3-4b-base', 'qwen3-8b-base', 'qwen3-14b-base',
                        'llama2-7b', 'llama2-13b', 'llama2-70b',
                        'llama2-7b-chat', 'llama2-13b-chat', 'llama2-70b-chat'
                    ],
                    help='Model to analyze')
parser.add_argument('--probe', type=str, default='dim',
                    help='Probe algorithm (default: dim)')
parser.add_argument('--layer', type=int, default=None,
                    help='Layer to use (default: auto-select best performing layer)')
args = parser.parse_args()

# Setup paths
script_dir = Path(__file__).parent
repo_root = script_dir.parent if script_dir.name == "experiments" else script_dir
output_dir_name = args.model
output_path = repo_root / f"output/comparison/{output_dir_name}"

# Find available layer subdirectories
layer_dirs = sorted([d for d in output_path.glob("layer_*") if d.is_dir()])

if not layer_dirs:
    print(f"Error: No layer result directories found in {output_path}")
    print(f"Please run:")
    print(f"  1. python experiments/extract_vectors.py --model {args.model}")
    print(f"  2. python experiments/evaluate_vectors.py --model {args.model}")
    sys.exit(1)

print(f"Found {len(layer_dirs)} layer result directories: {', '.join([d.name for d in layer_dirs])}")

print("=" * 80)
print(f"CROSS-DATASET GENERALIZATION MATRIX: {args.model.upper()} - {args.probe.upper()}")
print("=" * 80)

# Load cached results from all layer subdirectories
print(f"\n[1/4] Loading cached probe results...")
import jsonlines
results = []

for layer_dir in layer_dirs:
    layer_results_file = layer_dir / "probe_evaluate-v2.jsonl"
    if layer_results_file.exists():
        with jsonlines.open(layer_results_file) as reader:
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

# Determine which layers to process
if args.layer is None:
    # Process all available layers
    layers_to_process = sorted(df["layer"].unique())
    print(f"  Processing all {len(layers_to_process)} layers: {layers_to_process}")
else:
    # Process only specified layer
    layers_to_process = [args.layer]
    print(f"  Processing specified layer: h{args.layer}")

# Print dataset info
datasets = sorted(df["train"].unique())
print(f"\n  Found {len(datasets)} datasets:")
for ds in datasets:
    print(f"    - {ds}")

# STEP 1: Calculate thresholds (best in-distribution accuracy)
print(f"\n[2/4] Computing thresholds (best probe per dataset)...")
# For each dataset, find the best probe configuration (any layer) when train==eval
thresholds_idx = df.query("train == eval").groupby("eval")["accuracy"].idxmax()
thresholds = (
    df.loc[thresholds_idx][["eval", "accuracy"]]
    .rename(columns={"accuracy": "threshold"})
    .set_index("eval")
)

print("  Thresholds (best in-distribution accuracy):")
for ds, row in thresholds.iterrows():
    print(f"    {ds:30s}: {row['threshold']:.1%}")

# STEP 2: Calculate recovered accuracy
print(f"\n[3/4] Computing recovered accuracies...")
# Join thresholds and calculate recovered_accuracy = accuracy / threshold
df = df.join(thresholds, on="eval")
df["recovered_accuracy"] = (df["accuracy"] / df["threshold"]).clip(0, 1)

# STEP 3 & 4: Process each layer
print(f"\n[4/4] Generating matrix visualizations for {len(layers_to_process)} layer(s)...")

for current_layer in layers_to_process:
    print(f"\n{'='*80}")
    print(f"Processing Layer h{current_layer}")
    print(f"{'='*80}")

    # Filter to specific layer
    df_filtered = df.query(f"layer == {current_layer}").copy()
    print(f"  Filtered to layer {current_layer}: {len(df_filtered)} evaluations")

    # Get train/eval pairs (no aggregation needed since single layer)
    df_matrix = (
        df_filtered
        .groupby(["train", "eval"])["recovered_accuracy"]
        .mean()
        .reset_index()
    )
    print(f"  {len(df_matrix)} train/eval pairs")

    # Pivot to matrix format
    matrix = df_matrix.pivot(index="train", columns="eval", values="recovered_accuracy")

    # Order datasets by generalization performance (mean recovered accuracy when training on that dataset)
    # This matches the paper's visualization approach
    train_performance = df_filtered.groupby("train")["recovered_accuracy"].mean().sort_values(ascending=False)
    dataset_order = train_performance.index.tolist()

    # Reorder matrix
    matrix = matrix.reindex(dataset_order, axis=0).reindex(dataset_order, axis=1)

    # Determine output directory for this layer
    layer_output_path = output_path / f"layer_h{current_layer}"
    layer_output_path.mkdir(parents=True, exist_ok=True)

    # Print statistics
    print(f"\n  Matrix statistics (Recovered Accuracy):")
    print(f"    Shape: {matrix.shape[0]}x{matrix.shape[1]}")
    print(f"    Range: {matrix.min().min():.1%} to {matrix.max().max():.1%}")
    print(f"    Mean (all): {matrix.mean().mean():.1%}")

    # Calculate OOD statistics (excluding diagonal)
    ood_matrix = matrix.copy()
    np.fill_diagonal(ood_matrix.values, np.nan)
    print(f"    Mean (OOD only): {np.nanmean(ood_matrix.values):.1%}")
    print(f"    Probes >80% OOD: {(ood_matrix.values > 0.8).sum() / (~np.isnan(ood_matrix.values)).sum():.1%}")

    # Create AUC matrix if available
    if "auc" in df.columns:
        df_auc_matrix = (
            df_filtered
            .groupby(["train", "eval"])["auc"]
            .mean()
            .reset_index()
        )
        auc_matrix = df_auc_matrix.pivot(index="train", columns="eval", values="auc")
        auc_matrix = auc_matrix.reindex(dataset_order, axis=0).reindex(dataset_order, axis=1)

        # Print AUC statistics
        auc_ood_matrix = auc_matrix.copy()
        np.fill_diagonal(auc_ood_matrix.values, np.nan)
        print(f"\n  AUC Statistics (absolute probe quality):")
        print(f"    Mean AUC (all): {auc_matrix.mean().mean():.1%}")
        print(f"    Mean AUC (OOD only): {np.nanmean(auc_ood_matrix.values):.1%}")
        print(f"    Probes AUC >0.8: {(auc_ood_matrix.values > 0.8).sum() / (~np.isnan(auc_ood_matrix.values)).sum():.1%}")

    # Create visualization 1: Recovered Accuracy
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
        f"Cross-Dataset Generalization: Recovered Accuracy (Layer h{current_layer})\n{args.model.upper()} - {args.probe.upper()} Probe",
        fontsize=16,
        pad=20,
        weight='bold'
    )
    ax.set_xlabel("Evaluation Dataset (test on)", fontsize=12, weight='bold')
    ax.set_ylabel("Training Dataset (train on)", fontsize=12, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    # Save recovered accuracy matrix
    output_file_recovered = layer_output_path / "cross_dataset_matrix_recovered.png"
    plt.savefig(output_file_recovered, dpi=300, bbox_inches="tight")
    print(f"\n  ✓ Saved Recovered Accuracy Matrix: {output_file_recovered}")
    print(f"    File size: {output_file_recovered.stat().st_size / 1024:.1f} KB")
    plt.close()

    # Create visualization 2: AUC (absolute probe quality)
    if "auc" in df.columns:
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(
            auc_matrix * 100,  # Convert to percentages for display
            annot=True,
            fmt=".0f",
            cmap="RdYlGn",
            vmin=50,  # 50% = random guessing
            vmax=100,
            cbar=True,
            ax=ax,
            linewidths=0.5,
            annot_kws={"size": 7},
            cbar_kws={"label": "AUC (%)"}
        )

        ax.set_title(
            f"Cross-Dataset Generalization: AUC (Absolute Quality, Layer h{current_layer})\n{args.model.upper()} - {args.probe.upper()} Probe",
            fontsize=16,
            pad=20,
            weight='bold'
        )
        ax.set_xlabel("Evaluation Dataset (test on)", fontsize=12, weight='bold')
        ax.set_ylabel("Training Dataset (train on)", fontsize=12, weight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

        # Save AUC matrix
        output_file_auc = layer_output_path / "cross_dataset_matrix_auc.png"
        plt.savefig(output_file_auc, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved AUC Matrix: {output_file_auc}")
        print(f"    File size: {output_file_auc.stat().st_size / 1024:.1f} KB")
        plt.close()

    # Print top generalizing datasets for this layer
    print(f"\n  TOP GENERALIZING DATASETS (Layer h{current_layer}):")
    print(f"  Best datasets for generalization (train on these):")
    for i, ds in enumerate(dataset_order[:5], 1):
        score = train_performance[ds]
        print(f"    {i}. {ds:30s}: {score:.1%} mean recovered accuracy")

    print(f"\n  Worst datasets for generalization:")
    for i, ds in enumerate(dataset_order[-5:], 1):
        score = train_performance[ds]
        print(f"    {i}. {ds:30s}: {score:.1%} mean recovered accuracy")

print("\n" + "=" * 80)
print(f"✓ COMPLETED: Generated visualizations for {len(layers_to_process)} layer(s)")
print("=" * 80)
