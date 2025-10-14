#!/usr/bin/env python3
"""
Visualize sanity check results - compare h27 vs h35 generalization
"""
import jsonlines
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Load results
results_file = Path("output/comparison/qwen3-4b/sanity_check_results.jsonl")
results = []
with jsonlines.open(results_file) as reader:
    for item in reader:
        results.append(item["value"])

df = pd.DataFrame(results)
print(f"Loaded {len(df)} results\n")

# Get datasets
datasets = sorted(df['train_dataset'].unique())
print(f"Datasets: {', '.join(datasets)}\n")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('Cross-Dataset Generalization: h27 vs h35 (Qwen3-4B)',
             fontsize=16, weight='bold', y=1.02)

for ax, layer in [(ax1, 'h27'), (ax2, 'h35')]:
    # Filter to this layer
    df_layer = df[df['layer'] == layer]

    # Pivot to matrix
    matrix = df_layer.pivot(index='train_dataset', columns='eval_dataset', values='accuracy')

    # Reorder by mean generalization (train dataset performance)
    train_perf = df_layer.groupby('train_dataset')['accuracy'].mean().sort_values(ascending=False)
    matrix = matrix.reindex(train_perf.index, axis=0).reindex(train_perf.index, axis=1)

    # Create heatmap
    sns.heatmap(
        matrix * 100,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 9},
        cbar_kws={"label": "AUC (%)"}
    )

    ax.set_title(f'Layer {layer}', fontsize=14, weight='bold', pad=10)
    ax.set_xlabel('Evaluation Dataset', fontsize=11, weight='bold')
    ax.set_ylabel('Training Dataset', fontsize=11, weight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)

    # Calculate statistics
    in_dist = matrix.values[np.eye(len(matrix), dtype=bool)]
    ood_mask = ~np.eye(len(matrix), dtype=bool)
    ood_vals = matrix.values[ood_mask]

    stats_text = f"In-dist: {np.mean(in_dist):.1f}%\nOOD: {np.mean(ood_vals):.1f}%\nRatio: {np.mean(ood_vals)/np.mean(in_dist):.1%}"
    ax.text(1.15, 0.5, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save
output_file = Path("output/comparison/qwen3-4b/sanity_check_comparison.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")

# Print detailed comparison
print("\n" + "="*80)
print("DETAILED COMPARISON: h27 vs h35")
print("="*80)

for train_ds in datasets:
    print(f"\nTrained on {train_ds}:")
    df_train = df[df['train_dataset'] == train_ds]

    for eval_ds in datasets:
        if eval_ds == train_ds:
            continue

        h27_acc = df_train[(df_train['eval_dataset'] == eval_ds) & (df_train['layer'] == 'h27')]['accuracy'].values[0]
        h35_acc = df_train[(df_train['eval_dataset'] == eval_ds) & (df_train['layer'] == 'h35')]['accuracy'].values[0]

        diff = h27_acc - h35_acc
        winner = "h27" if diff > 0 else "h35"
        symbol = "✓" if winner == "h27" else "✗"

        print(f"  → {eval_ds:20s}: h27={h27_acc:.1%}, h35={h35_acc:.1%}, diff={diff:+.1%} {symbol}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for layer in ['h27', 'h35']:
    df_layer = df[df['layer'] == layer]
    in_dist = df_layer[df_layer['train_dataset'] == df_layer['eval_dataset']]['accuracy']
    ood = df_layer[df_layer['train_dataset'] != df_layer['eval_dataset']]['accuracy']

    print(f"\n{layer}:")
    print(f"  In-distribution mean: {in_dist.mean():.1%}")
    print(f"  OOD mean: {ood.mean():.1%}")
    print(f"  Generalization ratio: {ood.mean() / in_dist.mean():.1%}")

# Which layer wins more often?
h27_wins = 0
h35_wins = 0

for train_ds in datasets:
    for eval_ds in datasets:
        if train_ds == eval_ds:
            continue

        df_pair = df[(df['train_dataset'] == train_ds) & (df['eval_dataset'] == eval_ds)]
        h27_acc = df_pair[df_pair['layer'] == 'h27']['accuracy'].values[0]
        h35_acc = df_pair[df_pair['layer'] == 'h35']['accuracy'].values[0]

        if h27_acc > h35_acc:
            h27_wins += 1
        elif h35_acc > h27_acc:
            h35_wins += 1

total_ood = len(datasets) * (len(datasets) - 1)
print(f"\nOOD Pairwise Wins:")
print(f"  h27 wins: {h27_wins}/{total_ood} ({h27_wins/total_ood:.1%})")
print(f"  h35 wins: {h35_wins}/{total_ood} ({h35_wins/total_ood:.1%})")
print(f"  Ties: {total_ood - h27_wins - h35_wins}/{total_ood}")

print("\n" + "="*80)
