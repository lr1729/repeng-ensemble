#!/usr/bin/env python3
"""
Visualize saliency experiment results
"""
import argparse
import jsonlines
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

parser = argparse.ArgumentParser(description='Visualize saliency experiment results')
parser.add_argument('results_file', type=str, help='Path to saliency results jsonl file')
parser.add_argument('--output-dir', type=str, default=None, help='Output directory for plots')
args = parser.parse_args()

# Load results
print(f"Loading results from: {args.results_file}")
results = []
with jsonlines.open(args.results_file) as reader:
    for item in reader:
        results.append(item["value"])

df = pd.DataFrame(results)
print(f"Loaded {len(df)} results\n")

# Determine output directory
if args.output_dir:
    output_dir = Path(args.output_dir)
else:
    output_dir = Path(args.results_file).parent

output_dir.mkdir(parents=True, exist_ok=True)

# Get model names
models = df['model_type'].unique()
datasets = df['dataset'].unique()

print(f"Models: {', '.join(models)}")
print(f"Datasets: {', '.join(datasets)}")
print()

# Normalize layers to 0-1 range for comparison
def normalize_layer(row):
    """Normalize layer number to 0-1 range"""
    max_layer = df[df['model_id'] == row['model_id']]['layer'].max()
    return row['layer'] / max_layer if max_layer > 0 else 0

df['layer_normalized'] = df.apply(normalize_layer, axis=1)

# ========================================
# Plot 1: Saliency Ratio by Layer
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Saliency Metrics: Base vs Instruct Models', fontsize=16, weight='bold')

for idx, dataset in enumerate(datasets[:4]):  # Max 4 datasets
    ax = axes[idx // 2, idx % 2]

    df_dataset = df[df['dataset'] == dataset]

    for model_type in models:
        df_model = df_dataset[df_dataset['model_type'] == model_type]
        if len(df_model) > 0:
            ax.plot(
                df_model['layer_normalized'],
                df_model['saliency_ratio'],
                marker='o',
                label=model_type.capitalize(),
                linewidth=2,
                markersize=4
            )

    ax.set_title(f'Dataset: {dataset}', fontsize=12, weight='bold')
    ax.set_xlabel('Layer (normalized)', fontsize=10)
    ax.set_ylabel('Saliency Ratio (truth_var / PC1_var)', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

plt.tight_layout()
output_file = output_dir / 'saliency_ratio_by_layer.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

# ========================================
# Plot 2: Saliency Rank by Layer
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Saliency Rank: How Many PCs Beat Truth? (Lower = Better)', fontsize=16, weight='bold')

for idx, dataset in enumerate(datasets[:4]):
    ax = axes[idx // 2, idx % 2]

    df_dataset = df[df['dataset'] == dataset]

    for model_type in models:
        df_model = df_dataset[df_dataset['model_type'] == model_type]
        if len(df_model) > 0:
            ax.plot(
                df_model['layer_normalized'],
                df_model['saliency_rank'],
                marker='o',
                label=model_type.capitalize(),
                linewidth=2,
                markersize=4
            )

    ax.set_title(f'Dataset: {dataset}', fontsize=12, weight='bold')
    ax.set_xlabel('Layer (normalized)', fontsize=10)
    ax.set_ylabel('Saliency Rank (# PCs with more variance)', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

plt.tight_layout()
output_file = output_dir / 'saliency_rank_by_layer.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

# ========================================
# Plot 3: Summary Bar Chart
# ========================================
if len(models) > 1:
    summary_data = []
    for model_type in models:
        df_model = df[df['model_type'] == model_type]
        for dataset in datasets:
            df_dataset = df_model[df_model['dataset'] == dataset]
            if len(df_dataset) > 0:
                summary_data.append({
                    'model_type': model_type.capitalize(),
                    'dataset': dataset,
                    'saliency_ratio': df_dataset['saliency_ratio'].mean(),
                    'saliency_rank': df_dataset['saliency_rank'].mean(),
                })

    df_summary = pd.DataFrame(summary_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Average Saliency Metrics: Base vs Instruct', fontsize=16, weight='bold')

    # Saliency ratio
    df_pivot = df_summary.pivot(index='dataset', columns='model_type', values='saliency_ratio')
    df_pivot.plot(kind='bar', ax=ax1, width=0.7)
    ax1.set_title('Saliency Ratio (Higher = More Salient)', fontsize=12, weight='bold')
    ax1.set_xlabel('Dataset', fontsize=10)
    ax1.set_ylabel('Saliency Ratio (avg)', fontsize=10)
    ax1.legend(title='Model Type')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    # Saliency rank
    df_pivot = df_summary.pivot(index='dataset', columns='model_type', values='saliency_rank')
    df_pivot.plot(kind='bar', ax=ax2, width=0.7)
    ax2.set_title('Saliency Rank (Lower = More Salient)', fontsize=12, weight='bold')
    ax2.set_xlabel('Dataset', fontsize=10)
    ax2.set_ylabel('Saliency Rank (avg)', fontsize=10)
    ax2.legend(title='Model Type')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    output_file = output_dir / 'saliency_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

# ========================================
# Plot 4: Correlation with Probe Accuracy
# ========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Saliency vs Probe Accuracy', fontsize=16, weight='bold')

for model_type in models:
    df_model = df[df['model_type'] == model_type]
    ax1.scatter(
        df_model['saliency_ratio'],
        df_model['probe_accuracy'],
        alpha=0.6,
        label=model_type.capitalize(),
        s=50
    )

ax1.set_xlabel('Saliency Ratio', fontsize=10)
ax1.set_ylabel('Probe Accuracy', fontsize=10)
ax1.set_title('Saliency Ratio vs Probe Accuracy', fontsize=12, weight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

for model_type in models:
    df_model = df[df['model_type'] == model_type]
    ax2.scatter(
        df_model['saliency_rank'],
        df_model['probe_accuracy'],
        alpha=0.6,
        label=model_type.capitalize(),
        s=50
    )

ax2.set_xlabel('Saliency Rank (# PCs beating truth)', fontsize=10)
ax2.set_ylabel('Probe Accuracy', fontsize=10)
ax2.set_title('Saliency Rank vs Probe Accuracy', fontsize=12, weight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_file = output_dir / 'saliency_vs_accuracy.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

print(f"\nAll plots saved to: {output_dir}")
