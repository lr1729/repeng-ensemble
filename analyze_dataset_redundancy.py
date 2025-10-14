#!/usr/bin/env python3
"""
Analyze dataset redundancy and signal quality from existing experimental results.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_results():
    """Load all probe_evaluate-v2.jsonl files"""
    results = []
    output_dir = Path("/root/repeng/output/comparison")

    for jsonl_file in output_dir.glob("*/layer_*/probe_evaluate-v2.jsonl"):
        model = jsonl_file.parent.parent.name
        layer = jsonl_file.parent.name

        with open(jsonl_file) as f:
            for line in f:
                data = json.loads(line)
                value = data['value']
                value['model'] = model
                value['layer'] = layer
                results.append(value)

    return pd.DataFrame(results)

def compute_probe_quality(df):
    """In-distribution performance (diagonal)"""
    diagonal = df[df['train_dataset'] == df['eval_dataset']]

    quality = diagonal.groupby('train_dataset').agg({
        'accuracy': ['mean', 'std', 'count'],
        'auc': ['mean', 'std']
    }).round(3)

    quality.columns = ['_'.join(col).strip() for col in quality.columns.values]
    quality = quality.sort_values('auc_mean', ascending=False)

    return quality

def compute_generalization(df):
    """Off-diagonal generalization scores"""
    off_diagonal = df[df['train_dataset'] != df['eval_dataset']]

    # As source (train on X, test on all others)
    source_gen = off_diagonal.groupby('train_dataset')['auc'].agg(['mean', 'std', 'count'])
    source_gen.columns = ['as_source_mean', 'as_source_std', 'source_count']

    # As target (train on all others, test on X)
    target_gen = off_diagonal.groupby('eval_dataset')['auc'].agg(['mean', 'std', 'count'])
    target_gen.columns = ['as_target_mean', 'as_target_std', 'target_count']

    # Combine
    generalization = pd.concat([source_gen, target_gen], axis=1)
    generalization['overall_mean'] = (generalization['as_source_mean'] + generalization['as_target_mean']) / 2
    generalization = generalization.sort_values('overall_mean', ascending=False)

    return generalization

def compute_redundancy(df):
    """Find redundant dataset pairs via generalization pattern similarity"""
    datasets = sorted(df['train_dataset'].unique())

    # Build cross-generalization matrix (how each dataset generalizes to others)
    matrix = df.pivot_table(
        index='train_dataset',
        columns='eval_dataset',
        values='auc',
        aggfunc='mean'
    )

    # Fill NaN with mean (some pairs might be missing)
    matrix = matrix.fillna(matrix.mean().mean())

    # Ensure same order
    matrix = matrix.reindex(datasets, axis=0).reindex(datasets, axis=1)

    # Compute cosine similarity between rows (generalization patterns)
    similarity = cosine_similarity(matrix)

    # Create similarity dataframe
    sim_df = pd.DataFrame(similarity, index=datasets, columns=datasets)

    # Find highly similar pairs (>0.85 similarity, excluding diagonal)
    similar_pairs = []
    for i, d1 in enumerate(datasets):
        for j, d2 in enumerate(datasets):
            if i < j:  # Upper triangle only
                sim_val = similarity[i, j]
                if sim_val > 0.85:
                    similar_pairs.append({
                        'dataset1': d1,
                        'dataset2': d2,
                        'similarity': sim_val,
                        'recommend_remove': d2  # Arbitrary: keep first, remove second
                    })

    similar_pairs_df = pd.DataFrame(similar_pairs).sort_values('similarity', ascending=False)

    return sim_df, similar_pairs_df, matrix

def analyze_by_model(df):
    """Compare performance across models"""
    model_stats = df.groupby(['model', 'train_dataset', 'eval_dataset']).agg({
        'auc': 'mean',
        'accuracy': 'mean'
    }).reset_index()

    # Overall performance by model
    model_summary = df.groupby('model').agg({
        'auc': ['mean', 'std'],
        'accuracy': ['mean', 'std']
    }).round(3)

    return model_summary

def identify_weak_datasets(probe_quality, generalization, threshold_diagonal=0.75, threshold_gen=0.65):
    """Identify datasets with weak performance"""
    weak = []

    # Weak probe quality (can't even learn in-distribution)
    weak_probe = probe_quality[probe_quality['auc_mean'] < threshold_diagonal].index.tolist()

    # Poor generalization
    poor_gen = generalization[generalization['overall_mean'] < threshold_gen].index.tolist()

    weak_datasets = {
        'weak_probe_quality': weak_probe,
        'poor_generalization': poor_gen,
        'union': list(set(weak_probe) | set(poor_gen))
    }

    return weak_datasets

def main():
    print("="*80)
    print("DATASET REDUNDANCY AND SIGNAL QUALITY ANALYSIS")
    print("="*80)

    # Load all results
    print("\n[1/5] Loading results...")
    df = load_all_results()
    print(f"  Loaded {len(df)} result rows")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Layers: {sorted(df['layer'].unique())}")
    print(f"  Datasets: {len(df['train_dataset'].unique())} unique")

    # Probe quality
    print("\n[2/5] Computing probe quality (in-distribution)...")
    probe_quality = compute_probe_quality(df)
    print("\n=== TOP 10 DATASETS (by in-distribution AUC) ===")
    print(probe_quality.head(10))
    print("\n=== BOTTOM 5 DATASETS (by in-distribution AUC) ===")
    print(probe_quality.tail(5))

    # Generalization
    print("\n[3/5] Computing generalization scores...")
    generalization = compute_generalization(df)
    print("\n=== TOP 10 DATASETS (by overall generalization) ===")
    print(generalization.head(10))
    print("\n=== BOTTOM 5 DATASETS (by overall generalization) ===")
    print(generalization.tail(5))

    # Redundancy
    print("\n[4/5] Computing redundancy (similarity > 0.85)...")
    sim_df, similar_pairs, gen_matrix = compute_redundancy(df)

    if len(similar_pairs) > 0:
        print(f"\n=== FOUND {len(similar_pairs)} REDUNDANT PAIRS ===")
        print(similar_pairs.to_string(index=False))
    else:
        print("\n=== NO HIGHLY REDUNDANT PAIRS FOUND (similarity > 0.85) ===")

    # Weak datasets
    print("\n[5/5] Identifying weak datasets...")
    weak = identify_weak_datasets(probe_quality, generalization)
    print(f"\n=== WEAK PROBE QUALITY (AUC < 0.75) ===")
    print(weak['weak_probe_quality'] if weak['weak_probe_quality'] else "None")
    print(f"\n=== POOR GENERALIZATION (mean < 0.65) ===")
    print(weak['poor_generalization'] if weak['poor_generalization'] else "None")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Critical datasets (never remove)
    critical = ['diverse_truth', 'complex_truth', 'likely', 'counterfact_true_false']

    # Candidates for removal
    remove_candidates = set()

    # Add redundant pairs (keep first, remove second)
    if len(similar_pairs) > 0:
        redundant_to_remove = similar_pairs['recommend_remove'].tolist()
        remove_candidates.update([d for d in redundant_to_remove if d not in critical])

    # Add weak performers
    remove_candidates.update([d for d in weak['union'] if d not in critical])

    # Additional heuristic removals based on task type
    heuristic_removals = []
    datasets_list = df['train_dataset'].unique()

    # Check for sentiment duplicates
    if 'imdb' in datasets_list and 'amazon_polarity' in datasets_list:
        heuristic_removals.append(('amazon_polarity', 'Redundant with imdb (both sentiment)'))

    # Check for topic classification duplicates
    if 'ag_news' in datasets_list and 'dbpedia_14' in datasets_list:
        heuristic_removals.append(('dbpedia_14', 'Redundant with ag_news (both topic classification)'))

    # Check for arc duplicates
    if 'arc_challenge' in datasets_list and 'arc_easy' in datasets_list:
        heuristic_removals.append(('arc_easy', 'Redundant with arc_challenge (easier version)'))

    print("\n=== DATA-DRIVEN REMOVALS ===")
    if remove_candidates:
        print(f"Based on weak performance or high similarity:")
        for d in sorted(remove_candidates):
            print(f"  - {d}")
    else:
        print("  None identified from data")

    print("\n=== HEURISTIC REMOVALS (task-based) ===")
    if heuristic_removals:
        for dataset, reason in heuristic_removals:
            print(f"  - {dataset}: {reason}")
            remove_candidates.add(dataset)
    else:
        print("  None")

    print("\n=== FINAL RECOMMENDATION ===")
    print(f"Current dataset count: {len(datasets_list)}")
    print(f"Recommended to remove: {len(remove_candidates)}")
    print(f"Resulting count: {len(datasets_list) - len(remove_candidates)}")
    print(f"\nRemove: {sorted(remove_candidates)}")
    print(f"Keep: {sorted(set(datasets_list) - remove_candidates)}")

    # Model comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    model_summary = analyze_by_model(df)
    print(model_summary)

    # Save results
    output_file = Path("/root/repeng/dataset_analysis_results.txt")
    with open(output_file, 'w') as f:
        f.write("DATASET ANALYSIS RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write("PROBE QUALITY (top 10):\n")
        f.write(probe_quality.head(10).to_string() + "\n\n")
        f.write("GENERALIZATION (top 10):\n")
        f.write(generalization.head(10).to_string() + "\n\n")
        f.write("REDUNDANT PAIRS:\n")
        f.write(similar_pairs.to_string(index=False) + "\n\n")
        f.write(f"RECOMMENDED REMOVALS: {sorted(remove_candidates)}\n")

    print(f"\n✓ Detailed results saved to: {output_file}")

    # Create visualization of similarity matrix
    plt.figure(figsize=(16, 14))
    sns.heatmap(sim_df, annot=False, cmap='coolwarm', center=0.7,
                vmin=0.5, vmax=1.0, square=True, cbar_kws={'label': 'Similarity'})
    plt.title("Dataset Similarity Matrix (based on generalization patterns)", fontsize=14)
    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("Dataset", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    viz_file = Path("/root/repeng/dataset_similarity_heatmap.png")
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✓ Similarity heatmap saved to: {viz_file}")

if __name__ == "__main__":
    main()
