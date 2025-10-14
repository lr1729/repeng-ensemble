#!/usr/bin/env python3
"""
Saliency Experiment: Does instruction tuning make truth more salient?

Tests whether the "truth direction" (DIM probe) captures more variance in
instruction-tuned models compared to base models.

Research Question:
- Base models: Truth exists but is "hidden" among many features
- Instruction-tuned models: Truth becomes a prominent, high-variance feature

This would explain why unsupervised probes (PCA) work on chat models but not base models.
"""
import sys
sys.path.insert(0, "/root/repeng")

import argparse
import torch
import jsonlines
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.decomposition import PCA

from repeng.datasets.elk.utils.collections import resolve_dataset_ids
from repeng.datasets.elk.utils.fns import get_dataset
from repeng.activations.inference import get_model_activations
from repeng.activations.probe_preparations import ActivationArrays
from repeng.models.loading import load_llm_oioo
from repeng.models.points import get_points
from repeng.probes.difference_in_means import train_dim_probe
from repeng.evals.probes import eval_probe_by_question, eval_probe_by_row

# Parse arguments
parser = argparse.ArgumentParser(description='Saliency experiment comparing base vs instruct models')
parser.add_argument(
    '--base-model',
    type=str,
    required=True,
    help='Base model (e.g., Qwen/Qwen3-4B)'
)
parser.add_argument(
    '--instruct-model',
    type=str,
    default=None,
    help='Instruction-tuned model (e.g., Qwen/Qwen3-4B-Instruct)'
)
parser.add_argument(
    '--datasets',
    type=str,
    nargs='+',
    default=['boolq', 'imdb', 'race', 'got_cities'],
    help='Datasets to test'
)
parser.add_argument(
    '--n-pca-components',
    type=int,
    default=1024,
    help='Number of PCA components to compute'
)
args = parser.parse_args()

# Configuration
NUM_PCA_COMPONENTS = args.n_pca_components
TRAIN_LIMIT = 400
VAL_LIMIT = 2000
LAYERS_SKIP = 2  # Sample every 2nd layer

@dataclass
class SaliencyResult:
    """Results for one model/dataset/layer combination"""
    model_id: str
    model_type: str  # 'base' or 'instruct'
    dataset: str
    layer: int
    layer_name: str

    # Saliency metrics
    saliency: float  # truth_variance / total_variance (0-1, higher = more salient)
    saliency_rank: int  # How many PCA components have MORE variance than truth
    saliency_ratio: float  # truth_variance / PC1_variance (0-1, higher = more salient)

    # Probe performance
    probe_accuracy: float
    probe_n: int

    # Additional context
    truth_variance: float
    pc1_variance: float
    total_variance: float


def get_activations_for_split(dataset_id: str, split: str, llm, limit: int, layers_skip: int):
    """Get activations for a dataset split"""
    dataset_rows = get_dataset(dataset_id)
    rows = [r for k, r in dataset_rows.items() if r.split == split][:limit]

    activations_by_layer = defaultdict(list)
    labels = []
    groups = []

    for row in tqdm(rows, desc=f"    {split[:12]:12s}", leave=False):
        activation_row = get_model_activations(
            llm,
            text=row.text,
            last_n_tokens=1,
            points_start=1,
            points_end=None,
            points_skip=layers_skip,
        )

        for point_name, acts in activation_row.activations.items():
            activations_by_layer[point_name].append(acts[-1])

        labels.append(row.label)
        groups.append(row.group_id if row.group_id else None)

    return activations_by_layer, labels, groups


def calculate_saliency(
    model_id: str,
    model_type: str,
    dataset_id: str,
    layer_name: str,
    train_acts,
    train_labels,
    val_acts,
    val_labels,
    val_groups,
) -> SaliencyResult:
    """Calculate saliency metrics for one layer"""

    # Train DIM probe to get "truth direction"
    train_acts_np = np.stack(train_acts).astype(np.float32)
    train_labels_np = np.array(train_labels)

    probe = train_dim_probe(
        activations=train_acts_np,
        labels=train_labels_np,
    )

    # Normalize truth direction
    truth_direction = probe.probe.copy()
    if np.linalg.norm(truth_direction) > 0:
        truth_direction = truth_direction / np.linalg.norm(truth_direction)

    # Fit PCA on training data
    pca = PCA(n_components=min(NUM_PCA_COMPONENTS, train_acts_np.shape[1]))
    pca.fit(train_acts_np)

    # Calculate saliency on validation data
    val_acts_np = np.stack(val_acts).astype(np.float32)
    val_labels_np = np.array(val_labels)

    # Truth variance: variance along the truth direction
    truth_activations = val_acts_np @ truth_direction
    truth_variance = float(np.var(truth_activations))

    # Total variance: sum of variances across all dimensions
    total_variance = float(np.var(val_acts_np, axis=0).sum())

    # Saliency: what fraction of total variance is explained by truth?
    saliency = truth_variance / total_variance if total_variance > 0 else 0.0

    # PCA variances: variance along each principal component
    pca_variances = pca.transform(val_acts_np).var(axis=0)
    pc1_variance = float(pca_variances[0])

    # Saliency rank: how many PCs have MORE variance than truth?
    saliency_rank = int(np.sum(pca_variances > truth_variance))

    # Saliency ratio: truth variance relative to top PC
    saliency_ratio = truth_variance / pc1_variance if pc1_variance > 0 else 0.0

    # Evaluate probe accuracy
    if val_groups and any(g is not None for g in val_groups):
        val_groups_np = np.array([hash(str(g)) % 100000 if g else i for i, g in enumerate(val_groups)])
        unique_groups = len(np.unique(val_groups_np))

        if unique_groups > 1:
            eval_result = eval_probe_by_question(
                probe,
                activations=val_acts_np,
                labels=val_labels_np,
                groups=val_groups_np,
            )
            probe_accuracy = eval_result.accuracy
            probe_n = eval_result.n
        else:
            eval_result = eval_probe_by_row(probe, activations=val_acts_np, labels=val_labels_np)
            probe_accuracy = eval_result.roc_auc_score
            probe_n = eval_result.n
    else:
        eval_result = eval_probe_by_row(probe, activations=val_acts_np, labels=val_labels_np)
        probe_accuracy = eval_result.roc_auc_score
        probe_n = eval_result.n

    # Extract layer number
    layer_num = int(layer_name.lstrip('h'))

    return SaliencyResult(
        model_id=model_id,
        model_type=model_type,
        dataset=dataset_id,
        layer=layer_num,
        layer_name=layer_name,
        saliency=saliency,
        saliency_rank=saliency_rank,
        saliency_ratio=saliency_ratio,
        probe_accuracy=probe_accuracy,
        probe_n=probe_n,
        truth_variance=truth_variance,
        pc1_variance=pc1_variance,
        total_variance=total_variance,
    )


def run_experiment(model_id: str, model_type: str, datasets: list[str]):
    """Run saliency experiment for one model"""
    print(f"\n{'='*80}")
    print(f"SALIENCY EXPERIMENT: {model_id} ({model_type})")
    print(f"{'='*80}\n")

    # Load model
    print("[1/3] Loading model...")
    llm = load_llm_oioo(model_id, device=torch.device("cuda"), use_half_precision=True)
    print("✓ Model loaded\n")

    # Get layer names
    all_points = get_points(model_id)
    sampled_points = all_points[1::LAYERS_SKIP]
    point_names = [p.name for p in sampled_points]
    print(f"Layers ({len(sampled_points)}): {', '.join(point_names[:3])}...{', '.join(point_names[-2:])}")
    print()

    results = []
    total = len(datasets) * len(sampled_points)

    print(f"[2/3] Computing saliency...")
    print(f"Total: {len(datasets)} datasets × {len(sampled_points)} layers = {total}")
    print()

    with tqdm(total=total, desc="Overall progress") as pbar:
        for dataset_id in datasets:
            print(f"\n  Dataset: {dataset_id}")

            # Get training activations
            train_acts_by_layer, train_labels, _ = get_activations_for_split(
                dataset_id, "train", llm, TRAIN_LIMIT, LAYERS_SKIP
            )

            # Get validation activations
            val_acts_by_layer, val_labels, val_groups = get_activations_for_split(
                dataset_id, "validation", llm, VAL_LIMIT, LAYERS_SKIP
            )

            # Calculate saliency for each layer
            for point_name in point_names:
                result = calculate_saliency(
                    model_id=model_id,
                    model_type=model_type,
                    dataset_id=dataset_id,
                    layer_name=point_name,
                    train_acts=train_acts_by_layer[point_name],
                    train_labels=train_labels,
                    val_acts=val_acts_by_layer[point_name],
                    val_labels=val_labels,
                    val_groups=val_groups,
                )
                results.append(result)
                pbar.update(1)

    return results


def main():
    all_results = []

    # Run base model
    print("="*80)
    print("RUNNING SALIENCY EXPERIMENT")
    print("="*80)
    print(f"Base model: {args.base_model}")
    print(f"Instruct model: {args.instruct_model or 'None'}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"PCA components: {NUM_PCA_COMPONENTS}")
    print()

    base_results = run_experiment(args.base_model, "base", args.datasets)
    all_results.extend(base_results)

    # Run instruct model if provided
    if args.instruct_model:
        instruct_results = run_experiment(args.instruct_model, "instruct", args.datasets)
        all_results.extend(instruct_results)

    # Save results
    output_dir = Path("output/saliency")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = args.base_model.split('/')[-1]
    output_file = output_dir / f"saliency_{base_name}.jsonl"

    print(f"\n[3/3] Saving results...")
    with jsonlines.open(output_file, mode='w') as writer:
        for i, result in enumerate(all_results):
            writer.write({
                "key": f"result_{i}",
                "value": {
                    "model_id": result.model_id,
                    "model_type": result.model_type,
                    "dataset": result.dataset,
                    "layer": result.layer,
                    "layer_name": result.layer_name,
                    "saliency": result.saliency,
                    "saliency_rank": result.saliency_rank,
                    "saliency_ratio": result.saliency_ratio,
                    "probe_accuracy": result.probe_accuracy,
                    "probe_n": result.probe_n,
                    "truth_variance": result.truth_variance,
                    "pc1_variance": result.pc1_variance,
                    "total_variance": result.total_variance,
                }
            })

    print(f"✓ Saved to: {output_file}")
    print()

    # Print summary statistics
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for model_type in ['base', 'instruct']:
        model_results = [r for r in all_results if r.model_type == model_type]
        if not model_results:
            continue

        print(f"\n{model_type.upper()} MODEL:")
        print(f"  Model: {model_results[0].model_id}")

        # Average across all datasets and layers
        avg_saliency = np.mean([r.saliency for r in model_results])
        avg_saliency_ratio = np.mean([r.saliency_ratio for r in model_results])
        avg_saliency_rank = np.mean([r.saliency_rank for r in model_results])

        print(f"  Average saliency: {avg_saliency:.6f} ({avg_saliency*100:.3f}% of total variance)")
        print(f"  Average saliency ratio: {avg_saliency_ratio:.6f} ({avg_saliency_ratio*100:.1f}% of PC1)")
        print(f"  Average saliency rank: {avg_saliency_rank:.1f} (out of {NUM_PCA_COMPONENTS} PCs)")

        # Per-dataset breakdown
        print(f"\n  By dataset:")
        for dataset_id in args.datasets:
            dataset_results = [r for r in model_results if r.dataset == dataset_id]
            if dataset_results:
                ds_saliency = np.mean([r.saliency for r in dataset_results])
                ds_ratio = np.mean([r.saliency_ratio for r in dataset_results])
                ds_rank = np.mean([r.saliency_rank for r in dataset_results])
                print(f"    {dataset_id:20s}: saliency={ds_saliency:.6f}, ratio={ds_ratio:.3f}, rank={ds_rank:.0f}")

    # Comparison
    if args.instruct_model:
        print(f"\n{'='*80}")
        print("COMPARISON: INSTRUCT vs BASE")
        print("="*80)

        base_results = [r for r in all_results if r.model_type == "base"]
        instruct_results = [r for r in all_results if r.model_type == "instruct"]

        base_saliency = np.mean([r.saliency for r in base_results])
        instruct_saliency = np.mean([r.saliency for r in instruct_results])
        improvement = (instruct_saliency - base_saliency) / base_saliency * 100

        print(f"Saliency improvement: {improvement:+.1f}%")
        print(f"  Base model:     {base_saliency:.6f}")
        print(f"  Instruct model: {instruct_saliency:.6f}")

        base_ratio = np.mean([r.saliency_ratio for r in base_results])
        instruct_ratio = np.mean([r.saliency_ratio for r in instruct_results])
        ratio_improvement = (instruct_ratio - base_ratio) / base_ratio * 100

        print(f"\nSaliency ratio improvement: {ratio_improvement:+.1f}%")
        print(f"  Base model:     {base_ratio:.6f}")
        print(f"  Instruct model: {instruct_ratio:.6f}")

        base_rank = np.mean([r.saliency_rank for r in base_results])
        instruct_rank = np.mean([r.saliency_rank for r in instruct_results])
        rank_change = instruct_rank - base_rank

        print(f"\nSaliency rank change: {rank_change:+.0f}")
        print(f"  Base model:     {base_rank:.0f}")
        print(f"  Instruct model: {instruct_rank:.0f}")
        print(f"  (Lower rank = more salient)")

        print(f"\n{'='*80}")
        print("INTERPRETATION:")
        print("="*80)
        if improvement > 10 and ratio_improvement > 10 and rank_change < -10:
            print("✓ STRONG EFFECT: Instruction tuning makes truth significantly more salient!")
            print("  This explains why unsupervised probes work on instruct models but not base.")
        elif improvement > 5 or ratio_improvement > 5:
            print("✓ MODERATE EFFECT: Instruction tuning increases truth saliency somewhat.")
        else:
            print("✗ WEAK/NO EFFECT: Truth saliency similar between base and instruct models.")

    print(f"\n{'='*80}\n")
    print(f"Results saved to: {output_file}")
    print("To visualize, run:")
    print(f"  python experiments/saliency_visualize.py {output_file}")
    print()


if __name__ == "__main__":
    main()
