"""
Experiment 1: Multi-Dataset DIM Concatenation

Tests whether training on multiple datasets improves generalization through
noise averaging: θ_ensemble = mean(θ_i) = θ_true + mean(n_i) ≈ θ_true

Expected improvement: √N reduction in noise
- N=1 (baseline): 71% ± 2%
- N=5: 77% ± 2% (+6 points)
- N=10: 76% ± 2% (diminishing returns from family mixing)

Success criteria:
✓ N=5 beats N=1 by >3 points: Validates noise-averaging hypothesis
⚠ N=5 achieves 73-75%: Modest improvement, check family effects
✗ N=5 ≤ 72%: Truth not shared, need different approach
"""

import sys
sys.path.insert(0, "/root/repeng")

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
import pickle

from repeng.activations.probe_preparations import ActivationArrayDataset
from repeng.datasets.elk.utils.collections import resolve_dataset_ids
from repeng.datasets.elk.utils.filters import DatasetIdFilter
from repeng.probes.difference_in_means import train_dim_probe
from repeng.probes.base import DotProductProbe
from repeng.evals.probes import eval_probe_by_row, eval_probe_by_question


@dataclass
class ExperimentResult:
    """Results from a single concatenation experiment."""
    n_datasets: int
    dataset_combination: List[str]
    train_accuracy: float
    test_accuracies: Dict[str, float]
    avg_test_accuracy: float
    probe: np.ndarray


def load_activation_dataset() -> ActivationArrayDataset:
    """Load pre-computed activation dataset."""
    print("="*60)
    print("Loading activation dataset...")
    print("="*60)

    possible_paths = [
        Path("/root/repeng/output/comparison/activations_results/value.pickle"),
        Path("/root/repeng/output/create-activations-dataset/activations/value.pickle"),
    ]

    for path in possible_paths:
        if path.exists():
            print(f"✓ Found activations at: {path}")
            with open(path, "rb") as f:
                activation_results = pickle.load(f)
            return ActivationArrayDataset(activation_results)

    print("⚠ No pre-computed activations found")
    return ActivationArrayDataset([])


def get_dataset_combinations() -> Dict[int, List[str]]:
    """Get strategically chosen dataset combinations.

    Strategy:
    - N=1: Best single dataset (dbpedia_14)
    - N=2: Best + anomaly (got_cities_cities_conj)
    - N=3: One from each family
    - N=5: Diverse selection
    - N=10: Balanced across families
    - N=18: All datasets
    """
    DLK_DATASETS = resolve_dataset_ids("dlk")
    REPE_DATASETS = resolve_dataset_ids("repe")
    GOT_DATASETS = resolve_dataset_ids("got")

    return {
        1: ["dbpedia_14"],  # Best single
        2: ["dbpedia_14", "got_cities_cities_conj"],  # Best + anomaly
        3: ["dbpedia_14", "got_cities_cities_conj", "open_book_qa"],  # One per family
        5: ["dbpedia_14", "got_cities_cities_conj", "open_book_qa",
            "amazon_polarity", "got_larger_than"],  # Diverse
        8: ["dbpedia_14", "amazon_polarity", "ag_news",  # DLK
            "open_book_qa", "arc_easy",  # RepE
            "got_cities", "got_cities_cities_conj", "got_larger_than"],  # GoT
        DLK_DATASETS + REPE_DATASETS + GOT_DATASETS:
            DLK_DATASETS + REPE_DATASETS + GOT_DATASETS,  # All datasets
    }


def train_concatenated_probe(
    dataset: ActivationArrayDataset,
    dataset_ids: List[str],
    llm_id: str = "Qwen/Qwen3-4B",
    layer: int = 21,
) -> DotProductProbe:
    """Train DIM probe on concatenated data from multiple datasets.

    Args:
        dataset: Activation dataset
        dataset_ids: List of dataset IDs to concatenate
        llm_id: Model identifier
        layer: Layer number

    Returns:
        Trained probe
    """
    print(f"\nTraining on {len(dataset_ids)} datasets:")
    for ds in dataset_ids:
        print(f"  - {ds}")

    all_activations = []
    all_labels = []

    for dataset_id in dataset_ids:
        try:
            arrays = dataset.get(
                llm_id=llm_id,
                dataset_filter=DatasetIdFilter(dataset_id),
                split="train",
                point_name=f"h{layer}",
                token_idx=-1,
                limit=None
            )

            all_activations.append(arrays.activations)
            all_labels.append(arrays.labels)
            print(f"  ✓ {dataset_id}: {len(arrays.labels)} examples")

        except Exception as e:
            print(f"  ✗ {dataset_id}: Error - {e}")
            continue

    if not all_activations:
        raise ValueError("No datasets loaded successfully")

    # Concatenate
    activations_concat = np.concatenate(all_activations, axis=0)
    labels_concat = np.concatenate(all_labels, axis=0)

    print(f"\nConcatenated: {activations_concat.shape[0]} total examples")

    # Train probe
    probe = train_dim_probe(
        activations=activations_concat,
        labels=labels_concat
    )

    return probe


def evaluate_probe(
    probe: DotProductProbe,
    dataset: ActivationArrayDataset,
    test_dataset_ids: List[str],
    llm_id: str = "Qwen/Qwen3-4B",
    layer: int = 21,
) -> Dict[str, float]:
    """Evaluate probe on multiple test datasets.

    Returns:
        Dictionary mapping dataset_id → accuracy
    """
    results = {}

    for dataset_id in test_dataset_ids:
        try:
            arrays = dataset.get(
                llm_id=llm_id,
                dataset_filter=DatasetIdFilter(dataset_id),
                split="validation",
                point_name=f"h{layer}",
                token_idx=-1,
                limit=None
            )

            if arrays.groups is not None:
                result = eval_probe_by_question(
                    probe,
                    activations=arrays.activations,
                    labels=arrays.labels,
                    groups=arrays.groups
                )
            else:
                result = eval_probe_by_row(
                    probe,
                    activations=arrays.activations,
                    labels=arrays.labels
                )

            results[dataset_id] = result.accuracy

        except Exception as e:
            print(f"  ✗ {dataset_id}: Error - {e}")
            results[dataset_id] = 0.0

    return results


def run_experiment(
    dataset: ActivationArrayDataset,
    n_datasets: int,
    dataset_ids: List[str],
    all_test_datasets: List[str],
) -> ExperimentResult:
    """Run a single concatenation experiment."""

    print("\n" + "="*60)
    print(f"Experiment: N = {n_datasets} datasets")
    print("="*60)

    # Train probe
    probe = train_concatenated_probe(dataset, dataset_ids)

    # Evaluate on all test datasets
    print(f"\nEvaluating on {len(all_test_datasets)} test datasets...")
    test_results = evaluate_probe(probe, dataset, all_test_datasets)

    # Compute statistics
    valid_results = [acc for acc in test_results.values() if acc > 0]
    avg_acc = np.mean(valid_results) if valid_results else 0.0

    print(f"\n{'-'*60}")
    print(f"Results for N={n_datasets}:")
    print(f"  Average accuracy: {avg_acc:.1%}")
    print(f"  Valid evaluations: {len(valid_results)}/{len(all_test_datasets)}")
    print(f"{'-'*60}")

    # Calculate within-family and cross-family accuracy
    DLK_DATASETS = resolve_dataset_ids("dlk")
    REPE_DATASETS = resolve_dataset_ids("repe")
    GOT_DATASETS = resolve_dataset_ids("got")

    within_family = []
    cross_family = []

    for train_ds in dataset_ids:
        for test_ds, acc in test_results.items():
            if acc == 0:
                continue

            # Determine families
            train_family = ("DLK" if train_ds in DLK_DATASETS else
                          "RepE" if train_ds in REPE_DATASETS else "GoT")
            test_family = ("DLK" if test_ds in DLK_DATASETS else
                          "RepE" if test_ds in REPE_DATASETS else "GoT")

            if train_family == test_family:
                within_family.append(acc)
            else:
                cross_family.append(acc)

    if within_family:
        print(f"  Within-family: {np.mean(within_family):.1%} (n={len(within_family)})")
    if cross_family:
        print(f"  Cross-family:  {np.mean(cross_family):.1%} (n={len(cross_family)})")

    return ExperimentResult(
        n_datasets=n_datasets,
        dataset_combination=dataset_ids,
        train_accuracy=0.0,  # Could compute but not critical
        test_accuracies=test_results,
        avg_test_accuracy=avg_acc,
        probe=probe.probe
    )


def main():
    """Run the complete multi-dataset concatenation experiment."""

    print("\n" + "="*80)
    print(" "*15 + "MULTI-DATASET DIM CONCATENATION EXPERIMENT")
    print("="*80)
    print("\nHypothesis: Training on N datasets reduces noise by √N")
    print("\nExpected results:")
    print("  N=1:  71% ± 2% (baseline)")
    print("  N=5:  77% ± 2% (+6 points from noise reduction)")
    print("  N=10: 76% ± 2% (diminishing returns)")
    print("\nSuccess criterion: N=5 beats N=1 by >3 points")
    print("="*80)

    # Load dataset
    dataset = load_activation_dataset()

    if len(dataset.rows) == 0:
        print("\n✗ No activation data available. Exiting.")
        return

    # Get all test datasets
    all_test_datasets = (
        resolve_dataset_ids("dlk") +
        resolve_dataset_ids("repe") +
        resolve_dataset_ids("got")
    )

    # Get dataset combinations
    combinations = get_dataset_combinations()

    # Run experiments
    results = []

    for n, dataset_ids in sorted(combinations.items()):
        try:
            result = run_experiment(dataset, n, dataset_ids, all_test_datasets)
            results.append(result)
        except Exception as e:
            print(f"\n✗ Error in N={n} experiment: {e}")
            continue

    # Save results
    output_dir = Path("/root/repeng/output/probe_ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "multi_dataset_results.pkl"
    print(f"\n{'='*60}")
    print(f"Saving results to: {output_file}")
    print(f"{'='*60}")

    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    # Summary
    print("\n" + "="*80)
    print(" "*30 + "SUMMARY")
    print("="*80)
    print(f"\n{'N':>3s} {'Datasets':<50s} {'Avg Acc':>10s}")
    print("-"*80)

    for result in results:
        datasets_str = ", ".join(result.dataset_combination[:3])
        if len(result.dataset_combination) > 3:
            datasets_str += f", ... (+{len(result.dataset_combination)-3})"
        print(f"{result.n_datasets:3d} {datasets_str:<50s} {result.avg_test_accuracy:9.1%}")

    print("="*80)

    # Analysis
    if len(results) >= 2:
        baseline = results[0].avg_test_accuracy
        for result in results[1:]:
            improvement = result.avg_test_accuracy - baseline
            print(f"\nN={result.n_datasets} vs N=1: {improvement:+.1%} improvement")

            if result.n_datasets == 5:
                if improvement > 0.03:
                    print("  ✓ SUCCESS: >3% improvement validates noise-averaging hypothesis")
                elif improvement > 0.01:
                    print("  ⚠ PARTIAL: Modest improvement, investigate family effects")
                else:
                    print("  ✗ FAILURE: No significant improvement, truth may not be shared")


if __name__ == "__main__":
    main()
