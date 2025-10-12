"""
Experiment 2: PCA on Probe Vectors

This is a NOVEL approach: instead of doing PCA on activations within a dataset,
we do PCA on the probe vectors themselves across datasets.

Mathematical formulation:
1. Train θ_i = mean(dataset_i.true) - mean(dataset_i.false) for each dataset i
2. Normalize: θ̃_i = θ_i / ||θ_i||
3. Stack: Θ = [θ̃_1; θ̃_2; ...; θ̃_n] ∈ ℝ^(n×d)
4. SVD: Θ = U S V^T
5. Meta-probe: θ_meta = V[:, 0] (first principal component)

Expected results:
- If S[0]/S[1] > 3: Truth is unified, PC1 alone is sufficient
- If S[0]/S[1] ∈ [2,3]: Truth is compositional, need PC1+PC2+PC3
- If S[0]/S[1] < 2: Truth is fragmented, need different approach
"""

import sys
sys.path.insert(0, "/root/repeng")

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pickle

# Import repeng modules
from repeng.activations.probe_preparations import ActivationArrayDataset, ActivationArrays
from repeng.datasets.elk.utils.collections import resolve_dataset_ids
from repeng.datasets.elk.utils.filters import DatasetIdFilter
from repeng.probes.difference_in_means import train_dim_probe
from repeng.probes.base import DotProductProbe
from repeng.evals.probes import eval_probe_by_row, eval_probe_by_question

@dataclass
class ProbeEvaluationResult:
    """Results from evaluating a probe on a dataset."""
    dataset_id: str
    accuracy: float
    n_examples: int
    method: str


def load_activation_dataset() -> ActivationArrayDataset:
    """Load pre-computed activation dataset.

    Note: This requires downloading the cached activations from S3.
    For now, we'll create a simplified version using local data.
    """
    print("="*60)
    print("Loading activation dataset...")
    print("="*60)

    # Try to load from multiple possible locations
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

    # If not found, return empty dataset and note we need to generate it
    print("⚠ No pre-computed activations found")
    print("  You'll need to run: python experiments/comparison_dataset.py")
    print("  Or download from S3: s3://repeng/datasets/activations/")
    return ActivationArrayDataset([])


def train_individual_probes(
    dataset: ActivationArrayDataset,
    llm_id: str = "Qwen/Qwen3-4B",
    layer: int = 21,
) -> Dict[str, np.ndarray]:
    """Train DIM probe on each dataset individually.

    Returns:
        Dictionary mapping dataset_id to probe vector (shape: (d_model,))
    """
    print("\n" + "="*60)
    print("Step 1: Training individual probes on each dataset")
    print("="*60)

    probes = {}

    # Get all datasets
    all_datasets = (
        resolve_dataset_ids("dlk") +
        resolve_dataset_ids("repe") +
        resolve_dataset_ids("got")
    )

    print(f"\nDatasets to process: {len(all_datasets)}")
    print(f"Model: {llm_id}")
    print(f"Layer: {layer}")

    for i, dataset_id in enumerate(all_datasets, 1):
        try:
            print(f"\n[{i}/{len(all_datasets)}] Training on {dataset_id}...", end=" ")

            arrays = dataset.get(
                llm_id=llm_id,
                dataset_filter=DatasetIdFilter(dataset_id),
                split="train",
                point_name=f"h{layer}",
                token_idx=-1,
                limit=None
            )

            probe = train_dim_probe(
                activations=arrays.activations,
                labels=arrays.labels
            )

            probes[dataset_id] = probe.probe
            print(f"✓ (shape={probe.probe.shape}, n_examples={len(arrays.labels)})")

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    print(f"\n✓ Successfully trained {len(probes)} probes")
    return probes


def pca_on_probes(
    probes: Dict[str, np.ndarray],
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Compute PCA on probe vectors.

    Args:
        probes: Dictionary mapping dataset_id to probe vector
        normalize: Whether to normalize probes to unit length before PCA

    Returns:
        U: Left singular vectors (n_datasets, n_datasets)
        S: Singular values (n_datasets,)
        Vt: Right singular vectors (n_datasets, d_model)
        dataset_names: List of dataset names in order
    """
    print("\n" + "="*60)
    print("Step 2: PCA on Probe Vectors")
    print("="*60)

    # Stack probes
    dataset_names = list(probes.keys())
    probe_list = [probes[name] for name in dataset_names]

    print(f"\nNumber of probes: {len(probe_list)}")
    print(f"Probe dimension: {probe_list[0].shape[0]}")

    # Normalize if requested
    if normalize:
        print("Normalizing probes to unit length...")
        probe_list = [p / np.linalg.norm(p) for p in probe_list]

    # Stack into matrix (n_datasets, d_model)
    Theta = np.stack(probe_list)
    print(f"Probe matrix shape: {Theta.shape}")

    # Compute SVD
    print("Computing SVD...")
    U, S, Vt = np.linalg.svd(Theta, full_matrices=False)

    # Analysis
    print("\n" + "-"*60)
    print("Singular Value Analysis")
    print("-"*60)
    print(f"S[0]/S[1] = {S[0]/S[1]:.3f}")
    print(f"S[1]/S[2] = {S[1]/S[2]:.3f}")
    print(f"S[2]/S[3] = {S[2]/S[3]:.3f}")

    # Interpretation
    if S[0]/S[1] > 3:
        print("\n✓ Interpretation: Truth appears UNIFIED (S[0]/S[1] > 3)")
        print("  → Single shared direction is sufficient")
        print("  → Use PC1 only for meta-probe")
    elif S[0]/S[1] > 2:
        print("\n⚠ Interpretation: Truth is PARTIALLY COMPOSITIONAL (S[0]/S[1] ∈ [2,3])")
        print("  → 2-3 meaningful components")
        print("  → Need adaptive weighting of PC1, PC2, PC3")
    else:
        print("\n✗ Interpretation: Truth is FRAGMENTED (S[0]/S[1] < 2)")
        print("  → No single shared direction")
        print("  → May need larger model (70B) or different approach")

    # Variance explained
    print("\n" + "-"*60)
    print("Variance Explained")
    print("-"*60)
    variance_explained = S**2 / np.sum(S**2)
    cumsum = np.cumsum(variance_explained)

    for i in range(min(5, len(S))):
        print(f"PC{i+1}: {variance_explained[i]:6.1%}  (cumulative: {cumsum[i]:6.1%})")

    print(f"\nTop 5 singular values: {S[:5]}")

    return U, S, Vt, dataset_names


def interpret_components(
    probes: Dict[str, np.ndarray],
    U: np.ndarray,
    S: np.ndarray,
    Vt: np.ndarray,
    dataset_names: List[str],
    n_components: int = 3
):
    """Interpret what each principal component captures.

    This analysis reveals:
    1. Which datasets load heavily on each component
    2. Whether components correspond to dataset families
    3. What semantic meaning each component has
    """
    print("\n" + "="*60)
    print("Step 3: Component Interpretation")
    print("="*60)

    # Normalize probes for correlation computation
    probe_list = [probes[name] / np.linalg.norm(probes[name]) for name in dataset_names]

    # Get dataset families
    DLK_DATASETS = resolve_dataset_ids("dlk")
    REPE_DATASETS = resolve_dataset_ids("repe")
    GOT_DATASETS = resolve_dataset_ids("got")

    for i in range(n_components):
        component = Vt[i]

        print("\n" + "="*60)
        print(f"Component {i+1} (explains {S[i]**2/np.sum(S**2):.1%} variance)")
        print("="*60)

        # Compute correlations with each probe
        correlations = {}
        for j, dataset in enumerate(dataset_names):
            # Correlation = dot product (since both normalized)
            corr = component @ probe_list[j]
            correlations[dataset] = corr

        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(),
                            key=lambda x: abs(x[1]),
                            reverse=True)

        print(f"\nTop 5 highest loading (positive):")
        for dataset, corr in sorted_corr[:5]:
            family = "DLK" if dataset in DLK_DATASETS else "RepE" if dataset in REPE_DATASETS else "GoT"
            print(f"  {dataset:30s} {corr:+.3f}  [{family}]")

        print(f"\nTop 5 highest loading (negative):")
        for dataset, corr in sorted_corr[-5:]:
            family = "DLK" if dataset in DLK_DATASETS else "RepE" if dataset in REPE_DATASETS else "GoT"
            print(f"  {dataset:30s} {corr:+.3f}  [{family}]")

        # Family analysis
        dlk_corrs = [correlations[d] for d in DLK_DATASETS if d in correlations]
        repe_corrs = [correlations[d] for d in REPE_DATASETS if d in correlations]
        got_corrs = [correlations[d] for d in GOT_DATASETS if d in correlations]

        print(f"\nFamily loadings:")
        print(f"  DLK:  {np.mean(dlk_corrs):+.3f} ± {np.std(dlk_corrs):.3f}  (n={len(dlk_corrs)})")
        print(f"  RepE: {np.mean(repe_corrs):+.3f} ± {np.std(repe_corrs):.3f}  (n={len(repe_corrs)})")
        print(f"  GoT:  {np.mean(got_corrs):+.3f} ± {np.std(got_corrs):.3f}  (n={len(got_corrs)})")

        # Semantic interpretation hint
        if i == 0:
            print(f"\nExpected: Universal truth component")
            print(f"  Should have positive loading across all families")
        elif i == 1:
            print(f"\nExpected: Factual vs sentiment component")
            print(f"  Factual datasets should be positive, sentiment negative")
        elif i == 2:
            print(f"\nExpected: Format component (Q&A vs statements)")
            print(f"  Q&A format should be positive, statement format negative")


def evaluate_meta_probes(
    dataset: ActivationArrayDataset,
    Vt: np.ndarray,
    S: np.ndarray,
    dataset_names: List[str],
    llm_id: str = "Qwen/Qwen3-4B",
    layer: int = 21,
) -> Dict[str, Dict[str, float]]:
    """Evaluate different meta-probe configurations.

    Tests:
    1. PC1 only
    2. PC1 + PC2 (equal weight)
    3. PC1 + PC2 + PC3 (equal weight)
    4. PC1 + PC2 + PC3 (weighted by singular values)

    Returns:
        Dictionary mapping method_name → {dataset_id → accuracy}
    """
    print("\n" + "="*60)
    print("Step 4: Evaluating Meta-Probes")
    print("="*60)

    # Create meta-probes
    meta_probes = {
        "pc1_only": Vt[0],
        "pc1_pc2_equal": (Vt[0] + Vt[1]) / 2,
        "pc1_pc2_pc3_equal": (Vt[0] + Vt[1] + Vt[2]) / 3,
        "pc1_pc2_pc3_weighted": (S[0]*Vt[0] + S[1]*Vt[1] + S[2]*Vt[2]) / (S[0]+S[1]+S[2]),
    }

    results = {method: {} for method in meta_probes.keys()}

    print(f"\nEvaluating {len(meta_probes)} meta-probe configurations...")
    print(f"on {len(dataset_names)} datasets\n")

    for method, meta_probe in meta_probes.items():
        print(f"\n{method}:")
        probe_obj = DotProductProbe(meta_probe)

        method_accuracies = []

        for dataset_id in dataset_names:
            try:
                arrays = dataset.get(
                    llm_id=llm_id,
                    dataset_filter=DatasetIdFilter(dataset_id),
                    split="validation",
                    point_name=f"h{layer}",
                    token_idx=-1,
                    limit=None
                )

                # Evaluate
                if arrays.groups is not None:
                    result = eval_probe_by_question(
                        probe_obj,
                        activations=arrays.activations,
                        labels=arrays.labels,
                        groups=arrays.groups
                    )
                else:
                    result = eval_probe_by_row(
                        probe_obj,
                        activations=arrays.activations,
                        labels=arrays.labels
                    )

                results[method][dataset_id] = result.accuracy
                method_accuracies.append(result.accuracy)

            except Exception as e:
                print(f"  ✗ {dataset_id}: Error - {e}")
                continue

        avg_acc = np.mean(method_accuracies)
        std_acc = np.std(method_accuracies)
        print(f"  Average: {avg_acc:.1%} ± {std_acc:.1%}")

    return results


def main():
    """Run the complete PCA on probe vectors experiment."""

    print("\n" + "="*80)
    print(" "*20 + "PCA ON PROBE VECTORS EXPERIMENT")
    print(" "*20 + "(NOVEL METHOD)")
    print("="*80)
    print("\nThis experiments with operating on probe vectors themselves")
    print("rather than on activations within a dataset.")
    print("\nExpected outcomes:")
    print("- If S[0]/S[1] > 3: Truth is unified → PC1 alone works")
    print("- If S[0]/S[1] ∈ [2,3]: Truth is compositional → need PC1+PC2+PC3")
    print("- If S[0]/S[1] < 2: Truth is fragmented → need different approach")
    print("="*80)

    # Load dataset
    dataset = load_activation_dataset()

    if len(dataset.rows) == 0:
        print("\n✗ No activation data available. Exiting.")
        print("\nTo generate activations, run:")
        print("  python experiments/comparison_dataset.py")
        return

    # Step 1: Train individual probes
    probes = train_individual_probes(dataset)

    if len(probes) < 3:
        print(f"\n✗ Only {len(probes)} probes trained. Need at least 3 for PCA.")
        return

    # Step 2: PCA on probe vectors
    U, S, Vt, dataset_names = pca_on_probes(probes, normalize=True)

    # Step 3: Interpret components
    interpret_components(probes, U, S, Vt, dataset_names, n_components=3)

    # Step 4: Evaluate meta-probes
    results = evaluate_meta_probes(dataset, Vt, S, dataset_names)

    # Save results
    output_dir = Path("/root/repeng/output/probe_ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "probe_vector_pca_results.pkl"
    print(f"\n{'='*60}")
    print(f"Saving results to: {output_file}")
    print(f"{'='*60}")

    with open(output_file, "wb") as f:
        pickle.dump({
            "probes": probes,
            "U": U,
            "S": S,
            "Vt": Vt,
            "dataset_names": dataset_names,
            "results": results,
        }, f)

    print("✓ Results saved successfully")

    # Summary
    print("\n" + "="*80)
    print(" "*30 + "SUMMARY")
    print("="*80)
    print(f"\nProbes trained: {len(probes)}")
    print(f"Singular value ratio S[0]/S[1]: {S[0]/S[1]:.3f}")
    print(f"Variance explained by PC1: {S[0]**2/np.sum(S**2):.1%}")
    print(f"\nMeta-probe performance:")
    for method, dataset_results in results.items():
        avg = np.mean(list(dataset_results.values()))
        print(f"  {method:25s}: {avg:.1%}")
    print("="*80)


if __name__ == "__main__":
    main()
