#!/usr/bin/env python3
"""
Master Script: Run All Truth Probe Generalization Experiments

This script orchestrates the complete experimental pipeline:
1. Verify activation data availability
2. Run Experiment 1: Multi-dataset DIM concatenation
3. Run Experiment 2: PCA on probe vectors (NOVEL)
4. Generate comprehensive analysis and visualizations

Usage:
    python experiments/probe_ensemble/run_all_experiments.py
"""

import sys
sys.path.insert(0, "/root/repeng")

import pickle
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import traceback

# Import experiment modules
from repeng.activations.probe_preparations import ActivationArrayDataset

print("\n" + "="*80)
print(" "*20 + "TRUTH PROBE GENERALIZATION")
print(" "*20 + "Complete Experimental Pipeline")
print("="*80)

# ============================================================================
# STEP 0: Check for Activation Data
# ============================================================================

def check_activation_data() -> Tuple[bool, ActivationArrayDataset]:
    """Check if activation data is available."""

    print("\n" + "="*80)
    print("STEP 0: Checking for Activation Data")
    print("="*80)

    # Possible locations for cached activations
    possible_paths = [
        Path("/root/repeng/output/comparison/activations_results/value.pickle"),
        Path("/root/repeng/output/create-activations-dataset/activations/value.pickle"),
        Path("output/comparison/activations_results/value.pickle"),
        Path("output/create-activations-dataset/activations/value.pickle"),
    ]

    for path in possible_paths:
        if path.exists():
            print(f"\n✓ Found activations at: {path}")
            try:
                with open(path, "rb") as f:
                    activation_results = pickle.load(f)

                dataset = ActivationArrayDataset(activation_results)
                n_rows = len(dataset.rows)

                if n_rows > 0:
                    print(f"✓ Loaded {n_rows} activation records")

                    # Check what datasets we have
                    datasets_available = set(row.dataset_id for row in dataset.rows)
                    print(f"✓ Datasets available: {len(datasets_available)}")
                    print(f"  {', '.join(sorted(list(datasets_available)[:5]))}...")

                    return True, dataset
                else:
                    print(f"⚠ Dataset loaded but empty")
            except Exception as e:
                print(f"✗ Error loading {path}: {e}")
                continue

    print("\n" + "="*80)
    print("✗ NO ACTIVATION DATA FOUND")
    print("="*80)
    print("\nTo run experiments, you need to either:")
    print("\n1. Generate activations (recommended for new experiments):")
    print("   python experiments/comparison_dataset.py")
    print("   Time: ~2-4 hours on A40 GPU")
    print("   Output: ~150MB of activations for 18 datasets")
    print("\n2. Download pre-computed activations from S3:")
    print("   aws s3 cp s3://repeng/datasets/activations/datasets_2024-02-14_v1.pickle \\")
    print("     output/comparison/activations_results/value.pickle")
    print("\n3. Use a subset (for quick testing):")
    print("   python experiments/probe_ensemble/generate_subset_activations.py")
    print("="*80)

    return False, ActivationArrayDataset([])


# ============================================================================
# STEP 1: Multi-Dataset Concatenation
# ============================================================================

def run_multi_dataset_experiment(dataset: ActivationArrayDataset) -> Dict:
    """Run multi-dataset concatenation experiment."""

    print("\n" + "="*80)
    print("STEP 1: Multi-Dataset DIM Concatenation")
    print("="*80)
    print("\nHypothesis: Training on N datasets reduces noise by √N")
    print("Expected: N=5 should achieve ~77% (vs 71% baseline)")

    try:
        # Import the module
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "multi_dataset_training",
            "/root/repeng/experiments/probe_ensemble/multi_dataset_training.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get dataset combinations (exclude piqa - deprecated HF script)
        from repeng.datasets.elk.utils.collections import resolve_dataset_ids
        all_test_datasets = [
            ds for ds in (
                resolve_dataset_ids("dlk") +
                resolve_dataset_ids("repe") +
                resolve_dataset_ids("got")
            ) if ds != "piqa"
        ]

        combinations = {
            1: ["dbpedia_14"],
            2: ["dbpedia_14", "got_cities_cities_conj"],
            3: ["dbpedia_14", "got_cities_cities_conj", "open_book_qa"],
            5: ["dbpedia_14", "got_cities_cities_conj", "open_book_qa",
                "amazon_polarity", "got_larger_than"],
        }

        results = []
        for n, dataset_ids in sorted(combinations.items()):
            try:
                print(f"\n{'-'*80}")
                print(f"Testing N={n} datasets")
                print(f"{'-'*80}")
                result = module.run_experiment(dataset, n, dataset_ids, all_test_datasets)
                results.append(result)
                print(f"✓ N={n}: {result.avg_test_accuracy:.1%} average accuracy")
            except Exception as e:
                print(f"✗ Error in N={n}: {e}")
                traceback.print_exc()
                continue

        # Save results
        output_file = Path("/root/repeng/output/probe_ensemble/multi_dataset_results.pkl")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "wb") as f:
            pickle.dump(results, f)

        print(f"\n✓ Results saved to: {output_file}")

        # Analysis
        if len(results) >= 2:
            print("\n" + "="*80)
            print("ANALYSIS: Multi-Dataset Concatenation")
            print("="*80)

            baseline = results[0].avg_test_accuracy
            print(f"\nBaseline (N=1): {baseline:.1%}")

            for result in results[1:]:
                improvement = result.avg_test_accuracy - baseline
                print(f"N={result.n_datasets}: {result.avg_test_accuracy:.1%} ({improvement:+.1%})")

                if result.n_datasets == 5:
                    if improvement > 0.03:
                        print("  ✓ SUCCESS: >3% improvement validates noise-averaging!")
                        status = "success"
                    elif improvement > 0.01:
                        print("  ⚠ PARTIAL: Modest improvement, check family effects")
                        status = "partial"
                    else:
                        print("  ✗ FAILURE: No improvement, truth may not be shared")
                        status = "failure"

        return {"status": "complete", "results": results}

    except Exception as e:
        print(f"\n✗ Experiment 1 failed: {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


# ============================================================================
# STEP 2: PCA on Probe Vectors (NOVEL)
# ============================================================================

def run_pca_experiment(dataset: ActivationArrayDataset) -> Dict:
    """Run PCA on probe vectors experiment."""

    print("\n" + "="*80)
    print("STEP 2: PCA on Probe Vectors (COMPLETELY NOVEL)")
    print("="*80)
    print("\nHypothesis: PC1 of probe vectors extracts consensus truth direction")
    print("Expected: S[0]/S[1] ∈ [2,3] indicates partial compositional structure")

    try:
        # Import the module
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "probe_vector_pca",
            "/root/repeng/experiments/probe_ensemble/probe_vector_pca.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Train individual probes
        print("\n[2.1] Training individual probes...")
        probes = module.train_individual_probes(dataset)

        if len(probes) < 3:
            print(f"✗ Only {len(probes)} probes trained, need at least 3")
            return {"status": "insufficient_data"}

        print(f"✓ Trained {len(probes)} probes")

        # PCA on probe vectors
        print("\n[2.2] Computing PCA on probe vectors...")
        U, S, Vt, dataset_names = module.pca_on_probes(probes, normalize=True)

        # Interpret components
        print("\n[2.3] Interpreting components...")
        module.interpret_components(probes, U, S, Vt, dataset_names, n_components=3)

        # Evaluate meta-probes
        print("\n[2.4] Evaluating meta-probes...")
        results = module.evaluate_meta_probes(dataset, Vt, S, dataset_names)

        # Save results
        output_file = Path("/root/repeng/output/probe_ensemble/probe_vector_pca_results.pkl")
        with open(output_file, "wb") as f:
            pickle.dump({
                "probes": probes,
                "U": U,
                "S": S,
                "Vt": Vt,
                "dataset_names": dataset_names,
                "results": results,
            }, f)

        print(f"\n✓ Results saved to: {output_file}")

        # Analysis
        print("\n" + "="*80)
        print("ANALYSIS: PCA on Probe Vectors")
        print("="*80)

        ratio = S[0]/S[1]
        variance_pc1 = S[0]**2 / np.sum(S**2)

        print(f"\nSingular value ratio S[0]/S[1]: {ratio:.3f}")
        print(f"Variance explained by PC1: {variance_pc1:.1%}")

        if ratio > 3.0:
            print("\n✓ INTERPRETATION: Truth is UNIFIED")
            print("  → Single shared direction exists")
            print("  → Use PC1 alone for meta-probe")
            interpretation = "unified"
        elif ratio > 2.0:
            print("\n⚠ INTERPRETATION: Truth is PARTIALLY COMPOSITIONAL")
            print("  → 2-3 meaningful components")
            print("  → Need adaptive weighting of PC1, PC2, PC3")
            interpretation = "compositional"
        else:
            print("\n✗ INTERPRETATION: Truth is FRAGMENTED")
            print("  → No single shared direction")
            print("  → May need 70B model or different approach")
            interpretation = "fragmented"

        print("\nMeta-probe performance:")
        for method, dataset_results in results.items():
            valid_accs = [acc for acc in dataset_results.values() if acc > 0]
            if valid_accs:
                avg = np.mean(valid_accs)
                print(f"  {method:25s}: {avg:.1%}")

        return {
            "status": "complete",
            "interpretation": interpretation,
            "singular_value_ratio": ratio,
            "variance_pc1": variance_pc1,
            "results": results
        }

    except Exception as e:
        print(f"\n✗ Experiment 2 failed: {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


# ============================================================================
# STEP 3: Generate Summary
# ============================================================================

def generate_summary(exp1_results: Dict, exp2_results: Dict):
    """Generate final summary and interpretation."""

    print("\n" + "="*80)
    print(" "*30 + "FINAL SUMMARY")
    print("="*80)

    print("\n" + "-"*80)
    print("Experiment 1: Multi-Dataset Concatenation")
    print("-"*80)

    if exp1_results.get("status") == "complete":
        results = exp1_results["results"]
        if len(results) >= 2:
            baseline = results[0].avg_test_accuracy
            print(f"Baseline (N=1): {baseline:.1%}")
            for r in results[1:]:
                improvement = r.avg_test_accuracy - baseline
                symbol = "✓" if improvement > 0.03 else "⚠" if improvement > 0.01 else "✗"
                print(f"N={r.n_datasets}: {r.avg_test_accuracy:.1%} ({improvement:+.1%}) {symbol}")
        else:
            print("Insufficient results for comparison")
    else:
        print(f"Status: {exp1_results.get('status', 'unknown')}")

    print("\n" + "-"*80)
    print("Experiment 2: PCA on Probe Vectors (NOVEL)")
    print("-"*80)

    if exp2_results.get("status") == "complete":
        print(f"S[0]/S[1]: {exp2_results['singular_value_ratio']:.3f}")
        print(f"PC1 variance: {exp2_results['variance_pc1']:.1%}")
        print(f"Interpretation: {exp2_results['interpretation'].upper()}")

        if exp2_results.get('results'):
            print("\nMeta-probe accuracies:")
            for method, dataset_results in exp2_results['results'].items():
                valid = [acc for acc in dataset_results.values() if acc > 0]
                if valid:
                    print(f"  {method}: {np.mean(valid):.1%}")
    else:
        print(f"Status: {exp2_results.get('status', 'unknown')}")

    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    findings = []

    # Finding 1: Does multi-dataset help?
    if exp1_results.get("status") == "complete" and len(exp1_results["results"]) >= 2:
        baseline = exp1_results["results"][0].avg_test_accuracy
        n5_result = next((r for r in exp1_results["results"] if r.n_datasets == 5), None)
        if n5_result:
            improvement = n5_result.avg_test_accuracy - baseline
            if improvement > 0.03:
                findings.append("✓ Multi-dataset training works (+{:.1%})".format(improvement))
            else:
                findings.append("⚠ Multi-dataset training shows limited improvement")

    # Finding 2: Is truth unified?
    if exp2_results.get("status") == "complete":
        ratio = exp2_results["singular_value_ratio"]
        if ratio > 3:
            findings.append("✓ Truth is unified at 13B (S[0]/S[1] > 3)")
        elif ratio > 2:
            findings.append("⚠ Truth is partially compositional (S[0]/S[1] ∈ [2,3])")
        else:
            findings.append("✗ Truth is fragmented (S[0]/S[1] < 2)")

    # Finding 3: Novel method
    if exp2_results.get("status") == "complete":
        findings.append("✓ PCA on probe vectors is novel and implementable")

    for i, finding in enumerate(findings, 1):
        print(f"{i}. {finding}")

    # Next steps
    print("\n" + "="*80)
    print("RECOMMENDED NEXT STEPS")
    print("="*80)

    if exp2_results.get("interpretation") == "unified":
        print("1. Use PC1 as meta-probe for deployment")
        print("2. Test on 70B model (expect S[0]/S[1] > 4)")
        print("3. Apply to misalignment detection")
    elif exp2_results.get("interpretation") == "compositional":
        print("1. Develop adaptive multi-component probe")
        print("2. Investigate what PC2 and PC3 capture")
        print("3. Test context-dependent weighting")
    else:
        print("1. Investigate why truth is fragmented")
        print("2. Consider testing on 70B model")
        print("3. May need non-linear probes")

    print("\n" + "="*80)
    print("Experiments complete!")
    print("="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all experiments."""

    # Step 0: Check data
    has_data, dataset = check_activation_data()

    if not has_data:
        print("\n⚠ Cannot proceed without activation data")
        print("Please follow instructions above to obtain data")
        sys.exit(1)

    # Step 1: Multi-dataset concatenation
    exp1_results = run_multi_dataset_experiment(dataset)

    # Step 2: PCA on probe vectors
    exp2_results = run_pca_experiment(dataset)

    # Step 3: Summary
    generate_summary(exp1_results, exp2_results)

    print("\n✓ All experiments complete!")
    print(f"Results saved to: /root/repeng/output/probe_ensemble/")


if __name__ == "__main__":
    main()
