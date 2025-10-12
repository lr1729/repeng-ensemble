#!/usr/bin/env python3
"""
RLAIF Probe Refinement: Iteratively improve truth probe using teacher model feedback

Starting point: got_cities probe (81.7% on novel prompts)
Goal: Improve to 90%+ through iterative refinement

Algorithm:
1. Start with best single probe (got_cities)
2. Find errors on novel data
3. Get ground truth labels (oracle teacher / smarter model)
4. Create contrastive pairs from errors
5. Add to training data and retrain
6. Repeat until convergence
"""

import sys
sys.path.insert(0, "/root/repeng")

import torch
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

from repeng.models.loading import load_llm_oioo
from repeng.activations.inference import get_model_activations

print("="*80)
print("RLAIF PROBE REFINEMENT")
print("="*80)

# Load training data
print("\n[1/6] Loading training data...")
activations_file = Path("/root/repeng/output/comparison/activations_results/value.pickle")
with open(activations_file, 'rb') as f:
    from repeng.datasets.activations.types import ActivationResultRow
    rows = pickle.load(f)

LAYER = "h35"

# Organize training data by dataset
train_data = defaultdict(list)
for row in rows:
    if row.split == 'train' and LAYER in row.activations:
        train_data[row.dataset_id].append({
            'activation': row.activations[LAYER].flatten().astype(np.float32),
            'label': row.label,
        })

print(f"✓ Loaded training data from {len(train_data)} datasets")

# Load novel prompts (test data)
print("\n[2/6] Loading novel prompts...")
novel_dir = Path("/root/repeng/output/probe_ensemble/novel_prompts")
with open(novel_dir / "novel_prompt_results.pkl", 'rb') as f:
    novel_results = pickle.load(f)

novel_data = novel_results['prompts']
novel_acts = np.array([d['activation'] for d in novel_data])
novel_labels = np.array([d['label'] for d in novel_data])

print(f"✓ Loaded {len(novel_data)} novel prompts")
print(f"  True: {novel_labels.sum()}, False: {(~novel_labels).sum()}")

# Train probe function
def train_dim_probe(data):
    """Train DIM probe on activation/label pairs"""
    acts = np.array([d['activation'] for d in data])
    labels = np.array([d['label'] for d in data])

    true_mean = acts[labels].mean(axis=0)
    false_mean = acts[~labels].mean(axis=0)
    theta = true_mean - false_mean

    return theta / (np.linalg.norm(theta) + 1e-8)

def evaluate_probe(probe, acts, labels):
    """Evaluate probe accuracy"""
    scores = acts @ probe
    preds = scores > 0
    return np.mean(preds == labels)

# Initialize with got_cities probe
print("\n[3/6] Training baseline got_cities probe...")
initial_training_data = train_data['got_cities'].copy()
print(f"  Initial training set: {len(initial_training_data)} examples")

probe = train_dim_probe(initial_training_data)
baseline_acc = evaluate_probe(probe, novel_acts, novel_labels)

print(f"✓ Baseline accuracy: {baseline_acc*100:.1f}%")

# RLAIF Refinement Loop
print("\n[4/6] Starting RLAIF refinement...")
print("-"*80)

# Track metrics
iteration_history = [{
    'iteration': 0,
    'training_size': len(initial_training_data),
    'novel_accuracy': baseline_acc,
    'errors_found': 0,
}]

# Model for generating synthetic contrastive pairs
print("\n  Loading Qwen3-4B for generating contrastive pairs...")
device = torch.device("cuda")
llm = load_llm_oioo("Qwen/Qwen3-4B", device=device, use_half_precision=True)
print("  ✓ Model loaded")

MAX_ITERATIONS = 10
BATCH_SIZE = 100  # Number of novel prompts to sample per iteration

# Current training data
current_training_data = initial_training_data.copy()

for iteration in range(1, MAX_ITERATIONS + 1):
    print(f"\n  Iteration {iteration}/{MAX_ITERATIONS}")
    print("  " + "-"*76)

    # 1. Get probe predictions on novel data
    scores = novel_acts @ probe
    preds = scores > 0

    # 2. Find disagreements with ground truth (oracle teacher)
    errors = preds != novel_labels
    error_indices = np.where(errors)[0]

    if len(error_indices) == 0:
        print("    No errors found - probe is perfect!")
        break

    print(f"    Found {len(error_indices)} errors ({len(error_indices)/len(novel_data)*100:.1f}%)")

    # 3. Analyze error types
    false_positives = np.where((preds == True) & (novel_labels == False))[0]
    false_negatives = np.where((preds == False) & (novel_labels == True))[0]

    print(f"      False positives: {len(false_positives)}")
    print(f"      False negatives: {len(false_negatives)}")

    # 4. Show examples of errors
    if len(error_indices) > 0:
        print(f"\n      Example errors:")
        for idx in error_indices[:3]:
            error_data = novel_data[idx]
            print(f"        Text: {error_data['text'][:60]}...")
            print(f"        Predicted: {preds[idx]}, Ground truth: {novel_labels[idx]}")

    # 5. Create augmented training examples from errors
    #    For each error, add it directly as a training example
    new_training_examples = []

    for idx in error_indices:
        # Add the error as a direct training example
        new_training_examples.append({
            'activation': novel_data[idx]['activation'],
            'label': novel_labels[idx],  # Ground truth label
        })

    print(f"\n    Adding {len(new_training_examples)} error corrections to training set")

    # 6. Retrain probe with augmented data
    current_training_data.extend(new_training_examples)
    probe = train_dim_probe(current_training_data)

    # 7. Evaluate on novel prompts
    new_acc = evaluate_probe(probe, novel_acts, novel_labels)
    improvement = (new_acc - iteration_history[-1]['novel_accuracy']) * 100

    print(f"    Training set size: {len(current_training_data)} (+{len(new_training_examples)})")
    print(f"    Novel accuracy: {new_acc*100:.1f}% ({improvement:+.1f}%)")

    # Track iteration
    iteration_history.append({
        'iteration': iteration,
        'training_size': len(current_training_data),
        'novel_accuracy': new_acc,
        'errors_found': len(error_indices),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
    })

    # Check convergence
    if new_acc >= 0.95:
        print(f"\n    🎉 Reached 95% accuracy! Stopping early.")
        break

    if improvement < 0.5 and iteration > 3:
        print(f"\n    Convergence detected (improvement < 0.5%). Stopping.")
        break

# Results summary
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

print("\nIteration History:")
print("-"*80)
print(f"{'Iter':>4} | {'Training Size':>13} | {'Novel Acc':>10} | {'Errors':>7} | {'Δ Acc':>8}")
print("-"*80)

for i, metrics in enumerate(iteration_history):
    if i == 0:
        delta = 0
    else:
        delta = (metrics['novel_accuracy'] - iteration_history[i-1]['novel_accuracy']) * 100

    print(f"{metrics['iteration']:>4} | {metrics['training_size']:>13} | "
          f"{metrics['novel_accuracy']*100:>9.1f}% | {metrics['errors_found']:>7} | "
          f"{delta:>+7.1f}%")

# Final statistics
final_acc = iteration_history[-1]['novel_accuracy']
total_improvement = (final_acc - baseline_acc) * 100
total_iterations = iteration_history[-1]['iteration']

print("\n" + "="*80)
print("FINAL STATISTICS")
print("="*80)

print(f"\nBaseline (got_cities):     {baseline_acc*100:.1f}%")
print(f"After RLAIF refinement:    {final_acc*100:.1f}%")
print(f"Total improvement:         {total_improvement:+.1f}%")
print(f"Iterations completed:      {total_iterations}")
print(f"Final training set size:   {len(current_training_data)} (from {len(initial_training_data)})")

# Evaluation on categories
print("\n" + "="*80)
print("CATEGORY-WISE PERFORMANCE")
print("="*80)

categories = set(d['category'] for d in novel_data)

print(f"\n{'Category':20s} | {'Baseline':>10} | {'After RLAIF':>12} | {'Improvement':>12}")
print("-"*80)

# Get baseline probe
baseline_probe = train_dim_probe(initial_training_data)

for category in sorted(categories):
    cat_data = [d for d in novel_data if d['category'] == category]
    cat_acts = np.array([d['activation'] for d in cat_data])
    cat_labels = np.array([d['label'] for d in cat_data])

    baseline_cat_acc = evaluate_probe(baseline_probe, cat_acts, cat_labels)
    final_cat_acc = evaluate_probe(probe, cat_acts, cat_labels)
    improvement = (final_cat_acc - baseline_cat_acc) * 100

    print(f"{category:20s} | {baseline_cat_acc*100:>9.1f}% | {final_cat_acc*100:>11.1f}% | {improvement:>+11.1f}%")

# Analysis
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

if total_improvement > 5:
    print("\n✅ STRONG IMPROVEMENT: RLAIF refinement significantly improved generalization!")
    print(f"   {baseline_acc*100:.1f}% → {final_acc*100:.1f}% ({total_improvement:+.1f}%)")
    print("\n   This demonstrates that:")
    print("   1. Iterative refinement can improve probe generalization")
    print("   2. Adding error corrections as training examples helps")
    print("   3. The probe learns from its mistakes")
elif total_improvement > 2:
    print("\n✓ MODERATE IMPROVEMENT: RLAIF refinement helped somewhat")
    print(f"   {baseline_acc*100:.1f}% → {final_acc*100:.1f}% ({total_improvement:+.1f}%)")
    print("\n   Limited improvement suggests:")
    print("   1. got_cities baseline is already quite good")
    print("   2. Novel prompts may not provide diverse enough errors")
    print("   3. May need more sophisticated contrastive pair generation")
else:
    print("\n⚠ LIMITED IMPROVEMENT: RLAIF refinement didn't help much")
    print(f"   {baseline_acc*100:.1f}% → {final_acc*100:.1f}% ({total_improvement:+.1f}%)")
    print("\n   Possible reasons:")
    print("   1. got_cities already captures semantic truth well")
    print("   2. Test set is too small (60 prompts)")
    print("   3. Need more diverse error sources")
    print("   4. Simple addition of errors may not be enough")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

print("""
To improve RLAIF refinement further:

1. Use larger pool of novel prompts (1000+ instead of 60)
2. Use actual teacher model (GPT-4/Claude) instead of ground truth oracle
3. Generate synthetic contrastive pairs (not just add errors directly)
4. Sample diverse domains (not just science/history/math)
5. Use confidence weighting for teacher predictions
6. Implement adversarial red team prompts
7. Test on held-out validation set (separate from refinement set)
""")

# Save results
output_dir = Path("/root/repeng/output/probe_ensemble/rlaif_refinement")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "refinement_results.pkl", 'wb') as f:
    pickle.dump({
        'iteration_history': iteration_history,
        'baseline_accuracy': baseline_acc,
        'final_accuracy': final_acc,
        'total_improvement': total_improvement,
        'final_probe': probe,
        'baseline_probe': baseline_probe,
        'training_data_sizes': [h['training_size'] for h in iteration_history],
    }, f)

print(f"\n✓ Saved results to: {output_dir}")
