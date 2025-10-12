#!/usr/bin/env python3
"""
Test all probe approaches on 260 diverse novel prompts.

This will show:
1. How well got_cities generalizes to many more domains
2. Whether RLAIF trained on 60 prompts transfers to new 260
3. Which categories are hardest/easiest
"""

import sys
sys.path.insert(0, "/root/repeng")

import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

print("="*80)
print("TESTING ON 260 DIVERSE NOVEL PROMPTS")
print("="*80)

# Load diverse prompts
print("\n[1/3] Loading diverse prompts...")
diverse_dir = Path("/root/repeng/output/probe_ensemble/diverse_novel_prompts")
with open(diverse_dir / "diverse_prompts.pkl", 'rb') as f:
    diverse_data = pickle.load(f)

diverse_prompts = diverse_data['prompts']
diverse_acts = np.array([d['activation'] for d in diverse_prompts])
diverse_labels = np.array([d['label'] for d in diverse_prompts])

print(f"✓ Loaded {len(diverse_prompts)} prompts")
print(f"  Categories: {len(diverse_data['categories'])}")

# Load original 60 prompts for comparison
novel_dir = Path("/root/repeng/output/probe_ensemble/novel_prompts")
with open(novel_dir / "novel_prompt_results.pkl", 'rb') as f:
    novel_results = pickle.load(f)

original_acts = np.array([d['activation'] for d in novel_results['prompts']])
original_labels = np.array([d['label'] for d in novel_results['prompts']])

print(f"✓ Original 60 prompts loaded for comparison")

# Load training data
activations_file = Path("/root/repeng/output/comparison/activations_results/value.pickle")
with open(activations_file, 'rb') as f:
    from repeng.datasets.activations.types import ActivationResultRow
    rows = pickle.load(f)

LAYER = "h35"
train_data = defaultdict(list)
for row in rows:
    if row.split == 'train' and LAYER in row.activations:
        train_data[row.dataset_id].append({
            'activation': row.activations[LAYER].flatten().astype(np.float32),
            'label': row.label,
        })

print(f"✓ Training data loaded")

# Probe functions
def train_dim_probe(data):
    acts = np.array([d['activation'] for d in data])
    labels = np.array([d['label'] for d in data])
    true_mean = acts[labels].mean(axis=0)
    false_mean = acts[~labels].mean(axis=0)
    theta = true_mean - false_mean
    return theta / (np.linalg.norm(theta) + 1e-8)

def evaluate_probe(probe, acts, labels):
    scores = acts @ probe
    preds = scores > 0
    return np.mean(preds == labels)

# Load probes
print("\n[2/3] Loading probes...")

# 1. got_cities baseline
got_cities_probe = train_dim_probe(train_data['got_cities'])
print("✓ got_cities probe")

# 2. RLAIF probe
rlaif_dir = Path("/root/repeng/output/probe_ensemble/rlaif_refinement")
with open(rlaif_dir / "refinement_results.pkl", 'rb') as f:
    rlaif_results = pickle.load(f)
rlaif_probe = rlaif_results['final_probe']
print("✓ RLAIF probe")

# 3. Other comparison probes
comparison_dir = Path("/root/repeng/output/probe_ensemble/rlaif_comparison")
with open(comparison_dir / "comparison_results.pkl", 'rb') as f:
    comparison_results = pickle.load(f)
print("✓ Comparison probes")

# Test all probes
print("\n[3/3] Testing probes...")
print("="*80)

probes_to_test = {
    'got_cities baseline': got_cities_probe,
    'RLAIF (trained on 60)': rlaif_probe,
    'Concat all 60': comparison_results['probes']['concat_all'],
    'Weighted ensemble': comparison_results['probes']['ensemble_weighted'],
}

# Test on original 60
print("\nPerformance on ORIGINAL 60 prompts:")
print("-"*80)
for name, probe in probes_to_test.items():
    acc = evaluate_probe(probe, original_acts, original_labels)
    print(f"{name:30s}: {acc*100:.1f}%")

# Test on diverse 260
print("\nPerformance on DIVERSE 260 prompts:")
print("-"*80)
diverse_results = {}
for name, probe in probes_to_test.items():
    acc = evaluate_probe(probe, diverse_acts, diverse_labels)
    diverse_results[name] = acc
    print(f"{name:30s}: {acc*100:.1f}%")

# Category breakdown for best probe
print("\n" + "="*80)
print("CATEGORY-WISE BREAKDOWN (RLAIF probe)")
print("="*80)

categories = diverse_data['categories']
print(f"\n{'Category':20s} | {'Accuracy':>10} | N")
print("-"*50)

category_accs = {}
for category in sorted(categories):
    cat_data = [d for d in diverse_prompts if d['category'] == category]
    cat_acts = np.array([d['activation'] for d in cat_data])
    cat_labels = np.array([d['label'] for d in cat_data])

    acc = evaluate_probe(rlaif_probe, cat_acts, cat_labels)
    category_accs[category] = acc
    print(f"{category:20s} | {acc*100:>9.1f}% | {len(cat_data)}")

avg_acc = np.mean(list(category_accs.values()))
print("-"*50)
print(f"{'AVERAGE':20s} | {avg_acc*100:>9.1f}% | {len(diverse_prompts)}")

# Identify hardest/easiest categories
print("\n" + "="*80)
print("HARDEST vs EASIEST CATEGORIES")
print("="*80)

sorted_cats = sorted(category_accs.items(), key=lambda x: x[1])

print("\nHardest (lowest accuracy):")
for cat, acc in sorted_cats[:5]:
    print(f"  {cat:20s}: {acc*100:.1f}%")

print("\nEasiest (highest accuracy):")
for cat, acc in sorted_cats[-5:]:
    print(f"  {cat:20s}: {acc*100:.1f}%")

# Compare to original performance
print("\n" + "="*80)
print("TRANSFER ANALYSIS")
print("="*80)

print("\nHow well did RLAIF (trained on 60) transfer to new 260?")
print("-"*80)

rlaif_original = 0.917  # From earlier results
rlaif_diverse = diverse_results['RLAIF (trained on 60)']
transfer_gap = (rlaif_original - rlaif_diverse) * 100

print(f"RLAIF on original 60:  {rlaif_original*100:.1f}%")
print(f"RLAIF on diverse 260:  {rlaif_diverse*100:.1f}%")
print(f"Transfer gap:          {transfer_gap:+.1f}%")

if abs(transfer_gap) < 5:
    print("\n✅ EXCELLENT TRANSFER: Performance maintained on new domains!")
    print("   The probe learned general semantic truth.")
elif abs(transfer_gap) < 10:
    print("\n✓ GOOD TRANSFER: Slight performance drop on new domains")
    print("   Some overfitting to original 60 prompts.")
else:
    print("\n⚠ POOR TRANSFER: Significant performance drop")
    print("   Probe overfitted to original 60 prompts.")

# Final summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\nKey findings:")
print("-"*80)

# 1. Best probe overall
best_probe = max(diverse_results.items(), key=lambda x: x[1])
print(f"1. Best probe on 260 diverse: {best_probe[0]} ({best_probe[1]*100:.1f}%)")

# 2. Transfer quality
print(f"2. RLAIF transfer: {rlaif_original*100:.1f}% → {rlaif_diverse*100:.1f}% ({transfer_gap:+.1f}%)")

# 3. Hardest domain
hardest = sorted_cats[0]
easiest = sorted_cats[-1]
print(f"3. Hardest category: {hardest[0]} ({hardest[1]*100:.1f}%)")
print(f"4. Easiest category: {easiest[0]} ({easiest[1]*100:.1f}%)")

# 4. Overall generalization
baseline = diverse_results['got_cities baseline']
improvement = (rlaif_diverse - baseline) * 100
print(f"5. Improvement over baseline: {improvement:+.1f}% ({baseline*100:.1f}% → {rlaif_diverse*100:.1f}%)")

# Save results
output_dir = Path("/root/repeng/output/probe_ensemble/diverse_evaluation")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "results.pkl", 'wb') as f:
    pickle.dump({
        'diverse_results': diverse_results,
        'category_accs': category_accs,
        'transfer_gap': transfer_gap,
    }, f)

print(f"\n✓ Saved to: {output_dir}")
