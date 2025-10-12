#!/usr/bin/env python3
"""
Test if our ensemble probe has truly isolated "general truth"

Generate diverse novel prompts across domains:
- Science facts
- Historical events
- Mathematical statements
- Common sense reasoning
- Counterfactuals
- Ambiguous/tricky cases
"""

import sys
sys.path.insert(0, "/root/repeng")

import torch
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

from repeng.models.loading import load_llm_oioo
from repeng.activations.inference import get_model_activations

print("="*80)
print("GENERALIZATION TEST: Novel Prompts")
print("="*80)

# Define diverse test cases
NOVEL_PROMPTS = {
    "science_facts": [
        # True
        ("Water boils at 100 degrees Celsius at sea level.", True),
        ("The speed of light in vacuum is approximately 299,792,458 meters per second.", True),
        ("DNA contains the genetic instructions for living organisms.", True),
        ("The Earth's core is primarily composed of iron and nickel.", True),
        ("Photosynthesis converts sunlight into chemical energy in plants.", True),
        # False
        ("The moon is made of cheese.", False),
        ("Humans only use 10% of their brain.", False),
        ("Lightning never strikes the same place twice.", False),
        ("Antibiotics are effective against viral infections.", False),
        ("The Great Wall of China is visible from the moon with the naked eye.", False),
    ],

    "historical_events": [
        # True
        ("World War II ended in 1945.", True),
        ("The Berlin Wall fell in 1989.", True),
        ("The first human landed on the moon in 1969.", True),
        ("The Roman Empire existed for over a thousand years.", True),
        ("The printing press was invented in the 15th century.", True),
        # False
        ("Napoleon Bonaparte was born in France.", False),  # Born in Corsica
        ("The American Civil War began in 1861 and ended in 1866.", False),  # Ended 1865
        ("Christopher Columbus discovered America in 1592.", False),  # 1492
        ("The Titanic sank in 1920.", False),  # 1912
        ("Albert Einstein won the Nobel Prize for his theory of relativity.", False),  # Photoelectric effect
    ],

    "mathematics": [
        # True
        ("2 + 2 = 4", True),
        ("A triangle has three sides.", True),
        ("The square root of 16 is 4.", True),
        ("Pi is approximately 3.14159.", True),
        ("1000 divided by 10 equals 100.", True),
        # False
        ("5 + 7 = 13", False),
        ("A circle has four corners.", False),
        ("3 multiplied by 4 equals 15.", False),
        ("The sum of angles in a triangle is 360 degrees.", False),
        ("10% of 200 is 30.", False),
    ],

    "common_sense": [
        # True
        ("Fire is hot.", True),
        ("Ice melts when heated.", True),
        ("Birds can fly.", True),
        ("A week has seven days.", True),
        ("Humans need oxygen to survive.", True),
        # False
        ("Fish can breathe air underwater.", False),
        ("The sun rises in the west.", False),
        ("Rocks are lighter than feathers.", False),
        ("Winter is the hottest season of the year.", False),
        ("Elephants are smaller than mice.", False),
    ],

    "counterfactuals": [
        # True
        ("If I drop a glass, it will likely break.", True),
        ("If you don't water plants, they will die.", True),
        ("If the temperature drops below freezing, water turns to ice.", True),
        ("If you study hard, you're more likely to do well on exams.", True),
        ("If it rains heavily, the ground gets wet.", True),
        # False
        ("If you freeze water, it turns into steam.", False),
        ("If you eat healthy food, you'll definitely get sick.", False),
        ("If the sun sets, it means the world is ending.", False),
        ("If you run fast, you'll travel back in time.", False),
        ("If ice cream is cold, it must be made of snow.", False),
    ],

    "tricky_cases": [
        # True (but requires reasoning)
        ("The statement '2+2=4' is true.", True),
        ("Whales are mammals, not fish.", True),
        ("The Pacific Ocean is larger than the Atlantic Ocean.", True),
        ("Tomatoes are fruits, not vegetables (botanically speaking).", True),
        ("Penguins are birds that cannot fly.", True),
        # False (common misconceptions)
        ("Goldfish have a 3-second memory.", False),
        ("Bulls are enraged by the color red.", False),
        ("Bats are blind.", False),
        ("Cracking knuckles causes arthritis.", False),
        ("Different parts of the tongue taste different flavors.", False),
    ],
}

# Flatten all prompts
all_prompts = []
categories = []
for category, prompts in NOVEL_PROMPTS.items():
    for text, label in prompts:
        all_prompts.append((text, label, category))

print(f"\n✓ Generated {len(all_prompts)} novel test prompts across {len(NOVEL_PROMPTS)} categories")
for category, prompts in NOVEL_PROMPTS.items():
    true_count = sum(1 for _, label in prompts if label)
    false_count = len(prompts) - true_count
    print(f"  {category:20s}: {len(prompts):2d} prompts ({true_count} true, {false_count} false)")

# Load model
print("\n[1/5] Loading Qwen3-4B model...")
device = torch.device("cuda")
llm = load_llm_oioo("Qwen/Qwen3-4B", device=device, use_half_precision=True)
print("✓ Model loaded")

# Extract activations
print("\n[2/5] Extracting activations...")
LAYER = "h35"

novel_activations = []
for i, (text, label, category) in enumerate(all_prompts):
    if i % 10 == 0:
        print(f"  Processing {i}/{len(all_prompts)}...")

    result = get_model_activations(
        llm,
        text=text,
        last_n_tokens=1,
        points_start=35,  # Just final layer
        points_end=36,
        points_skip=1,
    )

    novel_activations.append({
        'text': text,
        'label': label,
        'category': category,
        'activation': result.activations[LAYER].flatten().astype(np.float32),
    })

print(f"✓ Extracted activations for {len(novel_activations)} prompts")

# Load trained probes
print("\n[3/5] Loading trained probes...")

# Load individual probes from earlier analysis
analysis_dir = Path("/root/repeng/output/probe_ensemble/comprehensive_analysis")
with open(analysis_dir / "comprehensive_results.pkl", 'rb') as f:
    results = pickle.load(f)

# Recreate probes from training data
activations_file = Path("/root/repeng/output/comparison/activations_results/value.pickle")
with open(activations_file, 'rb') as f:
    from repeng.datasets.activations.types import ActivationResultRow
    rows: List[ActivationResultRow] = pickle.load(f)

# Train individual probes
train_data = defaultdict(list)
for row in rows:
    if row.split == 'train' and LAYER in row.activations:
        train_data[row.dataset_id].append({
            'activation': row.activations[LAYER].flatten().astype(np.float32),
            'label': row.label,
        })

def train_dim_probe(data):
    acts = np.array([d['activation'] for d in data])
    labels = np.array([d['label'] for d in data])
    true_mean = acts[labels].mean(axis=0)
    false_mean = acts[~labels].mean(axis=0)
    theta = true_mean - false_mean
    return theta / (np.linalg.norm(theta) + 1e-8)

individual_probes = {}
individual_accs = results['individual_accs']

for ds_id in train_data.keys():
    individual_probes[ds_id] = train_dim_probe(train_data[ds_id])

print(f"✓ Loaded {len(individual_probes)} individual probes")

# Create ensemble probes
print("\n[4/5] Creating ensemble probes...")

# Best method: Weighted averaging with top 3 datasets
sorted_datasets = sorted(individual_accs.items(), key=lambda x: x[1], reverse=True)
top3_datasets = [ds for ds, _ in sorted_datasets[:3]]

print(f"  Top 3 datasets: {top3_datasets}")
print(f"  Accuracies: {[individual_accs[ds] for ds in top3_datasets]}")

# Weighted ensemble
weights = np.array([individual_accs[ds] for ds in top3_datasets])
weights = weights / weights.sum()

ensemble_weighted = np.average(
    [individual_probes[ds] for ds in top3_datasets],
    axis=0,
    weights=weights
)
ensemble_weighted = ensemble_weighted / (np.linalg.norm(ensemble_weighted) + 1e-8)

# Simple average
ensemble_simple = np.mean([individual_probes[ds] for ds in top3_datasets], axis=0)
ensemble_simple = ensemble_simple / (np.linalg.norm(ensemble_simple) + 1e-8)

# PC2 (from PCA analysis)
fragmentation_dir = Path("/root/repeng/output/probe_ensemble/fragmentation_analysis")
with open(fragmentation_dir / "spectrum_analysis.pkl", 'rb') as f:
    spectrum = pickle.load(f)

pc2_probe = spectrum[LAYER]['Vt'][1]  # PC2 is the truth component

print("✓ Created 3 ensemble probes + individual baselines")

# Test on novel prompts
print("\n[5/5] Testing probes on novel data...")

def evaluate_probe(probe, data, probe_name=""):
    acts = np.array([d['activation'] for d in data])
    labels = np.array([d['label'] for d in data])

    scores = acts @ probe
    preds = scores > 0

    accuracy = np.mean(preds == labels)

    # Per-category breakdown
    category_accs = {}
    for category in set(d['category'] for d in data):
        cat_data = [d for d in data if d['category'] == category]
        cat_acts = np.array([d['activation'] for d in cat_data])
        cat_labels = np.array([d['label'] for d in cat_data])
        cat_scores = cat_acts @ probe
        cat_preds = cat_scores > 0
        category_accs[category] = np.mean(cat_preds == cat_labels)

    return accuracy, category_accs

# Test all probes
probe_results = {}

print("\nTesting ensemble probes:")
print("-"*80)

# Ensemble probes
probes_to_test = [
    ("Weighted N=3 Ensemble", ensemble_weighted),
    ("Simple Average N=3", ensemble_simple),
    ("PC2 (Truth Component)", pc2_probe),
]

for name, probe in probes_to_test:
    acc, cat_accs = evaluate_probe(probe, novel_activations, name)
    probe_results[name] = {'overall': acc, 'categories': cat_accs}
    print(f"\n{name}:")
    print(f"  Overall: {acc*100:.1f}%")
    for cat, cat_acc in sorted(cat_accs.items()):
        print(f"    {cat:20s}: {cat_acc*100:.1f}%")

# Test individual probes for comparison
print("\n\nTesting individual dataset probes:")
print("-"*80)

for ds_id in top3_datasets:
    acc, cat_accs = evaluate_probe(individual_probes[ds_id], novel_activations, ds_id)
    probe_results[f"Individual: {ds_id}"] = {'overall': acc, 'categories': cat_accs}
    print(f"\n{ds_id}:")
    print(f"  Overall: {acc*100:.1f}%")
    for cat, cat_acc in sorted(cat_accs.items()):
        print(f"    {cat:20s}: {cat_acc*100:.1f}%")

# Summary
print("\n" + "="*80)
print("SUMMARY: Novel Prompt Generalization")
print("="*80)

sorted_results = sorted(probe_results.items(), key=lambda x: x[1]['overall'], reverse=True)

print("\nOverall Performance (ranked):")
for i, (name, result) in enumerate(sorted_results, 1):
    print(f"  {i}. {name:30s}: {result['overall']*100:.1f}%")

best_name, best_result = sorted_results[0]
print(f"\n🏆 Best probe: {best_name} ({best_result['overall']*100:.1f}%)")

# Analyze strengths/weaknesses
print("\n" + "="*80)
print("CATEGORY ANALYSIS")
print("="*80)

for category in NOVEL_PROMPTS.keys():
    print(f"\n{category}:")
    cat_results = [(name, result['categories'][category])
                   for name, result in probe_results.items()]
    cat_results.sort(key=lambda x: x[1], reverse=True)

    for name, acc in cat_results[:3]:  # Top 3 per category
        print(f"  {acc*100:5.1f}%  {name}")

# Save results
output_dir = Path("/root/repeng/output/probe_ensemble/novel_prompts")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "novel_prompt_results.pkl", 'wb') as f:
    pickle.dump({
        'prompts': novel_activations,
        'probe_results': probe_results,
        'probes': {
            'weighted_ensemble': ensemble_weighted,
            'simple_ensemble': ensemble_simple,
            'pc2': pc2_probe,
        }
    }, f)

print(f"\n✓ Saved results to: {output_dir}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

baseline_avg = np.mean([result['overall'] for name, result in probe_results.items()
                        if name.startswith("Individual")])
ensemble_best = best_result['overall']

print(f"\nIndividual probes (avg): {baseline_avg*100:.1f}%")
print(f"Best ensemble: {ensemble_best*100:.1f}%")
print(f"Improvement: {(ensemble_best - baseline_avg)*100:+.1f}%")

if ensemble_best > baseline_avg + 0.05:
    print("\n✅ STRONG GENERALIZATION: Ensemble significantly outperforms on novel data")
    print("   → We've isolated a general truth direction!")
elif ensemble_best > baseline_avg:
    print("\n✓ MODERATE GENERALIZATION: Ensemble slightly better on novel data")
    print("   → Some general truth captured, but room for improvement")
else:
    print("\n⚠ LIMITED GENERALIZATION: Ensemble doesn't improve on novel data")
    print("   → May have learned dataset-specific patterns")
