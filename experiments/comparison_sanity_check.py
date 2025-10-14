#!/usr/bin/env python3
"""
Quick sanity check: Does the best in-distribution probe also generalize best?

Tests hypothesis:
- For each dataset, find the best layer (highest accuracy when train=eval)
- Test if this "best" probe also performs well on other datasets
- If true: we can use best in-distribution probe for generalization

Minimal scope:
- 6 datasets: 2 per family (DLK, RepE, GoT)
- 4 layers: early, early-mid, late-mid, late (h11, h19, h27, h35)
- 6×6×4 = 144 evaluations (~20 minutes)
"""
import sys
sys.path.insert(0, "/root/repeng")

import argparse
import torch
from pathlib import Path
from collections import defaultdict

import jsonlines
import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

from repeng.datasets.elk.utils.fns import get_dataset
from repeng.activations.inference import get_model_activations
from repeng.activations.probe_preparations import ActivationArrays
from repeng.models.loading import load_llm_oioo
from repeng.models.types import LlmId
from repeng.probes.collections import train_probe
from repeng.evals.probes import eval_probe_by_question, eval_probe_by_row

# Configuration
DATASETS = [
    # DLK family
    "imdb", "boolq",
    # RepE family
    "race", "arc_easy",
    # GoT family
    "got_cities", "got_sp_en_trans"
]

LAYERS = ["h11", "h19", "h27", "h35"]  # Early, early-mid, late-mid, late

MODEL_ID = "Qwen/Qwen3-4B"
TRAIN_LIMIT = 400
EVAL_LIMIT = 2000

print("="*80)
print("CROSS-DATASET GENERALIZATION: SANITY CHECK")
print("="*80)
print(f"Model: {MODEL_ID}")
print(f"Datasets ({len(DATASETS)}): {', '.join(DATASETS)}")
print(f"Layers ({len(LAYERS)}): {', '.join(LAYERS)}")
print(f"Total evaluations: {len(DATASETS)} train × {len(DATASETS)} eval × {len(LAYERS)} layers = {len(DATASETS) * len(DATASETS) * len(LAYERS)}")
print()

# Result structures
class ResultRow(BaseModel):
    llm_id: LlmId
    train_dataset: str
    eval_dataset: str
    layer: str
    accuracy: float
    n: int
    is_in_distribution: bool

def get_activations(dataset_id: str, split: str, layer_names: list[str], limit: int):
    """Get activations for specific layers"""
    dataset_rows = get_dataset(dataset_id)

    # Fallback logic for splits
    if split == "train":
        splits_to_try = ["train"]
    else:
        splits_to_try = ["test", "validation", "val", "train"]

    rows = []
    split_used = None
    for try_split in splits_to_try:
        rows = [r for k, r in dataset_rows.items() if r.split == try_split]
        if rows:
            split_used = try_split
            break

    if not rows:
        raise ValueError(f"No data found for {dataset_id}")

    # Skip training data if using train split for eval
    if split != "train" and split_used == "train":
        rows = rows[400:]

    rows = rows[:limit]

    if split_used != split:
        print(f"    {dataset_id}: using '{split_used}' split ({len(rows)} examples)")

    activations_by_layer = defaultdict(list)
    labels = []
    groups = []

    # Extract layer numbers for filtering
    layer_nums = {int(ln.lstrip('h')) for ln in layer_names}

    for row in tqdm(rows, desc=f"  {dataset_id:20s}", leave=False):
        activation_row = get_model_activations(
            llm,
            text=row.text,
            last_n_tokens=1,
            points_start=1,
            points_end=None,
            points_skip=1,  # Get all layers
        )

        # Filter to only the layers we care about
        for point_name, acts in activation_row.activations.items():
            layer_num = int(point_name.lstrip('h'))
            if layer_num in layer_nums:
                activations_by_layer[point_name].append(acts[-1])

        labels.append(row.label)
        groups.append(row.group_id if row.group_id else None)

    return activations_by_layer, labels, groups

def train_probe_fn(train_acts, train_labels):
    """Train DIM probe"""
    arrays = ActivationArrays(
        activations=np.stack(train_acts).astype(np.float32),
        labels=np.array(train_labels),
        groups=None,
        answer_types=None
    )
    return train_probe("dim", arrays)

def eval_probe_fn(probe, eval_acts, eval_labels, eval_groups):
    """Evaluate probe"""
    eval_acts_np = np.stack(eval_acts).astype(np.float32)
    eval_labels_np = np.array(eval_labels)

    # Handle groups
    if eval_groups and any(g is not None for g in eval_groups):
        eval_groups_np = np.array([hash(str(g)) % 100000 if g else i for i, g in enumerate(eval_groups)])
        unique_groups = len(np.unique(eval_groups_np))
    else:
        eval_groups_np = None
        unique_groups = 0

    # Evaluate
    if eval_groups_np is not None and unique_groups > 1:
        result = eval_probe_by_question(probe, activations=eval_acts_np, labels=eval_labels_np, groups=eval_groups_np)
        accuracy = result.accuracy
        n = result.n
    else:
        result = eval_probe_by_row(probe, activations=eval_acts_np, labels=eval_labels_np)
        accuracy = result.roc_auc_score
        n = result.n

    return accuracy, n

# Load model
print("[1/3] Loading model...")
llm = load_llm_oioo(MODEL_ID, device=torch.device("cuda"), use_half_precision=True)
print("✓ Model loaded\n")

# Train probes
print("[2/3] Training probes...")
trained_probes = {}
for train_dataset in DATASETS:
    print(f"\nTraining on: {train_dataset}")
    train_acts, train_labels, _ = get_activations(train_dataset, "train", LAYERS, TRAIN_LIMIT)

    for layer in LAYERS:
        probe = train_probe_fn(train_acts[layer], train_labels)
        trained_probes[(train_dataset, layer)] = probe

print("\n✓ Trained {} probes\n".format(len(trained_probes)))

# Evaluate
print("[3/3] Evaluating cross-dataset generalization...")
results = []

with tqdm(total=len(DATASETS) * len(DATASETS) * len(LAYERS), desc="Evaluations") as pbar:
    for train_dataset in DATASETS:
        for eval_dataset in DATASETS:
            eval_acts, eval_labels, eval_groups = get_activations(eval_dataset, "test", LAYERS, EVAL_LIMIT)

            for layer in LAYERS:
                probe = trained_probes[(train_dataset, layer)]
                acc, n = eval_probe_fn(probe, eval_acts[layer], eval_labels, eval_groups)

                results.append(ResultRow(
                    llm_id=MODEL_ID,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    layer=layer,
                    accuracy=acc,
                    n=n,
                    is_in_distribution=(train_dataset == eval_dataset)
                ))

                pbar.update(1)

# Save results
output_dir = Path("output/comparison/qwen3-4b")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "sanity_check_results.jsonl"

with jsonlines.open(output_file, mode='w') as writer:
    for i, result in enumerate(results):
        writer.write({"key": f"result_{i}", "value": result.model_dump()})

print(f"\n✓ Saved {len(results)} results to: {output_file}\n")

# Analyze results
print("="*80)
print("ANALYSIS: Does best in-distribution probe generalize best?")
print("="*80)

import pandas as pd
df = pd.DataFrame([r.model_dump() for r in results])

# For each dataset, find best layer (highest in-distribution accuracy)
print("\n1. Best layers per dataset (in-distribution):\n")
in_dist = df[df['is_in_distribution']]
for dataset in DATASETS:
    dataset_results = in_dist[in_dist['train_dataset'] == dataset].sort_values('accuracy', ascending=False)
    if len(dataset_results) > 0:
        best = dataset_results.iloc[0]
        print(f"  {dataset:20s}: {best['layer']} ({best['accuracy']:.1%})")

# Test hypothesis: Does best layer for dataset X also perform best on other datasets?
print("\n2. Cross-dataset generalization by layer:\n")

for train_dataset in DATASETS:
    print(f"\n  Trained on {train_dataset}:")

    # Get best layer for this dataset (in-distribution)
    dataset_in_dist = in_dist[(in_dist['train_dataset'] == train_dataset)]
    best_layer = dataset_in_dist.sort_values('accuracy', ascending=False).iloc[0]['layer']

    # For each eval dataset, compare performance of best_layer vs other layers
    for eval_dataset in DATASETS:
        if eval_dataset == train_dataset:
            continue

        eval_results = df[(df['train_dataset'] == train_dataset) & (df['eval_dataset'] == eval_dataset)]
        eval_results_sorted = eval_results.sort_values('accuracy', ascending=False)

        best_performing_layer = eval_results_sorted.iloc[0]['layer']
        best_layer_acc = eval_results[eval_results['layer'] == best_layer]['accuracy'].values[0]
        actual_best_acc = eval_results_sorted.iloc[0]['accuracy']

        match = "✓" if best_performing_layer == best_layer else "✗"
        print(f"    → {eval_dataset:20s}: best={best_layer} ({best_layer_acc:.1%}), actual_best={best_performing_layer} ({actual_best_acc:.1%}) {match}")

# Summary statistics
print("\n3. Summary Statistics:\n")
ood_results = df[~df['is_in_distribution']]
print(f"  Mean OOD accuracy: {ood_results['accuracy'].mean():.1%}")
print(f"  Mean in-distribution accuracy: {in_dist['accuracy'].mean():.1%}")
print(f"  Generalization ratio: {ood_results['accuracy'].mean() / in_dist['accuracy'].mean():.1%}")

print("\n" + "="*80)
print("Interpretation:")
print("  ✓ = Best in-distribution layer also performs best on that eval dataset")
print("  ✗ = Different layer performs better on that eval dataset")
print("="*80)
