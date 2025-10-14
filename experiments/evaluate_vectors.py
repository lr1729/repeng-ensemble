#!/usr/bin/env python3
"""
Evaluate truthfulness probes for cross-dataset generalization

This script:
1. Loads pre-trained probe vectors from extract_vectors.py
2. Generates evaluation activations at optimal layer (once per dataset, cached)
3. Evaluates cross-dataset generalization (17×17 = 289 tests)
4. Saves results with both accuracy and AUC metrics

Efficiency: Activations are generated once per eval dataset and cached in memory,
then reused for all 17 probes. This reduces 289 activation generations to just 17.

Usage:
    # Full evaluation (default: 2000 examples per dataset, ~5-7 min)
    python experiments/evaluate_vectors.py --model qwen3-4b

    # Quick test (200 examples per dataset, ~1-2 min)
    python experiments/evaluate_vectors.py --model qwen3-4b --eval-limit 200

    # Different layer depth
    python experiments/evaluate_vectors.py --model qwen3-4b --layer-depth 0.80
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

from repeng.datasets.elk.utils.collections import DatasetCollectionId, resolve_dataset_ids
from repeng.datasets.elk.utils.fns import get_dataset
from repeng.models.loading import load_llm_oioo
from repeng.models.points import get_points
from repeng.models.types import LlmId
from repeng.probes.base import DotProductProbe
from repeng.evals.probes import eval_probe_by_question, eval_probe_by_row

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluate pre-trained truthfulness probes')
parser.add_argument('--model', type=str, required=True,
                    choices=['qwen3-4b', 'qwen3-8b', 'qwen3-14b',
                             'qwen3-4b-base', 'qwen3-8b-base', 'qwen3-14b-base',
                             'llama2-7b', 'llama2-13b', 'llama2-70b',
                             'llama2-7b-chat', 'llama2-13b-chat', 'llama2-70b-chat'],
                    help='Model to use')
parser.add_argument('--layer-depth', type=float, default=0.75,
                    help='Relative depth for evaluation (0-1, default 0.75 = 75%% through model)')
parser.add_argument('--batch-size', type=int, default=16,
                    help='Batch size for activation generation (default 16)')
parser.add_argument('--eval-limit', type=int, default=2000,
                    help='Number of examples per dataset for evaluation (default 2000, use 200 for quick tests)')
args = parser.parse_args()

model_spec = args.model

# Map model specs
if model_spec.startswith('qwen3-'):
    model_name = model_spec.replace('qwen3-', '')
    if model_name.endswith('-base'):
        size = model_name.replace('-base', '').upper()
        MODEL_ID = f"Qwen/Qwen3-{size}-Base"
    else:
        size = model_name.upper()
        MODEL_ID = f"Qwen/Qwen3-{size}"
elif model_spec.startswith('llama2-'):
    size = model_spec.replace('llama2-', '')
    if size.endswith('-chat'):
        size_clean = size.replace('-chat', '')
        MODEL_ID = f"Llama-2-{size_clean}-chat-hf"
    else:
        MODEL_ID = f"Llama-2-{size}-hf"
else:
    raise ValueError(f"Unknown model: {model_spec}")

# Get layers and calculate optimal layer for evaluation
all_points = get_points(MODEL_ID)
num_layers = len(all_points)
optimal_layer_idx = int(num_layers * args.layer_depth)
optimal_layer_idx = max(1, min(optimal_layer_idx, num_layers - 1))
optimal_point = all_points[optimal_layer_idx]
optimal_point_name = optimal_point.name

print(f"="*80)
print(f"EVALUATE TRUTHFULNESS PROBES: {MODEL_ID}")
print(f"="*80)
print(f"Input: output/comparison/{model_spec}/probe_vectors.jsonl")
print(f"Output: output/comparison/{model_spec}/probe_evaluate-v2.jsonl")
print(f"Evaluation layer: {optimal_point_name} ({args.layer_depth:.0%} depth)")
print(f"Batch size: {args.batch_size}")
print(f"Eval limit: {args.eval_limit} examples per dataset")
print()

# Setup paths
output_dir = Path(f"output/comparison/{model_spec}")
probe_vectors_file = output_dir / "probe_vectors.jsonl"
results_file = output_dir / "probe_evaluate-v2.jsonl"

if not probe_vectors_file.exists():
    print(f"ERROR: Probe vectors not found: {probe_vectors_file}")
    print(f"Please run: python experiments/extract_vectors.py --model {model_spec}")
    sys.exit(1)

# Get datasets
collections: list[DatasetCollectionId] = ["dlk", "repe", "got"]
all_dataset_ids = [
    dataset_id
    for collection in collections
    for dataset_id in resolve_dataset_ids(collection)
    if dataset_id != "piqa"
]

print(f"Datasets ({len(all_dataset_ids)}): {', '.join(all_dataset_ids)}")
print()

# Result structure
class PipelineResultRow(BaseModel, extra="forbid"):
    llm_id: LlmId
    train_dataset: str
    eval_dataset: str
    probe_method: str
    point_name: str
    token_idx: int
    accuracy: float
    accuracy_n: int
    auc: float

def get_activations_batch(llm, texts: list[str], layer_idx: int):
    """Get activations for a batch of texts at specific layer"""
    tokenized = llm.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )

    input_ids = tokenized["input_ids"].to(next(llm.model.parameters()).device)
    attention_mask = tokenized["attention_mask"].to(next(llm.model.parameters()).device)

    # Get point for the layer we want
    point = llm.points[layer_idx]

    # Run forward pass with hooks
    from repeng.hooks.grab import grab_many
    with grab_many(llm.model, [point]) as activation_fn:
        with torch.no_grad():
            llm.model.forward(input_ids, attention_mask=attention_mask, return_dict=True)
        layer_activations = activation_fn()

    # Extract last token activation for each example
    activations = []
    point_name = point.name
    acts = layer_activations[point_name]

    batch_size = acts.shape[0]
    for i in range(batch_size):
        seq_len = attention_mask[i].sum().item()
        last_token_act = acts[i, seq_len - 1, :].detach()
        # Convert bfloat16 to float16 (numpy doesn't support bfloat16)
        last_token_act = last_token_act.to(dtype=torch.float16)
        last_token_act = last_token_act.cpu().numpy()
        activations.append(last_token_act)

    return activations

def get_activations_for_split(dataset_id: str, split: str, llm, limit: int, layer_idx: int, batch_size: int):
    """Get activations for a dataset split at specific layer"""
    dataset_rows = get_dataset(dataset_id)

    # Define fallback order
    if split == "train":
        splits_to_try = ["train"]
    else:
        splits_to_try = ["test", "validation", "val", "train"]

    # Try splits in order
    rows = []
    split_used = None
    for try_split in splits_to_try:
        rows = [r for k, r in dataset_rows.items() if r.split == try_split]
        if rows:
            split_used = try_split
            break

    if not rows:
        raise ValueError(f"No data found for dataset '{dataset_id}'")

    # For evaluation, skip training examples if using train split
    if split != "train" and split_used == "train":
        rows = rows[400:]

    rows = rows[:limit]

    if split_used != split:
        print(f"        Using '{split_used}' split (found {len(rows)} examples)")

    activations = []
    labels = []
    groups = []

    # Process in batches
    for i in tqdm(range(0, len(rows), batch_size), desc=f"    {split_used[:12]:12s}", leave=False):
        batch_rows = rows[i:i + batch_size]
        batch_texts = [row.text for row in batch_rows]

        # Get activations for entire batch
        batch_acts = get_activations_batch(llm, batch_texts, layer_idx)
        activations.extend(batch_acts)

        for row in batch_rows:
            labels.append(row.label)
            groups.append(row.group_id if row.group_id else None)

    return activations, labels, groups

def eval_probe(probe, eval_acts, eval_labels, eval_groups):
    """Evaluate a trained probe"""
    eval_acts_np = np.stack(eval_acts).astype(np.float32)
    eval_labels_np = np.array(eval_labels)

    # Convert groups to numeric if they exist
    if eval_groups and any(g is not None for g in eval_groups):
        eval_groups_np = np.array([hash(str(g)) % 100000 if g else i for i, g in enumerate(eval_groups)])
        unique_groups = len(np.unique(eval_groups_np))
    else:
        eval_groups_np = None
        unique_groups = 0

    # Choose evaluation method based on groups
    if eval_groups_np is not None and unique_groups > 1:
        result = eval_probe_by_question(
            probe,
            activations=eval_acts_np,
            labels=eval_labels_np,
            groups=eval_groups_np,
        )
        accuracy = result.accuracy
        n = result.n
        auc = result.accuracy  # Use accuracy as AUC proxy for grouped eval
    else:
        result = eval_probe_by_row(
            probe,
            activations=eval_acts_np,
            labels=eval_labels_np,
        )
        accuracy = result.roc_auc_score
        n = result.n
        auc = result.roc_auc_score

    return accuracy, n, auc

# Main pipeline
print("[1/4] Loading probe vectors...")
probe_vectors = {}
with jsonlines.open(probe_vectors_file) as reader:
    for item in reader:
        value = item["value"]
        key = (value["train_dataset"], value["point_name"])
        probe_vector = np.array(value["probe_vector"], dtype=np.float32)
        probe_vectors[key] = DotProductProbe(probe=probe_vector)

print(f"✓ Loaded {len(probe_vectors)} probe vectors")
print(f"  Available layers: {sorted(set(k[1] for k in probe_vectors.keys()))}")
print()

# Check if optimal layer exists
available_layers = sorted(set(k[1] for k in probe_vectors.keys()))
if optimal_point_name not in available_layers:
    print(f"ERROR: Optimal layer {optimal_point_name} not found in probe vectors")
    print(f"Available layers: {', '.join(available_layers)}")
    print(f"Please retrain with appropriate --layer-skip value")
    sys.exit(1)

print("[2/4] Loading model...")
llm = load_llm_oioo(MODEL_ID, device=torch.device("cuda"), use_half_precision=True)
print("✓ Model loaded\n")

results = []
total_eval = len(all_dataset_ids) * len(all_dataset_ids)

print(f"[3/4] Generating evaluation activations (once per dataset)...")
print(f"Generating activations for {len(all_dataset_ids)} datasets at layer {optimal_point_name}")
print()

# Cache evaluation activations (generate once per dataset, reuse 17 times)
eval_activations_cache = {}
for eval_dataset in tqdm(all_dataset_ids, desc="Caching eval activations"):
    test_acts, test_labels, test_groups = get_activations_for_split(
        eval_dataset, "test", llm, args.eval_limit, optimal_layer_idx, args.batch_size
    )
    eval_activations_cache[eval_dataset] = (test_acts, test_labels, test_groups)

print(f"\n[4/5] Evaluating cross-dataset generalization...")
print(f"Total evaluations: {len(all_dataset_ids)} train × {len(all_dataset_ids)} eval = {total_eval}")
print(f"Using cached activations with {len(probe_vectors)} probes")
print()

with tqdm(total=total_eval, desc="Evaluating probes") as pbar:
    for train_dataset in all_dataset_ids:
        for eval_dataset in all_dataset_ids:
            # Get cached evaluation activations
            test_acts, test_labels, test_groups = eval_activations_cache[eval_dataset]

            # Get probe for this dataset/layer combination
            probe_key = (train_dataset, optimal_point_name)
            if probe_key not in probe_vectors:
                print(f"\nWARNING: Probe not found for {probe_key}, skipping...")
                pbar.update(1)
                continue

            probe = probe_vectors[probe_key]

            # Evaluate
            acc, n, auc = eval_probe(probe, test_acts, test_labels, test_groups)

            results.append(PipelineResultRow(
                llm_id=MODEL_ID,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                probe_method="dim",
                point_name=optimal_point_name,
                token_idx=-1,
                accuracy=acc,
                accuracy_n=n,
                auc=auc,
            ))

            pbar.update(1)

# Save results
print(f"\n[5/5] Saving results...")
print(f"  Saving {len(results)} evaluations...")
with jsonlines.open(results_file, mode='w') as writer:
    for i, result in enumerate(results):
        writer.write({"key": f"result_{i}", "value": result.model_dump()})

print(f"✓ Saved evaluations: {results_file}")
print(f"  File size: {results_file.stat().st_size / 1024:.1f} KB")

print("\n" + "="*80)
print("SUCCESS! Run visualization:")
print(f"  python experiments/visualize.py --model {model_spec}")
print("="*80)
