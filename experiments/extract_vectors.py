#!/usr/bin/env python3
"""
Train truthfulness probes across multiple layers

This script:
1. Generates activations on-the-fly from sampled layers with batching
2. Trains DIM probes for all sampled layers (306 probes = 17 datasets × 18 layers)
3. Saves all trained probe vectors for future reuse (~5MB)

Usage:
    python experiments/train_probes.py --model qwen3-4b
    python experiments/train_probes.py --model qwen3-4b --layer-skip 4 --batch-size 16
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
from repeng.activations.probe_preparations import ActivationArrays
from repeng.models.loading import load_llm_oioo
from repeng.models.points import get_points
from repeng.models.types import LlmId
from repeng.probes.collections import train_probe

# Parse arguments
parser = argparse.ArgumentParser(description='Train truthfulness probes for all layers')
parser.add_argument('--model', type=str, required=True,
                    choices=['qwen3-4b', 'qwen3-8b', 'qwen3-14b',
                             'qwen3-4b-base', 'qwen3-8b-base', 'qwen3-14b-base',
                             'llama2-7b', 'llama2-13b', 'llama2-70b',
                             'llama2-7b-chat', 'llama2-13b-chat', 'llama2-70b-chat'],
                    help='Model to use')
parser.add_argument('--layer-skip', type=int, default=2,
                    help='Sample every Nth layer for training probes (default 2)')
parser.add_argument('--batch-size', type=int, default=16,
                    help='Batch size for activation generation (default 16)')
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

# Get layers: sample for training
all_points = get_points(MODEL_ID)
num_layers = len(all_points)

# Sample layers for training probes (every Nth layer)
sampled_points = all_points[1::args.layer_skip]
point_names = [p.name for p in sampled_points]
sampled_layer_indices = [1 + i * args.layer_skip for i in range(len(sampled_points))]

print(f"="*80)
print(f"TRAIN TRUTHFULNESS PROBES: {MODEL_ID}")
print(f"="*80)
print(f"Output directory: output/comparison/{model_spec}/")
print(f"  - probe_vectors.jsonl (trained probe vectors)")
print(f"Training: {len(sampled_points)} layers (every {args.layer_skip} layers)")
print(f"  Layers: {', '.join(point_names[:3])}...{', '.join(point_names[-2:])}")
print(f"Batch size: {args.batch_size}")
print(f"Probe: DIM (Difference in Means)")
print()

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

# Setup output
output_dir = Path(f"output/comparison/{model_spec}")
output_dir.mkdir(parents=True, exist_ok=True)

# Probe vector structure
class ProbeVectorRow(BaseModel, extra="forbid"):
    llm_id: LlmId
    train_dataset: str
    probe_method: str
    point_name: str
    probe_vector: list[float]

def get_activations_batch(llm, texts: list[str], layer_indices: list[int]):
    """Get activations for a batch of texts with proper batching"""
    tokenized = llm.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )

    input_ids = tokenized["input_ids"].to(next(llm.model.parameters()).device)
    attention_mask = tokenized["attention_mask"].to(next(llm.model.parameters()).device)

    # Get points for the layers we want
    points = [llm.points[idx] for idx in layer_indices if idx < len(llm.points)]

    # Run forward pass with hooks to grab activations
    from repeng.hooks.grab import grab_many
    with grab_many(llm.model, points) as activation_fn:
        with torch.no_grad():
            llm.model.forward(input_ids, attention_mask=attention_mask, return_dict=True)
        layer_activations = activation_fn()

    # Extract last token activation for each example
    activations_by_layer = defaultdict(list)

    for point_name, acts in layer_activations.items():
        batch_size = acts.shape[0]
        for i in range(batch_size):
            seq_len = attention_mask[i].sum().item()
            last_token_act = acts[i, seq_len - 1, :].detach()
            # Convert bfloat16 to float16 (numpy doesn't support bfloat16)
            last_token_act = last_token_act.to(dtype=torch.float16)
            last_token_act = last_token_act.cpu().numpy()
            activations_by_layer[point_name].append(last_token_act)

    return activations_by_layer

def get_activations_for_split(dataset_id: str, split: str, llm, limit: int, layer_indices: list[int], batch_size: int):
    """Get activations for a dataset split"""
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

    activations_by_layer = defaultdict(list)
    labels = []

    # Process in batches with REAL batching
    for i in tqdm(range(0, len(rows), batch_size), desc=f"    {split_used[:12]:12s}", leave=False):
        batch_rows = rows[i:i + batch_size]
        batch_texts = [row.text for row in batch_rows]

        # Get activations for entire batch at once
        batch_acts = get_activations_batch(llm, batch_texts, layer_indices)

        # Accumulate results
        for point_name, acts_list in batch_acts.items():
            activations_by_layer[point_name].extend(acts_list)

        for row in batch_rows:
            labels.append(row.label)

    return activations_by_layer, labels

def train_dim_probe(train_acts, train_labels):
    """Train DIM probe"""
    arrays = ActivationArrays(
        activations=np.stack(train_acts).astype(np.float32),
        labels=np.array(train_labels),
        groups=None,
        answer_types=None
    )
    return train_probe("dim", arrays)

# Main pipeline
print("[1/3] Loading model...")
llm = load_llm_oioo(MODEL_ID, device=torch.device("cuda"), use_half_precision=True)
print("✓ Model loaded\n")

probe_vectors = []
total_train = len(all_dataset_ids) * len(sampled_points)

print(f"[2/3] Training probes...")
print(f"Total probes to train: {len(all_dataset_ids)} datasets × {len(sampled_points)} layers = {total_train}")
print()

with tqdm(total=total_train, desc="Training probes") as pbar:
    for train_dataset in all_dataset_ids:
        print(f"\n  Training on: {train_dataset}")

        # Get training activations (400 examples) from all sampled layers
        train_acts_by_layer, train_labels = get_activations_for_split(
            train_dataset, "train", llm, 400, sampled_layer_indices, args.batch_size
        )

        # Train probe for each layer
        for point_name in point_names:
            probe = train_dim_probe(
                train_acts_by_layer[point_name],
                train_labels
            )

            # Save probe vector
            probe_vectors.append(ProbeVectorRow(
                llm_id=MODEL_ID,
                train_dataset=train_dataset,
                probe_method="dim",
                point_name=point_name,
                probe_vector=probe.probe.tolist(),
            ))

            pbar.update(1)

# Save probe vectors
probe_vectors_file = output_dir / "probe_vectors.jsonl"
print(f"\n[3/3] Saving {len(probe_vectors)} probe vectors...")
with jsonlines.open(probe_vectors_file, mode='w') as writer:
    for i, pv in enumerate(probe_vectors):
        writer.write({"key": f"probe_{i}", "value": pv.model_dump()})

print(f"✓ Saved probe vectors: {probe_vectors_file}")
print(f"  File size: {probe_vectors_file.stat().st_size / (1024*1024):.1f} MB")

print("\n" + "="*80)
print("SUCCESS! Probe vectors saved.")
print("Next step: Run evaluation")
print(f"  python experiments/evaluate_vectors.py --model {model_spec}")
print("="*80)
