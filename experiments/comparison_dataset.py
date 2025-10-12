import sys
sys.path.insert(0, "/root/repeng")

import torch

from repeng.datasets.activations.creation import create_activations_dataset
from repeng.datasets.elk.utils.collections import (
    DatasetCollectionId,
    resolve_dataset_ids,
)
from repeng.datasets.elk.utils.limits import Limits, SplitLimits

"""
17 datasets (excluding piqa due to HuggingFace dataset script deprecation)
 * 20 layers
 * 1 token
 * 4400 (400 + 2000 + 2000) questions
 * 3 answers
 * 5120 hidden dim size
 * 2 bytes
= ~46GB
"""

collections: list[DatasetCollectionId] = ["dlk", "repe", "got"]

# Get all dataset IDs but exclude piqa (deprecated loading script)
all_dataset_ids = [
    dataset_id
    for collection in collections
    for dataset_id in resolve_dataset_ids(collection)
    if dataset_id != "piqa"  # Skip piqa - uses deprecated HuggingFace dataset script
]

print(f"Generating activations for {len(all_dataset_ids)} datasets:")
for ds in all_dataset_ids:
    print(f"  - {ds}")
print()

results = create_activations_dataset(
    tag="datasets_2025-qwen3-4b_v1",
    llm_ids=["Qwen/Qwen3-4B"],  # Newly released Qwen3-4B with reasoning capabilities
    dataset_ids=[
        *all_dataset_ids,
        "truthful_qa",
    ],
    group_limits=Limits(
        default=SplitLimits(
            train=400,
            train_hparams=2000,
            validation=2000,
        ),
        by_dataset={
            "truthful_qa": SplitLimits(
                train=0,
                train_hparams=0,
                validation=2000,
            )
        },
    ),
    num_tokens_from_end=1,
    device=torch.device("cuda"),
    layers_start=1,
    layers_end=None,
    layers_skip=2,
)

# Save results locally in the format expected by experiments
print(f"\n{'='*80}")
print(f"Saving {len(results)} activation rows locally...")

import pickle
from pathlib import Path

output_dir = Path("output/comparison/activations_results")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "value.pickle"

with open(output_file, 'wb') as f:
    pickle.dump(results, f)

print(f"✓ Saved to: {output_file}")
print(f"✓ File size: {output_file.stat().st_size / 1e9:.2f} GB")
print("="*80)
print("SUCCESS! You can now run:")
print("  python experiments/probe_ensemble/run_all_experiments.py")
print("="*80)
