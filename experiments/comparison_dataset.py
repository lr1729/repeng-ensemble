import sys
sys.path.insert(0, "/root/repeng")

import argparse
import torch
from pathlib import Path

from repeng.datasets.activations.creation import create_activations_dataset
from repeng.datasets.elk.utils.collections import (
    DatasetCollectionId,
    resolve_dataset_ids,
)
from repeng.datasets.elk.utils.limits import Limits, SplitLimits

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Generate activation datasets for Qwen3 and Llama-2 models',
    epilog="""
Examples:
  # Qwen3 models
  python comparison_dataset.py --model qwen3-4b
  python comparison_dataset.py --model qwen3-8b
  python comparison_dataset.py --model qwen3-14b

  # Llama-2 models
  python comparison_dataset.py --model llama2-7b
  python comparison_dataset.py --model llama2-13b
  python comparison_dataset.py --model llama2-70b
"""
)
parser.add_argument(
    '--model',
    type=str,
    required=True,
    choices=['qwen3-4b', 'qwen3-8b', 'qwen3-14b', 'llama2-7b', 'llama2-13b', 'llama2-70b'],
    help='Model to use (e.g., qwen3-8b, llama2-13b)'
)
args = parser.parse_args()

model_spec = args.model

# Map shortcuts to HuggingFace model IDs
if model_spec.startswith('qwen3-'):
    size = model_spec.replace('qwen3-', '').upper()
    MODEL_ID = f"Qwen/Qwen3-{size}"
    layers_skip = 2  # Sample every 2nd layer for Qwen3
    output_dir_name = f"qwen3-{model_spec.replace('qwen3-', '')}"
elif model_spec.startswith('llama2-'):
    size = model_spec.replace('llama2-', '')
    MODEL_ID = f"Llama-2-{size}-hf"
    layers_skip = 4  # Sample every 4th layer for Llama-2
    output_dir_name = f"llama2-{size}"
else:
    raise ValueError(f"Unknown model: {model_spec}")

print(f"="*80)
print(f"MODEL: {MODEL_ID}")
print(f"OUTPUT DIR: output/comparison/{output_dir_name}/")
print(f"LAYERS SKIP: {layers_skip}")
print(f"="*80)

collections: list[DatasetCollectionId] = ["dlk", "repe", "got"]

# Get all dataset IDs but exclude piqa (deprecated loading script)
all_dataset_ids = [
    dataset_id
    for collection in collections
    for dataset_id in resolve_dataset_ids(collection)
    if dataset_id != "piqa"  # Skip piqa - uses deprecated HuggingFace dataset script
]

print(f"\nGenerating activations for {len(all_dataset_ids)} datasets:")
for ds in all_dataset_ids:
    print(f"  - {ds}")
print()

results = create_activations_dataset(
    tag=f"activations_{output_dir_name}",
    llm_ids=[MODEL_ID],
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
    layers_skip=layers_skip,
)

# Save results locally
print(f"\n{'='*80}")
print(f"Saving {len(results)} activation rows locally...")

import pickle

output_dir = Path(f"output/comparison/{output_dir_name}/activations_results")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "value.pickle"

with open(output_file, 'wb') as f:
    pickle.dump(results, f)

print(f"✓ Saved to: {output_file}")
print(f"✓ File size: {output_file.stat().st_size / 1e9:.2f} GB")
print("="*80)
print("SUCCESS! You can now run:")
print(f"  python experiments/comparison.py --model {model_spec}")
print("="*80)
