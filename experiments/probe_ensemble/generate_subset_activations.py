#!/usr/bin/env python3
"""
Generate activation data for a subset of 5 datasets for quick testing.

This creates enough data to run experiments in ~30 minutes instead of 2-4 hours.
"""

import sys
sys.path.insert(0, "/root/repeng")

import torch
from repeng.datasets.activations.creation import create_activations_dataset
from repeng.datasets.elk.utils.limits import Limits, SplitLimits

print("="*80)
print("Generating Subset Activation Data (5 datasets)")
print("="*80)
print("\nThis will take ~30 minutes and use ~48GB VRAM")
print("\nDatasets:")
print("  1. dbpedia_14 (factual)")
print("  2. amazon_polarity (sentiment)")
print("  3. open_book_qa (knowledge)")
print("  4. got_cities_cities_conj (relational)")
print("  5. got_larger_than (numerical)")
print("\n" + "="*80)

create_activations_dataset(
    tag="subset_5_datasets",
    llm_ids=["Qwen/Qwen3-4B"],
    dataset_ids=[
        "dbpedia_14",           # DLK - factual
        "amazon_polarity",      # DLK - sentiment
        "open_book_qa",         # RepE - knowledge
        "got_cities_cities_conj",  # GoT - relational
        "got_larger_than",      # GoT - numerical
    ],
    group_limits=Limits(
        default=SplitLimits(
            train=400,
            train_hparams=0,  # Skip to save time
            validation=2000,
        ),
    ),
    num_tokens_from_end=1,
    device=torch.device("cuda"),
    layers_start=1,
    layers_end=None,
    layers_skip=2,
)

print("\n" + "="*80)
print("✓ Subset activation data generated!")
print("="*80)
print("\nOutput location: output/create-activations-dataset/activations/value.pickle")
print("\nNext step:")
print("  python experiments/probe_ensemble/run_all_experiments.py")
print("="*80)
