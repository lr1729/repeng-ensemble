# Cross-Dataset Generalization Experiments

Replicate mishajw's cross-dataset generalization experiments with Qwen3 and Llama-2 models using DIM (Difference in Means) probes.

## Quick Start

### Step 1: Extract Probe Vectors

This script generates activations on-the-fly with batching and trains probes efficiently:
- **Training:** Extracts all sampled layers (every 2nd layer) and trains 306 probes (17 datasets × 18 layers)
- **Batching:** Uses batch_size=32 for optimal throughput
- Saves all trained probe vectors for future reuse (~5MB)

**Why train all layers?**
- Enables future experimentation without re-running (only ~25-30 min with batching)
- Can evaluate at different layer depths later using saved probe vectors
- Middle layers (~75% depth) generalize better than final layers

```bash
# Qwen3 models (default: layer_skip=2, batch_size=16)
python experiments/extract_vectors.py --model qwen3-4b
python experiments/extract_vectors.py --model qwen3-8b
python experiments/extract_vectors.py --model qwen3-14b

# Llama-2 models (requires HuggingFace authentication)
python experiments/extract_vectors.py --model llama2-7b
python experiments/extract_vectors.py --model llama2-13b
python experiments/extract_vectors.py --model llama2-70b

# Optional: Customize layer sampling or batch size
python experiments/extract_vectors.py --model qwen3-4b --layer-skip 4     # Sample every 4th layer
python experiments/extract_vectors.py --model qwen3-4b --batch-size 16    # Smaller batches if OOM
```

**Time:** ~25-30 minutes per model (requires GPU, with proper batched inference)
**Output:** `output/comparison/{model}/probe_vectors.jsonl` (~5MB trained probe vectors, 306 probes)

### Step 2: Evaluate Cross-Dataset Generalization

This script loads pre-trained probes and evaluates them on all datasets:
- **Efficient caching:** Generates activations once per eval dataset (17 total), not once per train/eval pair (289)
- **Fast evaluation:** Reuses cached activations for all 17 probes per dataset
- **Flexible testing:** Use `--eval-limit 200` for quick tests (~1-2 min)

```bash
# Full evaluation (2000 examples per dataset, ~5-7 min)
python experiments/evaluate_vectors.py --model qwen3-4b

# Quick test (200 examples per dataset, ~1-2 min)
python experiments/evaluate_vectors.py --model qwen3-4b --eval-limit 200

# Different layer depth
python experiments/evaluate_vectors.py --model qwen3-4b --layer-depth 0.80
```

**Time:** ~5-7 minutes per model (full evaluation), ~1-2 minutes (quick test)
**Output:** `output/comparison/{model}/probe_evaluate-v2.jsonl` (~15KB, 289 evaluations)

### Step 3: Visualize Results

Auto-selects best performing layer (or specify manually with `--layer`):

```bash
# Auto layer selection (recommended)
python experiments/visualize.py --model qwen3-4b
python experiments/visualize.py --model llama2-13b

# Or specify layer manually
python experiments/visualize.py --model qwen3-4b --layer 35
```

**Time:** <1 minute per model
**Output:** Two heatmaps showing different metrics:
- `cross_dataset_matrix_recovered.png` - Recovered accuracy (relative generalization)
- `cross_dataset_matrix_auc.png` - AUC (absolute probe quality)

## Output Files

After running both scripts for a model, you'll find:

```
output/comparison/{model}/
├── probe_evaluate-v2.jsonl                # Evaluation results at optimal layer (~15KB, 289 tests)
├── probe_vectors.jsonl                    # All trained probe vectors (~5MB, 306 probes)
├── cross_dataset_matrix_recovered.png     # Recovered accuracy heatmap
└── cross_dataset_matrix_auc.png           # AUC (absolute quality) heatmap
```

## What Gets Analyzed

- **17 datasets** from DLK, RepE, and GoT families (piqa excluded)
- **1 probe algorithm:** DIM (Difference in Means) - supervised probe
- **Training:** All sampled layers (every 2nd layer = 18 layers for 36-layer models)
- **Evaluation:** Single optimal layer at 75% depth (h27 for 36-layer models)
- **Cross-dataset generalization:** Train DIM probe on dataset A, test on dataset B
- **Recovered accuracy metric:** Normalizes for dataset difficulty
- **Layer selection strategy:**
  - Train probes at all sampled layers for flexibility
  - Evaluate only at ~75% depth (empirically best for generalization)
  - Can re-evaluate at different layers later without retraining

## Understanding Layer Depth Selection

The script automatically selects the optimal layer based on model architecture:
- **Qwen3-4B/8B (36 layers):** 75% = layer 27 (h27)
- **Qwen3-14B (40 layers):** 75% = layer 30 (h30)
- **Llama-2-7B (32 layers):** 75% = layer 24 (h24)
- **Llama-2-13B (40 layers):** 75% = layer 30 (h30)
- **Llama-2-70B (80 layers):** 75% = layer 60 (h60)

You can override this with `--layer-depth` (e.g., `--layer-depth 0.80` for 80% depth).

## Hardware Requirements

- **GPU:** 16GB+ VRAM for inference (A40, V100, A100)
- **RAM:** 32GB+ recommended
- **Disk:** ~10-50GB for model weights (no activation caching)
- **Time:** ~45-60 minutes per model with proper batched inference

## Datasets Used

**DLK (Discovering Latent Knowledge) - 7 datasets:**
- imdb, amazon_polarity, ag_news, dbpedia_14, rte, copa, boolq

**RepE (Representation Engineering) - 5 datasets:**
- open_book_qa, common_sense_qa, race, arc_challenge, arc_easy

**GoT (Geometry of Truth) - 5 datasets:**
- got_cities, got_sp_en_trans, got_larger_than
- got_cities_cities_conj, got_cities_cities_disj

**Note:** piqa is excluded due to deprecated HuggingFace loading script

## How It Works

### Layer Selection
- The pipeline uses a single optimal layer at ~75% depth through the model
- This layer provides the best trade-off between in-distribution and OOD performance
- Based on empirical findings from sanity check experiments
- You can experiment with different depths using `--layer-depth`

### Cross-Dataset Generalization

1. **Train:** DIM probe trained on dataset A using 400 examples from train split
2. **Evaluate:** Probe tested on dataset B using 2000 examples from test split (with fallback to validation/train)

### Metrics Explained

**Recovered Accuracy (shown in the matrix):**
```
recovered_accuracy = (probe accuracy on dataset B) / (best in-distribution accuracy on dataset B)
```

- **Purpose:** Normalizes for dataset difficulty
- **Range:** 0-100%
- **100% means:** Probe generalizes perfectly (as good as training directly on target dataset)
- **Example:**
  - Best probe on IMDB (trained & tested on IMDB): 90%
  - Probe trained on AG_NEWS, tested on IMDB: 81%
  - Recovered accuracy = 81% / 90% = **90%**

**⚠️ Important caveat:** If the best in-distribution probe is poor (say 60%), then a 60% probe would show 100% recovered accuracy even though it's objectively bad.

**AUC (ROC-AUC) - also reported:**
- **Purpose:** Absolute measure of probe quality
- **Range:** 0.5 (random) to 1.0 (perfect)
- **Independent of dataset difficulty**
- **Use this to check:** Are the probes actually good, or just relatively good?

**Both metrics together** give you the full picture:
- **Recovered accuracy:** How well does generalization work?
- **AUC:** Are the probes actually performing well?

## Original Paper

mishajw (2024). "How well do truth probes generalise?"
https://www.lesswrong.com/posts/rbyvSZTvYqyqf5ksB/how-well-do-truth-probes-generalise
