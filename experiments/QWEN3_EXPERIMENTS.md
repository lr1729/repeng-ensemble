# Qwen3 Cross-Dataset Generalization Experiments

Replicate mishajw's cross-dataset generalization experiments with Qwen3-4B, 8B, and 14B models.

## Quick Start

### 1. Generate Activations

```bash
# Qwen3-4B (4 billion parameters)
python experiments/comparison_dataset.py --model 4b

# Qwen3-8B (8 billion parameters)
python experiments/comparison_dataset.py --model 8b

# Qwen3-14B (14 billion parameters)
python experiments/comparison_dataset.py --model 14b
```

**Time:** 1-2 hours per model (requires GPU)
**Output:** `output/comparison/qwen3-{model}/activations_results/value.pickle`

### 2. Run Analysis & Generate Figures

```bash
# Analyze Qwen3-4B results
python experiments/comparison.py --model 4b

# Analyze Qwen3-8B results
python experiments/comparison.py --model 8b

# Analyze Qwen3-14B results
python experiments/comparison.py --model 14b
```

**Time:** 30-60 minutes per model
**Output:** All figures saved to `output/comparison/qwen3-{model}/`

## Output Files

After running analysis for a model (e.g., 4b), you'll find:

```
output/comparison/qwen3-4b/
├── r0_acc_by_layer.png              # ECDF by layer
├── r1a_by_algorithms.png            # Best probe by algorithm
├── r1b_by_train.png                 # Best probe by training dataset
├── r1c_by_layer.png                 # Best probe by layer
├── r2_probes.png                    # Algorithm performance
├── r3a_datasets.png                 # Dataset performance bar chart
├── r3b_matrix.png                   # Cross-dataset generalization matrix
├── r3c_to_and_from.png              # Generalizes from/to scatter
├── r3d_clustering.png               # Hierarchical clustering
├── r4_truthful_qa.png               # TruthfulQA correlation
├── v1_arc_easy.png                  # ARC-easy validation
├── v2_got.png                       # GeometryOfTruth validation
├── v3_repe.png                      # RepE validation
└── v4_lda.png                       # LDA investigation
```

## What Gets Analyzed

- **17 datasets** from DLK, RepE, and GoT families
- **8 probe algorithms:** DIM, LDA, LR, LR-G, CCS, PCA, PCA-G, LAT
- **Cross-dataset generalization:** Train on dataset A, test on dataset B
- **Recovered accuracy metric:** Normalizes for dataset difficulty

## Expected Results (Qwen3-4B vs Llama-2-13B)

Based on existing experiments:

| Metric | Llama-2-13B | Qwen3-4B | Change |
|--------|-------------|----------|--------|
| Mean cross-dataset accuracy | 70% | **84.3%** | +14.3% |
| Probes >80% recovered | 36% | **70%** | +34% |
| Best probe (ag_news) | 92.8% | **94.3%** | +1.5% |

## Hardware Requirements

- **GPU:** 16GB+ VRAM (A40, V100, A100)
- **RAM:** 32GB+ recommended
- **Disk:** ~50GB per model
- **Time:** ~2-3 hours total per model

## Datasets Used

**DLK (Discovering Latent Knowledge):**
- imdb, amazon_polarity, ag_news, dbpedia_14, rte, copa, boolq

**RepE (Representation Engineering):**
- openbook_qa, common_sense_qa, race, arc_challenge, arc_easy

**GoT (Geometry of Truth):**
- got_cities, got_sp_en_trans, got_larger_than
- got_cities_cities_conj, got_cities_cities_disj

**Plus:** truthful_qa for validation

## Key Figures Explained

- **r3b_matrix.png:** Main cross-dataset generalization heatmap
- **r3c_to_and_from.png:** Which datasets generalize well?
- **r2_probes.png:** Which probe algorithms work best?
- **r3a_datasets.png:** Dataset performance comparison

## Original Paper

mishajw (2024). "How well do truth probes generalise?"
https://www.lesswrong.com/posts/rbyvSZTvYqyqf5ksB/how-well-do-truth-probes-generalise
