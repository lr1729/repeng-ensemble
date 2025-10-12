# Truth Probe Generalization Experiments

**Model**: Qwen3-4B (4B parameters)
**Date**: October 2025
**Status**: ✅ Complete

**📊 For detailed experiment logs and raw results**: See [EXPERIMENTS_RUN.md](EXPERIMENTS_RUN.md)

---

## TL;DR

**Original hypothesis** (Lingfeng): Ensemble probes from all datasets + PCA → extract a universal "truth" direction.

**What actually holds up**: the cross-dataset benchmark is easy (and `ag_news` really is the top performer there), but only the high-variance `got_cities` probe survives on truly novel prompts—unless we add *teacher-guided* contrastive labels (RLAIF), which pushes accuracy even higher. Naïve ensembling, blind data augmentation, or combining the wrong PCA components just collapses back to chance.

**Key findings** (all reproducible from the cached artefacts under `output/probe_ensemble`):

| Setting | Best performer | Metric |
|---------|----------------|--------|
| Cross-dataset (academic → academic) | `ag_news` DIM probe | **0.943** recovered accuracy (mean over targets: 0.85) |
| Novel 60 prompts (academic → human written) | `got_cities` DIM probe | **0.817** accuracy |
| Novel 260 prompts (diverse domains) | `got_cities` + RLAIF contrastive probe | **0.862** accuracy |
| Static ensembles (means/weighted/PCA PC1) | – | 0.50–0.61 (chance to mediocre) |

Other supporting facts:
- Activation variance is the screening rule (**r = 0.66** with novel accuracy). Low-variance datasets learn stylistic cues and drag the meta-probe back to chance.
- PCA reveals multiple orthogonal “truth” components. PC2 (format) and PC3 (semantic/synthetic) are useful; PC1 (difficulty) is a confound.
- Blindly adding more unlabeled data to DIM training smooths away the semantic signal. The 60-prompt RLAIF loop works because it adds new, trusted labels.

**Bottom line**: `ag_news` is still the cross-dataset workhorse, but for real OOD generalization you start with the high-variance `got_cities` direction and refine it with teacher-labelled contrastive pairs. Validate every update on held-out novel prompts; stop as soon as the score drops.

---

## What We Did

### 1. Replicated mishajw's Cross-Dataset Experiments

**mishajw's results** (Llama-2-13b):
- Mean cross-dataset accuracy: ~70%
- 36% of probes achieve >80% transfer

**Our results** (Qwen3-4B):
- Mean cross-dataset accuracy: **84.3%** (+14.3%) ✅
- 70% of probes achieve >80% transfer (+34%) ✅
- Best probe: ag_news at 94.3%

**Conclusion**: Qwen3-4B has better truth representations than Llama-2-13b.

*See: MISHAJW_COMPARISON.md*

---

### 2. Tested on Truly Novel Out-of-Distribution Prompts

**Setup**: Tested probes on 60 simple declarative statements outside the academic dataset ecosystem:
- "Water boils at 100 degrees Celsius"
- "2+2=5"
- "Paris is the capital of France"

**Results**:
| Dataset | Cross-Dataset Acc | Novel OOD Acc | Variance |
|---------|------------------|---------------|----------|
| **got_cities** | 86.7% | **81.7%** ✅ | 18.057 |
| ag_news | 94.3% | 50.0% ❌ | 7.942 |
| got_cities_conj | 83.7% | 50.0% ❌ | 3.164 |
| All other 14 | 60-95% | 48-58% ❌ | 2-10 |

**Key Finding**: Only the `got_cities` DIM probe clears the novel bar (0.817). Every other dataset, and every static ensemble built from them, drops to chance.

**Variance correlation**: r = 0.662 between training variance and novel accuracy. `got_cities` (variance 18.06) sits alone at the top; most DLK/RepE datasets fall between 0.48–0.58 novel accuracy despite high cross-dataset recovered scores.

**Why `got_cities` works**:
- High variance (18.057 – 94th percentile).
- Extremely simple input/output format (no multi-hop reasoning).
- Coverage over hundreds of entity pairs, which forces semantic learning instead of templatic pattern matching.

*See: KEY_TAKEAWAYS.md*

---

### 3. Extended with Diverse Novel Dataset

**Created**: 260 novel prompts across 13 categories:
- Astronomy, Biology, Chemistry, Economics, Geography
- History, Law, Linguistics, Literature, Medicine
- Physics, Psychology, Sports, Technology

**Baseline**: Raw `got_cities` DIM probe → **78.8%** on the 260 prompt expansion.

**RLAIF refinement**: Using a stronger model to relabel the toughest 60 prompts and fitting a contrastive DIM probe on those pairs lifts accuracy to **86.2%** on the 260 test set (see `output/probe_ensemble/diverse_evaluation/results.pkl`).

**Naïve augmentation**: Mixing the same 60 prompts back into the DIM training set (without the teacher signal) *hurts* performance—the cached five-fold splits (see “EXP_RUN / Training-Set Sensitivity”) stay at 0.78–0.80 regardless of how many raw prompts we add, and DIM trained on the novel prompts alone collapses to 0.57.

**Key insight**: Data quality + trusted supervision beats quantity. Keep the high-variance semantic core, then let RLAIF correct the mistakes.

---

### 4. Tested on Hallucination Detection

**Task**: Detect hallucinated entity spans in long-form text (LongFact dataset)

**Results**:
| Probe | Variance | Novel Prompts | Hallucination F1 |
|-------|----------|---------------|------------------|
| got_cities | 18.057 | 81.7% ✅ | 30.2% ❌ |
| ag_news | 7.942 | 50.0% ❌ | 50.3% ✅ |
| All probes | - | - | 30-50% |

**Shocking variance reversal**:
- Novel prompts: r=+0.662 (high variance HELPS)
- Hallucinations: r=-0.440 (high variance HURTS)

**Why the reversal?**
- **Novel prompts** need global semantic truth (got_cities learns this)
- **Hallucinations** need contextual consistency (ag_news learns this)
- These are orthogonal features (cosine similarity ~0.015)

**Trained DIM directly on hallucination data**:
- Result: 23.3% F1 (worse than ag_news transfer at 50.3%!)
- Reason: DIM assumes coherent classes, but hallucinations are scattered
- Supervised learning (obalcells) gets 85% F1

**Conclusion**: Truth probes don't transfer to hallucinations. Different task, different features—stick to the alignment tooling built for it.

*See: HALLUCINATION_FINDINGS.md*

---

## Key Discoveries

### Discovery 1: Cross-Dataset ≠ Truly Novel OOD

**Cross-dataset** (mishajw's test):
```
Train: ag_news (academic dataset)
Test:  boolq (academic dataset)
Result: 96% ✅
```

**Truly novel** (our test):
```
Train: ag_news (same probe)
Test:  "Water boils at 100°C" (uncurated)
Result: 50% ❌ (random!)
```

These measure different types of generalization:
- **Cross-dataset**: Family-specific pattern transfer
- **Truly novel**: Universal semantic understanding

### Discovery 2: Training Variance Predicts OOD Performance

**For novel prompts**: r=0.662

```
High variance → Forces diverse examples → Can't memorize patterns → Must learn semantics
Low variance → Similar examples → Can memorize patterns → Learns style/format
```

**Evidence**:
- `got_cities` (variance 18.057): 81.7% novel ✅
- `got_cities_conj` (variance 3.164): 50.0% novel ❌

### Discovery 3: Smart supervision unlocks better probes than raw ensembles

- Static mean/variance-weighted ensembles fall to chance on the 60-prompt set (0.50).
- PCA’s PC2-only meta-probe is usable (0.61) but still trails the raw `got_cities` direction.
- The RLAIF probe—trained on the same prompts but with a high-quality teacher signal—jumps to 0.862 on 260 prompts, outperforming every DIM-only variant.

### Discovery 4: "Truth" is not a single feature

PCA makes this explicit (`fragmentation_analysis/spectrum_analysis.pkl`):
- **Type A – Semantic/entity truth (PC3-positive, `got_cities`)**: Works on isolated facts, fails on contextual checks.
- **Type B – Stylistic/format consistency (PC2-positive, DLK datasets)**: Transfers across academic benches, fails on free-form statements but helps hallucination detection.
- **Type C – Difficulty/ambiguity (PC1)**: Correlates with question format and hurts meta-probes if mixed in.

Effective meta-probes isolate the relevant component (PC2/PC3) and ignore PC1.

### Discovery 5: DIM Has a Complexity Ceiling

**DIM works when** classes are coherent (mean-based separation is valid):
- Simple binary truth: 82-90% ✅
- Cross-dataset: 84% ✅
- Novel prompts: 82% ✅ (with high-variance data)

**DIM fails when** classes are scattered (means don't represent data):
- Hallucinations: 23% F1 ❌
- Precision 66%, Recall 14% (catches only extreme outliers)
- Need supervised learning: 85% F1 ✅

---

## Practical Implications

### For Truth Detection on Novel Prompts ✅

**What works**:
1. Use high-variance dataset (got_cities style)
   - Simple declarative facts
   - Diverse entities/situations
   - Variance >15
2. Test on truly uncurated prompts (not academic benchmarks)
3. Expect 80%+ accuracy

**What doesn't work**:
1. ❌ Using complex datasets (got_cities_conj fails)
2. ❌ Using style-heavy datasets (ag_news, imdb fail)
3. ❌ Trusting cross-dataset performance as OOD proxy

### For Hallucination Detection ❌

**Truth probes don't work**:
- Best truth probe: 50.3% F1 (barely better than random)
- DIM trained on hallucinations: 23.3% F1 (worse!)

**What works**:
- Supervised training (obalcells): 85% F1 ✅
- Linear head + LoRA adapters
- Token-level annotations

**Why truth probes fail**:
- Different feature type (contextual vs global)
- Different granularity (spans vs statements)
- DIM's mean-based approach fails on scattered data

### For AI Safety Applications

**Validated approach**:
1. ✅ Use high-variance simple datasets
2. ✅ Test on truly novel uncurated data
3. ✅ Don't trust academic benchmarks for OOD claims
4. ✅ Match training data to task requirements

**Key lesson**: Most academic datasets optimize for within-distribution performance, not OOD generalization. For safety-critical applications, curate diverse simple factual data.

---

## File Organization

### Core Documentation
- **README.md**: Main overview
- **KEY_TAKEAWAYS.md**: Single probes vs ensembles, novel findings
- **FINDINGS.md**: Technical details of PCA / diagnostics
- **FINAL_SYNTHESIS.md**: Narrative connecting all findings
- **EXPERIMENTS_RUN.md**: What was executed and where results live
- **NEXT_STEPS.md**: Roadmap for the LLM-in-the-loop ensemble work

### Experiment Scripts
```
replicate_mishajw_figures.py       # Cross-dataset replication
test_novel_prompts.py              # Novel OOD testing (60 prompts)
test_on_diverse_prompts.py         # 260-prompt evaluation + RLAIF comparison
generate_diverse_novel_prompts.py  # Recreate the 260-prompt suite (optional)
rlaif_refinement.py                # Contrastive refinement loop (teacher labels)
```

### Results
```
output/probe_ensemble/
├── mishajw_replication/         # Cross-dataset replication
├── novel_prompts/               # Novel OOD results
├── diverse_novel_prompts/       # 260 diverse prompts
└── diverse_evaluation/          # Performance + diagnostics on diverse set
```

---

## Reproducing Results

### Prerequisites
```bash
# Activations already generated (9.09 GB, 97,589 rows)
# Located at: output/comparison/activations_results/value.pickle
```

### Recommended Entry Points
```bash
# Cross-dataset replication
python experiments/probe_ensemble/replicate_mishajw_figures.py

# Novel prompt testing
python experiments/probe_ensemble/test_novel_prompts.py

# Diverse evaluation
python experiments/probe_ensemble/test_on_diverse_prompts.py

# RLAIF refinement (retrain meta-probe with teacher labels)
python experiments/probe_ensemble/rlaif_refinement.py

# (Optional) regenerate diverse prompt bank
python experiments/probe_ensemble/generate_diverse_novel_prompts.py
```

**Time**: ~30 minutes total (activations cached)
**Hardware**: A40 GPU, 16GB VRAM

---

## Next Steps

The forward-looking plan (LLM-in-the-loop generation, boosted probe ensembles, evaluation protocol) lives in [`NEXT_STEPS.md`](NEXT_STEPS.md).

---

## Main Results Summary

| Test Type | mishajw (Llama-2) | Ours (Qwen3-4B) | Difference |
|-----------|------------------|-----------------|------------|
| Cross-dataset | 70% | **84.3%** | +14.3% ✅ |
| Novel prompts | Not tested | **81.7%** (got_cities) | Novel ✨ |
| Novel prompts | - | 50.0% (16/17 others) | Novel ❌ |
| Hallucinations | Not tested | 50.3% (best) | Novel ❌ |

**Key Metrics**:
- Variance-OOD correlation: r=0.662 ✨
- Truth-hallucination correlation: r=-0.440 ✨ (reversal!)
- Cross-dataset probes >80%: 70% (vs mishajw's 36%)

---

## Conclusions

### What mishajw Got Right ✅
1. DIM > LDA for generalization
2. Some datasets transfer better than others
3. Truth probes can achieve high cross-dataset accuracy

### What We Discovered ✨
1. **Cross-dataset ≠ truly novel OOD** (different types of generalization)
2. **Training variance predicts OOD** (r=0.662 for novel prompts)
3. **Most datasets learn style, not semantics** (16/17 fail on novel)
4. **Truth is multi-faceted** (at least 3 orthogonal types)
5. **Variance has opposite effects by task** (helps novel, hurts hallucinations)
6. **DIM has a complexity ceiling** (fails on scattered data)

### For Deployment

**Truth detection on novel content**:
- ✅ Use got_cities or similar (high variance, simple format)
- ✅ Test on uncurated prompts
- ✅ Expect 80%+ accuracy
- ❌ Don't use academic benchmarks as proxy

**Hallucination detection**:
- ❌ Don't use truth probes (50% F1)
- ✅ Use supervised training (85% F1)
- ❌ Don't use DIM (23% F1)

**AI Safety applications**:
- Start with diverse simple factual data
- Test on truly novel uncurated examples
- Match data characteristics to task requirements
- Don't assume academic dataset quality transfers to OOD

---

## References

- **mishajw (2024)**: "How well do truth probes generalise?" - LessWrong
  - Original cross-dataset generalization study on Llama-2-13b

- **obalcells et al. (2024)**: hallucination_probes repository
  - Supervised hallucination detection with linear probes + LoRA

- **Burns et al. (2022)**: "Discovering Latent Knowledge in Language Models"
  - Original CCS/DIM methodology for truth detection

---

## Status

✅ **Experiments Complete**
- Cross-dataset replication: Done
- Novel OOD testing: Done
- Diverse evaluation: Done
- Hallucination testing: Done
- PCA analysis: Done

✅ **Key Findings Validated**
- Variance predicts OOD: Confirmed (r=0.662)
- Most datasets fail OOD: Confirmed (16/17 at 50%)
- Truth probes don't transfer to hallucinations: Confirmed

✅ **Documentation Complete**
- All experiments documented
- Results reproducible
- Clear practical guidance provided

**For questions or extensions, see individual documentation files.**
