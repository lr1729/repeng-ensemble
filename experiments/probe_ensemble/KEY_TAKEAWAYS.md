# Key Takeaways: Single Probes, Ensembles, and Smart Refinement

**Date**: 2025-10-12  
**Model**: Qwen/Qwen3-4B

---

## TL;DR

- Qwen3-4B nails the original cross-dataset benchmark (mean recovered accuracy **0.85**)—better than Llama‑2‑13b in mishajw’s study.
- The same probes crash to **~0.50** on truly novel, human-written statements **except** for one dataset: `got_cities`, which hits **0.817** thanks to its huge activation variance.
- Blind ensembling (means, weighted averages, PCA with PC1) drags the score back to chance; the variance rule is unforgiving.
- **Teacher-guided contrastive labels (RLAIF)** are the only safe way to refine the direction: the RLAIF probe sweeps the 260-question stress test at **0.862**, beating both the raw `got_cities` probe (0.788) and every static ensemble (≤0.842).
- Adding the same prompts as unlabeled DIM training examples actually *hurts* (see the cached training-split analysis in `EXPERIMENTS_RUN.md`).

Bottom line: start from the high-variance semantic probe, validate every change on held-out novel prompts, and only add data that comes with a reliable teacher signal.

---

## 1. Singles vs Ensembles

| Setting | Best Single Probe | Static Ensemble | RLAIF / Teacher Probe |
|---------|------------------|-----------------|-----------------------|
| Cross-dataset (academic → academic) | `ag_news` recovered 0.943 | N/A (singles already 0.85+) | N/A |
| Novel 60 prompts | `got_cities` **0.817** | Mean / weighted / PCA: 0.50–0.61 | **0.862** after RLAIF |
| Diverse 260 prompts | `got_cities` 0.788 | Weighted ensemble 0.842 | **0.862** |

Naïve ensembles (mean, weighted, “high variance top‑3”) never exceed the best single probe on novel data. PCA confirms why: PC2 and PC3 hold the task-relevant signal, while PC1 (difficulty) just adds noise.

---

## 2. Why `got_cities` Is the Only Useful Base Dataset

- Activation variance: **18.06** (over double the next dataset).  
- Format: trivial geography statements → forces semantic learning instead of prompt-template matching.  
- Novel accuracy: **0.817** on the 60-prompt set, **0.788** on the 260-prompt set.  
- Cross-dataset recovered accuracy: **0.867**—so it still plays nicely inside the academic benchmark.

Every other dataset tops out between 0.48 and 0.58 on the novel prompts despite strong cross-dataset scores because they mostly encode stylistic or logical patterns, not semantics. The variance/novel correlation is **r = 0.662** (see `systematic_dataset_analysis.py`).

---

## 3. Smart Refinement Beats Blind Augmentation

- **RLAIF contrastive probe**: Train DIM on the 60 human prompts relabelled by a stronger model → **0.862** on the 260 test set.  
- **Same prompts without the teacher**: Add them directly to the DIM training pool → ~0.80 in five-fold CV (no gain).  
- **Random resampling** (documented in `EXPERIMENTS_RUN.md`): Reducing the `got_cities` training set or mixing in unlabeled novel samples steadily lowers accuracy (50 examples → 0.59; 200 → 0.71; 600 → 0.78).  
- **Conclusion**: Extra data only helps when it carries new, high-quality information about the semantic direction.

---

## 4. Interpreting “Truth” Components

PCA on probe vectors (layer h35):

- **PC1 (difficulty)** – dominated by RepE question complexity, useless for truth (meta accuracy 0.36).  
- **PC2 (format consistency)** – separates DLK statements from Q&A, yields 0.61 meta accuracy.  
- **PC3 (synthetic vs natural)** – highlights GoT datasets (0.59 meta accuracy).  

The `got_cities` probe lives on the PC3 axis; DLK probes live on PC2. Since these vectors are almost orthogonal (cosine ≈ 0.015), averaging them erases the semantic signal—hence the collapse to 0.50.

---

## 5. Practical Checklist

1. **Start with `got_cities`** for DIM probes on factual truth.  
2. **Screen future datasets** by activation variance (≥10) and by their recovered accuracy when held out from the academic matrix.  
3. **Use teacher feedback (RLAIF)** to expand coverage; treat false positives / negatives as new contrastive pairs.  
4. **Validate every change** on a held-out novel set (e.g., the 60 prompt harness or the 260 expansion).  
5. **Avoid blind ensembling**—project into the PC2/PC3 subspace before combining if you must, or you’ll land back at chance.  
6. **For other tasks (hallucination detection)**, accept that the useful feature is different (stylized consistency, not global semantics); reuse the DLK probes or train a dedicated head.
