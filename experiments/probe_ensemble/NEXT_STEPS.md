# Probe Ensemble Roadmap

This document captures the agreed next steps for the “truth / misalignment” probe project.  
The goal is to move from single-direction DIM probes to a small, interpretable ensemble that:

1. Covers multiple behavioural axes (format‐truth, semantic‐truth, overt misalignment, subtle sabotage, …)  
2. Combines them with a lightweight meta-classifier, instead of collapsing everything into one vector  
3. Uses an LLM-in-the-loop workflow to continuously generate, audit, and incorporate high-quality contrastive data  
4. Reports reproducible metrics on held-out validation sets and cross-dataset matrices

---

## 1. Validation Sets (must exist before auto-generation)

- **Truth:**  
  - 60 human-written declaratives (current harness).  
  - 260 prompt expansion (per `test_on_diverse_prompts.py`).  
  - Keep 20–30% as a blind hold-out.

- **Misalignment / Deception:**  
  - Curated seed prompts for each facet (eg “edgelord jailbreak”, “subtle sabotage”, “malicious advice”).  
  - Minimum 10 per facet, labelled by humans.

Every new probe or ensemble is evaluated on these sets first.

---

## 2. LLM-in-the-loop Data Generation

We treat the language model as both **generator** and **critic**. A high-level loop:

```
repeat until validation metrics saturate:
    select target behaviour b (truth, deception, …)
    select subset S of validation failures for b (residuals)
    prompt generator LLM_G to produce aligned/misaligned pairs targeting S
    for each pair:
        run critic LLM_C with rationale prompt → preliminary label + confidence
        if confidence < τ: escalate to human audit
        else: human spot-check 10–20%; if pass rate < α, discard batch
    append human-approved items to dataset D_b
```

Implementation notes:
- Track provenance (generator prompt version, critic prompt, model checkpoint).  
- Keep *Rejected* examples in a backlog; they often seed the next generation cycle.  
- Use separate generator/critic models (eg GPT-4o vs Claude) to avoid shared blind spots.  
- Log rationales from the critic to speed up human QA.

---

## 3. Probe Training & Orthogonalization

For each behaviour dataset \( D_b = \{(a(x), y)\} \):

1. Fit a DIM / logistic probe \( v_b \) (difference of means, regularised logistic, etc.).  
2. Normalise \( \|v_b\| = 1 \).  
3. Orthogonalise against existing probes using Gram–Schmidt:
   \[
   v_b \leftarrow v_b - \sum_{i < b} \langle v_b, v_i \rangle v_i;\quad v_b \leftarrow \frac{v_b}{\|v_b\|}
   \]
4. Record loadings on PCA components (PC2 = format, PC3 = semantic) to check we discovered new signal.

Keep, for audit:
- Raw (pre-orthogonal) vector,  
- Orthogonalised vector,  
- Mean projections on PC2/PC3.

---

## 4. Meta-classifier (boosted ensemble)

Let probe scores be \( p_i(x) = \langle v_i, a(x) \rangle \).  
Define meta-feature \( p(x) = (p_1(x), \dots, p_k(x)) \).

Train a logistic regression (or shallow tree) to predict target label \( y \) from \( p(x) \):

```
gθ(p) = sigmoid(θ0 + Σ θi * p_i)
θ* = argminθ Σ (y * log gθ(p(x)) + (1-y) * log(1 - gθ(p(x))))
```

Per boosting intuition:
- Inspect residuals (instances with high loss).  
- Target new probe generation toward those residuals.  
- Repeat orthogonalisation + re-fit until validation metrics no longer improve.

Optional safety net: per-task gating (simple decision tree) when different behaviours conflict strongly.

---

## 5. Evaluation Protocol

Every update (new probe or meta-classifier retrain) must report:

1. **Main metrics**  
   - Accuracy / AUROC / AUPRC on truth validation sets (60 + 260).  
   - Same metrics on misalignment validation sets.  
   - Cross-dataset recovered accuracy matrix (as in `replicate_mishajw_figures.py`).

2. **Probe diagnostics**  
   - PCA loadings (PC2, PC3) per probe.  
   - Pairwise dot products (to confirm orthogonalisation worked).  
   - Histograms of probe scores on held-out data (to catch drift).

3. **Ensemble diagnostics**  
   - Weight vector \( θ \) for interpretability.  
   - Performance versus number of probes (learning curve).  
   - Confusion matrix per behaviour facet.

4. **Human spot-check log**  
   - % of auto-labelled data that passed manual review.  
   - Notes on any systematic annotation failures.

---

## 6. Infrastructure Tasks

- [ ] Build a lightweight dataset tracker (metadata + hashes for generated examples).  
- [ ] Implement generator/critic prompts as separate YAML configs.  
- [ ] Add a CLI script to run a full “cycle” (generate → critic → human queue → train → evaluate).  
- [ ] Automate PCA / diagnostics notebooks (store in `output/probe_ensemble/diagnostics/`).  
- [ ] Maintain a “probe registry” listing current vectors, scores, and loadings.

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Garbage synthetic data | Keep core validation human-written; require manual spot-checks on critic output |
| Probes recapturing old axes | PCA loadings + Gram–Schmidt guardrail |
| Overfitting the meta-classifier | Strict held-out split; report metrics on blind set only after multiple iterations |
| Concept drift in generator | Log prompt versions; periodically regenerate seed examples |
| Misalignment definition ambiguity | Maintain separate sub-datasets (edgelord, subtle sabotage, phishing); do not collapse without evidence |

---

## 8. Immediate To-do List

1. Stand up generator/critic prompt configs and a human QA checklist.  
2. Train two new misalignment probes (overt edgelord, subtle sabotage) from curated seeds; log PCA positions.  
3. Fit the first meta-classifier on `{ format-truth, semantic-truth }` + evaluate on held-out sets.  
4. Run one boosted iteration: target the top 50 residuals, generate new data, train probe #3, re-fit classifier.  
5. Document the results (metrics + diagnostics) in `results/README.md`.

Once this loop is stable, we can expand to additional behaviours and consider stronger classifiers (eg mixture-of-experts) if needed.

