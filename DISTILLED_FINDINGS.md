# Distilled Findings and Next Experiments for Auto-Probe Pipeline

## Core Claims Backed by Current Analysis
- Truth features split into ~5 orthogonal factors rather than a single axis; PC1 captures ~25% shared signal while residual components encode domain-specific structure (`TRUTH_DECOMPOSITION_FINDINGS.md:11-44`, `AUTO_PROBE_PIPELINE_NEXT_STEPS.md:5-11`).
- Mean-diff probes alone cannot certify transfer—cosine similarity is only a proxy and breaks when layer statistics diverge; real generalization demands cached activations and cross-evals (`DECOMPOSITION_PARADIGMS_ANALYSIS.md:13-125`).
- Probe in-dist accuracy and transfer gap are tightly anti-correlated (r≈0.98); moderately hard datasets (60-70% in-dist) yield the most portable probes (`PROBE_QUALITY_INSIGHTS.md:7-89`).
- Spectral clustering on the Gram matrix delivers 4.2× better specialist separation than manual groupings and preserves semantic interpretability (`OPTIMAL_SPECIALIST_GROUPING.md:3-195`).
- Averaging probes within spectral groups produces seven centroids that improve recovered accuracy by +17% on average and yield a 9× better-conditioned basis (`ENSEMBLE_BASIS_ANALYSIS.md:5-144`).
- Layer depth matters: shared directions emerge around 60-65% depth, while later layers sacrifice generalization for specificity; hybrid architectures should gate with mid-layer PC1 and defer to late-layer specialists (`TRUTH_DECOMPOSITION_FINDINGS.md:49-72`, `DECOMPOSITION_PARADIGMS_ANALYSIS.md:459-592`).

## Practical Intuitions for Pipeline Design
- Treat PC1 similarity as a quality prior, but rely on full-vector evaluation for deployment decisions; medium cosine overlap (20-60%) between probes supports ensemble diversity (`TRUTH_DECOMPOSITION_FINDINGS.md:33-45`).
- Hard reasoning datasets produce “super-transfer” probes because they force models to represent abstract reasoning rather than surface heuristics; template-heavy synthetic data inflates in-dist accuracy and gap (`PROBE_QUALITY_INSIGHTS.md:136-198`).
- Whitening score space before combining probes stabilizes thresholds and reduces correlated false positives; downstream combination should operate on whitened scores, not raw activations (`AUTO_PROBE_PIPELINE_NEXT_STEPS.md:64-184`).
- Hybrid shared+specialist ensembles balance compute with precision: gate benign traffic with the universal axis, then apply orthogonal specialists under conformal/FDR control for low FPR regimes (`DECOMPOSITION_PARADIGMS_ANALYSIS.md:424-446`).

## Non-Mathematical Design Decisions to Nail Down
- **Concept definitions:** Formalize what constitutes “truth,” “misalignment,” etc., and decide which borderline cases should pass or fail before automating generation (`DECOMPOSITION_PARADIGMS_ANALYSIS.md:596-665`).
- **Quality bars:** Pick operating thresholds for generalization gap, per-domain recall, and allowable FPR so automated filters have measurable targets (`AUTO_PROBE_PIPELINE_NEXT_STEPS.md:165-193`).
- **Layer usage policy:** Decide whether production inference can afford multi-layer hooks or must stay single-pass; this choice determines whether a hybrid layer strategy is viable (`DECOMPOSITION_PARADIGMS_ANALYSIS.md:472-553`).
- **Human-in-the-loop:** Specify review cadence for red-team negatives and PCA component interpretations to keep the automated loop aligned with human notions of the concept.

## Concrete Experiment Directions
1. **Activation cache + exhaustive cross-generalization:** Materialize per-dataset activation caches and recompute the full 18×18 evaluation across candidate layers to validate similarity proxies and quantify tail FPR (`DECOMPOSITION_PARADIGMS_ANALYSIS.md:71-125`).
2. **Layer-hybrid validation sweep:** For each dataset, train probes at depths {0.50, 0.625, 0.75, 0.875}; measure in-dist vs cross-dist accuracy to confirm the predicted precision/recall tradeoff and identify best layers per domain (`DECOMPOSITION_PARADIGMS_ANALYSIS.md:570-592`).
3. **Shared-gate calibration:** Whiten score space, learn mid-layer PC1 gate, and benchmark Neyman–Pearson or conformal thresholds at α∈{1e-3,1e-4,1e-5}; record impact on downstream specialists (`AUTO_PROBE_PIPELINE_NEXT_STEPS.md:136-193`).
4. **Spectral-centroid ensemble benchmarking:** Replace individual probes with the seven centroids, rerun cross-dataset metrics, and compare condition numbers and tail failure rates against baselines (`ENSEMBLE_BASIS_ANALYSIS.md:18-151`).
5. **Automated PCA labeling loop:** Implement LLM-assisted labeling of PCA components/residuals and audit interpretability quality to close the loop on white-box analysis (`AUTO_PROBE_PIPELINE_NEXT_STEPS.md:26-134`).
6. **LLM blue/red-team generation pilot:** Use the scripted prompt templates to synthesize new contrastive pairs, score them with the quality evaluator, and inspect accepted vs rejected ratios to tune thresholds (`AUTO_PROBE_PIPELINE_NEXT_STEPS.md:200-345`).
7. **Negative example stress tests:** Feed red-team negatives and out-of-domain corpora through the hybrid ensemble to quantify FPR under distribution shift and adjust whitening statistics over time (`AUTO_PROBE_PIPELINE_NEXT_STEPS.md:248-345`).

## Infrastructure and Evaluation Needs
- Build persistent activation stores so experiments avoid repeated forward passes and can analyze cross-layer representations cheaply.
- Instrument evaluation scripts to report both AUC and recovered accuracy alongside extreme-tail FPR/TPR metrics relevant for safety deployment.
- Track probe geometry metrics (cosine overlap, condition number, PCA loadings) in the same dashboard as classification scores to monitor ensemble diversity (`ENSEMBLE_BASIS_ANALYSIS.md:102-151`).

## Summary
The current evidence supports a pipeline that (a) models universal truth as a mid-layer shared axis, (b) captures residual structure with spectrally clustered specialists, and (c) iteratively curates datasets via LLM-generated contrastive pairs filtered by whitened score-space metrics. The remaining work lies in validating these claims with cached activations, deciding operational thresholds, and standing up the automated blue/red-team loop so probes stay aligned with evolving safety definitions.
