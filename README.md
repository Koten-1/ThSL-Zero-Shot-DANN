# ThSL Zero-Shot Signer Generalization + Domain-Adversarial Training

Research companion repository for a study on Thai Sign Language (ThSL) recognition, evaluated the way Radford et al. (2022, *Robust Speech Recognition via Large-Scale Weak Supervision* — Whisper, arXiv:2212.04356) argue robustness claims should be evaluated: decontaminated, and tested zero-shot on distributions the model never saw in training.

This is a standalone snapshot of the research work from the [ThSL Bridge](https://github.com/Koten-1/ThSL-Bridge) project — the live product, real-time pipeline, and web app live in that repository; this one exists to give the accompanying research paper a citable, self-contained code reference.

## RQ1 — Honest evaluation (done)

Does a naive random train/test split overstate accuracy through data leakage, compared to a group-aware split that keeps a video's augmented copies entirely on one side of the split?

**Result**: 99.6% (naive) → 98.88% (group-aware). The gap is the leakage naive evaluation was hiding.

## RQ2 — Architecture depth (done)

Single-layer vs. 2-layer stacked LSTM, both evaluated under RQ1's honest split.

**Result**: 97.75% (single) vs. 98.88% (stacked) — stacked wins.

## RQ3 — Zero-shot signer generalization + domain-adversarial training (done)

Does training on multiple signers generalize to a signer the model has *never seen*, and does explicitly training the model to ignore signer identity (domain-adversarial training; Ajakan et al., 2014; Ganin et al., 2016) improve that generalization beyond passively adding more signers?

One signer was permanently held out — zero clips in any training condition — as the true zero-shot test set. Three model variants were evaluated on that same held-out signer:

| Model | Zero-shot accuracy |
|---|---|
| `solo-baseline` (single signer, quantity-matched) | 21% |
| `pooled-baseline` (2 signers pooled) | 62% |
| `pooled+DANN` (pooled + gradient-reversal signer-invariance objective) | 60% |

Signer diversity produced the largest generalization gain by a wide margin; the domain-adversarial objective did not improve on the pooled baseline in this narrow (2-signer) setting — reported as a rigorously-tested negative result rather than omitted. A risk-coverage / selective-accuracy analysis (Area Under the Risk-Coverage Curve) is also reported for all three models, evaluating each one's ability to abstain from uncertain predictions rather than answer confidently but incorrectly.

## File map

- `notebooks/🧩LuckBasedModel(2).ipynb` — RQ1 + RQ2
- `notebooks/🧩RQ3_DANN.ipynb` — RQ3 (self-contained, separate variable namespace from the above)
- `check_thresholds.py` — per-class confidence threshold inspection tool
- `figures/` — confusion matrices, data-quality comparison charts
- `ThSL_Bridge_Study_Notes.md` — background study notes

Note: the underlying video/keypoint dataset is not included here (large, and contains identifiable recordings of volunteer signers) — both notebooks expect `data/processed/` and `data/augmented/` mounted from Google Drive when run in Colab.

## Future direction

Phase 2: extend the vocabulary from conversational words to medical terms, positioning this work as a step toward a clinical communication tool — the risk-coverage/selective-accuracy analysis above (knowing when the model is uncertain) is deliberately laying groundwork for that use case, where a wrong prediction has higher stakes than a rejected one.
