# Attention Findings (§11–13, F8)

Status: **no trained QK attention experiment has been run yet** (Phase 2C).
This file records what IS established and what is explicitly not.

## Established (MEASURED)

- Identity-initialized QK attention (Phase 2 mode D) is a mathematical
  no-op relative to untrained gating: identical R@10/NDCG to 3 decimals
  (0.672 both, frozen Phase 2 ablation). This supports only the claim
  "identity-init QK adds nothing" (F8).
- The QK machinery exists and is O(C·(F+D)), tanh-bounded, finite-safe
  (`core/src/retrieval.rs::AttentionScorer`); weights are plain f32 vectors
  with no training path wired yet.

## Explicitly NOT established

- Whether TRAINED QK attention improves retrieval over trained gating.
- Whether candidate-level attention adds anything beyond head gating.
- Any latency/memory cost for a trained QK path at inference.

## Plan constraints carried forward (from the Phase 2B spec)

- QK training happens ONLY after gating works (it now does on controlled;
  partially on multiview). §31 applies: if trained QK does not improve
  held-out retrieval, report exactly that — no "promising" renaming.
- Evaluation will compare identity-QK / trained-QK / gating+trained-QK on
  the same cached datasets and splits; results will land as
  PH2C-QK-* runs and populate `results/qk-attention.csv` (currently an
  explicit pending marker — no numbers will be fabricated).
