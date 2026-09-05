# Failure Analysis (§4)

Collected record of things that failed, why, validity, and disposition.
Detailed mechanism-level records: `harness-corrections.md`.

| # | What happened | Why | Valid result? | Changed | Excluded? |
|---|---|---|---|---|---|
| 1 | Phase 2: multi-head fusion < best single head | equal weights let 6 noisy heads outvote the clean one | YES (real property of the corpus + method) | motivated 2B | no — headline finding |
| 2 | Phase 2: mode E regression | equal-head weighting of exact scores | YES (mechanism decomposed in PH2C-RERANK-*) | pipeline fix pending | no |
| 3 | PH2B-GATING-001/002: plausible-but-wrong recalls | GT hint-id shift (HC-1) | NO (harness) | id mapping fixed | yes — INVALIDATED |
| 4 | PH2B-NOISE-001: all-zero table incl. oracle | same shift, unmasked | NO (harness) — but it EXPOSED #3 | id mapping fixed | yes — INVALIDATED |
| 5 | PH2B-MULTIVIEW-001: oracle 0.26 | GT tie lottery (HC-2) | NO (harness) | fine structure added | yes — INVALIDATED |
| 6 | PH2B-MULTIVIEW-002: gate refuses to learn (corr −0.34) | all-view full-fidelity queries (HC-3) | NO (harness design) | one-modality queries | yes — INVALIDATED |
| 7 | PH2B-MULTIVIEW-003/004: gating learns nothing at 210 train queries | sample efficiency, not capacity (grid moved <2pp) | YES (negative result) | data ×4 | no — before-picture for F6 |
| 8 | PH2B-MULTISEED-002: seed variance ±0.036 | learner variance at moderate data size | YES | multi-seed now mandatory for multiview claims | no |
| 9 | Temperature calibration on a failed model picks T=4 | validation picks near-uniform when no signal | YES (safe fallback behavior) | none needed | no |
| 10 | Sandbox resets lost commit history twice | environment, not code | n/a | recovery commits + registry as canonical history | n/a |

Pattern worth recording: every INVALIDATED result was caught by an
INVARIANT (oracle ≈ 0 impossible; oracle ≪ 1 on its own view; corr < 0
with uniform weights; recall = chance level). The harness is now audited
by these invariants in `verify_consistency.py` so future silent breakage
fails loudly.
