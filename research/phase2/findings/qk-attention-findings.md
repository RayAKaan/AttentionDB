# QK-Attention Findings (Phase 2C)

Finding slots F11–F15 are RESERVED for the Phase 2C main comparison
(trained candidate-level QK vs trained gating on the real corpora —
controlled / noise / multiview) and will be numbered only when those
experiments complete. This file records completed Phase 2C evidence in
run order. Nothing here renumbers or rewrites F1–F10 (fixed reference
text in `gating-findings.md`, `attention-findings.md`,
`reranking-findings.md`, `multiview-findings.md`).

---

## QK-SANITY — the QK machinery can learn candidate-level ordering that the entire gating class provably cannot [PH2C-QK-001]

**Setup** (full construction + impossibility argument:
`methodology/qk-sanity-dataset.md`): synthetic 2-head corpus where the
relevant candidate ranks LAST under the signal-head cosine for every query
(exact invariant), head 1 is pure noise, and the only separating signal is
the bilinear form `qᵀMx`, `M = v₀⊗u₀ + v₁⊗u₁`. Head selection is easy;
ordering requires query–candidate interaction.

**Observed** (held-out test, 250 queries, seeds 42/7/1):

| arm | R@1 |
|---|---|
| uniform / global-best-head0 | 0.0000 |
| gating (shipped SoftTarget-recall objective) | 0.0027 ± 0.0038 |
| gating trained with the QK objective (fairness control) | 0.1000 ± 0.0000 (exactly chance; learns w = (0,1)) |
| untrained QK (random init) | 0.1160 |
| **trained linear QK (W_Q, W_K, no hidden)** | **1.0000 ± 0.0000** |
| oracle | 1.0000 |

**Claims this supports (bounded to this dataset):**
1. The Phase 2C QK implementation (linear W_Q/W_K projections, `Q·K/√d`,
   InfoNCE with temperature softmax) can learn a candidate-level
   query–candidate interaction to held-out perfection. The machinery is
   not the bottleneck.
2. The isolation is a **function-class impossibility**, not a training
   failure: gating trained end-to-end with the QK objective still cannot
   exceed chance — it correctly collapses all weight onto the noise head
   (the class-optimal move) and still cannot order. Both gating objective
   regimes land ≤ chance, as the construction requires.
3. The pre-registered gate for proceeding to the main evaluation
   (QK ≥ 0.9 AND gating ≤ 0.25) **PASSED**.

**Claims this does NOT support:** any statement about the real corpora.
Real candidate pools may contain no query-dependent structure beyond
per-head cosines — in which case QK and gating collapse to the same
effective class on real data and Rule Zero's "no significant win" outcome
is live. That question is exactly what PH2C-QK-002+ measures.

**Diagnostics recorded:** gating weight distributions per arm
(`raw/runs/PH2C-QK-001/gating_weights_test.csv`), full val grid
(`val_grid.csv`; flat for both arms — no selection sensitivity),
training curve (`trainlog_qk.csv`), per-seed variability
(`variability.csv`).

**Negative-control note:** the first recorded run exposed HC-5
(`rank_metrics` consumes slice order as the ranking). The defect was
caught by cross-checking `qk_untrained` against an independent Python
replication of the generator RNG BEFORE any number entered the registry;
no invalid number was recorded (see `harness-corrections.md` HC-5).
