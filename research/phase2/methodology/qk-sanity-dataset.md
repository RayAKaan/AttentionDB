# PH2C-QK-001 — QK-Attention Sanity Dataset (Phase 2C §6)

**Status:** COMPLETED 2026-09-04 · **Run:** `raw/runs/PH2C-QK-001/` · **Seeds:** 42, 7, 1 (dataset seed `0x5A17`, fixed)
**Gate:** pre-registered — QK must reach test R@1 ≥ 0.9 AND every gating variant must stay ≤ 0.25; otherwise the candidate-level QK track stops here (§19/§41). **Gate result: PASSED (QK 1.0000; gating 0.0027 / 0.1000).**

## 1. Purpose

Phase 2C's central question is whether *candidate-level* query–candidate
interaction (QK attention: `Q = W_Q·q`, `K_i = W_K·x_i`, `score_i = Q·K_i/√d`)
adds value over trained head *gating* (per-head weights over per-head cosine
scores). Before spending the main corpora on that question, this experiment
checks that the QK machinery can learn an ordering task that **provably
requires candidate-level interaction**, on a dataset where the answer is
knowable in advance.

## 2. Construction

Two heads, head dimension 8, pool of 10 candidates per query (1 relevant +
9 distractors), 600/150/250 train/val/test queries, families balanced.
Coordinate axes of head-0 space: `u_0 = e_0`, `u_1 = e_1` (query family
directions), `v_0 = e_2`, `v_1 = e_3` (relevance directions). For a query of
family `f`:

- **Query** `q`: head-0 = `normalize(e_{u_f} + 0.03·n)`, head-1 = pure noise
  (unit Gaussian). Family is thus trivially decodable — head selection is easy.
- **Relevant candidate** `x⁺`: head-0 = `normalize(e_{v_f} + 0.03·n)`, head-1 =
  noise. `v_f ⊥ u_f` ⇒ `cos(x⁺, q) ≈ 0` (std ≈ 0.042).
- **Distractors** `x⁻`: head-0 = `normalize(0.3·e_{u_f} + w)` with `w` a unit
  vector projected orthogonal to **both** `u_f` and `v_f`; head-1 = noise.
  Every distractor therefore has `cos(x⁻, q) ≈ 0.287` — mildly attractive to
  the query, with **zero** `v_f` component.

Noise (0.03) is sized so the anti-cosine invariant is exact, not statistical:
the relevant candidate ranks **last** under the signal-head cosine for every
generated query (verified by unit test `sanity_geometry_anti_cosine` and by an
independent Python replication of the generator RNG; margin ≈ 6.8σ).

## 3. Why this isolates candidate-level attention

The gating function class computes, for candidate `i`,

```
score_i = Σ_h w_h(q) · ŝ_h(x_i),      ŝ_h = per-head minmax cosine
```

a per-candidate score that is a **query-independent functional of the per-head
cosine features**, combined with query-dependent mixture weights. On this
dataset `ŝ_{head0}` ranks the relevant candidate **last for every query**
(cos ≈ 0.0 vs 0.287), so no mixture weight vector — including an oracle over
weights — can place it first. Any residual ordering ability comes from the
noise head only, i.e. at most chance (R@1 = 1/10).

The only separating signal is the bilinear form

```
score_i = qᵀ M x_i   with   M = v_0⊗u_0 + v_1⊗u_1  (M ≠ scalar·I)
```

which a linear QK layer expresses exactly (`score_i = (W_Q q)·(W_K x_i)/√8`
realizes `M = W_QᵀW_K`): the relevant candidate has the family-matching `v_f`
component (≈ 1), distractors have none (≈ 0). Head selection plays no role in
the separation — the entire gap is a **query–candidate interaction** that
per-head-cosine features do not contain and head weights cannot recover.

This is the cleanest possible instantiation of Phase 2C §6's requirement:
"head selection easy + ordering requires query-dependent interaction."
Gating's failure here is not a training failure, an objective failure, or a
capacity failure — it is a **function-class impossibility**.

## 4. Arms (identical data, splits, eval)

| arm | class | objective | selection |
|---|---|---|---|
| `uniform` | w = (½, ½) | — | — |
| `global_best_head0` | w = (1, 0) | — | — |
| `gating_qualityreg` | GatingMlp 16→16→2, shipped Phase 2B recipe (SoftTarget on per-head recall, val-fitted T ∈ {0.25,0.5,1,2}) | quality regression | seeds 42/7/1 |
| `gating_infnce` | same MLP, trained **end-to-end with the QK objective** (InfoNCE on fused scores) — fairness control | InfoNCE, lr×T grid on val (seed 42, frozen) | seeds 42/7/1 |
| `qk_untrained` | linear W_Q/W_K, random init (seed 999) | — | — |
| `qk_trained` | linear W_Q/W_K (8×16 each), no hidden layers | InfoNCE, lr×T grid on val (seed 42, frozen) | seeds 42/7/1 |
| `oracle` | rank relevant first | — | — |

Shared: Adam (batch 32, ≤300 epochs, early stop patience 30 on val R@1),
T grid {0.25, 0.5, 1.0, 2.0}, lr grid {0.05, 0.01} (val-only selection, §9/§10;
the selected config per arm is frozen before any test evaluation).
Candidate recall is 1.0 by construction for every arm (the relevant candidate
is in the pool); the task measures ORDERING only — exactly the quantity Phase
2C attributes to the candidate-interaction stage.

## 5. Results (held-out test, 250 queries)

From `results/qk-sanity.csv` (verbatim copy of the run's `eval_test.csv`):

| arm | R@1 | NDCG@10 | MRR |
|---|---|---|---|
| uniform | 0.0000 | 0.2914 | 0.1022 |
| global_best_head0 | 0.0000 | 0.2891 | 0.1000 |
| gating_qualityreg (mean of 3 seeds) | 0.0027 ± 0.0038 | 0.3183 | 0.1306 |
| gating_infnce (mean of 3 seeds) | 0.1000 ± 0.0000 | 0.4528 | 0.2909 |
| qk_untrained | 0.1160 | 0.4497 | 0.2890 |
| **qk_trained (mean of 3 seeds)** | **1.0000 ± 0.0000** | **1.0000** | **1.0000** |
| oracle | 1.0000 | 1.0000 | 1.0000 |

Diagnostics (`gating_weights_test.csv`):
- `gating_infnce` collapses its weights to **exactly w = (0, 1)** — it
  *correctly* identifies that any weight on the signal head only hurts, puts
  everything on the noise head, and lands at exactly chance (0.1000). The
  class boundary is visible in the trained parameters, not just the metric.
- `gating_qualityreg` blurs (mean w = (0.37, 0.63), entropy 0.65) and lands
  *below* chance — both gating behaviors are ≤ chance, as the impossibility
  argument requires.
- `qk_untrained` at 0.1160 ≈ 1/10 confirms the task is not leaked by the
  machinery itself; learning the bilinear form is what closes the gap.
- The val grid is flat for both arms (QK 1.0 at every lr×T; gating_infnce
  0.0867 everywhere) — no selection sensitivity, no hidden tuning room.

## 6. Threats to validity, and controls run

- *"Gating just used the wrong objective."* → `gating_infnce` trains the
  gating architecture with the QK objective end-to-end; it still cannot exceed
  chance (it is not supposed to — the impossibility is in the features).
- *"The task leaks through candidate identity."* → candidate ids are random
  per query; only content geometry separates classes; `qk_untrained` ≈ chance.
- *"Selection tuned on test."* → all selection (lr, T, early stop) on the 150
  validation queries; test evaluated once per seed after freezing.
- *"A stronger per-candidate feature could rescue gating."* → Not on this
  dataset by construction: any functional of the per-head cosines alone is
  order-invariant to the relevant/distactor distinction (both classes expose
  the same cosine profile up to noise). This is precisely the property that
  makes the dataset a *sanity check for the representation*, and it is also
  why a pass here does NOT predict a pass on the real corpora — real candidate
  sets may contain no such query-dependent structure to exploit. That question
  is exactly what PH2C-QK-002+ measures.

## 7. Verdict and gate decision

**PASSED.** The QK machinery (linear W_Q/W_K, InfoNCE, temperature softmax)
learns to perfect held-out ordering on a task where the entire gating class —
under its shipped objective *and* under the QK objective — is bounded at
chance. Per the Phase 2C plan this authorizes proceeding to the main
candidate-level QK evaluation on the real corpora (PH2C-QK-002+), with Rule
Zero unchanged: the main question remains whether QK adds value over gating
*there*, and "no significant win" remains a permitted outcome.

## 8. Provenance

- Implementation: `benchmarks/phase2b/src/qk_sanity.rs` (subcommand
  `qk-sanity`); exact-rerun command in `PAPER_PACKAGE.md`.
- Invariant tests: `sanity_geometry_anti_cosine`,
  `sanity_head_quality_targets_reflect_true_head_ranking` (HC-5 regression),
  `sanity_qk_learns_gating_cannot`.
- Raw artifacts: `dataset.json` (GatingDataset schema), `qk_content.json`
  (candidate-content sidecar — required by candidate-level K),
  `eval_test.csv`, `variability.csv`, `val_grid.csv`, `trainlog_qk.csv`,
  `gating_weights_test.csv`, `config.txt`, `metrics.json`, `models/`.
- Harness correction HC-5 (this experiment): `rank_metrics` consumes slice
  ORDER as the ranking; every caller must sort by score before calling.
  Caught by cross-checking `qk_untrained` against an independent Python
  replication (exact RNG) before any number was recorded.
