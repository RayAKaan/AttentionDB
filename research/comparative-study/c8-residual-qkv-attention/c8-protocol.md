# C8 Protocol: Residual Candidate-Level Q/K/V Attention

**Version**: 1.0.0 (frozen)
**Branch**: `comparative-study/c8-residual-qkv-attention`
**Depends on**: C0–C7 complete; C7 artifacts frozen in `raw/`; `attentiondb-attention` crate (C8 math core) integrated into `attentiondb-core`

---

## 1. Research Question

**H0**: A residual candidate-level attention correction does not materially change
ranking relevance over the multi-head union baseline (C7-B / C5-B / C6-B) once the
candidate set, budget, and latency are controlled.

**H1**: A residual correction added *after* fusion,

```text
S_final = S_base + lambda * dS_attention
```

where `S_base` is the untouched retrieval score and `dS_attention` is a genuine
candidate-level Q/K/V attention-derived quantity, materially improves ranking
quality (nDCG@10, R@10, MRR) versus the union baseline, within an identical
candidate set and budget.

Sub-questions (ablation):
- **C vs B**: does the C8 machinery, at `lambda = 0`, reproduce the baseline exactly?
- **D vs E**: does an unrestricted learned projection (`W = dW`) beat a residual
  learned projection (`W = W0 + alpha*dW`) that starts at the identity base?
- **E vs F**: does retrieval evidence added to the correction help?
- **F vs G**: does cross-head disagreement variance add to evidence?
- **E vs H**: does training with KL distillation change ranking?
- **E vs I**: does the in-memory K/V cache preserve results exactly while cutting
  projection latency?

---

## 2. Mandatory Arms (9)

| Arm | Name | RetrievalMode | C8 correction | Projection | Evidence | Trained? |
|-----|------|---------------|---------------|-----------|----------|----------|
| C8-A | Canonical single-head | `SingleHead` | OFF | — | — | No |
| C8-B | Independent multi-head union (control) | `FixedFusion` | OFF | — | — | No |
| C8-C | Truncated-identity machinery, `lambda = 0` | `FixedFusion` | ON, `lambda=0` | truncated-identity | none | No |
| C8-D | Unrestricted learned projection | `FixedFusion` | ON | `W = dW` | none | Yes |
| C8-E | Learned residual projection | `FixedFusion` | ON | `W = W0 + alpha*dW` | none | Yes |
| C8-F | C8-E + retrieval evidence | `FixedFusion` | ON | `W = W0 + alpha*dW` | aggregate `agg(r_d)` | Yes |
| C8-G | C8-F + disagreement variance | `FixedFusion` | ON | `W = W0 + alpha*dW` | `agg(r_d)` + `Var(r)` | Yes |
| C8-H | C8-E trained with KL distillation | `FixedFusion` | ON | `W = W0 + alpha*dW` | none | Yes (distill) |
| C8-I | C8-E + in-memory K/V cache | `FixedFusion` | ON | `W = W0 + alpha*dW` | none | Yes (cached inference) |

Invariants:
- **Candidate membership**: arms B, C, D, E, F, G, H, I share an *identical* union
  per query (same `FixedFusion` path, same heads, ef, budgets). Only the C8
  correction score differs. Rank differences are attributable to the correction.
- **`lambda = 0` parity**: arm C sets `residual_scale = 0.0`, so
  `S_final = S_base` bit-for-bit for every candidate. The C8 trace still computes
  the (pre-scale) `dS_attention`, so the machinery is exercised, but
  `applied_correction` is exactly `0.0`.
- **C8-A** is the single-head canonical control. **C8-B** is the union control.
- **`lambda` is not a fusion weight.** It lives in `C8AttentionConfig` and is
  applied after `fuse_candidate`, so it cannot perturb the present-weight
  renormalization of the baseline.

Final score equation (documented, no hidden multipliers):

```text
S_base   = fuse_candidate(attention=C7-or-legacy ch, mhs, bm25)   [arms B..I]
dS_attn  = w_attn * <q_a, O_d> + w_evidence * agg(r_d) + w_disagree * Var(r_d) + bias
S_final  = S_base + lambda * dS_attn
```

With the vector-only study (`bm25_raw = None`) and C8 enabled, `S_base` for B..I is
the `FixedFusion` score of the union pipeline. Arm C's `lambda = 0` makes the
correction term vanish.

C8 config per arm (frozen, hashed by `C8AttentionConfig::fingerprint`):
- C8-B: `c8_attention = None`.
- C8-C: `C8AttentionConfig::truncated_identity_control(3, 384, d_k, d_v)` with
  `residual_scale = 0.0`, `use_evidence = false`.
- C8-D: `arch = unrestricted`, projection from the D model card, `residual_scale = lambda_D`.
- C8-E/F/G/H/I: `arch = residual`, projection from the respective model card,
  `residual_scale = lambda_*`, scorer per arm.

---

## 3. Budget Constraints (FROZEN)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `candidate_budget` | 500 | Union cap (same as C5/C6/C7) |
| `ef_search` | 64 | Per-head HNSW EF (same as C7-B) |
| `min_candidates_per_head` | 20 | Floor |
| `max_candidates_per_head` | 300 | Ceiling |
| `attention_dim` (d_a) | 384 | = collection dim |
| `key_dim` (d_k) | 64 | **reduced** (primary config; support 32/64/128/384) |
| `value_dim` (d_v) | 64 | **reduced** (primary config; support 32/64/128/384) |
| `heads` | TITLE, BODY, CITE | 3 heads (384-d each) |
| `bm25_raw` | None | Vector-only study (same as C6/C7) |

The C8 correction adds no candidate work — it re-uses the membership vectors already
fetched for the union (no extra HNSW calls). The only cost adder is the attention
compute and projection, recorded per candidate in the C8 trace timings.

---

## 4. Training / Tuning Policy

Training and tuning occur on the **VALIDATION split only**. Everything is frozen
before TEST.

### C8-C: machinery control
No training. `lambda = 0`, truncated-identity base. Establishes that the C8 plumbing
is a no-op on ranking when the correction is scaled out.

### C8-D/E/F/G/H: residual QKV training
- Optimizer: deterministic Adam (no RNG) via `ResidualQkvTrainer`.
- Loss (frozen):
  ```text
  L = L_contrastive + alpha_r * L_residual + beta * L_distill
  L_contrastive = InfoNCE over S_final within a query's candidate set (1 pos + k neg)
  L_residual    = mean(dW^2)                          [learned correction only]
  L_distill     = KL(softmax(S_base/tau_d) || softmax(S_final/tau_d))
  ```
- **Real mini-batches**: gradients accumulate over `batch_size` examples and one Adam
  step is taken per batch (`optimizer_steps = epochs * batches`), not one step per
  example.
- **The baseline is part of the loss**: `S_base` is a constant; the contrastive
  gradient reaches the QKV weights only through `lambda * dS`.
- Init:
  - C8-D: `ResidualQkvProjection::unrestricted(384, d_k, d_v, seed)` (`W = dW`, no base).
  - C8-E/F/G/H: `ResidualQkvProjection::truncated_identity_residual(384, d_k, d_v, alpha)`.
- Negative selection: deterministic stratified miner (`HardNegativeMiner`) over the
  union (`top_k = candidate_budget`, C8 OFF), excluding positives. Strata and quotas
  are fixed by `HardNegativeConfig` defaults (50% high-baseline, 20% disagreement,
  20% near-positive, 10% uniform). Quotas are enforced by largest-remainder, so the
  total is exact.
- HYPERPARAMETERS (FROZEN):
  - seed = 20260925
  - learning_rate = 1e-2
  - epochs = 5
  - temperature (tau) = 0.07
  - l2 = 1e-4
  - batch_size = 8
  - negatives_per_query = 8
  - residual_scale (lambda) = 0.1 (primary); support {0.05, 0.1, 0.25, 0.5}
  - residual_alpha = 0.1
  - residual_regularization (alpha_r) = 1e-3
  - distillation_weight (beta) = 0.0 (E/F/G) / 1.0 (H)
  - distillation_temperature (tau_d) = 0.5
  - use_distillation = false (E/F/G) / true (H)
  - scorer: w_attn = 1.0; w_evidence = 0.0 (E/D/H/I) / 1.0 (F/G);
    w_disagree = 0.0 (E/F/H/I) / 1.0 (G)
- Output: `C8ModelCard` JSON with `dataset_hash` = FNV-1a of the VALIDATION
  materialization, `negative_provenance_hash`, training meta, loss history, and the
  residual projection. The runtime config is reconstructed from the card.
- **No data leakage**: training reads only VALIDATION qrels/queries/vectors; TEST
  queries and TEST qrels never enter the trainer.

### Dimension / depth sweep (support, non-primary)
- key/value dims: {32, 64, 128, 384}.
- candidate depths: {25, 50, 100, 250, 500}.
These are recorded as support cells; the primary contrast is the frozen config above.

---

## 5. Experimental Design

### Splits (Per C4/C5/C6/C7)
| Split | SciFact | NFCorpus |
|-------|---------|----------|
| TEST | 300 qrels (test.tsv) | 323 qrels (test.tsv) |
| VALID | 200 seeded train-sample | 324 dev.tsv |
| PROBE/SMOKE | 20 from VALID | 20 from VALID |

### Replications
- 5 fresh-process runs per TEST cell (seed 20260925)
- Query order: `shuffled_indices(n, seed + rep * 0x9E3779B9)`

### Observability (mandatory per query)
For every run the probe records:
- `candidate_count` (union size), `c8_fingerprint`, effective fusion weights
- Per returned candidate: `id`, per-head normalized sims, `multi_head_similarity`,
  `bm25` (None), `final_score`, rank
- C8 trace (when enabled): per-candidate `baseline_score`, `attention_delta`,
  `residual_scale`, `applied_correction`, `final_score`, and (full trace) `weights`
  `A_d`, `output` `O_d`, `logits`, `entropy`; per-head mean; `mean_entropy`;
  `mean_abs_correction`; timings; cache stats.
- Per query: Spearman(`delta`, `final`) and Spearman(`delta`, `baseline`).
- Rank-change vs C8-B over the shared candidate set (Kendall-tau / top-k Jaccard).

### Plan Cells
| Track | Cells | Description |
|-------|-------|-------------|
| SAFETY | 4 | EFPROBE: ef 16 vs 128 on both datasets (MODE-B) |
| SAFETY | 18 | SMOKE: all 9 arms on both datasets (20 qids) |
| TUNE | 10 | VALIDATION: D/E/F/G/H train (2 datasets x 5) |
| TRK-A | 18 | TEST: 9 arms x 2 datasets x 5 reps |
| SUPPORT | 16 | dim/depth sweep on E (2 datasets x 8) |

---

## 6. Prove the Attention Is Genuine (Mandatory)

For C/D/E/F/G/H/I arms, demonstrate:
- `features` carry the additive C8 correction; `applied_correction != 0` for `lambda != 0`.
- Attention weights `A_d` and outputs `O_d` **vary across candidates and across
  queries** (entropy > 0, std > 0).
- `dS_attention` correlates with but is not identical to `S_base` (report Spearman;
  not >= 0.999).
- The fusion identity `S_final == S_base + lambda*dS` holds per candidate
  (`mean_abs_correction` and the verifier enforce it).
- Determinism: identical config + identical inputs -> bit-identical scores
  (integration tests and the harness re-run on SMOKE).
- `lambda = 0` parity: arm C reproduces arm B bit-for-bit.
- Cache parity: arm I reproduces arm E bit-for-bit.
- Identifiability: training fixtures give each head a distinct aligned vector; with
  identical heads the softmax shift-invariance makes Q/K gradients vanish exactly.

---

## 7. No Test-Set Tuning

Frozen before TEST (no TEST queries touched during TUNE):
- residual_scale `lambda` grid and selection rule (VALIDATION only, tie -> smaller lambda)
- QKV training hyperparameters (table in §4)
- fusion weights for non-gated arms: 0.3 / 0.5 / 0.2 (defaults, unchanged from C5/C6/C7)
- attention dims: d_a = 384, d_k = d_v = 64, 3 heads
- `ef_search = 64`, budgets as §3
- TUNED ONLY ON VALIDATION (10 TUNE cells).

---

## 8. Stop Conditions (Any -> Halt + Preserve + Report)

1. Anchor hash mismatch
2. Exact-oracle discrepancy
3. ef-sensitivity probe failing (ef not a real knob)
4. Candidate-set mismatch among B/C/D/E/F/G/H/I in any TEST cell
5. TEST data leakage during TUNE (qrels/queries from a TEST split in a training artifact)
6. Non-determinism detected on re-run (hash mismatch on SMOKE)
7. C8 fingerprints identical between ablated arms (config collision)
8. NaN / Inf / non-finite scores or attention weights
9. `lambda = 0` parity violated (C != B anywhere)
10. Cache parity violated (I != E anywhere)
11. Raw evidence overwritten
12. 3 consecutive harness crashes
13. Irreproducible statistics (bootstrap CIs empty)

---

## 9. Statistics (Per C4/C5/C6/C7 Framework)

- Unit = query; per-query metric = mean of 5 reps
- Paired bootstrap 10,000 (seed 20260925) -> 95% CI on the mean difference
- Wilcoxon signed-rank test (two-sided)
- Holm-Bonferroni across primary contrasts: **B vs C/D/E/F/G/H/I x 2 datasets = 14 contrasts**
- Cohen's dz effect sizes
- `|delta| < 0.01` -> negligible
- No winner labels; no post-hoc exclusion
- Primary metric: nDCG@10; secondary: R@10, MRR

---

## 10. Deliverables

1. `c8-protocol.md` (this file)
2. `c8-run-plan.csv` (frozen)
3. `probe/` — `c8pilot` (reproducible release build + JSON contract)
4. `harness/c8_test_run.py` (orchestrator)
5. `harness/c8_analyze.py` (statistics)
6. `harness/c8_verify.py` (invariant checks)
7. 10 reports in `analysis/reports/`:
   `1_results_report.md` ... `10_closure_report.md`
8. `analysis/statistical_results.json`
9. Raw evidence in `raw/C8-*/artifacts/`
10. `raw/RUN-INDEX.yaml` registrations (additive)
11. Updated `research/comparative-study/README.md`
12. PR into main (no auto-merge)
