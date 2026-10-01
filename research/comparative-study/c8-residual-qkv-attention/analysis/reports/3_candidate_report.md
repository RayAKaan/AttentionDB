# C8 Candidate Report

Phase: **C8 — Residual candidate-level Q/K/V attention** · Branch: `comparative-study/c8-residual-qkv-attention`

## 1. Union identity (stop condition #4)

The post-fusion residual stage must **not** change which candidates are collected.
Gate result on SMOKE (20 qids) and SMOKE-rerun, both datasets:

| Dataset | B vs C..I candidate-set mismatches |
|---|---|
| SciFact | 0 (all 8 target arms) |
| NFCorpus | 0 (all 8 target arms) |

`union_identity_ok = true`. Arm A (single canonical head) has no candidate union and
is excluded from this gate by construction.

## 2. Candidate counts

SMOKE mean `candidate_count` (budget 500, `ef_search=64`): **500.0** for every C8
arm on both datasets — i.e. the frozen budget binds and no arm silently truncates or
expands the union. EFPROBE (report 4) confirms the `ef` knob is a real lever at the
*search* layer (EF16 vs EF128 changes raw candidate depth and recall), independent of
the post-fusion C8 stage.

## 3. Feature and score integrity

For every candidate the trace carries:

- `baseline_score` (`S_base`, post-fusion pre-C8),
- `attention_delta` (pre-scale ΔS, non-zero even when `residual_scale=0`),
- `applied_correction = residual_scale · ΔS`,
- `final_score = baseline_score + applied_correction`.

Invariants verified: `final == baseline + correction` at `residual_scale=1`
(bit-level equality of the float sum), and for **C** (`λ=0`) both
`applied_correction == 0.0` and `final_score == baseline_score` for all candidates.

## 4. Distinctness of correction arms

Final-score maps differ per arm on 20/20 queries for every pair among
{E, F, G, H, I} (with I==E under cache parity, so I is compared to E as expected),
and D differs from all. The top-10 *ranking* happens to coincide on the 20-query
SMOKE sample for the E/F/G/H/I families, but the scores that induce it do not —
so the arms are genuinely distinct and TEST ranking differences are meaningful.
