# C7 Protocol: Genuine Candidate-Level Q/K/V Attention

**Version**: 1.0.0 (frozen)
**Branch**: `comparative-study/c7-genuine-qkv-attention`
**Depends on**: C0–C6 complete; C6 artifacts frozen in `raw/`; `attentiondb-attention` crate (C7 math core) integrated into `attentiondb-core`

---

## 1. Research Question

**H0**: The genuine candidate-level Q/K/V attention channel does not materially change ranking relevance over the multi-head union baseline (C5-B / C6-B) when the candidate set, budget, and latency are controlled.

**H1**: A candidate-level attention channel — `s_d = w_attn·⟨q_a, O_d⟩ + w_evidence·agg(r_d) + bias`, with `O_d = A_d V_d`, `A_d = softmax(Q_d K_dᵀ/√d_k)` — computed over the same per-head candidate vectors used for membership, materially improves ranking quality (nDCG@10, R@10, MRR) versus the union baseline, within an identical candidate set and identical budget.

Sub-questions (ablation):
- **C vs D**: does tuning the channel fusion weights (gate) matter, holding identity QKV fixed?
- **D vs E**: does contrastively-learned QKV improve over identity QKV?
- **E vs F**: does retrieval evidence added to the attention score improve over attention alone?

---

## 2. Mandatory Arms (6)

| Arm | Name | RetrievalMode | Attention channel | Fusion weights | Learned? |
|-----|------|---------------|-------------------|----------------|----------|
| C7-A | Canonical single-head | `SingleHead` | OFF | fixed | No |
| C7-B | Independent multi-head union (control) | `FixedFusion` | OFF | fixed 0.3/0.5/0.2 | No |
| C7-C | Union + identity-QKV attention + learned gate | `FixedFusion` | identity QKV | gate `g*` on VALIDATION | Yes (gate only) |
| C7-D | Union + identity-QKV attention | `FixedFusion` | identity QKV | fixed 0.3/0.5/0.2 | No |
| C7-E | Union + learned-QKV attention | `FixedFusion` | learned QKV (contrastive) | fixed 0.3/0.5/0.2 | Yes (QKV) |
| C7-F | C7-E + retrieval evidence | `FixedFusion` | learned QKV + evidence | fixed 0.3/0.5/0.2 | Yes (QKV) |

Invariants:
- **Candidate membership**: arms B, C, D, E, F share an *identical* union per query (same `FixedFusion` path, same heads, ef, budgets). Only the attention channel score differs among C/D/E/F and B. Rank differences are attributable to attention scores (§ core docstring contract).
- **C7-A** is the single-head canonical control (same effective candidate budget path as C5-A/C6-A). **C7-B** reproduces C5-B / C6-B candidate semantics with the attention channel absent.

Final score equation (documented, no hidden multipliers):
`S = (w_attention·attn + w_multi_head_similarity·mhs + w_bm25·bm25)`, renormalized over present components by `fuse_candidate`. In this study `bm25_raw = None` (vector-only), so the effective equation for C/D/E/F is `S = (w_attn_gate·attn + w_mhs_gate·mhs)/(w_attn_gate+w_mhs_gate)`.

Attention config per arm (frozen, hashed by `config_fingerprint`):
- C7-B: `attention=None` (or `AttentionConfig::disabled`).
- C7-C, C7-D: `AttentionConfig::fixed_identity(3, 384, 384, 384)`; scorer `AttentionScorer::new(1.0, 0.0, 0.0)`; `use_evidence=false`.
- C7-E: `AttentionConfig::learned(3, 384, 384, 384, identity_aligns, identity_q_align, learned_qkv, use_evidence=false, AttentionScorer::new(1.0,0.0,0.0))`.
- C7-F: as E but `use_evidence=true`, scorer `AttentionScorer::new(1.0, 1.0, 0.0)` (evidence term weighted equally).

---

## 3. Budget Constraints (FROZEN)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `candidate_budget` | 500 | Union cap (same as C5/C6) |
| `ef_search` | 64 | Per-head HNSW EF (same as C6-B) |
| `min_candidates_per_head` | 20 | Floor |
| `max_candidates_per_head` | 300 | Ceiling |
| `attention_dim` (d_a) | 384 | = collection dim |
| `key_dim` (d_k) | 384 | = collection dim |
| `value_dim` (d_v) | 384 | = collection dim |
| `heads` | TITLE, BODY, CITE | 3 heads (384-d each) |
| `bm25_raw` | None | Vector-only study (same as C6) |

All arms operate under identical candidate budget and identical per-head vectors. The attention channel adds no candidate work — it re-uses the membership vectors already fetched for the union (no extra HNSW calls), so the only cost adder is the attention compute (recorded as `compute_time_us`).

---

## 4. Training / Tuning Policy

Training and tuning occur on the **VALIDATION split only**. Everything is frozen before TEST.

### C7-C: fused-channel gate (`g*`)
The effective ratio between the attention channel and the multi-head-similarity channel is selected on VALIDATION by a deterministic grid scan:
```
g ∈ {0.0, 0.1, ..., 1.0}
for each g: fusion = {attention: g, multi_head_similarity: 1-g, bm25: 0}
            run FULL VALIDATION set, identity-QKV attention
            score = mean nDCG@10
g* = argmax_g score ; ties -> larger g
```
The candidate set is identical across g (FixedFusion with the same budget), so the scan isolates only the fusion weight. `C7-C` TEST uses `fusion = {g*, 1-g*, 0}`.

### C7-E / C7-F: contrastive QKV training
- Optimizer: deterministic Adam (no RNG) via `ContrastiveQkvTrainer` (analytic gradients, `candidate_gradient`), module `attentiondb-attention`.
- Init: `QkvProjection::identity(384)` — C7-E starts exactly at C7-D; any TEST delta is attributable to learning.
- Negative selection: for each VALIDATION query, the negative set is the union candidates (retrieved at `top_k = candidate_budget`, identity-QKV off) **excluding** relevant documents (per qrels); deterministic sample of 8 negatives per query (seed 20260925). Evidence per candidate = `agg(r_d)` = mean of present normalized head similarities.
- Loss: InfoNCE over the 9 candidates (1 positive + 8 negatives): `loss = -ln(softmax(s_pos/τ)_pos)`.
- HYPERPARAMETERS (FROZEN):
  - seed = 20260925
  - learning_rate = 1e-2
  - epochs = 5
  - temperature (τ) = 0.07
  - l2 = 1e-4
  - batch_size (recorded) = 8
  - w_attn = 1.0, w_evidence = 0.0 (E) / 1.0 (F)
- Output: `C7ModelCard` JSON with `dataset_hash` = FNV-1a of the VALIDATION materialization, training meta, loss history, and hash of final weights. The runtime `AttentionConfig` is reconstructed via `TrainedModel::to_config`.
- **No data leakage**: training reads only VALIDATION qrels/queries/vectors; TEST queries and TEST qrels never enter the trainer.

---

## 5. Experimental Design

### Splits (Per C4/C5/C6)
| Split | SciFact | NFCorpus |
|-------|---------|----------|
| TEST | 300 qrels (test.tsv) | 323 qrels (test.tsv) |
| VALID | 200 seeded train-sample | 324 dev.tsv |
| PROBE/SMOKE | 20 from VALID | 20 from VALID |

### Replications
- 5 fresh-process runs per TEST cell (seed 20260925)
- Query order: `shuffled_indices(n, seed + rep × 0x9E3779B9)`

### Observability (mandatory per query)
For every run the probe records:
- `candidate_count` (union size), `attention_config_fingerprint`, effective fusion weights
- Per returned candidate: `id`, per-head normalized sims, `multi_head_similarity`, `bm25` (None), `attention` channel score, `final_score`, rank
- C7 attention trace (when attention enabled): per-candidate per-head attention weights `A_d`, output `O_d`, logits, entropy; per-head mean; `mean_entropy`; `compute_time_us`
- Attention statistics per query: min/mean/max/std of the attention channel over all union candidates; Spearman correlation between attention channel and final score
- Rank-change vs C7-B over the shared candidate set (Kendall-τ / top-k Jaccard)

### Plan Cells (34 total)

| Track | Cells | Description |
|-------|-------|-------------|
| SAFETY | 4 | EFPROBE: ef 16 vs 128 on both datasets (MODE-B) |
| SAFETY | 12 | SMOKE: all 6 arms on both datasets (20 qids) |
| TUNE | 6 | VALIDATION: C gate scan (2), E train (2), F train (2) |
| TRK-A | 12 | TEST: 6 arms × 2 datasets × 5 reps |

---

## 6. Prove the Attention Is Genuine (Mandatory)

For C/D/E/F arms, demonstrate:
- `features.attention` is `Some` for every returned candidate; it is absent (None) for C7-A/B.
- Attention scores (and per-head weight vectors `A_d`) **vary across candidates and across queries** (entropy > 0, std > 0).
- Attention channel correlates with but is not identical to `multi_head_similarity` (report Spearman; not ≥ 0.999).
- Determinism: identical config + identical inputs → bit-identical attention scores (verified by the integration tests and re-checked by the harness on SMOKE).
- Backward compatibility: `attention = None` (C7-B) reproduces the legacy `FixedFusion` scoring exactly (parity test), so C5-B/C6-B candidate semantics are preserved.

---

## 7. No Test-Set Tuning

Frozen before TEST (no TEST queries touched during TUNE / gate scan / training):
- gate scan: `g ∈ {0.0,0.1,...,1.0}`, tie → larger g, selected on VALIDATION
- QKV training hyperparameters (table in §4)
- fusion weights for non-gate arms: 0.3 / 0.5 / 0.2 (defaults, unchanged from C5/C6)
- attention dims: 384/384/384, identity alignments, 3 heads
- `ef_search = 64`, budgets as §3
- TUNED ONLY ON VALIDATION (6 TUNE cells).

---

## 8. Stop Conditions (Any → Halt + Preserve + Report)

1. Anchor hash mismatch (B0)
2. Exact-oracle discrepancy
3. ef-sensitivity probe failing (ef not a real knob)
4. Candidate-set mismatch among B/C/D/E/F in any TEST cell
5. TEST data leakage during TUNE (qrels/queries from a TEST split in a training artifact)
6. Non-determinism detected on re-run (hash mismatch on SMOKE)
7. Attention fingerprints identical between ablated arms (config collision)
8. NaN / Inf / non-finite scores or attention weights
9. Raw evidence overwritten
10. 3 consecutive harness crashes
11. Irreproducible statistics (bootstrap CIs empty or non-overlapping with sanity bounds)
12. Run-ID inconsistency in `raw/RUN-INDEX.yaml`

---

## 9. Statistics (Per C4/C5/C6 Framework)

- Unit = query; per-query metric = mean of 5 reps
- Paired bootstrap 10,000 (seed 20260925) → 95% CI on the mean difference
- Wilcoxon signed-rank test (two-sided)
- Holm-Bonferroni across primary contrasts: **B vs C/D/E/F × 2 datasets = 8 contrasts**
- Cohen's dz effect sizes
- `|Δ| < 0.01` → negligible
- No winner labels; no post-hoc exclusion
- Primary metric: nDCG@10; secondary: R@10, MRR

---

## 10. Deliverables

1. `c7-protocol.md` (this file)
2. `c7-run-plan.csv` (34 cells, frozen)
3. `probe/` — `c7pilot` (reproducible `/Brepro` build + JSON contract, mirrors `c6pilot`)
4. `harness/c7_test_run.py` (orchestrator)
5. `harness/c7_analyze.py` (statistics)
6. `harness/c7_verify.py` (invariant checks)
7. 10 reports in `analysis/reports/`:
   - 1_results_report.md
   - 2_statistical_report.md
   - 3_candidate_report.md
   - 4_environment_report.md
   - 5_dataset_report.md
   - 6_resource_report.md
   - 7_implementation_report.md
   - 8_failure_report.md
   - 9_reproducibility_report.md
   - 10_closure_report.md
8. `analysis/statistical_results.json`
9. Raw evidence in `raw/C7-*/artifacts/`
10. `raw/RUN-INDEX.yaml` registrations
11. Updated `research/comparative-study/README.md`
12. PR into main (no auto-merge)