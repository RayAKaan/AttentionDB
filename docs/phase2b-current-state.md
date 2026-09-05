# Phase 2B — Current State (audit before code)

Date: 2026-09-03. Baseline: Phase 2 ablation (accepted by maintainer; results
in `benchmarks/phase2/{ablation.csv,heads_scaling.csv,ablation.md}` — those
numbers are frozen and will not be altered; new experiments land separately).

## 1. What Phase 2 established (frozen results)

| mode | R@10 | p50 |
|---|---|---|
| A0 best single head (σ=.02) | 0.763 | 186 µs |
| B multi-head fixed fusion | 0.672 | 1734 µs |
| C learned gating (untrained) | 0.672 | 1744 µs |
| D +QK attention (identity) | 0.672 | 1764 µs |
| E exact rerank (equal heads) | 0.558 | 2075 µs |

Parallel head execution: 1.39× (4 heads) / 1.72× (8 heads) p50, recall
identical to serial. Diagnosis: **quality-blind fusion** — per-head MinMax
normalization makes every head's champion equal (1.0), so 6 noisy heads
outvote the clean one, and equal-weight exact rerank repeats the mistake on
raw scores.

## 2. Existing assets (verified in code)

**`multihead/src/gating.rs` — `GatingNetwork`**
- Architecture: single linear layer `W[num_heads × input_dim] + b`, softmax.
  (`forward` = stable softmax; `train_step` = cross-entropy + momentum SGD.)
- `GatingTrainer::train_online`: per-example steps, early stop on loss stall.
- Serialization: `save_to_file` = bincode of `(weights, bias)` — **no model
  id, no format version, no architecture metadata, no compat validation**
  (§15, §16 gaps). `load_weights` **silently ignores** mismatched sizes —
  violates §16 (explicit error required; silent fallback forbidden).
- Initialization uses `rand::thread_rng()` — **not reproducible** (§24 gap).

**`core/src/collection.rs`**
- `gating_network: RwLock<Option<GatingNetwork>>` + `load_gating_network_from`.
- Stage 4 of `attend_detailed_inner`: `gate_override` (explicit caller-supplied
  weights) > fixed uniform (FixedFusion) > `get_gated_weights(query, heads)`
  (network path). Profile renormalized; non-finite/zero-sum → uniform fallback.
  This is the natural injection point for a trained model (§14 separation:
  inference reads a loaded model; no training at query time).

**`core/src/retrieval.rs` — `AttentionScorer`**
- QK attention: `W_q[heads × attn_dim]`, `W_k[(2·heads) × attn_dim]`,
  identity-initialized, tanh-bounded, O(C·(F+D)). Untrained ⇒ D ≡ C (measured:
  identical to 3 decimals). Weights are plain `Vec<f32>` — trainable, but no
  training path exists (§11–12 gaps).

**`benchmarks/phase2` harness** (`phase2-bench`)
- Deterministic seeded corpus (xorshift64*), GT = exact cosine over
  data-generating vectors, per-head σ ladder. Produces rankings per mode.
- Missing for 2B: per-head candidate dumps (training dataset), query splits,
  training loop, oracle/uniform/RRF offline evaluators, diagnostics.

## 3. Gap list → Phase 2B plan of record

| § | Gap | Plan |
|---|---|---|
| 2 | no head-quality target | per-head Recall@K/NDCG@K/MRR from GT; soft target = softmax(quality/τ); justify vs sparse one-hot |
| 3 | no splits | seeded 70/15/15 train/val/test; test touched once |
| 4 | no training dataset | serializable JSON: per-query per-head candidates + raw/norm/exact scores + GT; cache so training never re-runs HNSW |
| 5–6 | one loss (CE), non-deterministic init | new deterministic trainer (seeded): A) MSE→quality on logits, B) soft-target CE, C) pairwise logistic; Adam; early stop on val loss |
| 7 | no collapse monitoring | avg/entropy/selection-freq of weights + weight↔quality correlation; report dominance honestly |
| 8–10 | no query-conditional benchmark | controlled corpus: 3 query groups, best head = f(group); oracle per-query baseline |
| 11–13 | QK untrained | train AttentionScorer weights AFTER gating verdict; normalization variants benchmarked; explicit fusion equation |
| 14 | fallback undefined | no model ⇒ configured deterministic fusion; documented |
| 15–18 | model metadata/registry/hot-swap | JSON model cards (format version, dims, head count, seed, dataset hash, commit, timestamp); validation with typed errors; per-collection registry (save/activate/deactivate/inspect; active-delete guard); atomic Arc swap = hot-swap without restart |
| 19–22 | no calibration/diversity diagnostics | weight↔quality correlation/calibration; pairwise overlap@K, Jaccard, score correlation; diversity AND utility reported separately |
| 23–26 | no train CLI/reproducibility/curves/overfitting protocol | `phase2b` binaries with `generate-dataset` / `train` / `evaluate`; every run writes seed+dataset hash+config+timestamp+hardware; per-epoch curves to `benchmarks/phase2b/training/` |
| 27–29 | no extended ablation | rerun with C2/D2/F/G(RRF)/H(oracle) on same corpus/queries/seed; new CSVs only |
| 32 | rerank regression unexplained | offline rerank-weighting experiments (equal/learned/oracle/best-head) on cached exact scores |
| 33 | RRF not compared on identical candidates | add to evaluator |
| 34–37 | no multi-view benchmark | synthetic 3-view corpus (semantic/lexical-hashing/metadata-buckets) with query types; label exclusion enforced (no query-type feature into gating; oracle-only) |
| 38–40 | no latency/size accounting | gating overhead µs-level target; params/bytes/inference latency recorded; CPU-only |
| 41–42 | no report/checklist | `docs/phase2b-final-report.md` with fixed verdict vocabulary |

Known violations to fix in existing code (not silently): `GatingNetwork`
non-reproducible init and silent load-ignore stay for backward compatibility
of the Phase 1 API but are **not** used by the 2B trainer; the 2B trainer and
model format are new modules in `learned` (deterministic, validated).

## 4. Ordering (spec-mandated sequencing)

Gating first (§5–10). If gating cannot beat uniform on the **controlled**
benchmark and correlate with per-query head quality (§30.1–2), STOP per §31 —
no QK training, no extra layers; investigate target/data/loss first.

## 5. Explicit non-goals this phase

HNSW tuning, new heads, new query syntax, GPU, external ML runtimes
(tch feature stays off), sharding/distributed (Phase 1 §55 carry-over).
