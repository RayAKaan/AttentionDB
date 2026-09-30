# C7 Candidate Report

Scope: the candidate-generation and attention-observable layer of the C7 engine (one shared in-process HNSW engine; B/C/D/E/F share the SAME candidate union per query by construction).

## 1. Shared candidate unions (B/C/D/E/F identical by construction)

Config: per-head budget 500 candidates, min 20, max 300 per head; 3 heads (CANONICAL-decomposed HEAD-TITLE / HEAD-BODY / HEAD-CITE); ef_search 64; k=10.

| Dataset | Union size per query (any attention arm) | Min / Max | n_queries (TEST) |
|---|---|---|---|
| SciFact | 500.0 (saturated) | 500 / 500 | 300 |
| NFCorpus | 500.0 (saturated) | 500 / 500 | 323 |

Union identity gate: B vs C/D/E/F mismatches = 0 over every rep × dataset in EFPROBE(na), SMOKE, and TEST (protocol stop condition #4 — not triggered). The union is budget-saturated, so candidate sets are invariant whether attention is applied or not (attention happens *after* candidate generation, per candidate).

## 2. Genuine-attention evidence (protocol §6)

For every candidate returned by attention arms, `features.attention` is `Some`; absent for A/B. Measured over the full TEST ledger (150,000 SCI / 161,500 NFC candidates per arm, rep1):

| Dataset | Arm | Spearman(attention, mhs) | Spearman(attention, final) | mean head-entropy (queries) |
|---|---|---|---|---|
| SciFact | C identity+gate | 0.7232 | 1.0000 * | 1.0986 |
| SciFact | D identity | 0.7232 | 0.8414 | 1.0986 |
| SciFact | E learned | 0.1322 | 0.9173 | 1.0986 |
| SciFact | F learned+evid | 0.3269 | 0.9438 | 1.0986 |
| NFCorpus | C identity+gate | 0.5568 | 0.9948 | 1.0986 |
| NFCorpus | D identity | 0.5568 | 0.7794 | 1.0986 |
| NFCorpus | E learned | 0.0812 | 0.9249 | 1.0986 |
| NFCorpus | F learned+evid | 0.2257 | 0.9362 | 1.0986 |

\* C on SciFact has g*=1.0, so `final = attention` by construction — this rank-1 correlation is a gate artifact, not the mechanism; use D for the mechanism interpretation.

Interpretation:
- Identity attention (C/D) is a nonlinear function of the same per-head similarity dot products it re-scores, hence a moderate-but-not-identical correlation with `multi_head_similarity` (0.56–0.72) — satisfies "correlates but not ≥ 0.999".
- Learned attention (E) is nearly orthogonal to the static similarity score (0.08–0.13) — a genuinely different signal that, as §1 report shows, *hurts* ranking under the frozen budgets/hyperparameters.
- Mean 3-head entropy = ln(3) = 1.0986 for all attention arms: on average the post-softmax head weights are uniform across TITLE/BODY/CITE (variance exists per candidate, but no systematic head preference).

## 3. ef knob probe (EFPROBE, MODE-B, 20 qids VALIDATION)

Because hnsw_rs clamps its search beam to max(ef, k) and per-head k=500 left ef inert, arms carry `search_k = ef` to exercise the true beam:

| Leg | Dataset | recall@10 | cand mean (min/max) | p50 µs |
|---|---|---|---|---|
| EF16 | SciFact | 0.6750 | 30.4 (20–39) | 911 |
| EF128 | SciFact | 0.7250 | 231.4 (182–269) | 2438 |
| EF16 | NFCorpus | 0.1004 | 34.4 (24–44) | 887 |
| EF128 | NFCorpus | 0.1158 | 242.1 (203–275) | 2470 |

The knob is real: beam grows ~8×, candidate count and latency grow accordingly, recall is strictly higher at EF128 on both datasets.

## 4. Attention fingerprints
- Attention arms each carry a distinct `attention_fingerprint` (identity vs learned E vs learned F differ); gate #7 (config collision) not triggered. Fingerprints are stable across reps.