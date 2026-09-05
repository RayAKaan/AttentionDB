# Experimental Protocol (§6)

Every value below is the actual value used by the code
(`benchmarks/phase2b`, `learned`, `core`). Nothing is "standard settings".

## 1. Corpus generation

Deterministic per (corpus, seed) via xorshift64* (`Rng`, `corpora.rs`,
`benchmarks/phase2b/src/corpora.rs`); Gaussians are 4-uniform sums scaled by
0.866 (σ≈0.577 per draw), centered. All vectors are L2-normalized after
construction (`l2normalize`; zero-norm guarded at 1e-12).

- **controlled**: dim 32, 4 heads, 4 groups × 12 clusters × 8 docs = 384 docs.
  Group-g centroids carry signal only in block g (dims `g·8..g·8+8`, values
  N(0.6, 0.289) vs N(0, 0.1) elsewhere). Doc true vector = normalize(centroid
  + 0.35·u), u a distinct unit Gaussian direction per doc. Head h observes
  normalize(true_vec + N(0, σ)) with σ = 0.02 for h = the doc's group and
  σ = 0.30 otherwise. Queries = 300 noised centroids (noise 0.05).
- **noise**: dim 32, 8 heads, 100 clusters, 10 000 docs (Phase 2 corpus
  regenerated with per-doc true vectors). True vector = normalize(centroid +
  0.35·u). Head h observes normalize(true_vec + N(0, σ_h)),
  σ = [0.02, 0.04, 0.07, 0.12, 0.05, 0.08, 0.11, 0.14] — head 0 globally
  best. Queries = 300 noised centroids of the first 300 docs' clusters
  (noise 0.1).
- **multiview**: 3 views × dim 64 (head input dim 64, gating input 192 =
  concatenation), 40 topics, 3 000 docs. Doc true vector per view v =
  coarse(v) + 0.6·u_v: semantic coarse = unit topic centroid; lexical coarse
  = 10 topic keywords hashed into 64 buckets (FNV-mixed, salt 1); metadata
  coarse = category one-hot (8 categories, category = topic mod 8). The
  0.6·u_v term is a doc-UNIQUE direction per view. Query of type V targets a
  specific doc and carries its view-V vector (+N(0, 0.03) per dim); the
  other two views get only the topic-level coarse vector (+N(0, 0.05)) —
  what a cross-modal extractor could supply. Query type cycles
  semantic/lexical/metadata by query index; the type is NOT a model input.

## 2. Ingestion and HNSW

Docs are inserted into a fresh in-memory `AttentionEngine`
(`Durability::Async`, temp dir, destroyed after the run). Collection
settings: ef_search 64, ef_construction 400, M 16, cosine metric (API
defaults, `core` engine). One head per corpus head name; per-head vectors
are inserted as k_vecs. Insert order = doc index order.

**Known non-reproducibility**: `hnsw_rs` seeds its layer-assignment RNG from
OS entropy, so candidate pools vary slightly between processes for the same
corpus seed. Consequence: the CACHED DATASET (§4) is the reproducibility
unit — every downstream step (training, calibration, evaluation) is
deterministic given `dataset.json` + code commit. See `reproducibility.md`.

## 3. Candidate generation

Per query and per head, one independent `RetrievalMode::SingleHead`
`attend_detailed` call with top_k = 100 (POOL). No union is performed for
the dataset; fusion studies union the cached pools offline with the exact
same weighted-sum arithmetic as the pipeline (`learned/src/eval.rs`).

## 4. Cached dataset (§4 of the spec)

`dataset.json` per run: per query — engine-id ground truth (best 10 by
exact cosine), gating input, split tag, and per head: candidate ids (rank
order), raw scores, per-head MinMax normalized scores, exact scores
(cosine of the query vector against the generator's own head view of each
candidate doc), and per-head Recall@10 / NDCG@10 / MRR against the
ground truth. Dataset format `attentiondb-gating-dataset` v1; parse-time
validation (dim, head count, array lengths) with typed errors.

## 5. Normalization

Per-head MinMax over the candidate list; degenerate range (< 1e-6) maps to
zeros; non-finite raw scores excluded from min/max and set to 0. Identical
arithmetic to pipeline stage 3 (`core/src/retrieval.rs::normalize_scores`,
MinMax path).

## 6. Splitting (§3)

Fisher–Yates shuffle of query indices with seed 42 ^ 0x5EED, then 70% /
15% / 15% train / validation / test. Test is consumed exactly once per run,
after hyperparameter and temperature selection.

## 7. Training (§5, §6)

Model: `GatingMlp` — input → ReLU(hidden) → head logits → stable softmax.
Init: seeded xorshift64* Gaussians, σ = 1/√fan_in, zero biases.
Grid: objectives {quality-regression (MSE on logits), soft-target
cross-entropy (targets softmax(quality/τ), τ = 0.1), pairwise logistic
(margin 0.05)} × {hidden 32, lr 0.01, batch 32} and {hidden 64, lr 0.003,
batch 16}. Adam (β1 0.9, β2 0.999, ε 1e-8), L2 1e-4 on weight matrices,
max 200 epochs, early stopping on validation loss (patience 15, min_delta
1e-4). Targets are per-head Recall@10 stored in the dataset. Selection
criterion across the grid: validation R@10 of the temperature-calibrated
model. Training seed default 42 (multi-seed run varies it, split fixed).

## 8. Temperature calibration (§19)

After training: T ∈ {4, 2, 1, 0.75, 0.5, 0.35, 0.25}; prediction =
softmax(logits/T). T selected by validation R@10 only. Typical fitted T:
0.25–0.5 (controlled/noise), 0.25–4 (multiview; when the model fails to
learn, validation picks T = 4, i.e. near-uniform — a safe fallback).

## 9. Evaluation

Metrics: Recall@{1,5,10,50}, NDCG@10 (log2 discount, rank r weight
1/log2(r+1), IDCG over the GT-10 ideal), MRR. Fusion:
score(c) = Σ_h w_h · norm_h(c) over heads containing c; ties broken by
smaller engine id. RRF: score(c) = Σ_h 1/(60 + rank_h(c)). All evaluation
arithmetic lives in `learned/src/eval.rs` and is unit-tested.

## 10. Latency measurement

Model inference: 200 random probe vectors × 2000 repetitions, wall clock
(`std::time::Instant`), after warmup — reported µs/query (PH2B-LATENCY-001).
End-to-end pipeline latencies (p50/p95/p99, QPS) come from the frozen
Phase 2 ablation (`results/ablation.csv`): 100 queries, in-run wall clock
per query, release build, same-process, percentiles by nearest-rank.
Hardware: shared CI-class sandbox VM (`run-manifest.json` records cpu
count); absolute latencies are NOT comparable across machines — treat
ratios as indicative only.

## 11. Ground truth

Defined formally in `ground-truth.md`.
