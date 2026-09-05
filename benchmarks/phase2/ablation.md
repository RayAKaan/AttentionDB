# Phase 2 Ablation Results (§23–28)

Run: see `run_info.txt` (unix timestamp, corpus, seed). Everything below is
measured in-run on 2026-09-03; no number is invented, copied, or extrapolated.

## Setup

- Corpus: 10 000 docs, dim 32, 100 clusters, **8 heads** with observation
  noise σ = [0.02, 0.04, 0.07, 0.12, 0.05, 0.08, 0.11, 0.14] per dimension.
  Each doc's *true* vector v_i = normalize(centroid + 0.35·u_i); head h
  observes normalize(v_i + N(0, σ_h)). Head 0 ("default") is the cleanest —
  the single-head baseline is deliberately STRONG.
- 100 queries = noised centroids of the first 100 docs' clusters, top_k = 10.
- Ground truth: exact cosine(v_i, query) over the TRUE vectors — well-defined
  intra-cluster ordering (earlier all-same-centroid GT was a tie lottery and
  produced a meaningless ~0.1 recall; that harness bug is fixed, see git log).
- Defaults everywhere (ef_search 64, MinMax, parallel on). Single run,
  release build. Ablation modes share the identical query set and corpus.

## Results (ablation.csv)

| mode | R@10 | NDCG@10 | MRR | p50 µs | p99 µs | QPS |
|---|---|---|---|---|---|---|
| A0 head default (σ .02) | **0.763** | **0.826** | **1.000** | 186 | 277 | 5198 |
| A1 head semantic (σ .04) | 0.562 | 0.631 | 0.931 | 268 | 377 | 3654 |
| A2 head lexical (σ .07) | 0.348 | 0.392 | 0.750 | 295 | 423 | 3282 |
| A3 head graph (σ .12) | 0.249 | 0.257 | 0.486 | 350 | 531 | 2760 |
| A4 h4 (σ .05) | 0.487 | 0.536 | 0.841 | 276 | 421 | 3555 |
| A5 h5 (σ .08) | 0.341 | 0.375 | 0.688 | 315 | 463 | 3148 |
| A6 h6 (σ .11) | 0.247 | 0.273 | 0.553 | 341 | 550 | 2810 |
| A7 h7 (σ .14) | 0.194 | 0.201 | 0.425 | 370 | 584 | 2616 |
| B multi-head fixed | 0.672 | 0.750 | 0.985 | 1734 | 2198 | 573 |
| C learned gating | 0.672 | 0.750 | 0.985 | 1744 | 2014 | 574 |
| D +QK attention | 0.672 | 0.750 | 0.985 | 1764 | 2013 | 572 |
| E full (exact rerank) | 0.558 | 0.626 | 0.951 | 2075 | 3509 | 476 |

Head scaling (`heads_scaling.csv`, mode B): recall identical serial vs
parallel at every head count (1/2/4/8) — parallel execution does not change
semantics, confirmed empirically as well as by unit test.

| heads | R@10 | p50 µs (parallel) | p50 µs (serial) | parallel speedup |
|---|---|---|---|---|
| 1 | 0.763 | 253 | 224 | ~1.0 (single head: no parallelism) |
| 2 | 0.731 | 426 | 461 | 1.08× |
| 4 | 0.667 | 945 | 1312 | 1.39× |
| 8 | 0.672 | 1892 | 3253 | **1.72×** |

Corpus build: 129.1 s for 10 000 docs × 8 heads ≈ **77 docs/s insert
throughput** (async durability). Peak RSS ≈ 367 MB.

## Answers to the four questions (§28)

**1. Does multi-head help? — NO WIN on this corpus (TRADEOFF).**
B (0.672) LOSES to the best single head A0 (0.763) while costing ~9× the
p50 latency (1734 vs 186 µs). It beats the other seven single heads (mean of
A1–A7 = 0.339), so multi-head union is *robust to picking the wrong head* —
but if you know the best head, a single HNSW is both more accurate and much
cheaper here. This is exactly the §41 warning: more heads ≠ better; equal
weights let the six noisier heads outvote the clean one.

**2. Does gating help? — NO SIGNIFICANT WIN (C ≡ B to 3 decimals).**
Untrained gating produces a (near-)uniform profile, so C is mathematically
close to fixed fusion. The mechanism exists and is wired end-to-end; the
*learning* is what's missing. No benefit is claimed.

**3. Does QK attention help? — NO SIGNIFICANT WIN (D ≡ C).**
By design the AttentionScorer is identity-init (untrained): D's candidate
attention cannot change C's ranking yet. "Attention does not help" is the
correct finding for the current system and we report it as such.

**4. Does exact rerank help? — NO (negative on this dataset).**
E (0.558) is WORSE than D (0.672): the exact-rerank mixture weights every
head equally, so noisy-head exact cosines drag the final ordering below the
normalized per-head-champion fusion. This is a real, unfavorable result and
it stands in the report. Candidate fix (future work, untested): rerank with
quality-weighted head mixtures, or rerank only the best gate-selected heads.

**Parallelism: WIN.** Identical recall at all head counts serial vs parallel
(semantics preserved), 1.39–1.72× p50 speedup at 4–8 heads.

**Latency/memory cost of the pipeline:** single head 186 µs → 8 heads 1892 µs
p50 (~10×). Exact rerank adds ~180–300 µs. RSS 367 MB for 80 k vectors +
8 HNSW indexes — recorded, not judged.

## Honest conclusion to "does attention measurably help?"

Not yet. With untrained gating and identity-init attention, modes C and D are
indistinguishable from fixed fusion — the measured deltas are zero by
construction, and this report will not dress that up. The staged pipeline's
proven value in this run is candidate ROBUSTNESS (never worse than the 2nd-best
single head) and parallel execution; its weakness is quality-blind weighting,
which currently *forfeits* accuracy to the best single head. The lever is
gating/attention trained or calibrated per query (or per head-quality
estimates); until that lands, any "attention" superiority claim would be
dishonest and is not made.

## Limitations

- Synthetic clustered Gaussians; GT favors centroid-aligned structure. Heads
  with genuinely different semantics (not just noise levels) may behave
  differently — that experiment needs real multi-view data.
- 10 k docs, ef_search 64: HNSW recall is near-exact here; scale effects
  (100 k/1 M) belong to the scale benchmark, not this ablation.
- Single run, shared CI-class sandbox: latencies have run-to-run variance;
  relative ordering across modes was stable across our three runs.
- Head-quality spread is engineered (σ ladder); real deployments must measure
  their own head quality (§41 head-diversity report) before enabling modes.
