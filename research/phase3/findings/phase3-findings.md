# Phase 3 findings (PROVISIONAL — no finding numbers until §33 evidence validation)

Status vocabulary per spec: SUPPORTED / NOT SUPPORTED / OPEN QUESTION.
Scope is stated per finding; nothing here reopens Phase 2 conclusions.

## F-P3-1 (SUPPORTED, scope: PH3-DS-FM-S) — On a corpus where one view dominates, trained gating CORRECTLY collapses to it

Trained gating converges to w(full)=1.0000 (all weight on the full-image
view) and matches single-head ANN (R@10 0.9853 vs 0.9860; NDCG 0.9906 vs
0.9911; per-seed 0.9840–0.9860). Uniform multi-head fusion DEGRADES quality
(0.6293). This replicates the Phase 2 noise-corpus collapse behavior on
REAL images: the gate does not hallucinate benefit from dominated heads.
Evidence: raw/runs/PH3-QUAL-FM-S/{results.csv,gating_weights.csv}.
Limitation: this corpus cannot exhibit per-query complementarity gains —
that requires a head set with semantically complementary views (pending
text/multi-view dataset family).

## F-P3-2 (SUPPORTED, scope: PH3-DS-FM-S) — Retrieval quality is candidate-generation-bound at the ceiling

Pool-union candidate recall = 0.9992; best arm = 0.9860 (single) / 0.9853
(gating) / 1.0000 (exact, non-approximate). The 1.4pp gap between the ANN
arms and exact is HNSW top-10 miss (ef_search=64) plus ordering inside
pools — i.e., recall is bounded by candidate generation, not fusion.
Evidence: results.csv, latency.csv, config.json candidate_recall.

## F-P3-3 (SUPPORTED, scope: PH3-DS-FM-S, measurement) — Latency structure of the frozen path

p50 per query (warm, release, 2-CPU sandbox): single-head ANN 766 µs;
5-head ANN + fusion 2544 µs; gating MLP forward 36 µs; exact brute-force
reference 3567 µs. The frozen multi-head path costs ~3.3× the single-head
path; the learned component is ~1.4% of the pipeline. Quality-per-cost:
multi-head adds NO quality on this corpus over single-head (F-P3-1), so
its cost is currently unjustified HERE (scope-limited: corpora with
complementary heads may differ — Phase 2 multiview showed gating gains
there).
Evidence: latency.csv.

## F-P3-4 (SUPPORTED, scope: PH3-DS-FM tiers) — Memory is the scaling wall (§19)

Peak RSS 571 MB at 10K docs (raw vectors ≈ 66 MB → ≈8.6× multiplier).
30K docs: OOM-killed at ~82 s of build. 60K docs: OOM-killed at ~111 s.
On a 1984 MB sandbox the largest reproducible scale for this configuration
is 10K docs. This is the primary §9/§19 limitation; per §41 the response
is systems work (memory), not new retrieval mechanisms.
Evidence: PH3-QUAL-FM-M / PH3-QUAL-FM-T30 runs (FAILED, preserved),
results/memory.csv, tables/table-scaling-memory.md.

## OPEN QUESTIONS

- Does gating add value on corpora with semantically complementary views
  (text title/body, multi-field products)? Pending the text dataset family
  (also unlocks BM25/hybrid baselines D/E). [§5/§6]
- Where exactly (between 10K and 30K docs) is the memory wall, and which
  component dominates (HNSW vectors+graph vs document store vs WAL)? [§19]
- Head-count scaling (1/2/4/8) and serial/parallel on realistic workloads.
  [§10]
- Candidate-budget plateau on realistic data. [§11]
- Filtering, mutations-under-query, restart/recovery, concurrency —
  experiment families defined; runs pending. [§12–§24]
