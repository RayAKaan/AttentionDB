# C11 — End-to-End Workload Evaluation and Paper-Ready Evidence

**Status:** implementation plan and evidence tooling in progress. This PR must not be described as empirical completion until the frozen SciFact and NFCorpus TEST runs have been executed and validated on a named machine. CI verifies the tooling, not the scientific result.

## Research question

Under a frozen retrieval workload, do C7/C8 attention variants improve retrieval quality over the canonical single-head and independent multi-head union controls, and at what end-to-end query-latency and memory cost? Does the C10 cache lifecycle microbenchmark explain any measured serving-cost change?

## Scope and integration

C11 is the integration and evidence phase after C7–C10. It does not replace or rewrite the C7/C8 execution engines and does not mutate the closed Phase 3E evidence.

- Reuse the C8 multi-arm probe because it evaluates arms A–I in one process and verifies the shared candidate union for B–I.
- Use the frozen C8 test splits and run protocol: SciFact TEST (300 qrels), NFCorpus TEST (323 qrels), k=10, ef_search=64, candidate budget=500, 5 fresh-process repetitions, and the C8 frozen attention dimensions/hyperparameters.
- Compare A (canonical single-head), B (independent multi-head union control), and C–I (C8 variants). B is the primary attention-channel control; A is the canonical retrieval baseline.
- Capture per-query recall@10, nDCG@10, MRR@10, and query latency; summarize p50/p90/p95 latency, memory/provenance, deadlines, and union-identity gates.
- Ingest C10 JSONL output as a separate microbenchmark track. Never merge kernel/cache timings into end-to-end query latency.
- Generate machine-readable aggregate JSON/CSV and a paper-ready Markdown report from immutable raw artifacts.
- Use paired per-query bootstrap confidence intervals only where both arms have complete aligned per-query records. Mark missing/incomplete evidence explicitly rather than silently substituting aggregate estimates.
- Keep raw run directories immutable; reruns receive new IDs. Never invent or check in synthetic results as empirical results.

## Frozen TEST design

| Factor | Frozen value |
|---|---|
| Datasets | SciFact, NFCorpus |
| Split | TEST only for primary comparisons |
| Arms | A, B, C, D, E, F, G, H, I |
| Repetitions | 5 fresh-process repetitions per dataset/arm |
| Retrieval cutoff | k=10 |
| Candidate budget | 500 |
| ef_search | 64 |
| Per-head candidate floor/cap | 20 / 300 |
| Warm-up | 20 queries |
| Metrics | recall@10 (qrels), nDCG@10 (qrels), MRR@10, p50/p90/p95 latency (µs), deadline count |
| Primary comparisons | B vs A; C–I vs B, within dataset |
| Uncertainty | Paired query bootstrap, 10,000 resamples, deterministic seed 20261011; only when per-query pairing is verified |
| Multiplicity | Report all contrasts; treat secondary contrasts as exploratory and state that no multiplicity correction is applied unless a predeclared correction is explicitly run |
| Cache track | C10 lifecycle results are reported separately with their own machine/build/config provenance |

## Run procedure

1. Build the existing C8 probe in release mode from the exact commit under evaluation.
2. Verify input dataset hashes, qrels, split membership, model/config fingerprints, and probe binary hash.
3. Execute the C8 frozen TEST cells for both datasets and all A–I arms, with five fresh-process repetitions. Preserve every run directory and raw artifact.
4. Run the C8 integrity verifier and require all candidate-union identity, deterministic smoke, and exact-parity gates to pass before analysis.
5. Run the C11 analyzer with the comparative-study raw root and output directory. See the usage section in the C11 README.
6. Run the evidence validator against the generated outputs. A missing dataset/arm/repetition, failed union gate, missing provenance, or incomplete raw file is a validation failure, not a zero or an omitted observation.
7. Optionally ingest a completed C10 JSONL file. Report cache microbenchmarks in a separate table.
8. Review the generated report and explicitly disclose negative, neutral, incomplete, or resource-limited results.

## Evidence/provenance contract

Each run must retain the original raw artifacts plus environment metadata, configuration, input hashes, probe hash, commit SHA, OS/kernel, CPU, RAM, Rust/Python versions, build profile, seed, timestamps, and all measured metrics. The aggregator records source paths and hashes in its manifest. Raw inputs are never rewritten by C11.

## Acceptance gates

1. CI passes for the full workspace and C11 evidence-tool tests.
2. Frozen run plan and validator agree on all required dataset/arm/repetition cells.
3. No test-set tuning; all C7/C8 model training and tuning remains validation-only.
4. All primary runs include full-query latency and retrieval-quality metrics.
5. Union identity and C8 correctness/parity gates pass for every eligible run.
6. All reported uncertainty intervals are computed from verified paired per-query samples; otherwise they are marked unavailable.
7. C10 microbenchmarks are not conflated with end-to-end latency.
8. Report contains run counts, exclusions, missingness, confidence intervals, provenance, limitations, and no unsupported speedup/quality claim.
9. Preserve raw evidence and add a new run ID for every rerun.
10. Do not mark C11 empirically complete until both datasets' complete frozen TEST matrix is executed and evidence validation passes.

## Non-goals and interpretation boundary

This phase does not assert that attention improves quality or latency. It does not turn CI smoke tests into research evidence, and it does not treat C10 projection/cache timing as a substitute for full query latency. External baseline systems are not claimed unless independently executed under equivalent data, hardware, and workload conditions; the frozen in-repository controls A/B are the primary baseline comparison.