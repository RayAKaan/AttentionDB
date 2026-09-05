# Claim Ledger (§21, §22)

Every major claim maps to experiment IDs in `raw/experiment-index.json`.
Statuses: SUPPORTED / NOT SUPPORTED / OPEN QUESTION. Statistical confidence
is reported as mean ± std over training seeds where a multi-seed run exists;
all other entries are single-run point estimates (stated).

## SUPPORTED

**C1.** "Query-dependent learned gating improves retrieval over fixed
multi-head fusion on the evaluated controlled and multiview corpora."
→ PH2B-GATING-004 (0.9533 vs 0.6244, +33pp R@10), PH2B-MULTIVIEW-005
(0.5322 vs 0.2128, +32pp), PH2B-MULTISEED-002 (0.4365 ± 0.0361 vs 0.2128).
Limitation: synthetic corpora, small query counts, single split per corpus.

**C2.** "Gating can recover oracle head-selection behavior on the
controlled corpus." → PH2B-GATING-004: gating R@10/NDCG@10 identical to
oracle (0.9533/0.9700). PH2B-MULTISEED-001: 0.9556 ± 0.0000 across seeds.
Limitation: 4 heads, strong block-structured signal.

**C3.** "RRF loses to trained gating on the evaluated corpora." → F7
evidence chain; margins +6.9pp to +28.4pp R@10. Limitation: RRF k fixed at
60 (the standard constant [CITE: RRF]); no k sweep.

**C4.** "Trained gating adds negligible inference cost." → PH2B-LATENCY-001:
0.87 µs/query, 1188 params, 20 288 bytes serialized, CPU-only. Limitation:
micro-benchmark on one machine class; end-to-end adds one matmul per query.

**C5.** "On a globally-dominated corpus, trained gating reduces to selecting
the dominant head." → PH2B-NOISE-003 (avg weight 0.894, selection 100%,
quality ≈ global best). Limitation: one corpus family.

**C6.** "Parallel head execution preserves semantics and speeds up
multi-head search." → Phase 2 `heads-scaling.csv` (identical recall serial
vs parallel; 1.39×/1.72× p50 at 4/8 heads) + unit tests (parallel==serial).
Limitation: shared VM, cpu-count dependent.

**C7.** "Ground-truth construction and id-mapping bugs were found, fixed,
and re-run." → PH2B-GATING-001/002 (invalidated), harness-corrections.md,
PH2B-GATING-003+. Limitation: none (documented).

## NOT SUPPORTED

**N1.** "Candidate-level QK attention improves retrieval." → No trained-QK
experiment exists (PH2C-QK-* pending). The only QK data point is the
UNTRAINED identity scorer (Phase 2 mode D = mode C), which supports only:
"identity-init QK adds nothing" (F8). Claim status until trained results
exist: NOT SUPPORTED (insufficient evidence).

**N2.** "More heads improve retrieval." → Phase 2 ablation: 8-head fixed
fusion (0.672) LOSES to the best single head (0.763) on its own corpus.
Multi-head helps only as robustness-to-wrong-head-choice, not accuracy.

**N3.** "Exact reranking as implemented in the Phase 2 pipeline improves
retrieval." → Phase 2 mode E 0.558 < mode D 0.672 (regression). The offline
study attributes it to equal-head weighting of exact scores
(PH2C-RERANK-*), but the pipeline itself has not been re-weighted and
re-evaluated — the pipeline-level fix remains unvalidated.

## OPEN QUESTIONS

**Q1.** Does learned gating generalize to real-world multi-view retrieval
workloads (real embeddings, real lexical features, real structured fields)?

**Q2.** Where exactly between 420 and 840 training queries does the
multiview transition occur, and is it a true threshold or a smooth curve?

**Q3.** Can a rerank mixture with learned/oracle head weights fix mode E
inside the live pipeline (the offline evidence says the scores are fine;
the pipeline change is untested)?

**Q4.** Does trained QK attention add value beyond learned gating (§11–13)?

**Q5.** Does temperature calibration transfer (one T per corpus vs per
query-band), and is validation-fitted T stable across datasets?

**Q6.** How do these results scale beyond 10k docs / 300–1200 queries?

---

## Dated addenda (append-only; never rewrite entries above)

**Addendum to N1 (2026-09-04, PH2C-QK-001).** A trained-QK data point now
exists — on the SYNTHETIC sanity dataset only: linear QK reaches test
R@1 = 1.0000 where every gating variant is ≤ chance (0.0027 / 0.1000),
confirming the machinery can learn candidate-level interaction that the
gating class provably cannot express [PH2C-QK-001]. N1 itself is UNCHANGED
for real corpora: no trained-QK result on controlled/noise/multiview
exists yet, so "candidate-level QK improves retrieval (on the real
corpora)" remains NOT SUPPORTED until PH2C-QK-002+ measures it (Rule
Zero: outcome open).
