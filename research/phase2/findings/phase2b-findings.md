# Phase 2B Findings (§20)

Each finding states its evidence (experiment ID → registry). Status words:
MEASURED (numbers exist in raw results), INTERPRETATION (our reading),
HYPOTHESIS (testable, untested).

## F1 — Learned gating can recover query-conditional head quality. MEASURED

Evidence: PH2B-GATING-004 (controlled) + PH2B-MULTIVIEW-005 (multiview).
On controlled test queries, predicted argmax head matched the query's
group head (selection frequency matches the group distribution
[0.267, 0.244, 0.356, 0.133]); weight–quality Pearson r = 0.903. The model
infers usefulness from the query representation alone — the type label is
never an input (verified by construction; §36).

## F2 — On the controlled corpus, gating recovered 100% of the uniform→oracle gap. MEASURED

R@10: uniform 0.6244, oracle 0.9533, trained gating 0.9533 (identical to
oracle to 4 decimals; NDCG@10 0.9700 both). Gap recovered: (0.9533−0.6244)/
(0.9533−0.6244) = 1.000. Evidence: PH2B-GATING-004; seed-robust
(PH2B-MULTISEED-001: 0.9556 ± 0.0000 over seeds 42/7/1).

## F3 — On the multiview corpus, gating recovered approximately 41% of the gap (single run). MEASURED, with seed caveat

Single run (PH2B-MULTIVIEW-005): uniform 0.2128 → oracle 0.9933; gating
0.5322 = 40.9% of the gap. Multi-seed (PH2B-MULTISEED-002): 0.4365 ± 0.0361
(range 0.394–0.482) ⇒ 28–34% of the gap; the single-run 0.5322 sits near
the upper end of the seed distribution. Both numbers are reported; neither
is hidden.

## F4 — On the noise ladder, gating collapses toward the globally dominant head. MEASURED

PH2B-NOISE-003: average weight on head 0 = 0.894, selection frequency
1.000, R@10 0.8489 ≈ global best single head 0.8378 ≈ oracle 0.8400.
INTERPRETATION: collapse is the CORRECT response when no query-conditional
structure exists; we did not force diversity to look multi-head (§7).

## F5 — Head diversity alone does not imply retrieval utility. MEASURED

The noise corpus's six noisy heads are diverse (pairwise Jaccard@10 well
below 1; see `results/noise-head-diversity.csv`) yet individually near-useless
(mean test recall 0.04–0.35; `results/noise-head-utility.csv`). The trained
model discounts them to ~0 weight. "Diverse" and "useful" are separate axes
(§22).

## F6 — Training-data quantity dominated the tested architecture changes. MEASURED

Multiview at 210 training queries: gating learned nothing (R@10 ≈ uniform;
corr −0.34, PH2B-MULTIVIEW-003/004). At 840: decisive preferences
(+28–32pp, PH2B-MULTIVIEW-005). The sweep (PH2B-SAMPLE-001) shows a PLATEAU
(0.24–0.26 R@10) for train sizes 105/210/420 and a transition at 840
(0.4928) — the boundary lies between 420 and 840. Meanwhile the
architecture grid (hidden 32→64, lr 0.01→0.003, batch 32→16) moved
validation R@10 by <2pp at fixed data. HYPOTHESIS: the bottleneck is
sample efficiency of a linear-softmax gate over a 192-dim concatenated
representation, not capacity.

## F7 — RRF was beaten by trained gating on all evaluated corpora. MEASURED

Test R@10, gating vs RRF k=60: controlled 0.9533 vs 0.6689 (+28.4pp);
noise 0.8489 vs 0.7800 (+6.9pp); multiview 0.5322 vs 0.2622 (+27.0pp)
(single-run protocol; multiview multiseed gating mean 0.4365 still exceeds
RRF by +17.4pp). Evidence: PH2B-GATING-004 / PH2B-NOISE-003 /
PH2B-MULTIVIEW-005 / PH2B-MULTISEED-002.

## F8 — Untrained mechanisms establish nothing about trained attention. MEASURED (the negative)

Phase 2's untrained gating and identity-initialized QK attention scored
identically to fixed fusion (0.672). This is evidence ONLY that the
untrained mechanisms provide no improvement — it must not be cited as
"attention does not help". Trained-QK evaluation remains open (Phase 2C;
`results/qk-attention.csv` intentionally empty).

## F9 — Exact reranking initially regressed; the regression is a weighting problem, not an exactness problem. MEASURED

Phase 2 mode E (equal-head exact rerank in the live pipeline): 0.558 vs
0.672 (mode D). Offline decomposition on cached candidates
(PH2C-RERANK-001/002/003): uniform EXACT fusion is fine on controlled
(0.6511 > 0.6244 norm-uniform) and slightly worse on noise (0.7133 <
0.7378) — and oracle-weighted exact fusion equals the oracle ranking
(0.9533 / 0.8400 / 0.9933). INTERPRETATION: exact similarity is not
intrinsically harmful; the regression comes from equal weighting across
heads of unequal quality inside the pipeline's exact mixture. Full analysis:
`reranking-findings.md`.

## F10 — Parallel multi-head execution produced a measured speedup while preserving semantics. MEASURED

Phase 2 (`results/heads-scaling.csv`): recall identical serial vs parallel
at 1/2/4/8 heads; p50 speedup 1.39× (4 heads), 1.72× (8 heads); plus unit
tests assert parallel==serial assembly. Trained-gating inference adds
0.87 µs/query (1188 params, 20 KB serialized, PH2B-LATENCY-001) — ~3 orders
of magnitude below HNSW search cost.
