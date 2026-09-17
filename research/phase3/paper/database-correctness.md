# Database Correctness Validation (Phase 3D)

## Claim
AttentionDB behaves as a correct single-node document database for the operation families
tested: a deterministic mixed workload of inserts, updates, upserts, deletes, queries and
metadata filters — interleaved with checkpoints, offline compactions and restarts — leaves
logical state exactly equal to a reference model, never returns dead documents, never leaks
documents across collections, and never returns a document excluded by a filter.

## Method
Model-based testing (§33): a reference `Map<LogicalID,(version,cat,num)>` driven by a seeded
PRNG (3 × 1000-op runs, seeds 42/7/1, plus a 100-op anchor), ≈30% mutation ops, ≈27%
query/filter ops, ≈8% checkpoint/compact/restart; mandatory engine+directory consistency
gates at build, after every compaction, and at the end. Filters use a soundness oracle
(every returned doc matches; zero-match ⇒ empty) plus determinism; completeness is reported
as a measured recall, not asserted.

## Results
0 failed checks out of 24 gated checks across the model suite; state equality after every
restart/compact gate (925–936 logged ops per run); 0 filter leaks in ~380 probes; measured
filter recall 0.967–1.0 on 60–120-doc eligibility sets (candidate-bound by design — the
frozen retrieval architecture's ANN candidate pool is the recall limiter, consistent with
Phase 3C's candidate-recall measurements).

## Relation to retrieval validation
Phase 3B/3C validated that the frozen retrieval architecture (per-head candidates → union →
trained gating → fusion → top-K) retrieves well and bounded its costs. Phase 3D validates
that the same engine also *stores, mutates and recovers* state correctly. The two are
complementary: retrieval quality claims assume a correct substrate; this phase establishes
the substrate properties measured here.

## Limitations
Small corpus (≤ 434 live docs), one collection pair, DIM 32, one head; no multi-client
driver; no isolation/ordering claims (see systems-limitations).
