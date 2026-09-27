# C5 — Cross-Head Candidate-Generation Research (protocol)

Study `comparative-study-001`, protocol v1.0.0. Branch
`comparative-study/c5-cross-head` cut from the C4 closure commit
`cc661615c854fb5f8c6e6a35b0287cdf1d2a80f7`. This document freezes the C5
hypotheses, ablation ladder, budget-matching rules, run schema, and statistical
plan **before** any experiment or tuning measurement. It supersedes nothing from
C1–C4; it re-uses the C1 statistics protocol verbatim (as C4 did).

## 1. Research question

AttentionDB's multi-head search performs one **independent** per-head ANN search
on the same query vector and then unions candidates + fuses scores
(see `c5-architecture.md` §2). The cross-head interaction that exists today
happens **after** candidate generation (union/fusion/attention = scoring only).
C5 asks the causal question:

> Does adding a genuine cross-head interaction step that CHANGES THE CANDIDATE
> SET (not just scores) improve retrieval, once total candidate budget, per-head
> budget, ef effort, union size, rerank budget, memory, and latency are equalized?

The unit of causal attribution is the candidate set, not the fused score. An
interaction arm must provably surface an id that the independent baseline would
not have surfaced, only then can a quality difference be attributed to
cross-head candidate generation.

## 2. Hypotheses (frozen)

- **H0**: after budget/latency/memory equalization, the cross-head interaction
  arm (C5-C) shows no meaningful improvement over the independent multi-head
  arm (C5-B) on either primary dataset. Meaningful = |mean paired diff| ≥ 0.01
  absolute recall@10 AND bootstrap CI excluding 0 (see §7).
- **H1**: cross-head interaction changes candidate generation and improves
  candidate-set oracle recall and/or final recall@10 (C5-C ≻ C5-B).
- **H2**: if H1 holds, the improvement concentrates on queries whose relevant
  docs are spread across multiple per-head candidate lists (multi-space-distributed
  queries), measured by a per-query overlap/entropy index defined in §6.

Decision rule (preregistered, no winner labels): report C5-C vs C5-B as
IMPROVED / COMPARABLE / WORSE per dataset on the paired statistics; H1 is
supported only if the primary contrast on BOTH candidate-set recall AND final
recall@10 crosses the meaningfulness thresholds. A neutral result (H0 not
rejected) is a valid, reported outcome.

## 3. Ablation ladder (frozen, strict subsumption)

| Arm | Candidate generation | Cross-head interaction | Rerank |
|-----|----------------------|------------------------|--------|
| C5-A | single-head (one index) | none | off (primary) |
| C5-B | independent multi-head, per-head search + union | **none** | off (primary); on in B' arm |
| C5-C | multi-head + ONE interaction step (round-2/injection from other heads' round-1 candidates) | **yes — set-changing** | off (primary); on in C' arm |
| C5-D | optional: C5-C iterated / adaptive-step variant | yes, multi-step | per arm |
| C5-E | learned interaction weights | candidate — only if C5-D shows signal | per arm |

C5-B == C4 B2 candidate generation semantics (independent per-head HNSW search,
overfetch 5x, min 20, union capped at candidate_budget.max(top_k), fixed or
identity fusion) and therefore C5-A..E reproduces the C4 B1→B2 transition at C5
scale. C5-D is **not run** unless C5-C shows a candidate-set-change signal on a
held-back proof-of-concept; C5-E is **not run** unless C5-D justifies it. The
frozen ladder requires C5-A, C5-B, C5-C (primary) on both datasets; C5-D/C5-E
are contingent.

## 4. Budget equalization (hard, frozen)

C5-B is the independent control. C5-C must spend the **same** per-query resources:

- total candidate budget (`candidate_budget`) identical across arms;
- per-head budget identical: every arm uses same heads (TITLE/BODY/CITE for the
  multi-embedding datasets per C4), same `candidate_multiplier` (5) and same
  `min_candidates_per_head` (20), so the round-1 forward width is identical;
- ef effort identical: C5 harness applies per-head ef explicitly (see §5;
  `update_settings` on each head index before querying, because the collection
  level `ef_search` setting is not consumed by the engine query path — audit
  finding). Same ef value for round-1 in every arm. C5-C's interaction step may
  **re-use** round-1 hits only; it may NOT raise ef or k. An additional round-2
  search is only allowed at equal or lower ef, consuming the same total ef
  budget as an equivalent additional width would require — implemented as a
  pre-check in the harness (`total_ef_work(C5-C) ≤ total_ef_work(C5-B)` for the
  same width);
- union size capped identically by `candidate_budget`;
- rerank: rerank shared exact-affine or HNSW-affine path; primary arms rerank OFF
  (so the B2-vs-B7 lesson is not re-confounded); a secondary B' vs C' cell runs
  the SAME rerank on both to guard interaction×rerank interactions;
- memory: both arms use same loaded indexes and parallel search threads;
- latency: primary comparison is per-query p50 latency with the C1 protocol
  (5 fresh-process reps, warmup 20, seeded paired order); a **same-latency
  secondary table** reports C5-C at its natural latency vs C5-B at the latency
  to which C5-B can be tuned to match (tuning on validation only).

## 5. Harness contract (frozen)

`c5pilot` is an additive probe crate under `research/comparative-study/c5-cross-head/`
(rooted workspace, not in CI scope — mirrors C2/C3/C4 pilots) that drives
`AttentionEngine` + `Collection::attend_detailed_with_stats` and:

- registers every run pre-execution as `C5-…` in append-only `RUN-INDEX.yaml`;
- applies per-head ef via `idx.update_settings(...)` so ef is a REAL knob
  (audit finding §2), then verifies ef sensitivity on a tiny probe cell
  (C5-EFPROBE) before real runs;
- implements C5-A/C5-B (no interaction) and C5-C (interaction) deterministically
  and identical to C5-B except for the single interaction step;
- **C5-C primary mechanism (frozen): one-step cross-head query refinement
  ("retrieve → interact → retrieve").** Round 1 = per-head HNSW search (same
  query vector into each head index, exactly as C5-B, but round-1 search runs at
  ef/2 so that round-1 + round-2 total ef effort == C5-B's single ef; C5-B runs
  at full ef). The interaction step computes, for each head H, the
  "cross-head surprise" set `S_H` = ids returned by OTHER heads in round 1 that
  H itself did NOT return, and builds a refined query for H:
  `q'_H = normalize(q_H + λ · mean_{d∈S_H} v_H(d))` (v_H(d) = H-space vector via
  `idx.get_vector(d)`; λ ∈ {0.10, 0.25, 0.50} tuned on VALIDATION only; λ=0.0 is
  the sanity arm that must reproduce C5-B round-2 behavior). Round 2 = per-head
  HNSW search with `q'_H` at ef/2. Each head's round-1 ∪ round-2 list is then
  capped deterministically back to `per_head_k` by best raw score (so per-head
  width and union budget are identical to C5-B). Guard rules: q'_H must be
  normalized; empty S_H ⇒ no refinement (round 2 == round 1 query, pure
  re-search, still budget-documented); centroid computed over a bounded
  canonical order of S_H sorted by (id ASC) so the float sum is order-independent.
  The ONLY difference between C5-C and C5-B is the two-round split with the
  cross-head-conditional round-2 query — scoring method (HNSW-approximate)
  stays identical across arms, so no scoring confound is introduced;
- C5-C switching decision (frozen): if the VALIDATION evidence shows the
  refinement is inert (C5-C glue == C5-B glue at λ=0.0, or λ>0 never changes
  the candidate set on >99% of validation queries), the runner must SWITCH the
  single interaction step to the contingent variant 2 (cross-head candidate
  propagation: promote S_H members into H's list when their H-space exact score
  clears the head's own round-1 min score) and re-tune on VALIDATION. Only the
  variant that provably changes the candidate set on validation proceeds to the
  frozen TEST cells; this is a preregistered VALIDATION-gate, not test-tuning.
- emits the per-query causal ledger (§6) for every interaction cell;
- reproduces the C4 fidelity anchors: B0 exact oracle in-process;
  `recall10_exact` vs canonical brute-force top-10; exact-top10 hashes;
- guardrails: sampler abort, 85% MemAvailable, disk floor, per-run timeout,
  PH3E envs unset — as C4.

## 6. Per-query causal ledger (frozen)

For every C5-C query, record: round-1 per-head candidate ids+scores+ranks;
interaction-output ids with source-head and tag `own|cross`; union size
pre/post; **additions** (ids present post, absent pre), **removals** (if any
replacement policy), overlap across heads; final top-k ids+scores; candidate-set
oracle recall (union vs exact oracle top-50 reference) and final recall@10 vs
qrels and vs oracle; per-head effort (k used, ef used); p50 latency; total
union/rerank counts; configuration_id. From these rows derive the per-query
cross-head contribution index `chi(q) = (#additions with `cross` tag that are
in the oracle reference) / union_size_pre` and the multi-space spread index
`s(q) = 1 − (max_h |cands_h ∩ oracle| / |union_oracle ∩ oracle|)`; H2 analysis
uses s(q) and chi(q) splits on VALIDATION for hypothesis-neutral exploration and
on TEST only as a preregistered secondary contrast (no TEST exploration).

## 7. Statistics (preregistered, frozen — re-uses C1 verbatim)

- Unit = query; per-query recall = mean over 5 fresh-process reps (C4 lesson:
  intranun determinism yes, cross-process no; mean-of-5 is the stable estimator).
- Paired bootstrap 10,000 (seed 20260925) 95% CI on the mean paired difference;
- two-sided Wilcoxon signed-rank; Holm step-down within the primary family
  (C5-A↔C5-B, C5-B↔C5-C on SCI; same on NFC — 4 primary contrasts);
- Cohen's dz; |diff| < 0.01 absolute = negligible regardless of p;
- power note for n.s. cells: min detectable mean recall diff reported, C4
  pattern;
- primary family on TEST only; tuning exclusively on VALIDATION (LODO/W01 split
  semantics as C2/C3/C4);
- no winner labels, no composite scores, no post-hoc exclusion; failed runs
  preserved with new-ID reruns.

## 8. Run schema & registration (frozen)

Every C5 run: unique `C5-…` ID in `RUN-INDEX.yaml`, `environment.yaml`,
`config.yaml`, `manifest.yaml`, `status.txt`, `metrics.json`, per-query JSON
rows (with §6 ledger for interaction cells), exact-oracle reference, hashes.
Status values same taxonomy as C4 (PASS / FAILED / FAILED-HARNESS / ABORTED /
INVALID-STARTUP / BLOCKED-* / SUPERSEDED). Runs are append-only; reruns get new
IDs; raw evidence immutable.

## 9. Stop conditions (frozen — any → halt, preserve, report)

anchor mismatch (branch/tree/C4 hashes); dataset/embedding hash mismatch;
exact-oracle discrepancy; unpreregistered workload/config; harness silently
changing config or metrics; ef-sensitivity probe failing (ef not a real knob);
budget-leak detected in C5-C (total_ef_work or width exceeded); raw evidence
overwritten; guardrail abort; 3 consecutive harness crashes on same config;
silent-wrong-result indicator. A blocked scope isolates only that scope. C5
never auto-promotes beyond the frozen plan; scope extension requires user
amendment.

## 10. Deliverables (frozen list)

`c5-protocol.md` (this), `c5-architecture.md`, `c5-environment.md`,
`c5-dataset-validation.md`, `c5-resource-budget.yaml`, `c5-run-plan.csv`
(frozen), `c5-implementation-audit.md`, `c5-candidate-analysis.md`,
`c5-results.md`, `c5-statistical-analysis.md`, `c5-resource-analysis.md`,
`c5-failure-register.md`, `c5-reproducibility.md`, `c5-closure-decision.md`,
`harness/`, `raw/`, plus a `research/comparative-study/README.md`
index/changelog update. Reports are produced only from immutable raw rows.