# C2 Finding & Disposition: BM25 tied-score ordering nondeterminism

Audit: comparative-study C2 (protocol v1.0.0), branch `comparative-study/c2-validation`.

## 1. Finding ID
**BM25-TIE-ORDER-001** — confirmed engine-core defect.

- Component: `attentiondb-core` — `bm25_index_search_v2` (`core/src/bm25.rs`, `search`, `search_phrase`, and
  `reciprocal_rank_fusion`).
- Behavior: scores for tied documents were sorted *by score only* (`b.1.partial_cmp(&a.1)` with no doc-id
  tiebreaker) over a per-call `HashMap<u64, f32>` whose iteration order is not stable, so the **top-k doc-id
  order was nondeterministic per call** under strict score ties.
- Downstream: deterministic tie rule is part of the preregistered contract for BM25 search and Reciprocal
  Rank Fusion (B5); a nondeterministic ordering propagates into fused scores and fused top-k membership.

## 2. Evidence (pre-fix)
Run **`C2-BM25-REPRO-001`** — registered **FAILED** (intentional defect capture).
- Stimulus: 12 identical docs (identical BM25 scores = strict ties), 200 samples, top_k 4.
- `search_distinct_orderings_seen` = **198** / 200 (2 searches coalesced)
- `phrase_search_distinct_orderings_seen` = **197** / 200
- `rrf_distinct_fused_topk_memberships` = **153** / 200
- `rrf_distinct_fused_max_score_ids` = all 12 document ids (no stable winner)
- `strict_score_ties_observed` = true

## 3. Fix
Committed as **`7cbd16ddd62b671eaefccf4db44bf464164d79ca`** on branch `comparative-study/c2-validation`.
Diff (`git diff 7788067..7cbd16d -- core/src/bm25.rs`): append `.then_with(|| a.0.cmp(&b.0))` at all three
sort sites (score desc, id asc) — matches the preregistered tie rule.

Regression pin added: `core/tests/regression_bm25_tie_order.rs` (search / search_phrase / RRF; expected
RRF top-4 `[0, 11, 1, 10]`).
Result: **3/3 PASS** (`cargo test -p attentiondb-core --test regression_bm25_tie_order`).

## 4. Evidence (post-fix)
Run **`C2-BM25-REPRO-002`** — registered **PASS** (identical stimulus).
- `search/ phrase / rrf_distinct_fused_topk_memberships` = **1** each; `rrf_distinct_fused_max_score_ids` = **[11]**.

Run **`C2-MODES-TEST-002`** — registered **PASS** (8/8 battery runs; artifacts `modes-test-r1..r8.json`).
- TEST-C2-008 (B5, harness assertion corrected to the token-doc-set contract after the pre-v2 target was
  found vacuous — `xyloquery` appears in 8 docs): top-1 among token-doc set asserted, deterministic 5 calls.
- TEST-C2-003 / TEST-C2-009 re-scoped (see §6): determinism asserted unconditionally; exact-equality to the
  exhaustive reference asserted only when candidate pools cover all docs; pool coverage recorded.

## 5. Disposition
**FIXED — VERIFIED.** The preregistered deterministic tie rule (score desc, id asc) is enforced at all three
BM25 sort sites, pinned by a regression test, and independently reproduced PASS/FAIL on identical stimuli
(pre-fix `C2-BM25-REPRO-001` vs post-fix `C2-BM25-REPRO-002`). No other engine behavior was modified.

## 6. Related recorded observation (not modified by this fix)
**HNSW-RECALL-OBS-001** — `hnsw_rs` search at *k == element count* is **not exhaustive**: 0..7 docs per head
can be unreachable for a given query, and *which* docs varies across processes (missing sets observed include
`[25]`, `[27]`, `[35]`, `[30]`, `[37, 40]`, `[33, 34, 35, 37, 38, 39, 40]`). This pre-dates the BM25 fix and
is independent of it. It makes an unconditional "k == corpus ⇒ candidate pool is exact" harness assumption
undecidable, which is why TEST-C2-003/-009 assert exact-equality conditionally and record `coverage_min_per_head`.
Evidence artifact: `C2-MODES-TEST-002/artifacts/hrecall-recall-observations.txt`. Mode A (single-head ANN)
already records approximate agreement by design (`TEST-C2-002 exact_set_agreement_rate` recorded, not asserted).

## 7. git_commit evidence-integrity note
The three post-fix runs were registered while `HEAD` was still the pre-fix commit `7788067`; their
`config.yaml`/`manifest.yaml` `git_commit` fields truthfully record HEAD-at-registration. Per the
no-overwrite rule on run dirs these were **not** rewritten. The fix commit `7cbd16d` is anchored here and in
the closure decision; future runs on this branch record the post-fix HEAD.

## 8. Files touched
- `core/src/bm25.rs` (fix; committed)
- `core/tests/regression_bm25_tie_order.rs` (new regression pin; committed)
- `research/comparative-study/c2/probe/main.rs` (harness: TEST-C2-008 v2 assertion; TEST-C2-003/-009
  re-scope; `hrecall` subcommand; committed)
- `research/comparative-study/c2/probe/Cargo.toml`, `bm25repro.rs` (BM25 repro + `attentiondb-hnsw` dep; committed)
- staged repro inputs preserved under `c2/_staging/bm25-repro-001-staging`, `-002-staging`.