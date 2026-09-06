# Phase 3B datasets — AG News multi-field text retrieval (PH3B-DS-AG)

Prose mirror of `datasets/build_agnews.py` (keep in sync).

## Source & license
- AG News corpus, Xiang Zhang's CharCNN distribution (120,000 train /
  7,600 test; columns class_index 1–4, title, description). Mirrors:
  mhjabreel/CharCnn_Keras (primary), AaronCCWong/Char-CNN (fallback).
  License: freely available for research (Zhang et al. 2017; original AG
  corpus, Gulli 2004). Downloaded 2026-09-06; source-file sha256 recorded
  in every tier's meta.json.

## Preprocessing
- Text cleaning: literal backslash escapes (`\n`, `\r`, `\t`, `\\`, `\"`)
  → space; whitespace collapsed. (Artifacts of the original CSV release.)
- Vector tokenizer: lowercase, `[a-z0-9]+` tokens, NO stopwords, NO
  stemming. Embedding: head-salted FNV-1a unigram hashing → dim buckets,
  tf, l2-normalized; the salt is the head name, so each field view has its
  own dictionary (per-head query vectors differ, as in fielded multi-
  encoder search).
- Heads: `title` = title text; `body` = description; `full` = title +
  description. Engine collection dim uniform (512 standard; 256 for the
  M20D256 memory-constrained probe).
- Category/metadata view NOT included: 4 classes cannot justify a metadata
  embedding view (degenerate/dominated head). Recorded per spec §1
  "where justified".

## Query protocol (§2)
- Queries are held-out TEST rows, one field at a time:
  - title-type: query text = the doc's TITLE only;
  - body-type: the doc's DESCRIPTION only;
  - mixed-type: TITLE + " " + DESCRIPTION.
- Each query is embedded in ALL THREE head spaces (same tokens, per-head
  salt) — matching a real system where one user query hits several field
  indexes. The full-document representation is never privileged: no query
  carries more than its own field text.
- Statistical verification of differentiated utility BEFORE training
  (§2, in meta.json `differentiated_utility_exact_R@10`): for every type,
  the defining head's exact R@10 = 1.0 and the next-best head is ≤ 0.36
  (S tier) — no single view dominates every query; the complementarity
  precondition holds by a wide margin (asserted at build time).

## Tiers
| id | docs | queries/type | dim | note |
|---|---|---|---|---|
| S | 10,000 | 150 raw | 512 | primary |
| M20 | 20,000 | 200 raw | 512 | build OOM on 2 GB sandbox (preserved) |
| M | 30,000 | 200 raw | 512 | build OOM on 2 GB sandbox (preserved) |
| M20D256 | 20,000 | 200 raw | 256 | memory-constrained probe; completed |

## Ground truth (§3)
- Definition: per query type t, the EXACT cosine top-10 over the corpus in
  the type's DEFINING head space (title→title, body→body, mixed→full).
  Rationale: relevance = same-field semantic neighborhood; the oracle head
  is known by construction while the gate must INFER it from per-head
  query vectors (no metadata leakage — `query_group` is dataset metadata
  used only for by-type reporting and oracle baselines, never a model
  input).
- Determinism: cosines rounded to 1e-4 (round-ties-even in BOTH numpy and
  rust — HC-P3-4) so independent implementations produce identical
  orderings; ties break by lower doc index; boundary-tie events counted
  (recorded in meta.json).
- Independent validation: argpartition-vs-full-argsort check caught a real
  nondeterminism (HC-P3-4) before registration; after the fix, 30
  queries/type re-ranked by full argsort match the chunked path exactly.
- Leakage: corpus = train.csv rows, queries = test.csv rows; AG News
  contains cross-split duplicate stories → any query whose title OR
  description md5 appears in the corpus is EXCLUDED (recorded per type:
  S tier excluded 11/17/21 → 401 queries kept; overlap re-checked = 0).
- No engine-ID mismatches: GT stored as doc indices; engine-id conversion
  happens once through the same mapper used by all arms; the exact
  anchor (R@10 = 1.0000) asserts cross-space consistency at every run.

## Known limitations
1. Relevance is defined by the benchmark's own embedding spaces — legitimate,
   deterministic, and fully documented, but not human judgment; lexical
   (BM25) systems are structurally disadvantaged on it (reported honestly,
   C-3).
2. AG News fields are short news text; long-document field behavior is
   untested.
3. 512-dim medium tiers are memory-bound on this sandbox (§14 preserved
   runs); the dim-256 probe changes the collision regime — cross-tier
   comparisons are directional, not exact.
