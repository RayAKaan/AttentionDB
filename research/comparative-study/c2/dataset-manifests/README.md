# Dataset Manifests — C2 materialized datasets

Study `comparative-study-001`, protocol v1.0.0.
Each manifest records the sourced dataset as a C2 run materialized + verified
it into `../raw/datasets/`, with sha256 + provenance. Source: the specific
`C2-DATA-*` run's `metrics.json` (immutable evidence).

## DS-SCIFACT — [C2-DATA-SCIFACT-007 PASS]

- corpus_docs: 5183; queries.jsonl rows: 1109 (all split queries; C1 'queries'
  count refers to TEST qrels queries)
- qrels_rows: test 339, train 919 (hf-BeIR-qrels official, positives-only)
- corpus_fields: `_id`, `metadata`, `text`, `title`
- graded_levels: [1] — BEIR qrels are positives-only; SciFact ships score=1 only
  (binary). C1 recorded "graded 0/1/2" → recorded as a **C1-vs-data discrepancy**
  (vocabulary vs shipped levels); acceptance rule (defensible labels) unaffected.
- qrel_test_queries: 300; dangling qrel query/doc ids: 0; counts match C1: true
- Source manifest hash (embeddings): `dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6`

## DS-NFCORPUS — [C2-DATA-NFCORPUS-003 PASS]

- corpus_docs: 3633; queries.jsonl rows: 3237
- qrels_rows: dev 11385, test 12334, train 110575 (hf-BeIR-qrels official,
  positives-only)
- corpus_fields: `_id`, `metadata`, `text`, `title`
- graded_levels: [1, 2] (NFCorpus ships graded 1 and 2)
- qrel_test_queries: 323; dangling qrel query/doc ids: 0; counts match C1: true
- Note: NFCorpus TEST queries are used as B3 gating TRAINING inputs (LODO arm,
  disclosed — C1 LODO rule; SciFact untouched).

## DS-ANN-GLOVE-25 — [C2-DATA-GLOVE25-003 PASS]

- upstream: `https://ann-benchmarks.com/glove-25-angular.hdf5` (ann-benchmarks
  official pre-split HDF5)
- file_sha256 `51004cb0ae962159f0db507a51fec2b395de14b166f55976c89f16bd2f8b6391`,
  127,359,688 bytes; train (1,183,514 × 25) float32; test (10,000 × 25)
- neighbors/distances (10,000 × 100) shipped; dim 25 matches id
- NN-GT recompute: full exact 10,000 queries vs shipped neighbors, set equality;
  top100_set_mismatches 0, exact_boundary_tie_swaps 2, match_rate 1.0
- split: tune 2000 / test 8000, seed 20260925
- track_label: NN-GROUND-TRUTH / EFFICIENCY TRACK (never semantic)

## DS-ANN-GLOVE-50 — [C2-DATA-GLOVE50-003 PASS]

- upstream: `https://ann-benchmarks.com/glove-50-angular.hdf5` (same family)
- file_sha256 `388b0aedc2dad689549e6587932c8c9efeaf8a95f383f36dc567a40233a11f40`,
  246,711,088 bytes; train (1,183,514 × 50) float32; test (10,000 × 50)
- neighbors/distances (10,000 × 100); dim 50 matches id
- NN-GT recompute: full exact 10,000 queries, set equality; top100_set_mismatches 0,
  exact_boundary_tie_swaps 1, match_rate 1.0
- split: tune 2000 / test 8000, seed 20260925

## Gating/derived datasets (B3, exact per-head targets)

- `gating-synth-concat.json` sha256 `317963d44f39d2f320bd5d377139ec9e6c7033efb2fa1f058f4f344580f1bd3d`
- `gating-synth-view0.json` sha256 `4bcff6b5bb7081e6d1404c942ca91a96f8a5a19fd8f76815e78abbde620b8a7f`
- `gating-lodo-nfcorpus.json` sha256 `a58bb437456d16e78c1f9592192d5918a512416c79a594c58b4c1648dcab6ef5`
- Sources: exact (B0) per-view recall targets computed only over the exported
  training corpora (INV-L3); LODO arm built from NFCorpus TEST queries
  (disclosed reuse, per preregistration).

## Re-procurement / integrity (see environment-reset-2026-09-24.md)

Raw download files are re-fetchable from the recorded upstreams; any
re-download must be sha256-verified against these records before use and the
hash recorded here.