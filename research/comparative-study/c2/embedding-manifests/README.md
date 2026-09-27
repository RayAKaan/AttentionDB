# Embedding Manifests — pinned MiniLM vectors (C2)

Study `comparative-study-001`, protocol v1.0.0.
Each manifest records the embedding export produced by the named run (sha256 of
each `.npy` in the run's `metrics.json`/"manifest" block). All vectors are
cosine metric, float32, dim 384, model `sentence-transformers/all-MiniLM-L6-v2`
revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

## EMB-SCIFACT — [C2-EMBED-SCIFACT-003 PASS]

- counts: HEAD-TITLE 5183, HEAD-BODY 5183, HEAD-CITE 5183, CANONICAL 5183, QUERIES 1109
- encode_seconds: HEAD-TITLE 33.31, HEAD-BODY 335.39, HEAD-CITE 105.8,
  CANONICAL 328.77, QUERIES 7.03
- head_view_definitions:
  - HEAD-TITLE = embed(title)
  - HEAD-BODY = embed(text)
  - HEAD-CITE = embed(first + final sentence of text; query-independent, fixed before any run)
  - CANONICAL = embed(title + ' ' + text); the single vector every external
    single-vector system receives
  - QUERIES = embed(query text)
- determinism_full_view_reencode_byte_equal: true; batch1_bitwise_differs: false
- export hashes: HEAD-TITLE `77084a4c…`, HEAD-BODY `7c8e4609…`, HEAD-CITE
  `a0960407…`, CANONICAL `94ae4fd9…`, QUERIES `b83301c1…`
  (full sha256 in C2-EMBED-SCIFACT-003 metrics.json)
- source dataset hash: `dec31c81…` (DS-SCIFACT)

## EMB-NFCORPUS — [C2-EMBED-NFCORPUS-003 PASS]

- same model/revision/dim; counts + per-view hashes in the run's metrics.json
- artifacts sha256-recorded and survived the environment reset (per
  environment-reset-2026-09-24.md)

## Determinism policy (applies to both)

- Exports pin batch=32 + order; regeneration uses the same script
  (`embed_minilm.py`).
- A single-text (batch-1) re-encode bitwise-differs from the batch-32 export
  only by padding effects; this is documented, not latent nondeterminism.
- Any re-encode must be byte-verified against the manifest per-view sha256
  (owed task from the reset; determinism was demonstrated in-run).