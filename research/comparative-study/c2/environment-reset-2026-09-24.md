# Environment reset 2026-09-24 — loss inventory & recovery procedure

The workspace snapshot is best-effort capped (~128 MB / ~10,000 files). The session of
2026-09-24 ended far over budget (4.3 GB / 70k files, mostly dataset downloads + Rust
build artifacts), so the snapshot kept only a prefix subset. This file records exactly
what was lost, what survives, and how to restore. **No run evidence was fabricated or
reconstructed — losses are documented, not papered over.**

## Survived (verified present after reset)

- Full git history (`.git`, 65 MB pack; phase2/phase3 raw runs are tracked and recoverable).
- All `raw/<run_id>/` dirs for every TERMINAL C2 run (manifests, logs, metrics, sampler
  traces, artifacts) — including all smoke runs C2-SMOKE-{QDRANT-006, PGVECTOR-005,
  ES-003, MILVUSLITE-002, WEAVIATE-004} and all B3/embedding/gate/oracle runs.
- `c2/` tree: harness scripts, modelcards (`-v2` set with head_names + TrainingMeta),
  training-config.yaml, READMEs, smoke registry + results.
- NFCorpus embedding exports (C2-EMBED-NFCORPUS-003 artifacts, sha256-recorded).
- BEIR scifact/nfcorpus processed corpora + hf-qrels under `raw/datasets/`.
- Non-terminal leftovers annotated, never overwritten: C2-SMOKE-QDRANT-002 (no manifest —
  harness defect, in RUN-INDEX), C2-DATA-GLOVE25-004 (reset killed verifier mid-run;
  INVALID-STARTUP; successor will be -005).

## Lost (must re-procure; URLs/sizes live in the sources cited)

| Item | Size | Re-procurement |
|---|---|---|
| `raw/datasets/downloads/*` (scifact/nfcorpus/fiqa/scidocs zips, glove-25/50 hdf5, COCO ann + val2017, esci parquets ×2, bin/{qdrant,ES,weaviate}) | ~2.4 GB | re-download from the sources recorded in `raw/datasets/README.md` and the harness scripts (`gates_conditional.py`, `embed_minilm.py`, `ann_materialize.py`); byte sizes in session dataset manifests; **sha256 each artifact at re-download time** (the owed sha256-all task now happens then, per artifact) |
| SciFact embedding exports (C2-EMBED-SCIFACT-003 artifacts: 4 views + queries, ~33 MB) | 33 MB | deterministic re-encode via `embed_minilm.py` (pinned MiniLM revision `1110a243…`, batch 32); **byte-identity vs the manifest's per-view sha256 must be re-verified after re-encode** (determinism was demonstrated in-run) |
| Rust toolchain + protoc (`/var/tmp/toolchain`, `/var/tmp/protoc`) | — | reinstall to /var/tmp before any probe rebuild; PROTOC env var as before |
| `/var/tmp` scratch (extracted servers, wv-diag) | — | ephemeral by design; smokes are terminal, no rerun needed |

## Working-tree policy (this reset's cleanup)

- `research/phase2` and `research/phase3` working trees are now **sparse-checkout-excluded**
  (git-tracked; recoverable anytime with:
  `git sparse-checkout set --no-cone '/*' '!/research' '/research/*'` then reapply, or
  `git sparse-checkout disable`). History untouched; `git status` stays clean.
- Residual `research/phase3/raw/*/stdout.log` (~434 KB) intentionally kept — tracked, tiny.
- Budget after cleanup: **127 MB / 998 files** (caps: ~128 MB / 10,000).

## Standing rules reaffirmed

- Downloads are re-fetchable; **raw run dirs are evidence and are never deleted/rewritten**.
- Any re-downloaded artifact must be sha256-verified against its recorded byte size/origin
  before use, and the sha256 recorded in `dataset-manifests/` at that time.
- Keep `/home/user` under budget: large fetches should live outside the snapshot scope
  (e.g. `/var/tmp`) or be re-downloaded per session; never stage GB-scale data under
  `raw/datasets/downloads/` at end of turn while over budget.
