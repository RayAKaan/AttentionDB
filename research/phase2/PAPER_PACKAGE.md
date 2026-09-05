# PAPER_PACKAGE.md — reconstruction manifest (§30)

State: **live document** — regenerated whenever experiments complete.
Phase 2B (gating) is complete; Phase 2C (trained QK, pipeline rerank
re-weighting) is pending and marked as such. A researcher should be able
to reconstruct every quoted result from this directory alone.

## How to reconstruct a result

1. Pick the experiment in `raw/experiment-index.json` (ID → corpus,
   dataset hash, commit, config, metrics, outputs).
2. Find its run directory `raw/runs/<ID>/` (config.json, metrics files,
   CSVs). Historical/invalidated runs keep their status and numbers.
3. Tables: `python3 research/phase2/tables/generate_tables.py` — reads
   `results/*.csv` only.
4. Figures: `python3 research/phase2/figures/generate_figures.py` — reads
   `results/*.csv` + per-corpus dataset/model JSONs only.
5. Verify: `python3 research/phase2/verify_consistency.py` — must PASS
   (tables ↔ raw ↔ registry; oracle-sanity invariants).
6. Determinism: given a `dataset.json` + code commit, training and
   evaluation are bit-reproducible (unit-tested); HNSW candidate
   generation is NOT cross-process reproducible (OS-seeded RNG), hence
   the dataset is the reproducibility unit (`methodology/reproducibility.md`).

## Contents inventory

| category | items |
|---|---|
| methodology | 5 docs in `methodology/` (protocol, datasets, ground truth, splits, reproducibility) |
| datasets | 3 corpora (controlled / noise / multiview) + per-run dataset.json with hashes |
| raw results | `results/*.csv` (28 files incl. per-corpus splits) + `raw/runs/` (10 run dirs) |
| registry | `raw/experiment-index.json` (23 experiments: 13 historical incl. 5 INVALIDATED, 10 on-disk) |
| tables | 6 generated + generator script |
| figures | 9 generated (fig10 PENDING) + generator script + captions |
| findings | F1–F10, claim ledger (SUPPORTED/NOT SUPPORTED/OPEN), negative results, harness corrections HC-1..4, failure analysis, per-topic deep dives |
| paper sections | methodology, results, discussion, limitations (modular; final assembly deliberately deferred per §31 of the Phase 2B spec) |
| limitations | `paper/phase2-limitations.md` |
| final report | `phase2b-final-report.md` (§41 structure, fixed-vocabulary verdicts) |

## Pending before the paper is assembled

- PH2C-QK-*: trained QK attention (figure 10, `results/qk-attention.csv`).
- PH2C-RERANK-004+: pipeline-level rerank re-weighting (ledger N3/Q3).
- Optional: RRF k sweep (C3 limitation), split replication, scale runs.
- Literature review replacing `[CITE: ...]` placeholders (none invented).

## Citation placeholders used

[CITE: HNSW] [CITE: RRF] [CITE: attention] [CITE: learned-to-rank]
(InfoNCE not yet needed — contrastive objectives unused in 2B.)

## PH2C-QK-001 (QK sanity) — reproduction

```
cargo build --release -p phase2b-bench
./target/release/phase2b-bench qk-sanity \
  --out research/phase2/raw/runs/PH2C-QK-001 --seeds 42,7,1
python3 research/phase2/scripts/record_run.py   # registry/manifest/results copy
python3 research/phase2/tables/generate_tables.py
python3 research/phase2/verify_consistency.py
```

- Implementation: `benchmarks/phase2b/src/qk_sanity.rs` (deterministic;
  dataset seed 0x5A17; reruns are byte-identical).
- Construction, impossibility argument, arms, and gate:
  `research/phase2/methodology/qk-sanity-dataset.md`.
- Findings: `research/phase2/findings/qk-attention-findings.md` (QK-SANITY).

## Dataset storage note

Cached `dataset.json` files are stored gzip-compressed (`.json.gz`) to fit
the workspace snapshot budget. Before any harness run that reads a dataset:

```
gunzip -k benchmarks/phase2b/<corpus>/dataset.json.gz
# verify against research/phase2/raw/datasets-manifest.sha256
```

See `research/phase2/raw/DATASET-STORAGE.md`.

