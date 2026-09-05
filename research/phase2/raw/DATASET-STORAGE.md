# Dataset storage policy (2026-09-05)

The workspace snapshot budget (~128 MB) cannot hold the uncompressed cached
datasets (~118 MB of JSON). All large `dataset.json` / `qk_content.json`
files are stored **gzip-compressed (`.json.gz`)**.

- **Lossless**: pre-compression sha256 of every file is recorded in
  `datasets-manifest.sha256`; each compressed file was verified to
  decompress byte-identically before the original was removed.
- **Restore**: `gunzip -k <path>.json.gz` (keeps the .gz; writes the .json),
  then optionally verify: `sha256sum <path>.json` against the manifest.
- **Regeneration is NOT a substitute**: hnsw layer RNG is OS-seeded, so a
  regenerated dataset is not byte-identical (documented eval deltas up to
  ~0.04 R@10). The compressed caches are the reproducibility units.
- Re-run `record_run.py`-produced flows only after restoring the datasets
  they read. All results/registry CSVs are plain text and unaffected.

Compressed files (see also run-manifest.json `storage_policy`):
benchmarks/phase2b/{controlled,noise,multiview}/dataset.json.gz;
PH2B-SAMPLE-001/q{150,300,600}/dataset.json.gz;
PH2B-MULTISEED-001/dataset.json.gz;
PH2C-QK-001/{dataset,qk_content}.json.gz
