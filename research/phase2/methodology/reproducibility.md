# Reproducibility (§24)

## What is deterministic

Given (dataset.json, code commit, configuration):

- Model init: seeded xorshift64* (no thread_rng anywhere in `gating_v2`).
- Epoch order: seeded Fisher–Yates.
- Optimizer: Adam, single-threaded, fixed operation order.
- Temperature selection, evaluation, metrics: pure functions of cached data.
- Bit-for-bit repeatability is TESTED: `gating_v2_test::
  training_is_deterministic_bit_for_bit` asserts identical weights across
  two identical runs.

## What is NOT deterministic across processes

HNSW candidate generation: `hnsw_rs` seeds layer assignment from OS
entropy, so two runs on the same corpus seed produce slightly different
graphs and candidate pools (same corpora produced eval deltas up to ~0.04
R@10 across fresh regenerations, e.g. multiview 840-train point 0.4928 vs
0.5322 in its original run). Therefore:

**The cached dataset is the reproducibility unit.** Every experiment ID
pins its `dataset.json` (hash in the registry). Re-running candidate
generation produces a NEW run ID with its own hash — never an overwrite
(spec §3).

## Recorded per run (§24 checklist)

seed · dataset hash · corpus configuration · code commit · model
configuration · optimizer/lr/batch/epochs · hardware (cpu count) · unix
timestamp — all present in each run's `run_info.txt` / `config.json` and
aggregated in `raw/experiment-index.json`.

## Environment

rustc 1.98.0, single binary `phase2b-bench`, no GPU, no external ML
runtime (the `tch` feature of `attentiondb-learned` is compiled OFF).
Hardware is a shared CI-class sandbox; absolute times are machine-specific.
