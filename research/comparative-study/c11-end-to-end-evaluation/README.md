# C11 tooling

This directory adds the evidence layer for C11. It consumes existing C8 TEST raw runs without modifying them and optionally imports C10 JSONL reports as a distinct microbenchmark track.

## Analyze existing raw runs

From the repository root:

    python3 research/comparative-study/c11-end-to-end-evaluation/c11_analyze.py --raw-root research/comparative-study/raw --output-dir research/comparative-study/c11-end-to-end-evaluation/results

Add `--c10-jsonl PATH` to include a completed C10 sweep. Add `--require-complete` to fail when any required dataset/arm cell or validation gate is missing. The outputs are `c11_results.json`, `c11_summary.csv`, and `C11_REPORT.md`.

## Required execution

Run the frozen C8 TEST plan with the existing C8 probe/harness before analyzing. The run plan here summarizes the C11 matrix; it does not launch or replace the C8 executor. The executor's per-arm JSON artifacts must remain present under each raw run's `artifacts/` directory for paired query-level analysis.

Required TEST matrix: 2 datasets × 9 arms (A–I), 5 fresh-process repetitions per cell. Frozen settings: k=10, ef_search=64, candidate budget=500, 20 warm-up queries; SciFact 300 TEST qrels and NFCorpus 323 TEST qrels. C8's exact configuration and model-training protocol remain authoritative.

## Evidence semantics

- C11 result generation is not evidence that the full experiment ran. `--require-complete` is the paper-evidence gate.
- Missing or invalid source records are surfaced; no missing metric is replaced with zero.
- Bootstrap intervals use paired per-query/per-repetition observations only when both arms contain aligned records. The script labels unavailable intervals when it cannot pair them.
- The analyzer records source-file SHA-256 hashes and environment metadata in its manifest.
- Cache lifecycle numbers are kept separate from end-to-end latency.
- CI fixtures are software tests only; never cite their synthetic values as experimental results.

## Validation

    python3 research/comparative-study/c11-end-to-end-evaluation/c11_validate.py
    python3 -m unittest discover -s research/comparative-study/c11-end-to-end-evaluation -p 'test_*.py'

See PLAN.md for the frozen protocol, acceptance gates, and interpretation limits.