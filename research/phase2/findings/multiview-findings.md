# Multiview Findings (§34–37, F3, F6)

## Corpus intent and honest history

Three genuinely different views (semantic / lexical / metadata) with
doc-unique fine structure per view. Two earlier designs were invalid and
are preserved (harness-corrections HC-2, HC-3). The final design gives a
type-V query the target doc's fine view-V vector and only coarse
projections elsewhere — the operational meaning of "different views are
useful for different queries".

## Results (test splits)

At 210 training queries the gate learned nothing (PH2B-MULTIVIEW-003/004).
At 840 (PH2B-MULTIVIEW-005): R@10 0.5322 single-run / 0.4365 ± 0.0361
multi-seed, vs uniform 0.2128, RRF 0.2622, global best 0.3428, oracle
0.9933. Gap recovered: ~41% (single run), 28–34% (multi-seed range).

## Why only partial (INTERPRETATION, labeled)

- The signal the gate must read is a fine-vs-coarse contrast across a
  192-dim concatenation — plausible but subtle; pooled corr is only 0.269.
- Per-query-type breakdown (`results/multiview-by-query-type.csv`):
  all three types benefit vs uniform, none approaches oracle; selection
  frequency [0.311, 0.489, 0.200] shows real but imperfect view routing.
- Data quantity dominates (F6): the grid moved <2pp; ×4 data moved +25–30pp.

## What this does and does not show

MEASURED: a linear gate over concatenated query views learns useful
view routing when given ~840 training queries on THIS construction.
NOT SHOWN: transfer to real multimodal embeddings; behavior with more
views; whether a view-specific encoder (instead of concatenation) would
sharpen the signal (HYPOTHESIS for 2C).
