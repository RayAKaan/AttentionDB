# C8 Resource Report

Phase: **C8 — Residual candidate-level Q/K/V attention** · Branch: `comparative-study/c8-residual-qkv-attention`

## 1. Latency (SMOKE, per-rep p50, microseconds)

| Arm | SciFact p50 | NFCorpus p50 | Config |
|---|---|---|---|
| A single-head | 3369 | 3036 | no C8 |
| B fixed fusion (ref) | 13774 | 11963 | no C8 |
| C λ=0 control | 342086 | 348860 | C8 machinery, λ=0 |
| D unrestricted | 348829 | 358464 | C8, unrestricted |
| E residual | 356282 | 364848 | C8, residual |
| F residual+evidence | 352058 | 357295 | C8 |
| G residual+evid+disagree | 351548 | 361193 | C8 |
| H residual distilled | 354263 | 358314 | C8 |
| I residual + K/V cache | 316994 | 314140 | C8, cache |

## 2. Observations

- **C8 dominates latency.** Turning on the residual stage costs two orders of
  magnitude over B (`~14 ms → ~350 ms` per query p50) because it materializes a
  per-candidate per-head attention pass over the full 500-candidate fusion-depth
  union. This is a *correctness-first* experimental stage, not a production cost
  claim; the honest number matters for the report.
- **Cache recovers ~11–14%.** The in-memory K/V cache (arm I) lowers p50 from
  E `356`→I `317` ms (SCI) and `365`→`314` ms (NFC) while reproducing E's scores
  bit-for-bit (cache-parity gate). Cache value is bounded because the projection is
  cheap relative to the per-candidate softmax/scoring pass.
- **`λ=0` is not free.** Arm C pays nearly the full C8 cost while contributing no
  correction: the projection + trace are still computed, then scaled by zero. This is
  the price of the frozen `λ=0`-parity design and is called out in the failure report.

## 3. Memory

K/V cache retains only present-document projections within the budgeted union and is
purged on `retain` (unit tests `retain_purges_retired_documents`,
`retain_keeps_the_fingerprint`). No disk persistence (protocol Q3: in-memory only).
