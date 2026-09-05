# Latency & parameters (model-only micro-bench, warm, release, this VM)

| path | p50 µs |
|---|---|
| gating forward | 13.0 |
| QK forward (union ≈ 250 cands) | 310.5 |
| fuse_weighted (pools) | +weighted-sum cost |
| RRF (pools) | +rank-fusion cost |

QK costs ~24× gating per query (K projection per candidate dominates) while
REDUCING quality. Parameters: gating grid winner (SoftTarget, lr .003,
hidden 64) vs QK 2×8×192 = 3,072 params — parameters are not the issue;
the signal is. Full percentiles/QPS: raw/runs/…/latency.csv.
Frontier verdict: QK is dominated (worse quality, higher latency) — F15.
