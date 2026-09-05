# Results (TEST; agg of seeds 42/7/1)

| arm | R@10 | NDCG@10 | MRR |
|---|---|---|---|
| uniform | 0.2128 | 0.2803 | 0.6435 |
| global_best | 0.3428 | 0.3521 | 0.4706 |
| RRF k=60 | 0.2622 | 0.3114 | 0.6805 |
| trained gating | **0.4983** | 0.5289 | 0.7009 |
| trained QK | 0.1113 | 0.1151 | 0.2944 |
| gating+QK | 0.2769 | 0.3187 | 0.6153 |
| oracle | 0.9933 | 0.9957 | 1.0000 |

Deltas (R@10): QK−gating = **−0.3870**; gating+QK−gating = −0.2214;
gating+QK−QK = +0.1656.
Per-seed gating .5050/.4939/.4961; QK .1100/.1078/.1156 (no seed flips the
sign). Candidate recall 0.9975 (100% queries ≥1) — not pool-limited.
Budgets (K=10/25/50/100), per-group, rerank diagnostics, latency:
raw/runs/PH2C-QK-002-MULTIVIEW/*.csv. Controlled/noise: canonical 2B
reference results only (HC-6).
