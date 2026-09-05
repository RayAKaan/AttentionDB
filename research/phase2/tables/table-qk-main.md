# Table: PH2C-QK-002 — trained candidate-level QK vs trained gating (multiview, TEST)

Caption draft: *Paired comparison on identical candidate pools (candidate recall 0.9975): trained linear QK loses to trained gating and to uniform fusion; gating+QK (RRF-60 blend) is below gating. Controlled/noise QK arms not executable on frozen caches (HC-6). Mean over seeds 42/7/1.* [PH2C-QK-002-MULTIVIEW]

| arm | R@10 | NDCG@10 | MRR |
|---|---|---|---|
| Uniform multi-head | 0.2128 | 0.2803 | 0.6435 |
| Global best head | 0.3428 | 0.3521 | 0.4706 |
| RRF (k=60) | 0.2622 | 0.3114 | 0.6805 |
| Oracle head selection | 0.9933 | 0.9957 | 1.0000 |
| Trained gating | 0.4983 | 0.5289 | 0.7009 |
| Trained QK | 0.1113 | 0.1151 | 0.2944 |
| Gating + QK (RRF-60) | 0.2769 | 0.3187 | 0.6153 |

Deltas (agg): QK−gating = -0.3870, gating+QK−gating = -0.2214, gating+QK−QK = +0.1656 (R@10).

Per-seed, budgets, diagnostics, latency: raw/runs/PH2C-QK-002-MULTIVIEW/{results,budgets,by_query_type,rerank_diagnostics,latency,variability,qk_sanity_checks}.csv. Source: results/qk-attention-multiview.csv.
