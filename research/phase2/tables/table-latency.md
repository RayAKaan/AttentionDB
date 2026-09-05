# Table: latency and cost

Caption draft: *Top: trained gating model cost (CPU micro-benchmark, 200 probe vectors × 2000 reps). Bottom: end-to-end pipeline latency from the frozen Phase 2 ablation (100 queries, one shared CI-class VM — machine-specific, ratios indicative).* [PH2B-LATENCY-001, PH2-ABLATION-001]


| component | heads | input dim | value | unit |
|---|---|---|---|---|
| model_inference | 4 | 32 | 0.87 | us_per_query |
| pipeline mode (Phase 2 corpus) | p50 (µs) | p95 (µs) | p99 (µs) | QPS |
|---|---|---|---|---|
| A_single_head | 239 | 323 | 336 | 4121 |
| A0_head_default | 186 | 250 | 277 | 5198 |
| A1_head_semantic | 268 | 364 | 377 | 3654 |
| A2_head_lexical | 295 | 406 | 423 | 3282 |
| A3_head_graph | 350 | 497 | 531 | 2760 |
| A4_head_h4 | 276 | 376 | 421 | 3555 |
| A5_head_h5 | 315 | 427 | 463 | 3148 |
| A6_head_h6 | 341 | 475 | 550 | 2810 |
| A7_head_h7 | 370 | 512 | 584 | 2616 |
| B_multi_head_fixed | 1734 | 2002 | 2198 | 573 |
| C_learned_gating | 1744 | 1907 | 2014 | 574 |
| D_qk_attention | 1764 | 1911 | 2013 | 572 |
| E_full_exact_rerank | 2075 | 2277 | 3509 | 476 |

| head scaling (Phase 2 corpus, mode B, parallel) | heads | p50 (µs) | p99 (µs) | R@10 |
|---|---|---|---|---|
| multi-head | 1 | 253 | 359 | 0.7630 |
| multi-head | 2 | 426 | 564 | 0.7310 |
| multi-head | 4 | 945 | 1246 | 0.6670 |
| multi-head | 8 | 1892 | 2170 | 0.6720 |
