# Table: Phase 2 ablation (FROZEN accepted baseline; noise-ladder corpus, degenerate-era GT — see HC-4)

Caption draft: *Phase 2 ablation on the 8-head noise-ladder corpus. Values are the accepted frozen baseline; the harness ground truth was later corrected (HC-4), so these absolutes are not comparable to Phase 2B numbers — mode ORDERING was stable across runs.* [PH2-ABLATION-001]


| mode | R@10 | NDCG@10 | MRR | p50 (µs) | p99 (µs) | QPS |
|---|---|---|---|---|---|---|
| A_single_head | 0.7630 | 0.8259 | 1.0000 | 239 | 336 | 4121 |
| A0_head_default | 0.7630 | 0.8259 | 1.0000 | 186 | 277 | 5198 |
| A1_head_semantic | 0.5620 | 0.6310 | 0.9308 | 268 | 377 | 3654 |
| A2_head_lexical | 0.3480 | 0.3920 | 0.7495 | 295 | 423 | 3282 |
| A3_head_graph | 0.2490 | 0.2568 | 0.4862 | 350 | 531 | 2760 |
| A4_head_h4 | 0.4870 | 0.5360 | 0.8415 | 276 | 421 | 3555 |
| A5_head_h5 | 0.3410 | 0.3754 | 0.6877 | 315 | 463 | 3148 |
| A6_head_h6 | 0.2470 | 0.2734 | 0.5532 | 341 | 550 | 2810 |
| A7_head_h7 | 0.1940 | 0.2008 | 0.4251 | 370 | 584 | 2616 |
| B_multi_head_fixed | 0.6720 | 0.7499 | 0.9850 | 1734 | 2198 | 573 |
| C_learned_gating | 0.6720 | 0.7499 | 0.9850 | 1744 | 2014 | 574 |
| D_qk_attention | 0.6720 | 0.7502 | 0.9850 | 1764 | 2013 | 572 |
| E_full_exact_rerank | 0.5580 | 0.6263 | 0.9508 | 2075 | 3509 | 476 |
