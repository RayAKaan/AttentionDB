# Table: candidate recall before fusion (§9) — PH3B runs

Caption draft: *Pool-union GT coverage before fusion. The gap between candidate recall (~0.96–0.98) and the oracle arm quantifies how much of the remaining headroom is candidate generation, not gating.* [PH3B-COMP-001, PH3B-COMP-002-D256, PH3B-COMP-003]

| Experiment | Tier | Query type | GT covered frac | Queries full | Avg union candidates |
|---|---|---|---|---|---|
| PH3B-COMP-001 | S-512 | title | 0.9818 | 19 | 273.7 |
| PH3B-COMP-001 | S-512 | body | 0.9889 | 16 | 259.4 |
| PH3B-COMP-001 | S-512 | mixed | 0.9810 | 18 | 259.4 |
| PH3B-COMP-001 | S-512 | ALL | 0.9836 | — | — |
| PH3B-COMP-002-D256 | M20-256 | title | 0.9760 | 21 | 280.6 |
| PH3B-COMP-002-D256 | M20-256 | body | 0.9400 | 15 | 280.3 |
| PH3B-COMP-002-D256 | M20-256 | mixed | 0.9581 | 19 | 279.7 |
| PH3B-COMP-002-D256 | M20-256 | ALL | 0.9580 | — | — |
| PH3B-COMP-003 | S-512-rerun | title | 0.9864 | 19 | 273.6 |
| PH3B-COMP-003 | S-512-rerun | body | 0.9833 | 15 | 259.4 |
| PH3B-COMP-003 | S-512-rerun | mixed | 0.9714 | 17 | 259.9 |
| PH3B-COMP-003 | S-512-rerun | ALL | 0.9803 | — | — |
