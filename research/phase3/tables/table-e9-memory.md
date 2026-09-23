| experiment | baseline | O1 purge-only | optimized | RSS reduction | elapsed before→after s |
|---|---|---|---|---|---|
| repro churn 150k | 205368.0 KB | 203080.0 KB | 85600.0 KB | 58.3% | 88→85 |
| update churn 100k @10k docs | 240448.0 KB | 229040.0 KB | 202856.0 KB | 15.6% | 121→123 |
| restart-reset | 165632.0 KB | 156660.0 KB | 105496.0 KB | 36.3% | 46→54 |
| growing 80k (expect ~0) | 494412.0 KB | 492172.0 KB | 496928.0 KB | -0.5% | 84→93 |
