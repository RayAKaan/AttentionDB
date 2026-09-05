# PH2C-QK-002 — Real-corpus main evaluation (QK vs gating)

**Verdict (Rule Zero): NO SIGNIFICANT WIN.** Trained candidate-level QK
LOSES to trained gating on the only corpus where the paired comparison is
valid (multiview: 0.1113 vs 0.4983 R@10). §29 STOP condition engaged — no
deeper architectures. Gating retained; QK optional (no measured benefit).

- methodology.md — protocol, arms, fairness, HC-6 gate
- results.md — numbers (traceable to raw/runs/PH2C-QK-002-MULTIVIEW)
- failure-analysis.md — why QK failed here after winning the sanity set
- latency-analysis.md — cost accounting
- Main analysis: ../findings/qk-main-analysis.md (F11–F15)
- Table: ../tables/table-qk-main.md · Raw: ../raw/runs/PH2C-QK-002-*
