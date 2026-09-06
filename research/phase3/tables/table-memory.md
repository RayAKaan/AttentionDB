# Table: memory scaling — PH3B text family (spec §14)

Caption draft: *Peak build RSS scales ≈14× raw vector bytes; the 2 GB sandbox cannot build the 512-dim medium corpus (two preserved OOM runs, plus one tmpfs-hygiene failure), while the dim-256 20K probe completes at 967 MB and replicates the gating result.* [PH3B-COMP-001, PH3B-COMP-002-*, PH3B-COMP-003]

| Experiment | Docs | Raw vector MB | Peak RSS MB | Engine×raw | Status |
|---|---|---|---|---|---|
| PH3B-COMP-001 | 10000 | 58.6 | 841.6 | 14.36 | COMPLETED |
| PH3B-COMP-002-D256 | 20000 | 58.6 | 966.7 | 16.50 | COMPLETED |
| PH3B-COMP-003 | 10000 | 58.6 | 840.5 | 14.34 | COMPLETED |
