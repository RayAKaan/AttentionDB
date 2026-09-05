# Table: memory scaling — PH3-DS-FM (spec §9/§19)

Caption draft: *Index build memory on a 1984 MB sandbox: the engine builds 10K docs at 571 MB peak (≈8.6× raw vector bytes) but OOM-kills between 10K and 30K docs — the dominant §19 limitation.* [PH3-QUAL-FM-S, PH3-QUAL-FM-T30, PH3-QUAL-FM-M]

| tier | docs | status | peak RSS MB | note |
|---|---|---|---|---|
| S | 10000 | OK | 571 | PH3-QUAL-FM-S (config.json peak_rss_mb) |
| T30 | 30000 | OOM_KILLED | — | PH3-QUAL-FM-T30 exit 137 ~82s into build |
| M | 60000 | OOM_KILLED | — | PH3-QUAL-FM-M exit 137 ~111s into build |
