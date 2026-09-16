# ds.json storage transformation (snapshot budget)

| run | sha256 before | gzipped MB | round-trip identical |
|---|---|---|---|
| PH3B-COMP-001 | `78ee590f3090425a…` | 1.1 | YES |
| PH3B-COMP-002-D256 | `5a9ac4c267596ef1…` | 1.4 | YES |
| PH3B-COMP-003 | `0bd472ea7b223f66…` | 1.1 | YES |
| PH3C-HEAD-001-H1-S5K | `e8bff55f73082485…` | 0.5 | YES |
| PH3C-HEAD-001-H1 | `d31a303a1c077621…` | 0.4 | YES |
| PH3C-HEAD-001-H2-S5K | `ea972873a827665e…` | 0.7 | YES |
| PH3C-HEAD-001-H2 | `3778beb20e5c2b9b…` | 0.7 | YES |
| PH3C-HEAD-001-H4-S5K | `4bd02c4ab6d7a310…` | 0.8 | YES |
| PH3C-HEAD-001-H8-S5K | `f503470f8f26a63c…` | 1.3 | YES |
| PH3C-REPRO-001 | `e1652069edd4e8c9…` | 1.1 | YES |

Total: 64.4 MB -> 9.0 MB (all sha-verified byte-identical on round-trip).
