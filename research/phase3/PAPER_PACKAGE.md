# Phase 3 paper package (reconstruction manifest)

Claim → table/figure → canonical result → experiment → raw run → dataset
hash → code commit → reproduction command.

## Current claims (provisional findings F-P3-1..F-P3-4)

| claim | table | result file | experiment | raw run | dataset | commit |
|---|---|---|---|---|---|---|
| gating collapses to dominant view on PH3-DS-FM (matches single-head) | tables/table-retrieval-quality.md | results/retrieval-quality.csv | PH3-QUAL-FM-S | raw/runs/PH3-QUAL-FM-S | PH3-DS-FM-S (sha256 via datasets/build_fashion.py meta.json) | see registry entry |
| uniform fusion degrades on dominated heads | tables/table-retrieval-quality.md | results/retrieval-quality.csv | PH3-QUAL-FM-S | raw/runs/PH3-QUAL-FM-S | PH3-DS-FM-S | " |
| latency structure of frozen path (gating fwd ≈1.4% of pipeline) | findings (F-P3-3) | results/latency.csv | PH3-QUAL-FM-S | raw/runs/PH3-QUAL-FM-S | PH3-DS-FM-S | " |
| memory wall between 10K and 30K docs (2 GB sandbox) | tables/table-scaling-memory.md | results/memory.csv | PH3-QUAL-FM-M, PH3-QUAL-FM-T30 | raw/runs/* (FAILED runs preserved) | PH3-DS-FM-M/T30 | " |

## Reproduction

```
python3 research/phase3/datasets/build_fashion.py S
cargo build --release -p phase3-bench
./target/release/phase3-bench quality --tier S --data /tmp/phase3/fashion-S \
  --out research/phase3/raw/runs/PH3-QUAL-FM-S --seeds 42,7,1
python3 research/phase3/tables/generate_tables.py
python3 research/phase3/verify_consistency.py
```

Datasets regenerate deterministically; verify sha256 against the printed
meta.json. Registry commit IDs pin the producing code.
