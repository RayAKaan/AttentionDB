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

---

## Phase 3B claims (complementary multi-view + text/hybrid)

| claim | table/figure | result file | experiment | raw run | dataset | commit |
|---|---|---|---|---|---|---|
| gating recovers 44% of oracle-head gap on complementary real text (0.6448±0.0025 vs uniform 0.3885 / gbest 0.4033 / oracle 0.9672) | tables/table-complementary-main.md, figures/figure-A-comparisons.svg, figure-C | results/complementary-retrieval.csv + gating-by-query-type.csv | PH3B-COMP-001 | raw/runs/PH3B-COMP-001 | PH3B-DS-AG-S | see registry |
| gate allocates largest mean weight to each type's defining head (agreement 0.55–0.62) | tables/table-gating-vs-oracle.md, figure-B | results/gating-weights-by-type.csv | PH3B-COMP-001 | raw/runs/PH3B-COMP-001 | PH3B-DS-AG-S | " |
| advantage replicates at 20K docs dim 256 (0.6021±0.0161 vs 0.3580/0.4333) | tables/table-memory.md | results/complementary-retrieval.csv | PH3B-COMP-002-D256 | raw/runs/PH3B-COMP-002-D256 | PH3B-DS-AG-M20D256 | " |
| 512-dim medium corpora OOM on 2 GB sandbox (2 preserved failures + 1 tmpfs-hygiene failure) | tables/table-memory.md | results/complementary-memory.csv | PH3B-COMP-002-M30/-M20/-D256-FAILED-TMPFS | raw/runs/* | PH3B-DS-AG-M*/ | " |
| BM25 channel verified correct (containment 0.9902, rare-token 0.7556, indep-overlap 0.5852) but trails dense arms vs semantic GT (0.1787) | tables/table-bm25-hybrid.md, figure-D | results/bm25.csv | PH3B-BM25-001 | raw/runs/PH3B-BM25-001 (analysis of COMP-001) | PH3B-DS-AG-S | " |
| hybrid RRF k=60 (0.3131) < gating (0.6448); k never tuned on test (val: k20 0.350/k60 0.330/k120 0.325) | figure-D | results/hybrid.csv | PH3B-HYBRID-001 | raw/runs/PH3B-HYBRID-001 | PH3B-DS-AG-S | " |
| reproduction: seed-42 rerun gating 0.6492 vs 0.6426 (pool wobble), BM25 identical | findings/complementary-analysis.md C-1 | results/complementary-retrieval.csv | PH3B-COMP-003 | raw/runs/PH3B-COMP-003 | PH3B-DS-AG-S | " |

Reproduction:
```
python3 research/phase3/datasets/build_agnews.py S          # + M20D256 / M20 / M
cargo build --release -p phase3-bench
./target/release/phase3-bench textquality --tier S --data /var/tmp/phase3b/agnews-S \
  --out research/phase3/raw/runs/<RUN_ID> --seeds 42,7,1
python3 research/phase3/baselines/bm25_verify.py <RUN_DIR> <DATA_DIR>
python3 research/phase3/results/sync_ph3b.py
python3 research/phase3/tables/generate_tables_ph3b.py
python3 research/phase3/figures/generate_figures_ph3b.py
python3 research/phase3/verify_consistency.py
```

---

## Phase 3C claims (memory forensics / head scaling / candidate budgets)

| claim | table/figure | result file | experiment | raw run | commit |
|---|---|---|---|---|---|
| wall is build-time (insertion loop), ~10.9x raw rate; bracket 15K OK / 20K OOM | figures/figure-1, figure-9 | results/memory-components.csv, memory-scaling.csv | PH3C-MEM-001, -002-C* | raw/runs/PH3C-MEM-* | see registry |
| memory ~linear in heads (245-260 MB/head @512) | figure-2 | results/memory-scaling.csv | PH3C-MEM-002-H* | " | " |
| memory ~linear in dim over fixed base | figure-3 | results/memory-scaling.csv | PH3C-MEM-002-D* | " | " |
| max reproduced: 1h/30K, 2h/20K, 4h/10K, 8h/5K @512 | paper/memory.md | results/memory-scaling.csv | PH3C-MEM-002-BUDGET-*, -H8-512-5K | " | " |
| duplication ≥2x raw + ~128MB retained build transient; exact split open | paper/memory.md | results/memory-components.csv | PH3C-MEM-001 | " | " |
| tmpfs/leak mechanism verified (SIGKILL/exit leak; Drop cleans) | paper/memory.md | (leaktest transcript in run_info) | PH3C-MEM-003 | " | " |
| head ladder: 0.519/0.658/0.715/0.709 (1/2/4/8, 5K) — saturation at 8 | figures/figure-4, -5, -6 | results/head-scaling-quality.csv | PH3C-HEAD-001-*-S5K | " | " |
| parallel-2 halves p50 at >=2 heads; hurts at 1 | figure-5 | results/head-scaling-latency.csv | PH3C-HEAD-001-*-S5K | " | " |
| budget plateau: gating flat K=10..200; CR 0.787->0.877 | figure-7 | results/candidate-budget.csv | PH3C-HEAD-001-H4(-S5K) | " | " |
| miss decomposition 34% absent / 66% ranked-out / 19% view-selection | figure-8 | results/candidate-decomposition.csv | PH3C-HEAD-001-* | " | " |
| reproduction: gating 0.6492 vs 0.6448 (+0.0044); BM25 identical; dataset hashes identical | — | results/reproduction.csv | PH3C-REPRO-001 | " | " |
