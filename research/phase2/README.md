# Research Record — Phase 2 / 2B / 2C (§29, §34)

This directory is the scientific record of the retrieval-learning research
program: methodology, raw runs, generated tables and figures, findings
(positive AND negative), paper components, and the experiment registry.
Every number quoted anywhere in the paper materials traces:
**paper → table/figure → results CSV → experiment ID → raw run → dataset
hash + code commit** (`raw/experiment-index.json`).

## Layout

| path | content |
|---|---|
| `methodology/` | experimental protocol, datasets, ground truth, split protocol, reproducibility |
| `results/` | canonical CSV/JSON result files (single source for tables/figures) |
| `tables/` | auto-generated Markdown tables (`tables/generate_tables.py`) |
| `figures/` | auto-generated figures + caption drafts (`figures/generate_figures.py`) |
| `findings/` | findings F1–F10, claim ledger, negative results, harness corrections, failure analysis, per-topic deep dives |
| `paper/` | modular paper sections (methodology / results / discussion / limitations) |
| `raw/experiment-index.json` | machine-readable registry of every experiment |
| `raw/run-manifest.json` | asset manifest for paper reconstruction |
| `raw/runs/<ID>/` | immutable per-run outputs (run_info, config, metrics, CSVs) |
| `verify_consistency.py` | §25 check: tables ↔ raw ↔ registry + oracle-sanity invariants |
| `phase2b-final-report.md` | Phase 2B final report (§41 structure) |
| `PAPER_PACKAGE.md` | reconstruction manifest for the eventual paper (§30) |

Rules enforced in this directory: raw results are never overwritten (new
run IDs on rerun); failed/invalid runs keep INVALIDATED status with reasons;
numbers in tables are generated, never transcribed; no fabricated
references (`[CITE: ...]` placeholders only); no fabricated data
(pending experiments are marked pending).

## Experiment timeline (chronological, §29)

| date | ID | event |
|---|---|---|
| 2026-09-03 | PH2-ABLATION-001 | Phase 2 ablation A–E + head scaling (accepted, frozen) |
| 2026-09-03 | PH2B-GATING-001 | first gating run on controlled corpus |
| 2026-09-03 | PH2B-GATING-002 | + temperature calibration (first attempt) |
| 2026-09-03 | PH2B-NOISE-001 | noise-corpus run → all-zero recalls **exposes GT id bug (HC-1)** |
| 2026-09-03 | PH2B-MULTIVIEW-001 | multiview v1 → oracle 0.26 **exposes GT tie lottery (HC-2)** |
| 2026-09-03 | PH2B-GATING-003 | first valid run after engine-id GT fix |
| 2026-09-03 | PH2B-MULTIVIEW-002 | fine-grained views v2 → gate refuses to learn **exposes all-view query flaw (HC-3)** |
| 2026-09-03 | PH2B-MULTIVIEW-003 | one-modality queries: negative result at 210 train queries |
| 2026-09-03 | PH2B-MULTIVIEW-004 | hyperparameter grid: <2pp — rules out capacity |
| 2026-09-03 | PH2B-GATING-004 | grid+calibration protocol: gating = oracle (controlled) |
| 2026-09-03 | PH2B-NOISE-002 | valid noise-corpus run |
| 2026-09-03 | PH2B-MULTIVIEW-005 | 840 train queries: gating learns (+28–32pp) |
| 2026-09-03 | PH2B-NOISE-003 | grid protocol rerun (current noise result; collapse to head 0) |
| 2026-09-04 | PH2B-MULTISEED-001/002 | 3-seed robustness: controlled ±0.0000; multiview ±0.0361 |
| 2026-09-04 | PH2B-SAMPLE-001 (q150..q1200) | sample-efficiency sweep: plateau → transition |
| 2026-09-04 | PH2C-RERANK-001/002/003 | exact-vs-normalized weighting study (§32) |
| 2026-09-04 | PH2B-LATENCY-001 | gating model micro-bench (0.87 µs, 1188 params) |
| 2026-09-04 | PH2C-QK-001 | QK sanity dataset: linear QK 1.0000 vs gating ≤ chance (class impossibility) — gate PASSED |
| pending | PH2C-QK-002+ | trained candidate-level QK on real corpora (gating verdict positive → unlocked) |
| pending | PH2C-RERANK-004+ | pipeline-level rerank re-weighting (ledger N3/Q3) |

Environment note: two workspace resets lost git history (tree preserved);
the registry is the canonical experiment history, commit IDs reference the
commits that contained the producing code.
