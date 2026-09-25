# C3 Preflight Report — handoff, host, and eligibility state

Study `comparative-study-001`, protocol v1.0.0. Captured 2026-09-24 at C3
start (before any C3 run). Companion: `c3-plan.md`, three eligibility CSVs,
`c3-resource-budget.yaml`, `c3-run-plan.csv`.

## 1. C2 handoff verification (C3-G1) — PASS

| Check | Result |
|---|---|
| Repo HEAD | `7cbd16ddd62b671eaefccf4db44bf464164d79ca` (C2 fix commit) |
| Branch | `comparative-study/c3-controlled-benchmark` (created from `7cbd16d` per C3 mandate) |
| Phase3E diff (`fe4f92b..HEAD` and `963170b..HEAD -- research/phase3`) | empty — C0/Phase3E untouched (C3-G15) |
| C2 narrow diff (`c0-audit...c2-validation`) | 0/1 commit (only the fix commit) |
| `verify_c2.py` rerun | 67 dirs; 0 inconsistencies; dataset 14/14 MATCH; statuses PASS 24 / FAILED 14 / INVALID-STARTUP 17 / SUPERSEDED 4 / FAILED-HARNESS-* 7 / ABORTED 1 |
| C2 closure docs | present under `c2/final-audit/` (bm25 finding, gate audit, report-builder audit, closure decision, reconciliation) |
| Working tree | 0 modified tracked files; untracked study dirs only (`research/comparative-study/{c2,raw}` — expected convention) |

C2 is therefore a verified, closed handoff; no discrepancy record required.

## 2. Host measurement (C3-G2) — recorded, NOT the preregistered envelope

| | Actual C3 execution host | Preregistered constrained envelope (C2 `environment.yaml`) |
|---|---|---|
| OS | Windows 11 Pro (build 26200) / NTFS (H:) | Linux kernel 6.1.158+ / ext4 |
| CPU | AMD Ryzen 3 3100 (4C/8T) | Intel Xeon 2.60 GHz |
| Logical CPUs | 8 | 2 |
| RAM total | 16,700,964 KB (~15.9 GiB) | 2,032,608 KB (~1.9 GiB) |
| MemAvailable (measured) | ≈2,693,405–4,243,100 KB (per-run preflight records exact) | ≈1,267,000 KB |
| Disk free | ~455.5 GB of 984.1 GB | ~14 GB |
| Python | 3.14.4 (numpy 2.4.6, scipy 1.17.1, pandas 2.3.3, torch 2.12.1+cpu, sentence-transformers 5.7.0) | 3.13.14 |
| rustc/cargo | 1.96.0 | recorded per-run |

**Envelope policy (decided):** the preregistered budget AXES are absolute
caps, unchanged (BUDGET-CAND {50,100,200,500}; BUDGET-EF {16,32,64,128};
BUDGET-MEM {512 MiB, 1.0 GiB, 1.6 GiB}; BUDGET-TIME {1,10,100 ms}). Per-run
enforcement uses C1 guardrails as written (85 % MemAvailable at preflight,
500 MiB disk floor, sampler abort, one external server at a time, build ≤
60 min / query ≤ 30 min timeouts, `PH3E_CRASH_*` asserted unset). All C3 runs
are NEW run IDs with their own `environment.yaml`; this host's figures are
never presented as the 2-vCPU envelope. No enlarged memory budget is invented;
any system that cannot hold within the matching caps is BLOCKED or
OBSERVED-LIMIT at that level.

## 3. Toolchain feasibility on this host

- Engine: `c2/probe/target/release/c2probe.exe` + `bm25repro.exe` present
  (Windows release build, real engine via `AttentionDB` crate). The C2
  post-fix modes battery ran on this host (C2-MODES-TEST-002 PASS 8/8;
  C2-BM25-REPRO-002 PASS 1/1/1).
- Exact oracle: `harness/oracle.py` (numpy) + independent reference; C2
  oracle battery PASS, engine agreement PASS (C2-ORACLE-AGREE-001).
- Re-embedding: sentence-transformers 5.7.0 + torch cpu present; model
  revision pinned `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

## 4. Data materialization state on this host (drives dataset eligibility)

| Dataset | Corpus/queries/qrels on disk | Embeddings on disk | Glove HDF5 / ESCI / COCO | Status |
|---|---|---|---|---|
| DS-SCIFACT | corpus/queries/qrels present | **absent** (C2-EMBED-SCIFACT-003 run dirs have no `.npy` artifacts) | — | CONDITIONAL (re-embed + sha256) |
| DS-NFCORPUS | corpus/queries/qrels present | **present** (5 `.npy`: CANONICAL/HEAD-*/QUERIES) | — | ELIGIBLE (verify hashes) |
| DS-ANN-GLOVE-25 | — | — | HDF5 **absent** (split idx npy present) | CONDITIONAL (re-fetch + sha256) |
| DS-ANN-GLOVE-50 | — | — | HDF5 **absent** | CONDITIONAL (re-fetch) |
| DS-ESCI-EN-SUB | — | — | parquet **absent** | CONDITIONAL (re-fetch) |
| DS-COCO-CAP-5K | — | — | images/annotations **absent** | CONDITIONAL (re-fetch) |
| DS-SYNTH-PH2B | generator in repo | n/a | — | ELIGIBLE (C2 validated) |

Retention caveat recorded in `raw/datasets/README.md` (large files excluded
from snapshot; every manifest records source URL + sha256, so byte-identical
re-materialization is scripted, not guessed). Any re-procured byte must match
the recorded sha256 before use (C3-G4), and the verification is itself
registered as a C3 preflight run.

## 5. What this authorizes

Eligible now: oracle/latency-floor + mode + synthetic workload cells on hosts
where the needed dataset bytes exist (NFCorpus embed present; synthetic
generator in repo; probe built). Conditional: SciFact embeddings, GloVe,
ESCI, COCO (re-procure + verify first). Blocked: Pinecone/MongoDB (auth),
Elasticsearch under the constrained envelope (C2 terminal OBSERVED-LIMIT; no
reduced-heap retry per preregistration); external multi-vector B6 adapter
(requires-implementation). See `c3-system-eligibility.csv`,
`c3-dataset-eligibility.csv`, `c3-workload-eligibility.csv`.