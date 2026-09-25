# C4 — Confirmed-Scale Environment Record

Study `comparative-study-001`, protocol v1.0.0. Captured 2026-09-25 at C4
start (before any C4 run). Companion: `c4-plan.md`, `c4-dataset-validation.md`,
`c4-resource-budget.yaml`, `c4-run-plan.csv`.

## 1. C3 closure handoff (C4-G1) — PASS

| Check | Result |
|---|---|
| Repo HEAD (closure base) | `5195133ca5fab7a660d403b009b565408d6a09b3` (C3 closure commit) |
| Branch | `comparative-study/c4-confirmed-benchmark` (cut from the C3 closure commit) |
| Initial tree sha16 | `93c1ccdef076aeca` |
| C3 work | committed at C3 closure `5195133` (docs + harness + c2pilot driver + B3 modelcards); raw evidence remains untracked by convention (hashes per-run) |
| C3 pilot status | CLOSED PASS (7 final cells; 5 fresh-process reps real-data; exact oracle; guardrails) |
| C3 report correction | `c3-execution-report.md` §4 peaks "2–284 MB" → "2–304 MB" (documentation-only, recorded in C3 change-history footer; no raw/hash/metrics value altered) |
| Working tree (tracked) | only C4 additions to be added; raw/ + c2 untracked as before |

## 2. Host measurement (C4-G2) — recorded, NOT the preregistered envelope

| | Actual C4 execution host | Preregistered constrained envelope (C2 `environment.yaml`) |
|---|---|---|
| OS | Microsoft Windows 11 Pro (build 26200.…), NTFS (H:) | Linux kernel 6.1.158+ / ext4 |
| CPU | AMD Ryzen 3 3100 (4C/8T; 8 logical processors) | Intel Xeon 2.60 GHz |
| RAM total | 16,700,964 KB (~15.9 GiB) | 2,032,608 KB (~1.9 GiB) |
| MemAvailable @ C4 preflight | 3,004,496 KB (~2.87 GiB) | ≈1,267,000 KB |
| 85% of MemAvailable @ preflight | 2,553,822 KB | ≈1,077,000 KB |
| Disk free @ C4 preflight | 466,046 MB (~455.1 GiB) | ~14 GB |
| Python | 3.14.4 (numpy 2.4.6, torch 2.12.1+cpu, sentence-transformers 5.7.0) | 3.13.14 |
| rustc/cargo | 1.96.0 (ac68faa20 2026-05-25) | recorded per-run |

## 3. Toolchain feasibility on this host (verified by C3)

- Engine: `c2/probe/target/release/c2pilot.exe` SHA-256 (Windows release
  build, real `AttentionDB` crate; matches every run's `environment.yaml`).
  C3-era binary: `56bf2bd8014bedd1f9761964cadc30a7b393cf33f70009bff9083423da1a97f4`
  (used by C3 pilot). C4-parameterized binary (adds `candidate_budget`,
  `min_candidates_per_head`, `candidate_multiplier`, `ef_search`,
  `ef_construction`, `m`, `configuration_id` knobs + emits per-query
  `candidate_count`/`heads_present`/`relevant_ids` via
  `attend_detailed_with_stats`):
  `9024974731d8d5c0b182348eae2601037768819ebae5eff459c82c6931f543ce`
  (smoke-tested on this host: budget=16/ef=4 cell PASS, exit 0).
  `c2probe.exe` / `bm25repro.exe` present for synthetic corpus export + BM25
  regression. C2-MODES-TEST-002 PASS 8/8 on this host.
- Exact oracle: `harness/oracle.py` (numpy) + pure-python reference + Rust
  brute force; C2-ORACLE-AGREE-001 PASS. Exact top-k deterministic
  (score desc, id asc). C3 confirmed 0/100 exact-top10 mismatches across
  NFCorpus cells.
- B3 gating: in-repo `learned/gating_v2` trainer + ModelCard activation
  (torch cpu present); INV-L1..L5 leak guards PASS; card
  `b3-lodo-nf-v2-s20260925-h32-lr0.01.json` (3 heads TITLE/BODY/CITE)
  validated + activation tested.
- Re-embedding: sentence-transformers 5.7.0 + torch cpu; model revision
  pinned `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

## 4. Envelope policy (decided at C3, unchanged, reaffirmed for C4)

- Execution host = this Windows 11 workstation. The preregistered constrained
  envelope is the Linux sandbox (2 vCPU / 1.9 GiB / 20 GB) captured in every
  C2 `environment.yaml`. C4 runs are NEW run IDs with their own
  `environment.yaml`; no cross-host conflation; deviation disclosed, not
  silenced.
- Per-run enforcement per C1 guardrails: 85% of preflight MemAvailable,
  500 MiB disk floor, sampler abort (500 ms interval, process tree),
  one external server at a time, build ≤60 min / query ≤30 min timeouts,
  `PH3E_CRASH_*` asserted unset.
- Budget axes unchanged and absolute: BUDGET-CAND {50,100,200,500};
  BUDGET-EF {16,32,64,128}; BUDGET-MEM {512 MiB, 1.0 GiB, 1.6 GiB};
  BUDGET-TIME {1,10,100 ms}. No enlarged memory budget is invented; systems
  that cannot fit are BLOCKED or OBSERVED-LIMIT at that level.
- Observed C3 peaks ≤304 MiB vs 2.5+ GiB abort threshold → no boundary hits
  expected on AttentionDB cells at confirmed scale; measured, not assumed.

## 5. Data materialization state at C4 start (drives C4 dataset validation)

| Dataset | Status at C3 close | C4 disposition (see `c4-dataset-validation.md`) |
|---|---|---|
| DS-SCIFACT | ELIGIBLE-ON-OWN-HASHES (`C3-EMBED-SCIFACT-001`, drift ~2–3e-7, sim ~0.99999997) | decision A vs B recorded in validation doc; hashes re-verified |
| DS-NFCORPUS | ELIGIBLE (5 .npy byte-verified) | re-verify hashes; canonical + 3 HEAD views + queries |
| DS-ANN-GLOVE-25/50 | CONDITIONAL (HDF5 absent) | stays non-primary for C4 confirmed scale unless re-procured + verified (not required for the 8 primary contrasts) |
| DS-ESCI/COCO | CONDITIONAL | not in C4 primary confirmed-scale scope (Track B optional; re-procure + verify first) |
| DS-SYNTH-PH2B | ELIGIBLE (in-repo) | DIAGNOSTIC only; not primary evidence |

## 6. What this authorizes for C4

Confirmed-scale primary scope (preregistered workloads W-01, W-02, W-03,
W-04, W-07 on DS-SCIFACT + DS-NFCORPUS; modes B0, B1, B2, B3, B4, B7; Track A
matched + BUDGET-CAND/EF/MEM/TIME sweeps). External systems stay by eligibility
status (Qdrant/pgvector/ES/Milvus-lite/Weaviate CONDITIONAL — a system is
benchmark-ready only after this-host smoke, else BLOCKED at C4; Pinecone/Mongo
BLOCKED-AUTH unchanged; B6 adapter still requires-implementation).