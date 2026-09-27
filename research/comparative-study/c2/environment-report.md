# C2 Environment Report — run hosts and guardrail envelopes

Study `comparative-study-001`, protocol v1.0.0 (commit `7788067`).
This report documents the environments in which every C2 run executed and the
guardrail envelope each run preflighted against. Per-run `environment.yaml`
files under `../raw/<run_id>/` are the authoritative, immutable records; this
document is a readable synthesis of them.

## Two environments, one study

| | Primary run host (all sandbox-originated runs) | Evidence re-registration host (probe rebuild + fresh runs) |
|---|---|---|
| OS / kernel | Linux 6.1.158+ | Windows 11 Pro (build 26200) |
| CPU | Intel(R) Xeon(R) Processor @ 2.60 GHz | AMD Ryzen 3 3100 (4C/8T) |
| Logical CPUs | 2 | 8 |
| RAM total | 2,032,608 KB (~1.9 GiB) | 16,700,964 KB (~15.9 GiB) |
| MemAvailable (representative) | 1,267,640 KB | 2,693,405 KB (falling, see below) |
| Filesystem | ext4 | NTFS (H:) |
| Disk free (representative) | 14,094 MB | ~471,677 MB (460.7 GB) |
| Python | 3.13.14 | 3.14.4 |

Both environments share the same repository working tree and the same
registration commit `7788067c1437ae8c75ced4dc1ffc2f49b2c22a88`; raw run dirs are
host-agnostic evidence and `environment.yaml` records the host truth per run.

## Guardrail envelope (as preregistered, C1 environment-guardrails + C1 charter §16)

Every run preflight asserted and recorded:

- `mem_budget_kb_85pct_available` = 85 % of MemAvailable at start;
  a child process tree may be SIGKILL'd when its tree RSS crosses the budget.
- `disk_floor_mb` = 500 MB usable-on-`/dev` free requirement (GLOVE50 materialization
  gate and all big-file runs enforced this).
- `phase3e_crash_env_asserted_unset` = true on every run (no Phase 3E crash
  env leakage).
- Sampler policy: `guarded_popen` samples tree RSS every 500 ms when attached;
  server-abort evidence includes sampler rows tied to stderr timing.

Observed budgets (KB): C2-ORACLE-TESTS-002 1,077,708; C2-DATA-SCIFACT-007 and
C2-DATA-NFCORPUS-003 ~1,087,839; the fresh Windows re-runs 2,293,816 (C2-MODES-TEST-001),
2,307,947 (C2-ORACLE-AGREE-001), 2,309,293 (C2-INTEGRITY-001).

## Abort / resource events actually observed (all preserved as evidence)

- **Elasticsearch 8.15.2** (C2-SMOKE-ES-003): tree RSS 1,149,344 KB reached ~10 s
  into JVM startup >= 1,089,169 KB budget; sampler SIGKILL'd the child at row 20;
  20 sampler rows + ES logs preserved. `vm.max_map_count=65530 < 262144` recorded
  as a second, independent constraint. Result: **ABORTED/OBSERVED-LIMIT, terminal
  on this host**; no reduced-heap retry (preregistered).
- **C2-B3-TRAIN-001** (run): killed by SIGPIPE from the invoking shell during
  config write (INVALID-STARTUP; superseded).
- **C2-DATA-GLOVE25-004**: INVALID-STARTUP — the session reset killed the verifier
  mid-run; recorded. The reset doc recorded intent for a successor (-005); no
  successor run was registered this session, so -004 remains the terminal
  INVALID-STARTUP entry for that leg.
- **Windows re-runs**: no abort; c2probe is an in-process lightweight executable
  (modes-test < 5 s, negligible RSS; reported ~2.7 MB RSS, 1.1 MB share).

## Host-capability summary (C1 Question G)

- Qdrant 1.12.4: feasible at this envelope (peak server RSS 59,604 KB under sampler).
- Weaviate 1.39.6: feasible (peak RSS 109,656 KB, ready 5.5 s).
- pgvector 0.8.0 + PostgreSQL 17: feasible; system service (no per-process
  sampler applicable; preflight env only).
- milvus-lite 3.2.1: feasible (in-process `MilvusClient` over local `.db`).
- Elasticsearch 8.15.2 default-JVM: NOT feasible at ~1.9 GiB RAM (OBSERVED-LIMIT).
- Pinecone / MongoDB Atlas: never requested (no credentials; C1, unchanged).
- B3 gating training + exact NN-GT recompute: feasible (wall times seconds-scale).

## Standing note for any future run

Re-runs on hosts with different envelopes are admissible as NEW run IDs with their
own `environment.yaml`; they never modify existing evidence. Fresh-mode runs do not
inherit the sandbox host's constraints (e.g. 2 vCPU) and must not be conflated with
it — any cross-host comparison (none performed in C2) would need that stated.