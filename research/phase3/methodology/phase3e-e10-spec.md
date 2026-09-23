# Phase 3E — E10 Specification: Scale-Envelope Expansion & Capacity Characterization
(FROZEN before measurement)

Date frozen: 2026-09-23. Baseline: commit `3e65d50f8fd1a81cda339d4608260d6be98ba708`,
E9 working tree (sha16 `a4ce3224bc9a3b3d`). Entering envelope (E9-sealed):
80,000 docs / 150,000 ops; ~9.5 KB/doc all-in memory; hygiene active at checkpoints.

## Environment (captured at freeze)

2 vCPU (e2b.local, kernel 6.1.158+), 1.9 GiB RAM total (1.4 GiB available at
capture), 20 GiB disk free on /, ext2/ext3, rustc 1.98.1 (restored after
sandbox reset; E9 used the identical version). All scale claims are scoped to
THIS resource class.

## M-markers

- M1 baseline reproduction: re-run the E9 sealed configuration (80k-doc
  insert-only ladder rung, PH3E-MEM-004 shape) with the E10 harness BEFORE any
  expansion; material divergence blocks E10.
- M2 ladder: document tiers 80K → 100K → 150K → 200K, one fixed primary
  configuration, each tier = build + verify + reopen + verify + retrieval
  probes + full telemetry. Tiers are attempted strictly in order.
- M3 budget gate (pre-flight, binding): projected peak = base + per-doc_marginal
  × docs, per-doc_marginal measured from the largest completed tier (initial
  estimate 7 KB/doc from E9). LAUNCH IS FORBIDDEN if projected peak > 80% of
  MemAvailable measured immediately before the run. A tier blocked by this gate
  is classified OBSERVED LIMIT (resource-bound, budgeted) — never silently
  skipped, never launched into uncontrolled OOM.
- M4 live guard (in-run): sample MemAvailable every 2 s; if live RSS exceeds
  85% of launch MemAvailable, the harness halts cleanly (final telemetry +
  census + verify + preserve) and the run is classified OBSERVED LIMIT.
- M5 OOM discipline: any kernel-kill evidence is preserved verbatim; OOM is a
  resource boundary, not an engine bug, unless correctness evidence shows
  otherwise (§27 of the charter). Reruns after any boundary take NEW run IDs.
- M6 correctness gates per tier (ALL required for VERIFIED): (a) checker clean
  post-build and post-reopen; (b) exact live count; (c) sampled model equality
  ≥ 5,000 docs (uuid + fields exact); (d) fresh-process reopen equality
  (count + sample); (e) retrieval self-hit rate ≥ 95% at k=10 over the frozen
  100-query self set (E9-strengthened S18); (f) no verification failure.
- M7 primary configuration (FIXED for the ladder): DIM=32, 1 semantic head
  ("h"), HNSWConfig::default() (M/efConstruction/efSearch as shipped —
  recorded from source, not tuned), store_vectors=true (implementation-fixed),
  Durability::Sync, default WAL segment size, E9 hygiene policy unchanged
  (purge + rebuild when pre-purge vstore > 1.5× live AND > 20k, or insert
  budget > max(20k, 4× live)). Any changed setting = separate configuration
  and separate run ID.
- M8 dataset generation (deterministic, independent of the engine): per idx,
  cluster c = idx mod 50; centroid = unit vector from FNV(idx/50); vector =
  normalize(centroid*0.8 + hash-noise(idx)*0.2); every 16th idx is a
  near-duplicate of idx-1 plus epsilon (hash-derived); payload fields
  idx/version/cat/num/title with mixed title lengths (idx mod 7 + 1 words);
  hot subset = idx mod 1000 < 50 (churn families only). UUID =
  version-stamped deterministic (uuid_for scheme). dataset_hash = FNV-1a-64
  over the canonical (idx|uuid|payload-hash) stream, computed pre-insert.
- M9 query set (frozen): 100 self-queries (vectors of live docs at fixed idx
  strides — correctness instrument) + 100 noise queries (latency instrument),
  identical across all comparable runs; per-query latency in µs; p50/p95/p99
  reported; QPS from the same batch.
- M10 families → run IDs (registry-checked, next valid):
  PH3E-SCALE-001 baseline repro (80k, E9 config);
  002..005 ladder tiers 80K/100K/150K/200K;
  006 retrieval scaling at the largest verified tier (1,000 queries);
  007 head-count 1/2/4 at 40K;
  008 dimension 128/256 at 20K (512 only if budget allows);
  009 mutation churn at 60K base (200k mixed ops, hygiene active);
  010 E9 hygiene at scale (checkpoint hygiene timings + recall at 100K/150K
      — measured inside 003/004 and cross-recorded here);
  011 checkpoint/compaction scaling (explicit timings at tiers);
  012 recovery scaling (reopen phase timings at tiers);
  013 bounded concurrency at 100K (1 writer + 1 reader + ckpt/compact);
  014 integrated scale validation at the largest verified tier
      (build→churn→txns→ckpt→hygiene→compact→backup→restart→verify).
- M11 telemetry: REUSES the E9 instruments (no second system): /proc status
  + smaps_rollup (RSS/PSS/anon/file), fds/threads, dir census, engine
  mem_census at boundaries; 2 s cadence; plus per-phase timers
  (build/ckpt/compact/reopen/rebuild) and per-query latencies.
- M12 reference model: harness-side lock-step map (idx → version) + exact
  live count + FNV state hash over the canonical stream (independent of the
  engine); sampled equality against engine reads (committed-only paths);
  the model is NEVER derived from engine state (anti-circular).
- M13 statistics: single-run observations are labeled; the 80K tier is run
  TWICE (001 baseline + 002 ladder) giving one reproduced pair; latency
  percentiles from the frozen query set; no cross-environment comparisons.
- M14 stop conditions: ladder stops at the first tier blocked by M3/M4
  (record OBSERVED LIMIT); families stop on any correctness-gate failure
  (preserve + diagnose before continuing); the integrated run requires all
  its stage gates green.
- M15 artifact policy: per-run telemetry.csv/census.csv/summary.json/verif.csv
  are the evidence; green-run db/ payload is regenerable (deterministic
  dataset + workload) and pruned post-run under the persistence-cap TRIAGE
  policy; failure/OOM runs preserve EVERYTHING including the db dir.

## Resource budget (initial, to be refined by M3 after each tier)

Estimated at freeze: raw vector bytes 32×4 = 128 B/doc; record+payload
~0.5–1 KB/doc; HNSW graph+store+arc ~2–6 KB/doc (E9-census-derived);
DocumentStore caches capped (50k entries); harness model ~0.1 KB/doc.
Projected 200k-doc peak ≈ 1.3–1.6 GB → likely violates M3 at 200K on this
host; the ladder proceeds tier-by-tier and the boundary is MEASURED, not
assumed.
