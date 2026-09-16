# Paper section: memory behavior (PH3C)

## What was measured
Lifecycle-instrumented builds (memprobe: checkpoints A–K + per-1000-doc
insertion curves + disk class sizes + smaps anon/file) across a controlled
matrix: corpus 5K–30K × dims 128–512 × 1–8 heads, all on the 2 GB sandbox,
disk-backed engine dirs, Async durability, engine-default HNSW
(M=16, efc=400, store_vectors=true).

## Answers (all scoped to this sandbox; canonical: results/memory-scaling.csv,
results/memory-components.csv; figures 1–3, 9)
1. **Where the wall is**: 3 heads × 512 → OK at 15K (1182 MB), OOM at 20K
   (last peak 1410 MB); 8 heads × 512 × 10K OOMs (5K completes, 960 MB).
2. **Where the memory goes**: the insertion loop (inline HNSW build) —
   ≈ 64 MB RSS per 1K docs ≈ 10.9× the raw vector rate; everything else
   (init, flush, BM25, gating model) is negligible. Build transient ≈
   128 MB above steady state; only ~82 MB released at engine drop
   (allocator retention).
3. **Scaling shape**: ≈ linear in heads (245–260 MB/head @512;
   125–155 @256), ≈ linear in dim over a ~130–160 MB fixed base, ≈ linear
   in corpus (10.9× raw rate). No nonlinear blowup at tested scale.
4. **Duplication**: on-disk footprint 1.4× raw; in-RSS ≥ 2× raw explained
   by vectors stored in BOTH HNSW and the document store; the remaining
   ≈ 6–8× is retained build workspace / allocator behavior that requires
   engine-side instrumentation to attribute exactly (open question;
   nothing invented — bounds only).
5. **Budget envelope**: max reproduced at dim 512: 1h/30K, 2h/20K,
   4h/10K, 8h/5K (heads×docs ≈ 30K head-docs).
6. **tmpfs mechanism**: unclean process death leaks engine dirs (Drop
   guards never run); on tmpfs the leak is RAM; harness now defaults to
   disk-backed dirs and never counts leaked temp files as DB size.

## Benchmark limitation (not hidden)
All statements hold for a 2 GB RAM / 2-CPU cgroup sandbox. They must not
be read as "AttentionDB cannot scale beyond 10–30K documents" — they
identify the memory model and the avoidable sources (duplication,
retained workspace) that the systems phase should address (§25:
baseline → change → identical benchmark → measured result).
