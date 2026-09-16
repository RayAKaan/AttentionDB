# PH3C memory findings (PROVISIONAL — §24/§33 discipline)

Statuses: SUPPORTED / NOT SUPPORTED / OPEN QUESTION. All claims are scoped
to "the 2 GB / 2-CPU sandbox and the tested configurations"; none of these
may be generalized into claims about AttentionDB at large.

## M-1 (SUPPORTED, scope: tested configs on this sandbox) — The memory wall is a build-time phenomenon driven by the insertion loop, not by any single component

Lifecycle checkpoints (PH3C-MEM-001, 3 heads × 512 × 10K): engine init,
collection creation, flush, BM25 query, and gating-model load are each
memory-negligible; the ENTIRE jump happens during document insertion
(inline HNSW construction), at ≈ 64 MB RSS per 1,000 docs — ≈ 10.9× the
raw vector rate (5.9 MB/1K docs). Peak (859.5 MB) exceeds steady state
(649.7 MB) by ~128 MB of build transient. The wall on this sandbox sits
between 15K docs (1182 MB, OK) and 20K docs (OOM, last peak 1410 MB) at
3 heads × 512.

## M-2 (SUPPORTED, scope: 10K–30K docs, dims 128–512, 1–8 heads) — Memory is approximately linear in head count, corpus size, and dimension, with a fixed base

- Head count @10K×512: 351 / 596 / 860 / 1092 MB for 1/2/3/4 heads
  (≈ 245–260 MB/head; 8 heads @10K OOMs, @5K completes at 960 MB).
- Dimension @10K×3h: 361 / 532 / 860 MB for 128/256/512 (≈ linear in dim
  over a ~130–160 MB fixed base).
- Corpus @3h×512: 507 / 860 / 1182 MB for 5K/10K/15K (≈ 10.9× raw rate).
- No nonlinear blowup component detected at the tested scale; the
  multiplier (12–18× raw) falls with scale/fixed-base dilution.

## M-3 (SUPPORTED, scope: measured from outside the process) — Vector duplication and allocator-retained workspace are the dominant avoidable sources; exact per-component split needs engine instrumentation

Evidence: logical raw vectors at 10K×3h×512 = 59 MB; steady engine-attributable
RSS ≈ 627 MB (≈ 10.6×); on-disk engine footprint only 84 MB (1.4×); ≥ 2× of
the RSS is explained by vectors living simultaneously in HNSW vector storage
(store_vectors=true) and the document store records (k_vecs); a further
build transient of ~128 MB appears at insertion time and only ~82 MB of RSS
is released at engine drop (allocator retention). The remaining ≈ 6–8×
resists attribution from outside the process (allocator fragmentation,
memtable/WAL buffering, HNSW build scratch) — flagged as an OPEN QUESTION
with the instrumented-checkpoint evidence attached. No invented component
values are reported (§3).

## M-4 (SUPPORTED, scope: this sandbox) — Maximum reproduced corpus sizes (dim 512, Async durability, disk-backed engine dir): 1 head 30K, 2 heads 20K, 4 heads 10K, 8 heads 5K

The product heads × docs ≈ 30K head-docs is the empirical envelope at dim
512 (941 / 1152 / 1092 / 960 MB peaks respectively). NOT a claim about
maximum supported corpus size in general (§12).

## M-5 (SUPPORTED, scope: verified mechanism) — SIGKILLed/uncleanly-exited runs leak their engine directories; on tmpfs that leak consumes RAM

PH3C-MEM-003: clean drop removes the dir; std::process::exit (no
destructors) and real SIGKILL leave it behind (10,547 B at toy scale,
∝ corpus at real scale — 632 MB found in Phase 3B from two killed
builds). Cleanup mechanism = Rust Drop guards; mitigation (already in
place): harness engine dirs default to DISK (/var/tmp); leaks are never
counted in database-size measurements (§11).

## OPEN QUESTIONS

- Exact in-engine component split (HNSW graph vs workspace vs memtable vs
  allocator retention) requires engine-side instrumentation (allocator
  hooks or component-aware allocation). The outside-process checkpoints
  bound the answers but do not separate them.
- Whether releasing HNSW build workspace and deduplicating vector copies
  (§25 candidates) recovers the 10.6× → target ~3–4×; NOT attempted yet
  (measure-first rule; a small baseline→change→measure experiment is the
  designated next step).
- Behavior at dims/corpora beyond the tested envelope (Tier L remains
  infeasible on this hardware).
