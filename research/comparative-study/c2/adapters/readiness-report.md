# C2 Adapter Readiness — Index & Summary

Study `comparative-study-001`, protocol v1.0.0 (commit `7788067`).
Vocabulary (§35): **IMPLEMENTED ≠ VALIDATED ≠ FEASIBLE ≠ BENCHMARK-READY**.
A smoke is a **feasibility observation only**; no performance conclusions are
drawn in C2 and no failed smoke may be converted to READY.

## Terminal smoke results (see smoke/results.md for full ledger)

| System | Terminal run | Status | Feasibility at this envelope |
|---|---|---|---|
| Qdrant 1.12.4 | C2-SMOKE-QDRANT-006 | PASS | end-to-end CRUD + exact self-top-1; **named-vector collection validated** (B6 config feasible) |
| pgvector 0.8.0 | C2-SMOKE-PGVECTOR-005 | PASS | distro-repo install; extension + HNSW + IVFFlat; exact `<=>` self-top-1 |
| Elasticsearch 8.15.2 | C2-SMOKE-ES-003 | ABORTED | OBSERVED-LIMIT: default-JVM tree RSS 1.10 GiB ≥ 85%-MemAvailable 1.04 GiB; terminal for this host |
| milvus-lite 3.2.1 | C2-SMOKE-MILVUSLITE-002 | PASS | MilvusClient over local `.db`; create/insert/search OK |
| Weaviate 1.39.6 | C2-SMOKE-WEAVIATE-004 | PASS | ready @5.5 s; schema/object/nearVector 200; peak RSS 107 MiB |
| Pinecone | — | BLOCKED-AUTH | never requested (C1, unchanged) |
| MongoDB Atlas Vector | — | BLOCKED-AUTH | never requested (C1, unchanged) |
| Milvus standalone | — | EXCLUDED | C1 decision (unchanged) |
| AttentionDB engine | C2-MODES-TEST-001 | FAILED (5/8) | see c2-validation-report.md §G; findings are engine-documented, not envelope-bounded |

## What C2 authorizes (ladder discipline, from smoke/results.md)

- **Feasible (smoke-level)**: qdrant, pgvector, milvus-lite, weaviate → these
  reports are IMPLEMENTED/VALIDATED smokes; none is BENCHMARK-READY by this alone.
- **Resource-bounded**: Elasticsearch → ABORTED/OBSERVED-LIMIT is terminal for
  this host; future ES work needs a larger envelope (documented, not attempted).
- **Blocked**: pinecone, mongodb (auth) — documented, unchanged.
- Next ladder discipline (if used): 10k→25k→40k→60k→80k only with ≥20% RAM
  headroom at the previous rung; one external server at a time; all guardrails
  as preregistered.

## Per-system reports

- qdrant.md (B6 named-vector adjacency)
- pgvector.md
- elasticsearch.md
- milvus-lite.md
- weaviate.md
- pinecone.md, mongodb.md (BLOCKED-AUTH, unchanged)