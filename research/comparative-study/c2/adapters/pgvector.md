# pgvector — Readiness Report

Study `comparative-study-001` · §15 (smoke-level feasibility).

## Status

**SMOKE-PASS (feasibility: IMPLEMENTED/VALIDATED).** Not BENCHMARK-READY alone.

## Evidence — C2-SMOKE-PGVECTOR-005 (terminal, PASS)

- install source: `postgresql-17-pgvector 0.8.0-1` from the **Debian distro
  repos** — no PGDG repo required (BLK-3 resolved on this host)
- `CREATE EXTENSION vector` succeeded; HNSW and IVFFlat indexes created
- exact cosine `<=>` vs ANN query returned exact self-top-1
- cluster start/stop clean under `sudo -u postgres` (supersedes -004's unsudoed
  stop dead-code path)
- superseded ledger: -001 (module SyntaxError pre-start), -002 FAILED
  (unprivileged apt rc=100 + version-glob IndexError), -003 FAILED (su auth),
  -004 PASS (superseded: unsudoed stop)

## Limitation

PostgreSQL 17 runs under `pg_ctlcluster` as a **system service** — the
per-process tree sampler does not apply; resource evidence is preflight
`environment.yaml` only. This is disclosed, not a defect to hide.

## Verdict

Feasible on this host as a benchmark target for a later phase. Single-vector
schema is a C1 constraint; multi-head/prefetch mapping is NOT in scope for
pgvector in C1 (B6 is Qdrant-specific).