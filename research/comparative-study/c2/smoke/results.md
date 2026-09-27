# C2 External Baseline Smoke Results (§14–§19)

**Status vocabulary**: exactly one of PASS / FAILED / BLOCKED / ABORTED per run; statuses are
never reclassified afterwards; failed attempts stay visible. Nothing in this document is a
comparative/benchmark result — these are feasibility observations only.

## Summary table (terminal runs)

| System | Terminal run | Status | Headline observation |
|---|---|---|---|
| Qdrant 1.12.4 | C2-SMOKE-QDRANT-006 | **PASS** | end-to-end vector CRUD + exact self-top-1; **named vectors validated** (B6 config feasible) |
| pgvector 0.8.0 | C2-SMOKE-PGVECTOR-005 | **PASS** | distro-repo install; extension + HNSW + IVFFlat; exact `<=>` self-top-1 |
| Elasticsearch 8.15.2 | C2-SMOKE-ES-003 | **ABORTED** | OBSERVED-LIMIT: default-JVM tree RSS 1.10 GiB ≥ 85%-MemAvailable budget 1.04 GiB |
| milvus-lite 3.2.1 | C2-SMOKE-MILVUSLITE-002 | **PASS** | pymilvus MilvusClient over local `.db`; create/insert/search OK |
| Weaviate 1.39.6 | C2-SMOKE-WEAVIATE-004 | **PASS** | ready @ 5.5 s; schema/object/nearVector 200; peak RSS 107 MiB |
| Pinecone | adapters/pinecone.md | **BLOCKED-AUTH** | no credentials; never requested (C1, unchanged) |
| MongoDB Atlas Vector | adapters/mongodb.md | **BLOCKED-AUTH** | no credentials; never requested (C1, unchanged) |

## Details worth preserving

### Elasticsearch — the OBSERVED-LIMIT is the result
The preregistered expectation was a guardrail abort at this host's envelope, and it fired with
**full evidence** this time: the server itself ran under `guarded_popen` (500 ms tree-RSS
sampler); at ~10 s into JVM startup the tree reached **1,149,344 KB ≥ 1,089,169 KB** budget
(85 % of MemAvailable); the sampler SIGKILL'd the child only; 20 sampler rows + ES logs are
preserved in the run dir. `vm.max_map_count=65530 < 262144` was additionally recorded — a
second, independent host constraint that production-mode ES would also hit.
Per prereg: **no reduced-heap retry**; the earlier `-002` metrics note announcing one is
**RETRACTED** in the registry. C1's CONDITIONAL-LOW classification is unchanged: ES remains
feasible only on a larger envelope, which this host is not.

Earlier ES attempts are preserved as evidence of harness defects, not engine behavior:
`-001` hit the 993 MiB `/tmp` tmpfs during extraction (OSError 28); `-002` ran the server
outside the sampler with a stray `chown nobody`, producing an **unsubstantiated**
OBSERVED-LIMIT label — the label was retracted and the attempt superseded rather than cited.

### Qdrant — named-vector validation (B6 adjacency)
Beyond the basic smoke, a 2×384-d named-vector collection (`HEAD-TITLE`/`HEAD-BODY`) was
created, populated, and queried successfully with exact self-top-1. This confirms the
**feasibility of the B6 proposed Qdrant named-vector/prefetch configuration**. It is a
configuration proposal, **not** an AttentionDB adapter, and carries no benchmark semantics.

### pgvector — BLK-3 resolved without PGDG
`postgresql-17-pgvector 0.8.0-1` exists in the **Debian distro repos**; no third-party repo
was needed. Extension, both index types (HNSW, IVFFlat), and exact cosine `<?>` self-query
all succeeded. Limitation recorded: PostgreSQL runs under `pg_ctlcluster` as a system service,
so the per-process tree sampler does not apply — resource evidence is preflight-only.

### Weaviate — documented env deltas, not behavior changes
v1.39's raft clustering requires an advertise address; this sandbox has no private IP, so the
smoke pinned `CLUSTER_ADVERTISE_ADDR=127.0.0.1` (+ gossip/data ports) and forced
`--scheme http --port 8080` (the embedded swagger default otherwise demands TLS certificates).
Every delta is recorded verbatim in the run's `config.yaml`. All deltas are startup
configuration, not engine modifications.

### milvus-lite — client package reality
`milvus-lite` 3.x ships no `milvus` module; the documented client is
`pip install pymilvus[milvus-lite]` → `MilvusClient("<path>.db")`. (Sandbox PyPI egress was
available this session; the earlier "no pip network" note is obsolete for this host.)
Milvus-standalone remains **EXCLUDED** per C1.

## Superseded-attempt ledger (never hidden)

| System | Run | Status | One-line cause |
|---|---|---|---|
| qdrant | -001 | PASS | valid ops, but no sampler evidence (pre-rebuild) |
| qdrant | -002 | INVALID-STARTUP | harness fake-thread verdict defect |
| qdrant | -003 | INVALID-STARTUP | `rest()` raised on HTTP 400 instead of recording |
| qdrant | -004/-005 | FAILED | named-search payload shape (`NamedVectorStruct`) |
| pgvector | -001 | (no run) | module SyntaxError pre-start |
| pgvector | -002 | FAILED | unprivileged apt (rc=100) + version-glob IndexError |
| pgvector | -003 | FAILED | `su postgres` auth failure (fixed → `sudo -u postgres`) |
| pgvector | -004 | PASS | superseded by -005 (unsudoed stop, dead code path) |
| ES | -001 | FAILED | OSError 28: /tmp tmpfs too small for 1.5 GiB extraction |
| ES | -002 | ABORTED | evidence hollow (server outside sampler) → superseded |
| milvus-lite | -001 | FAILED | `from milvus import MilvusClient` (3.x removed module) |
| weaviate | -002 | INVALID-STARTUP | memberlist advertise addr + swagger TLS/port defaults |
| weaviate | -003 | FAILED | `json.loads` on truncated `/v1/meta` |

## What these results authorize

- **Feasible (smoke-level)**: qdrant, pgvector, milvus-lite, weaviate → readiness reports may
  be written as IMPLEMENTED/VALIDATED smokes; none are BENCHMARK-READY by this alone.
- **Resource-bounded**: Elasticsearch → ABORTED/OBSERVED-LIMIT is terminal for this host;
  any future ES work requires a larger-envelope host (documented, not attempted).
- **Blocked**: pinecone, mongodb (auth) — documented, unchanged.
- **Next ladder discipline** (if/when used): advance 10k→25k→40k→60k→80k only with ≥20 % RAM
  headroom at the previous rung; one external server at a time; all guardrails as preregistered.
