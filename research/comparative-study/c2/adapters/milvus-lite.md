# Milvus-Lite — Readiness Report

Study `comparative-study-001` · §19 (feasibility for milvus-LITE only).

## Status

**SMOKE-PASS (feasibility: IMPLEMENTED/VALIDATED).** Milvus-**standalone**
remains **EXCLUDED per C1** (unchanged). Not BENCHMARK-READY alone.

## Evidence — C2-SMOKE-MILVUSLITE-002 (terminal, PASS)

- install: `pip install pymilvus[milvus-lite]` (sandbox PyPI egress available
  this session); resolved `milvus-lite 3.2.1`
- documented client reality: `milvus-lite` 3.x ships **no top-level `milvus`
  module**; correct path = `pymilvus[milvus-lite]` → `MilvusClient("<path>.db")`
- `create_collection`, `insert`, `search` all succeeded (5/5 hits) over a local
  `.db` file
- superseded ledger: -001 FAILED (`from milvus import MilvusClient` — 3.x
  removed the module)

## Scope note

Covers millivus-LITE only. In-process `MilvusClient` semantics differ from a
server; nothing here implies anything about milvus standalone, which is
out of scope for this study.