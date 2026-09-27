# C2 Baseline Adapters — Readiness Index

Per-system reports in this directory. Vocabulary is the C1/C2 discipline
(§35): IMPLEMENTED ≠ VALIDATED ≠ FEASIBLE ≠ BENCHMARK-READY. A smoke result
is a feasibility observation only; no performance conclusions are drawn in
C2 and no failed smoke may be converted to READY.

| system | C2 status | report |
|---|---|---|
| AttentionDB | engine mode battery executed on the real engine (C2-MODES-TEST-001; 5/8 PASS, findings recorded — dense paths deterministic; B5-BM25 tie-order and 2 strict-tolerance score-mismatch observations are documented, not hidden) | (engine, see c2-validation-report.md §G + readiness-report.md) |
| Qdrant | see qdrant.md | qdrant.md |
| pgvector | see pgvector.md | pgvector.md |
| Elasticsearch | see elasticsearch.md | elasticsearch.md |
| Milvus-Lite | see milvus-lite.md | milvus-lite.md |
| Weaviate | see weaviate.md | weaviate.md |
| Pinecone | BLOCKED-AUTH | pinecone.md |
| MongoDB Atlas | BLOCKED-AUTH | mongodb.md |
| Milvus standalone | EXCLUDED (C1) | (unchanged) |

## B6 note (Qdrant named-vector/prefetch-fusion mapping)

C1 defines B6 as a PROPOSED Qdrant configuration (named vectors per head +
prefetch fusion), distinct from the three adapter files that do not compile
(C0). C2 verifies only that the mapping is technically executable against
the real Qdrant API and documents whether it is comparable to AttentionDB's
B2 model — see qdrant.md. It is NOT an AttentionDB adapter and is never
reported as one.
