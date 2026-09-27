# Pinecone — BLOCKED-AUTH (external authorization blocker)

Status: **BLOCKED-AUTH** (unchanged from C1 baseline-feasibility.csv, BLK-1).

## What this means

- No account exists, no credentials were requested, no dataset was
  transmitted, no charge was incurred (charter + C2 prompt §19).
- Feasibility CANNOT be assessed from here: it requires the study owner to
  explicitly authorize (a) creating/using an account, (b) a budget, and
  (c) transmitting the study's embedding exports to a managed service.

## What C2 would need if authorized later (recorded for completeness)

1. Serverless or pod-based index, cosine, 384 dims (pinned MiniLM exports).
2. The same byte-identical embedding exports as Track A (fairness rule).
3. Client: `pinecone` python client; upsert + query smoke identical in
   shape to the Qdrant smoke (C2-SMOKE-QDRANT-001) before any benchmarking.
4. Data-egress disclosure in the run manifest (dataset hashes transmitted).

Nothing in the study depends on Pinecone for C2/C3 local work.
