# MongoDB Atlas — BLOCKED-AUTH (external authorization blocker)

Status: **BLOCKED-AUTH** (unchanged from C1 baseline-feasibility.csv, BLK-1).

## What this means

- Atlas requires a cloud account and incurred cost; none exists, none was
  requested, nothing was transmitted (C2 prompt §19). A local mongod with
  Atlas-style vectorSearch is NOT a substitute baseline (different engine
  build and search implementation) and was not presented as one.

## What C2 would need if authorized later

1. Atlas M10+ cluster (vectorSearch available), region + budget declared.
2. Same byte-identical Track A embedding exports (fairness rule).
3. `$vectorSearch` smoke: index definition (384 dims, cosine), collection
   insert, k-neighbor query — mirroring the Qdrant smoke shape.
4. Data-egress disclosure in the run manifest.

Local work (C3+) does not depend on Atlas.
