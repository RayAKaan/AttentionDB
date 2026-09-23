# PH3E-SCALE-014 — INVALIDATED (harness model defect; engine exonerated)

Mutation-churn run (260k ops over a 60k base). The harness's "reinsert" branch
called insert_new() for idx values that were ALREADY LIVE, minting a second
live uuid per idx; the engine faithfully stored BOTH versions while the
lock-step model tracked only the newest — final count check flagged
engine=81,419 vs model=48,148. Diagnosis: a minimal probe proved scan_filtered
correctly excludes deleted documents pre- and post-checkpoint, and every
model-tracked doc was present and exact (5,350/5,350 samples, checker clean).
The defect is the harness's upsert semantics, not the engine. Preserved per
the failure protocol; rerun with corrected upsert semantics as
PH3E-SCALE-018. The measured hygiene behavior (vstore stayed ≈ live despite
200k mutations; peak RSS 877,184 KB) remains directionally informative but
the run's classification is INVALIDATED.
