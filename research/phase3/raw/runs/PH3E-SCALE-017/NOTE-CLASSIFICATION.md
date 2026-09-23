# PH3E-SCALE-017 — classification note (harness gate gap; numbers valid)

The run's summary says VERIFIED, but the frozen M6(e) retrieval gate (self-hit
>= 95% at k=10) was NOT applied by the classification logic at the time — the
measured self-hit is 93/100, below the gate. The measurements (p50 428 us,
p95 817 us, p99 1428 us, QPS 2,100, 1,000-query batch at 80k docs) are valid;
the correct classification per the frozen spec is OBSERVED LIMIT (grown-graph
recall boundary near the hnsw_rs max_elements=100k cap; SCALE-001 shows
reopen-rebuild restores recall to 99-100/100). Re-run with the complete
gate-honoring classification as PH3E-SCALE-021. This note, not an edit of the
original summary, preserves the audit trail.
