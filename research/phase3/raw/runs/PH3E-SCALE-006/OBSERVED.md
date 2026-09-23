# PH3E-SCALE-006 — OBSERVED LIMIT (OOM during build, ~100k docs, slim config)

Slim-payload configuration, 120k target. Pre-flight M3 gate passed on an
UNMEasured per-doc marginal (6.0 KB estimate) — a methodology violation
caught by nature: the process was OOM-killed during the build at ~100k docs
(last hnsw_rs progress marker: 100,000 points; telemetry was lost because it
was buffered until run end — harness fixed afterward to flush incrementally).
The db dir (build state at the kill point) is preserved. Per spec M5 this is
a resource boundary of the slim configuration, not an engine bug; the
per-doc marginal for slim payloads is measured properly in PH3E-SCALE-007
before any higher tier is attempted (M3 with MEASURED marginal only).
