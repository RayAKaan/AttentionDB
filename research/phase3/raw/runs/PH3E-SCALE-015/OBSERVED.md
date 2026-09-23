# PH3E-SCALE-015 — OBSERVED LIMIT (run terminated at the harness wall-clock
# budget; configuration error in the experiment design)

The bounded-concurrency experiment was configured with a 100k-doc base + 30k
writer docs (130k total) — projected ~2.2 GB, violating the frozen M3 budget
discipline for this 1.9 GiB host (the writer loop also lacked the M4 live
guard). The outer harness wall-clock terminated it mid-run; partial artifacts
(config, incremental telemetry, db state) are preserved as-is. Redesigned to
a 40k base + 20k writer with in-loop guard checks and re-run as
PH3E-SCALE-019. Classification: OBSERVED LIMIT (resource/configuration
boundary; not an engine defect).
