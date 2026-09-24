# Comparative Benchmarking Study — Research Namespace

Initiative: research-grade comparative benchmarking of AttentionDB against
established database/retrieval systems. Separately scoped from (and posterior
to) the closed Phase 3E program; Phase 3E evidence is immutable and is NOT
comparative evidence.

- Entry commit: `fe4f92b2187bf631125e975a5db8a202b5316d00` (main @ E11 close)
- Tree sha16 at entry: `d12fa093bc294977` (established procedure)
- Working tree at entry: clean
- Initiative branch: `comparative-study/c0-audit`

## Structure (per initiative charter §10)

```
research/comparative-study/
├── README.md                  ← this file
├── repository-audit/          ← STAGE C0 (complete)
│   ├── architecture-audit.md
│   ├── retrieval-path.md
│   ├── storage-path.md
│   ├── baseline-readiness.md
│   └── known-limitations.md
├── methodology/               ← STAGE C1 (not started)
├── baselines/ datasets/ workloads/ experiments/
├── raw/                       ← NEW registry (never inside research/phase3/)
├── results/ tables/ figures/ findings/ limitations/ reports/ scripts/
```

Rules: raw runs immutable; reruns get new IDs; every claim traces
CLAIM → FINDING → ANALYSIS → TABLE → RAW RUNS → CONFIG → DATASET HASH →
COMMIT; no Phase 3E directory is modified; no fabrication; unfavorable
results preserved.

## Stage status

- [x] **C0 — Repository & Baseline Audit** (this commit; see
  `repository-audit/*.md`)
- [x] **C1 — Methodology freeze + preregistration** (protocol v1.0.0; 14 documents under `c1/`; validation PASS)
- [ ] C2 — Harness + baseline validation (blocked items carried: BLK-1..4)
- [ ] C3 — Core retrieval ablation (B0–B4)
- [ ] C4 — External system comparisons (feasibility matrix in
  `repository-audit/baseline-readiness.md`)
- [ ] C5–C9 — workload/scale/statistics/final report
