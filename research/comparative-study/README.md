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
├── c1/                        ← STAGE C1 (complete; protocol v1.0.0)
│   ├── c1-protocol.md
│   ├── c1-architecture.md
│   ├── c1-baseline-config.md
│   ├── c1-run-plan.csv
│   ├── c1-dataset-hashes.md
│   └── ... (14 docs total)
├── c2-embed/                  ← STAGE C2 (embedding pipeline; complete)
├── c3-embed/                  ← STAGE C3 (SciFact embed; complete)
├── c4/                        ← STAGE C4 (multi-head ablation; complete)
│   ├── c4-protocol.md
│   ├── c4-architecture.md
│   ├── c4-run-plan.csv
│   ├── harness/
│   └── analysis/
├── c5-cross-head/             ← STAGE C5 (cross-head interaction; COMPLETE)
│   ├── c5-protocol.md
│   ├── c5-architecture.md
│   ├── c5-run-plan.csv
│   ├── probe/
│   ├── harness/
│   ├── analysis/
│   │   ├── statistical_results.json
│   │   └── reports/ (10 reports)
│   └── c5-test_run.py
├── c6-adaptive-retrieval/     ← STAGE C6 (adaptive allocation; COMPLETE)
│   ├── c6-protocol.md
│   ├── c6-architecture.md
│   ├── c6-run-plan.csv
│   ├── probe/
│   ├── harness/
│   ├── analysis/
│   │   ├── statistical_results.json
│   │   └── reports/ (9 reports)
│   └── c6_test_run.py
├── methodology/               ← STAGE C1 docs
├── baselines/ datasets/ workloads/ experiments/
├── raw/                       ← Immutable run registry (C0–C6)
├── results/ tables/ figures/ findings/ limitations/ reports/ scripts/
```

Rules: raw runs immutable; reruns get new IDs; every claim traces
CLAIM → FINDING → ANALYSIS → TABLE → RAW RUNS → CONFIG → DATASET HASH →
COMMIT; no Phase 3E directory is modified; no fabrication; unfavorable
results preserved.

## Stage status

- [x] **C0 — Repository & Baseline Audit** (see `repository-audit/*.md`)
- [x] **C1 — Methodology freeze + preregistration** (protocol v1.0.0; 14 docs under `c1/`; validation PASS)
- [x] **C2 — Embedding pipeline** (C2-EMBED-NFCORPUS-003, C3-EMBED-SCIFACT-001 complete)
- [x] **C3 — SciFact embedding** (frozen artifacts)
- [x] **C4 — Multi-head ablation** (B0–B2 complete; C4-B2 = C5-B baseline)
- [x] **C5 — Cross-head candidate-generation** (COMPLETE: mechanism implemented, TEST executed, neutral result)
  - Protocol: `c5-cross-head/c5-protocol.md`
  - Engine: `core/src/retrieval.rs` + `collection.rs` (50 tests pass)
  - Probe: `c5-cross-head/probe/c5pilot.rs` (reproducible `/Brepro` build)
  - Primary result: interaction changes candidate set on >99% queries, but recall@10 Δ negligible (|Δ|<0.01, p>0.5 both datasets)
  - Reports: 10 reports in `c5-cross-head/analysis/reports/`
- [x] **C6 — Adaptive retrieval allocation** (COMPLETE: mechanism implemented, TEST executed, negative result)
  - Protocol: `c6-adaptive-retrieval/c6-protocol.md`
  - Engine: `core/src/adaptive.rs` + `collection.rs` (61 tests pass)
  - Probe: `c6-adaptive-retrieval/probe/c6pilot.rs` (reproducible `/Brepro` build)
  - Primary result: interaction-guided allocation changes candidate set on 100% queries, but recall@10 Δ negligible (|Δ|<0.01, p>0.05 both datasets); 8× latency overhead not justified
  - Reports: 9 reports in `c6-adaptive-retrieval/analysis/reports/`
- [x] **C7 — Genuine candidate-level Q/K/V attention** (COMPLETE: mechanism implemented, TEST executed, negative result)
  - Protocol: `c7-genuine-qkv-attention/c7-protocol.md`
  - Engine: `core/src/retrieval.rs` + `collection.rs` (candidate-level attention); `attention/` crate
  - Probe: `c7-genuine-qkv-attention/probe/`
  - Primary result: learned candidate-level QKV (E) significantly degrades nDCG@10 vs fixed fusion (B) on both datasets (SCI Δ +0.063, NFC Δ +0.093); identity arms C/D equivalent to B
  - Reports: 10 reports in `c7-genuine-qkv-attention/analysis/reports/`
- [ ] **C8 — Residual candidate-level Q/K/V attention** (IN PROGRESS: mechanism implemented; VALIDATION/SMOKE/EFPROBE verified; TRK-A TEST + SUPPORT pending Ubuntu CI)
  - Protocol: `c8-residual-qkv-attention/c8-protocol.md` v1.0.0 (9 arms A–I; 14 primary contrasts)
  - Plan: `c8-residual-qkv-attention/c8-run-plan.csv` (66 cells × 22 cols)
  - Engine: `attention/` (C8 modules) + `core/src/collection.rs` stage 8b (`S_final = S_base + λ·ΔS_attention`)
  - Probe/harness: `c8-residual-qkv-attention/probe/` + `harness/c8_{test_run,verify,analyze}.py`
  - Verified: λ=0 parity C==B bit-exact, cache parity I==E bit-exact, union identity 0 mismatches; `c8_verify.py` → `all_pass: true`
  - Reports: 10 reports in `c8-residual-qkv-attention/analysis/reports/` (TEST statistics pending CI)
- [x] **C9 — Efficient attention serving implementation** (merged to main; correctness/benchmark protocol registered, authoritative performance evidence pending)
  - Engine: `attention/src/qkv.rs` query projection reuse; `attention/src/c9_benchmark.rs` scalar-vs-batch microbenchmark
  - Protocol: `c9-efficient-attention/PLAN.md`; no speedup claim until measured on a named runner
- [x] **C10 — Cache lifecycle and projection-cost benchmark** (implementation merged; authoritative measurements pending)
  - Protocol: `c10-cache-lifecycle/PLAN.md`
  - Engine: `attention/src/c10_cache_benchmark.rs` configurable deterministic Cartesian sweep; uncached projection vs cold fill vs warm lookup; exact K/V parity, cache accounting, p50/p95, payload estimate
- [ ] **C11 — End-to-end workload evaluation and paper-ready evidence** (evidence tooling and CI checks implemented; authoritative TEST runs and paper-ready empirical conclusions still pending)
  - Protocol/run matrix: `c11-end-to-end-evaluation/PLAN.md` and `c11-run-plan.csv` (2 datasets × 9 arms; 5 repetitions per cell)
  - Aggregator: `c11-end-to-end-evaluation/c11_analyze.py` produces JSON, CSV, paired query bootstrap intervals where available, provenance hashes, and a Markdown report
  - Validation/tests: `c11-end-to-end-evaluation/c11_validate.py` and `test_c11_evidence.py`; wired into CI
  - C11 uses C8 raw per-query artifacts and keeps C10 cache microbenchmarks separate from end-to-end latency
  - No empirical completion claim until all 18 TEST cells, five repetitions, correctness gates, and provenance checks pass
