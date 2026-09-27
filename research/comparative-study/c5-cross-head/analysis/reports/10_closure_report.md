# C5 Closure Report

**Branch**: `comparative-study/c5-cross-head`  
**Protocol**: Frozen (`c5-protocol.md`)  
**Status**: **COMPLETE** — all phases executed, all reports generated

---

## Phase Completion Checklist

| Phase | Protocol § | Status | Evidence |
|-------|------------|--------|----------|
| Audit & Protocol Freeze | §1–§2 | ✅ | `c5-protocol.md` (SHA256 recorded) |
| Mechanism Implementation | §5 | ✅ | `core/src/retrieval.rs`, `collection.rs`; 50 tests pass |
| Unit Tests | §5 | ✅ | 5 cross-refine tests + 45 existing = 50 pass |
| Probe Harness | §6 | ✅ | `probe/c5pilot.rs`; reproducible build |
| EFPROBE (ef knob) | §9 | ✅ | 4 cells; ef real (latency +30%, NFC recall moves) |
| SMOKE (sanity) | §6 | ✅ | 8 cells; A/B/C/glue on both datasets |
| VALIDATION (λ gate) | §8 | ✅ | 10 cells; λ=0.10 chosen; mechanism NOT inert |
| TEST (primary) | §7, §10 | ✅ | 6 cells × 5 reps; all registered PASS |
| Statistical Analysis | §10 | ✅ | Bootstrap/Wilcoxon/Holm/dz; candidate analysis |
| Reports (10) | §10 | ✅ | `analysis/reports/1–10_*.md` |
| README Update | §10 | 🔄 | Pending (this report triggers it) |
| Commit & Push | — | 🔄 | Pending |
| PR (no auto-merge) | — | 🔄 | Pending |

---

## Primary Scientific Conclusion

> **The cross-head candidate-generation interaction (C5-C) changes the candidate set on >99% of queries (causal mechanism operates), but produces no statistically significant or practically meaningful improvement in recall@10 on either SciFact (+0.18pp, p=0.59) or NFCorpus (−0.17pp, p=0.87). The effect is negligible (|Δ| < 0.01) on both datasets.**

- **Causal claim**: ✅ SATISFIED (candidate set changes >99%)
- **Recall benefit**: ❌ NOT OBSERVED (neutral/negative, non-significant)
- **Cost**: ~2.8× latency overhead for zero recall gain

This is a **valid neutral result** per protocol: "neutral results are valid."

---

## Secondary Findings

1. **Multi-head union (A→B)** yields positive but non-significant recall gains (+1.3pp SCI, +0.9pp NFC), consistent with C4-B2.
2. **ef knob is real**: latency scales ~30% with ef; NFC recall sensitive.
3. **λ tuning**: λ=0.10 closest to B recall; λ≥0.25 degrades recall on both datasets.
4. **Glue arm (λ=0)**: Matches B exactly (0% candidate change, recall diff < 0.01).

---

## Artifacts Delivered

| Artifact | Location |
|----------|----------|
| Frozen protocol | `research/comparative-study/c5-cross-head/c5-protocol.md` |
| Run plan | `research/comparative-study/c5-cross-head/c5-run-plan.csv` |
| Engine implementation | `core/src/retrieval.rs`, `core/src/collection.rs` |
| Probe harness | `research/comparative-study/c5-cross-head/probe/` |
| Orchestrator | `research/comparative-study/c5-cross-head/harness/c5_test_run.py` |
| Analysis | `research/comparative-study/c5-cross-head/harness/c5_analyze.py` |
| Statistical results | `research/comparative-study/c5-cross-head/analysis/statistical_results.json` |
| Reports (10) | `research/comparative-study/c5-cross-head/analysis/reports/` |
| Raw evidence | `raw/C5-*/artifacts/` (23 runs) |
| Run index | `raw/RUN-INDEX.yaml` |

---

## Commit & PR

- **Commit message**: `C5: Cross-head candidate-generation research (neutral result)`
- **Branch push**: `comparative-study/c5-cross-head` (no force-push)
- **PR title**: `C5: Cross-head candidate-generation research`
- **PR body**: Links to all 10 reports; states neutral result; **no auto-merge**
- **CI**: Must pass Ubuntu full workspace before merge consideration

---

## Sign-Off

All protocol phases executed per frozen specification. No deviations.  
Neutral result reported faithfully. Ready for PR.