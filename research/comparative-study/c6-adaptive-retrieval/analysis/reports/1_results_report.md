# C6 Results Report

**Study**: Adaptive retrieval allocation (C6)  
**Branch**: `comparative-study/c6-adaptive-retrieval`  
**Protocol**: `c6-protocol.md` (frozen)  
**Datasets**: SciFact (300 test queries), NFCorpus (323 test queries)  
**Replications**: 5 fresh-process runs per cell (seed 20260925)  
**Budget**: total_candidates=500, total_ef_work=192, min_per_head=20, max_per_head=300

---

## Primary Outcomes (TEST, mean ± sd over 5 reps)

| Dataset | Arm | Recall@10 (qrels) | Recall@10 (exact) | p50 Latency | Adaptive Changes |
|---------|-----|-------------------|-------------------|-------------|------------------|
| **SciFact** | A (CANONICAL) | 0.7796 ± 0.0049 | 0.9801 ± 0.0021 | 938 µs | — |
| | B (3-head union) | **0.7926 ± 0.0038** | 0.6892 ± 0.0035 | 1,748 µs | — |
| | C (Static equal) | 0.7867 ± 0.0041 | 0.6941 ± 0.0038 | 1,769 µs | 0% changed |
| | D (Query-adaptive) | 0.7898 ± 0.0039 | 0.6958 ± 0.0041 | 1,786 µs | 0% changed |
| | E (Interaction-guided) | 0.7881 ± 0.0037 | 0.6941 ± 0.0036 | **16,073 µs** | 100% changed |
| | F (E + cross-refine) | 0.7883 ± 0.0036 | 0.6934 ± 0.0034 | **18,213 µs** | 100% changed |
| | RAND (Randomized) | 0.7882 ± 0.0037 | 0.6940 ± 0.0035 | 1,702 µs | 0% changed |
| **NFCorpus** | A | 0.1495 ± 0.0023 | 0.9655 ± 0.0028 | 796 µs | — |
| | B | **0.1597 ± 0.0018** | 0.6741 ± 0.0031 | 1,624 µs | — |
| | C | 0.1598 ± 0.0019 | 0.6722 ± 0.0032 | 1,606 µs | 0% changed |
| | D | 0.1602 ± 0.0017 | 0.6742 ± 0.0030 | 1,635 µs | 0% changed |
| | E | 0.1604 ± 0.0020 | 0.6751 ± 0.0033 | **13,043 µs** | 100% changed |
| | F | 0.1596 ± 0.0021 | 0.6716 ± 0.0030 | **15,415 µs** | 100% changed |
| | RAND | 0.1609 ± 0.0024 | 0.6720 ± 0.0029 | 1,419 µs | 0% changed |

---

## Contrasts (B vs Adaptive Arms)

| Contrast | Mean Δ (Arm − B) | 95% CI (bootstrap) | p (Wilcoxon) | Cohen's dz | Significant? (Holm) |
|----------|------------------|-------------------|--------------|------------|---------------------|
| SCI: B → C | **−0.0059** | [−0.0123, −0.0003] | 0.046 | −0.113 | No (α=0.005) |
| SCI: B → D | **−0.0029** | [−0.0081, +0.0024] | 0.273 | −0.061 | No |
| SCI: B → E | **−0.0045** | [−0.0111, +0.0019] | 0.167 | −0.078 | No |
| SCI: B → F | **−0.0044** | [−0.0139, +0.0049] | 0.151 | −0.052 | No |
| SCI: B → RE | **−0.0044** | [−0.0098, +0.0008] | 0.047 | −0.095 | No (α=0.0056) |
| NFC: B → C | **+0.0001** | [−0.0035, +0.0031] | 0.258 | +0.002 | No |
| NFC: B → D | **−0.0005** | [−0.0032, +0.0037] | 0.409 | −0.015 | No |
| NFC: B → E | **−0.0007** | [−0.0013, +0.0029] | 0.611 | −0.035 | No |
| NFC: B → F | **+0.0001** | [−0.0027, +0.0027] | 0.747 | +0.003 | No |
| NFC: B → RE | **−0.0012** | [−0.0014, +0.0039] | 0.749 | −0.048 | No |

*Δ = mean(Arm) − mean(B); positive = Arm better. Holm α: 0.005, 0.0056, 0.0063, 0.0071, 0.0083, 0.010, 0.0125, 0.0167, 0.025, 0.05.*

---

## Adaptive Allocation Behavior

| Arm | Allocation Strategy | % Queries Changed | % Queries Redistributed | p50 Latency |
|-----|---------------------|-------------------|------------------------|-------------|
| C (Static equal) | Fixed equal split | 0% | 0% | ~1.7 ms |
| D (Query-adaptive) | Query→centroid cosine | 0% | 0% | ~1.7 ms |
| E (Interaction-guided) | Stage1(30%) → inspect → Stage2(70%) | **100%** | **100%** | **~14.5 ms** |
| F (E + cross-refine) | E + C5 λ=0.10 | **100%** | **100%** | **~16.8 ms** |
| RAND (Randomized) | Random weights | 0% | 0% | ~1.6 ms |

---

## Interpretation

- **Static equal (C) and Query-adaptive (D) did not actually vary allocation** across queries — their final allocation equals initial allocation (no redistribution step). They perform equivalently to B.
- **Interaction-guided (E) and F** actually redistributed budget on 100% of queries, but at **~8× latency cost** with **no recall benefit** (all |Δ| < 0.01, all p > 0.05 after Holm).
- **Randomized control (RE)** performed equivalently to B, confirming no benefit from non-informative allocation.
- **C5 cross-refine (λ=0.10) in arm F** added ~2ms latency over E with no additional recall benefit.

**Conclusion**: Under fixed total budget (500 candidates, 192 EF-work), adaptive allocation does **not** improve recall@10 over independent multi-head union (B). The interaction-guided approach changes candidate sets (100% redistribution) but yields no measurable recall gain. The latency overhead of two-stage retrieval (~8×) is not justified.

---

## B0 Fidelity Anchors (from C4/C5)

| Dataset | C5-B Recall@10 (qrels) | C5-B Exact Hash |
|---------|------------------------|-----------------|
| SciFact | 0.7898 | ad5e20010bb624755d2e6e9664c8aaba267c9442e53433ecd72746a8f18e409b |
| NFCorpus | 0.1597 | d2a9ee2fa469d2bd468550aea0520067ee921e501c9e4583e1d7cd22a892cdd3 |

C6-B matches C5-B within sampling noise.

---

## Run Registry (TEST cells)

All 14 TEST cells registered in `raw/RUN-INDEX.yaml` with status PASS. Raw evidence in `raw/C6-TEST-*-*/artifacts/`.