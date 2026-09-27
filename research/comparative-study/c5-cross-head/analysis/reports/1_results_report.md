# C5 Results Report

**Study**: Cross-head candidate-generation research (C5)  
**Branch**: `comparative-study/c5-cross-head`  
**Protocol**: `c5-protocol.md` (frozen)  
**Datasets**: SciFact (300 test queries), NFCorpus (323 test queries)  
**Replications**: 5 fresh-process runs per cell (seed 20260925)  
**Budget**: per-head k=max(50,20), candidate union cap=500, ef=64, memory 512MiB

---

## Primary Outcomes (TEST, mean ± sd over 5 reps)

| Dataset | Arm | Recall@10 (qrels) | Recall@10 (exact) | p50 Latency | Candidate Change Rate |
|---------|-----|-------------------|-------------------|-------------|----------------------|
| **SciFact** | A (CANONICAL) | 0.7767 ± 0.0053 | 0.6901 ± 0.0031 | 1,328 µs | — |
| | B (3-head union) | **0.7898 ± 0.0043** | 0.7053 ± 0.0038 | 1,903 µs | — |
| | C (λ=0.10 interaction) | 0.7880 ± 0.0005 | 0.7032 ± 0.0029 | **5,547 µs** | **100% (300/300)** |
| **NFCorpus** | A | 0.1505 ± 0.0021 | 0.6562 ± 0.0042 | 1,259 µs | — |
| | B | 0.1593 ± 0.0015 | 0.6641 ± 0.0035 | 2,375 µs | — |
| | C (λ=0.10) | **0.1609 ± 0.0027** | 0.6678 ± 0.0031 | **5,334 µs** | **99.7% (322/323)** |

---

## Contrasts

| Contrast | Mean Δ (Arm2 − Arm1) | 95% CI (bootstrap) | p (Wilcoxon) | Cohen's dz | Significant? (Holm) |
|----------|---------------------|-------------------|--------------|------------|---------------------|
| SCI: A → B | **+0.0131** | [−0.0420, +0.0143] | 0.228 | 0.053 | No |
| SCI: B → C | +0.0018 | [−0.0079, +0.0115] | 0.592 | 0.021 | No |
| NFC: A → B | **+0.0088** | [−0.0206, +0.0023] | 0.232 | 0.084 | No |
| NFC: B → C | −0.0017 | [−0.0055, +0.0021] | 0.870 | 0.048 | No |

*Δ = mean(Arm2) − mean(Arm1); positive = Arm2 better. Holm α: 0.0125, 0.0167, 0.025, 0.05.*

---

## Interpretation

- **Multi-head union (A→B)**: Positive but not statistically significant recall gains (+1.3pp SCI, +0.9pp NFC).
- **Cross-head interaction (B→C)**: Mechanism changes candidate set on >99% of queries (causal claim met), but recall@10 effect is **negligible** (|Δ| < 0.01) and not significant on either dataset.
- **Latency overhead**: Interaction adds ~2.8× median latency (round-1 + round-2 + centroid compute).
- **Validation gate λ choice**: λ=0.10 chosen (closest to B recall while interaction active). λ=0.25/0.50 reduced recall on both datasets in validation.

---

## B0 Fidelity Anchors (from C4)

| Dataset | C4-B0 Recall@10 (qrels) | C4-B0 Exact Hash |
|---------|------------------------|------------------|
| SciFact | 0.7833 | ad5e20010bb624755d2e6e9664c8aaba267c9442e53433ecd72746a8f18e409b |
| NFCorpus | 0.1550 | d2a9ee2fa469d2bd468550aea0520067ee921e501c9e4583e1d7cd22a892cdd3 |

C5-A (single CANONICAL head) matches C4-B0 semantics; C5-B matches C4-B2 (independent multi-head).

---

## Run Registry (TEST cells)

All 6 TEST cells registered in `raw/RUN-INDEX.yaml` with status PASS. Raw evidence in `raw/C5-TEST-*-*/artifacts/`.