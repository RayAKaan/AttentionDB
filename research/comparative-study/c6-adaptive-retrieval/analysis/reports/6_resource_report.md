# C6 Resource Utilization Report

**Budget Constraints (Per Protocol)**: Matched total candidate budget (500), total EF work (192), memory (512 MiB).

---

## Latency (p50, median over 5 reps × 300/323 queries)

| Arm | SciFact (µs) | NFCorpus (µs) | Overhead vs B |
|-----|-------------|---------------|---------------|
| A (CANONICAL) | 938 | 796 | −46% / −51% |
| B (3-head union) | 1,748 | 1,624 | baseline |
| C (Static equal) | 1,769 | 1,606 | +1% / −1% |
| D (Query-adaptive) | 1,786 | 1,635 | +2% / +1% |
| E (Interaction-guided) | **16,073** | **13,043** | **+819% / +703%** |
| F (E + cross-refine) | **18,213** | **15,415** | **+942% / +849%** |
| RAND (Randomized) | 1,702 | 1,419 | −3% / −13% |

E/F arms run two search rounds (Stage 1 at 30% budget + Stage 2 at 70%) + overlap/entropy computation.

---

## Memory (Peak RSS, max over 5 reps)

| Arm | SciFact (MiB) | NFCorpus (MiB) |
|-----|--------------|---------------|
| A | 95 | 89 |
| B | 272 | 194 |
| C | 275 | 196 |
| D | 274 | 195 |
| E | 305 | 275 |
| F | 310 | 282 |
| RAND | 271 | 193 |

All under 512 MiB budget. E/F arms slightly higher (dual-round buffers + trace storage).

---

## Budget Utilization

| Arm | Total Candidates | Total EF Work | Stage-1 Allocation | Stage-2 Allocation |
|-----|-----------------|---------------|-------------------|-------------------|
| A | 50 | 64 | — | — |
| B | 150 | 192 | — | — |
| C | 150 | 192 | 100% (static) | 0% |
| D | 150 | 192 | 100% (static) | 0% |
| E | 150 | 192 | 30% (per-head) | 70% (adaptive) |
| F | 150 | 192 | 30% (per-head) | 70% (adaptive) |
| RAND | 150 | 192 | 100% (static) | 0% |

C and D show 0% Stage-2 allocation — no actual redistribution occurs.

---

## EFPROBE: ef Knob Sensitivity

| Dataset | ef=16 (p50) | ef=128 (p50) | Speedup | Recall Δ |
|---------|-------------|--------------|---------|----------|
| SciFact | 1,873 µs | 2,897 µs | 1.55× | −0.0125 |
| NFCorpus | 1,534 µs | 2,506 µs | 1.63× | −0.0025 |

**ef is a REAL knob**: latency scales ~55-63% with ef; recall moves slightly (not saturated at small probe).

---

## Cost-Benefit Summary

| Metric | B → E Delta | B → F Delta |
|--------|-------------|-------------|
| Recall@10 (SCI) | −0.0045 | −0.0044 |
| Recall@10 (NFC) | −0.0007 | +0.0001 |
| Latency p50 | +14.3 ms | +16.5 ms |
| Memory | +33 MiB | +38 MiB |
| Candidate changes | 100% queries | 100% queries |

**Verdict**: Interaction-guided allocation adds ~8–9× latency for negligible recall change. Not cost-effective for recall@10 on these datasets/embeddings.