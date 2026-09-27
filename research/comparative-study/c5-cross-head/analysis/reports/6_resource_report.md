# C5 Resource Utilization Report

**Budget Constraints (Per Protocol §4)**: Matched candidate budget (500), memory (512 MiB), ef effort (64), single-threaded HNSW search.

---

## Latency (p50, median over 5 reps × 300/323 queries)

| Arm | SciFact (µs) | NFCorpus (µs) | Overhead vs B |
|-----|-------------|---------------|---------------|
| A (CANONICAL) | 1,328 | 1,259 | −30% / −47% |
| B (3-head union) | 1,903 | 2,375 | baseline |
| C (λ=0.10 interaction) | **5,547** | **5,334** | **+192% / +125%** |

C arm runs two search rounds (ef=32 each) + centroid computation + re-search.

---

## Memory (Peak RSS, max over 5 reps)

| Arm | SciFact (MiB) | NFCorpus (MiB) |
|-----|--------------|---------------|
| A | 131 | 96 |
| B | 272 | 193 |
| C | 305 | 275 |

All under 512 MiB budget. C arm slightly higher (trace storage + dual-round buffers).

---

## Candidate Budget Utilization

| Arm | per_head_k | union_pre_cap | union_post_cap | Effective candidates/query |
|-----|-----------|---------------|----------------|---------------------------|
| A | 50 | 50 | 50 | 50 |
| B | 50 | 150 | 150 | 150 |
| C (round 1) | 50 | 150 | 150 | 150 |
| C (round 2) | 50 | 150 | 150 | +6–7 (interaction additions) |

C arm total candidates ≈ 157/query (150 round-1 + ~7 interaction additions). Union capped at 500.

---

## EFPROBE: ef Knob Sensitivity

| Dataset | ef=16 (p50) | ef=128 (p50) | Speedup | Recall Δ |
|---------|-------------|--------------|---------|----------|
| SciFact | 2,154 µs | 2,765 µs | 1.28× | 0.0000 |
| NFCorpus | 2,020 µs | 2,627 µs | 1.30× | +0.0072 |

**ef is a REAL knob**: latency scales ~30% with ef; NFC recall moves. Not blocked per §9.

---

## Cost-Benefit Summary

| Metric | B → C Delta |
|--------|-------------|
| Recall@10 (SCI) | −0.0018 (negligible) |
| Recall@10 (NFC) | +0.0016 (negligible) |
| Latency p50 | +3.6 ms (SCI) / +3.0 ms (NFC) |
| Memory | +33 MiB / +82 MiB |
| Candidate changes | >99% queries |

**Verdict**: Interaction adds ~2.8× latency for negligible recall change. Not cost-effective for recall@10 on these datasets/embeddings.