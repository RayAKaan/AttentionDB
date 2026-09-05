# Phase 2C final report (QK attention vs gating)

Q1 QK>gating CONTROLLED? — **Not answerable on frozen caches** (HC-6
  content loss, refused pre-training per §4); gating equals oracle there
  (0.9533), so headroom was ~nil regardless.
Q2 QK>gating NOISE? — Same as Q1 (content loss); gating collapsed to the
  one reliable head (0.8378) — no headroom.
Q3 QK>gating MULTIVIEW? — **NO: 0.1113 vs 0.4983 R@10** [PH2C-QK-002-MULTIVIEW].
Q4 QK>uniform? — **NO: 0.1113 vs 0.2128** — loses to the non-learned baseline.
Q5 gating+QK > components? — **NO: 0.2769 < gating 0.4983**.
Q6 Robust across seeds? — Yes, the LOSS is robust (QK .1078–.1156 in all seeds).
Q7 Candidate recall limiting? — **NO** (0.9975; 100% ≥1 relevant).
Q8 Latency cost? — QK p50 310.5 µs vs gating 13.0 µs (~24×).
Q9 Parameters added? — 3,072 (2×8×192) + training cost; parameters were not
   the constraint — signal was.
Q10 Exact rerank still regresses? — As a pipeline mode it remains harmful
   (Phase 2 E<D); in the offline study quality tracks the WEIGHTING
   (F14); QK weighting does not rescue it.
Q11 Which query types benefit? — **None** (QK loses in every group).
Q12 Improvement large enough to justify complexity? — There is NO
   improvement to justify; QK is dominated on quality AND latency.
Q13 SUPPORTED: "gating captures the exploitable multiview structure";
   reference-arm pipeline reproduction (exact match to registered 2B values).
Q14 OPEN: QK on REGENERATED corpora (new experiment family; not run —
   Rule Zero outcome determined by valid multiview result + nil headroom on
   controlled/noise); QK on genuinely interaction-structured domains.
Q15 NOT SUPPORTED: "candidate-level QK improves retrieval" (N1 closed:
   measured, negative); "QK adds value beyond gating" (Rule Zero).

**Decision (§29/§39): architecture = trained gating → topK (A). QK is
optional (no measured benefit). No transformer/cross-attention/deeper
variants — STOP condition honored. Exact rerank stays optional/off**
(unless a future weighting study reverses F14 evidence).

Artifacts: runs PH2C-QK-002-{MULTIVIEW,CONTROLLED,NOISE}; results/
qk-attention-multiview.csv; tables/table-qk-main.md; findings/
qk-main-analysis.md (F11–F15); HC-6; ledger addendum 2.
Figures for the QK main comparison: PENDING (marked, not fabricated).
