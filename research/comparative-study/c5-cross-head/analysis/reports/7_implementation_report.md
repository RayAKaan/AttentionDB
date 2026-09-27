# C5 Implementation Report

**Scope**: Engine changes in `attentiondb-core` + probe harness `c5pilot`

---

## Core Engine Changes (attentiondb-core)

### New Types (`retrieval.rs`)

| Type | Purpose |
|------|---------|
| `CrossRefineConfig { lambda: f32, per_head_k: usize }` | Interaction hyperparameters |
| `CrossHeadTrace { round1_counts, round2_added_counts, round2_added_ids, surprise_counts, surprise_ids, interaction_union_additions, refined, ... }` | Per-query causal ledger |
| `l2_normalize(v: &mut [f32])` | Unit L2 normalization (pure) |
| `cross_refine_query(q, centroids, lambda)` | q' = normalize(q + λ·centroid) (pure) |
| `cross_head_centroid(idx, ids, head)` | Mean vector of doc IDs in head (pure) |

### Collection Changes (`collection.rs`)

| Change | Location | Detail |
|--------|----------|--------|
| `RetrievalConfig.ef_search: Option<usize>` | struct field | Real per-head ef knob (None = legacy) |
| `RetrievalConfig.cross_refine: Option<CrossRefineConfig>` | struct field | Enable C5-C interaction (None = legacy) |
| `attend_detailed_inner(..., ef_search, cross_refine)` | private | 3-tuple return: (hits, stats, Option<Trace>) |
| `attend_detailed_traced(...)` | public | Returns trace for C5-C; discards for A/B |
| `run_round1(heads, q, k, ef)` | private | Stage-1 search at ef/2 per head |
| Cross-refine block | `collection.rs:540–620` | Round-2: centroids from OTHER heads → q' → re-search → union → per-head cap |

### Backward Compatibility

- `attend_detailed` / `attend_detailed_with_stats` signatures **unchanged** (discard trace)
- Default `ef_search=None`, `cross_refine=None` → pre-C5 byte-identical execution path
- All 50 core tests pass; clippy clean; fmt clean

---

## Unit Tests (5 new, all passing)

| Test | Verifies |
|------|----------|
| `cross_refine_lambda_zero_is_glue` | λ=0.0 → candidate set = control (B) |
| `cross_refine_is_normalized_and_direction_sensitive` | q' normalized; sign(λ) flips direction |
| `cross_refine_empties_and_degenerate_are_safe` | Empty S_H, empty corpus, λ extremes handled |
| `cross_refine_changes_candidate_set_not_scores` | Candidate IDs change; scores not just re-ranked |
| `cross_refine_glue_arm_matches_control_candidates` | λ=0.0 glue arm = B candidate semantics |

---

## Probe Harness (c5pilot)

| File | Purpose |
|------|---------|
| `probe/Cargo.toml` | Standalone workspace; deps: core, storage, serde_json |
| `probe/.cargo/config.toml` | `/Brepro`, `CARGO_INCREMENTAL=0` |
| `probe/c5pilot.rs` | Modes A/B/C → SingleHead/FixedFusion; `ef_search` real knob; ledger emission for C |

### c5pilot Contract (Per-Run JSON Output)

```json
{
  "run_id": "C5-...",
  "mode": "A|B|C",
  "per_query": [{ "recall10_qrels": f32, "recall10_exact": f32, "latency_us": u64, "ledger": {...} }],
  "candidate_change_count": u32,
  "surprise_empty_queries": u32,
  "deadline_exceeded_count": u32
}
```

### Modes

| Mode | RetrievalMode | collection_heads | attend_heads | cross_refine |
|------|---------------|------------------|--------------|--------------|
| A | SingleHead | ["CANONICAL"] | ["CANONICAL"] | None |
| B | FixedFusion | [TITLE, BODY, CITE] | [TITLE, BODY, CITE] | None |
| C | FixedFusion | [TITLE, BODY, CITE] | [TITLE, BODY, CITE] | Some(CrossRefineConfig{lambda, per_head_k}) |

---

## Build Verification

```
cargo build --release -p c5pilot   # 4.31s, 0 warnings
cargo test -p attentiondb-core     # 50 passed, 0 failed
cargo fmt --all --check            # OK
cargo clippy -p attentiondb-core   # OK
```

---

## Binary Reproducibility

- MSVC `/Brepro` → bit-identical `c5pilot.exe` across builds
- SHA256 recorded in every `environment.yaml`