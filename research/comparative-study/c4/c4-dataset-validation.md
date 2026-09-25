# C4 — Dataset Validation Record (confirmed scale)

Study `comparative-study-001`, protocol v1.0.0. Registered run:
`C4DATA-VERIFY-001` (PASS, 18/18 checks). Companion: `c4-environment.md`,
`c4-resource-budget.yaml`, `c4-run-plan.csv`.

## 1. Verification on this host (C4-G4) — PASS

Recomputed sha256 on this host against the recorded manifests, formalized as a
registered run (`raw/C4DATA-VERIFY-001`, status PASS, RUN-INDEX appended).

| Check | Expected | Observed | Match |
|---|---|---|---|
| DS-NFCORPUS corpus.jsonl | `10cc83ef…991f` | same | true |
| DS-NFCORPUS queries.jsonl | `d024e662…ef2a` | same | true |
| DS-NFCORPUS qrels dev/test/train | manifest | same | true ×3 |
| NFC EMBED HEAD-TITLE | `1f10d4eb…9eb` | same | true |
| NFC EMBED HEAD-BODY | `0fcaaf14…501` | same | true |
| NFC EMBED HEAD-CITE | `0fdadcb4…000` | same | true |
| NFC EMBED CANONICAL | `75bf5f3f…b1a6` | same | true |
| NFC EMBED QUERIES | `82be7079…e493` | same | true |
| DS-SCIFACT corpus.jsonl | `dec31c81…1ac6` | same | true |
| DS-SCIFACT queries.jsonl | `8ff84a7c…0313` | same | true |
| DS-SCIFACT qrels test/train | manifest | same | true ×2 |
| SCI EMBED HEAD-TITLE | `34c2290e…6365` | same | true |
| SCI EMBED HEAD-BODY | `f5792419…17e` | same | true |
| SCI EMBED HEAD-CITE | `22cafcc3…8348` | same | true |
| SCI EMBED CANONICAL | `c876d312…75a` | same | true |
| SCI EMBED QUERIES | `2ffd4fb2…989` | same | true |

Both datasets are therefore usable for confirmed-scale execution as-is.

## 2. Head/view provenance (per-head representations)

- DS-NFCORPUS: 5 views from `C2-EMBED-NFCORPUS-003` (canonical + 3 HEAD-*
  views + query set). Views are exact per-head embeddings derived from the
  native doc fields (title/body/cite) via the pinned model
  (revision `1110a243…`). B3 modelcard contract: 3 heads
  (HEAD-TITLE, HEAD-BODY, HEAD-CITE), num_heads 3 = collection heads 3.
- DS-SCIFACT: 5 views from `C3-EMBED-SCIFACT-001` (created on this host in C3;
  own hashes; NOT byte-equal to C2 bytes due to cross-env float drift ~2–3e-7,
  cosine similarity ~0.99999997). Full re-encode determinism verified in C3.

## 3. SciFact embedding decision (option A vs option B)

Per C1 dataset protocol ("decide according to existing protocol") the C3
precedent applies:

- **Decision: Option A — ACCEPT `C3-EMBED-SCIFACT-001` as the C4 SciFact
  representation, on its own registered hashes.**
- Rationale: (1) The existing protocol (C1 dataset-manifests + C3 eligibility
  CSV) already records `DS-SCIFACT` as **ELIGIBLE-ON-OWN-HASHES** with a
  registered, version-pinned re-embed on this host; C3 closed PASS consuming
  these exact vectors. (2) Option B (regenerate "canonical" to match C2 bytes)
  is impossible in practice — the C2 SciFact embeddings exist only as manifest
  hashes on a different OS/toolchain envelope whose float output is not
  byte-reproducible here (documented C3-EMBED-SCIFACT-001 run: model files
  byte-exact, export hashes differ by ~3e-7 float drift). (3) Re-running the
  same pipeline would yield a NEW artifact with NEW hashes and the SAME drift
  class — it would not remove the drift, only consume more wall time.
- **Scope/disclosure (unchanged from C3):** these SciFact vectors are a
  distinct, registered dataset artifact; never conflated with C2 bytes;
  cross-env drift disclosed in every run's `environment.yaml` + this doc;
  hash-validated (Section 1) at C4 start.
- Contingency: if any downstream check requires byte-identity with C2, that
  cell is reported BLOCKED-CANONICAL-UNAVAILABLE rather than silently
  substituted.

## 4. Query/split freeze for confirmed scale (C4-G8/statistical plan)

- Semantic datasets have no official train split → C1 leakage protocol uses
  (i) held-out training corpora for B3 supervision (LODO NFCorpus→SciFact,
  disclosed; SciFact untouched), and (ii) strictly untouched TEST queries for
  measurement. No model, tuning step, or analyst sees test labels before
  measurement.
- **Test (measurement) query sets, frozen before any C4 measurement:**
  - DS-SCIFACT: the 300 qrel-test query ids (`qrels/test.tsv`, 340 rows;
    verified count 300 distinct qids; disjoint from train qids).
  - DS-NFCORPUS: the 323 qrel-test query ids (`qrels/test.tsv`, 12,334 rows;
    verified 323 distinct qids; disjoint from dev qids).
- **Validation (tuning) query sets — validation-only tuning rule:**
  - DS-NFCORPUS: the 324 official DEV query ids (exists in BEIR release;
    verified disjoint from test). Tuning grids (ef_search, candidate
    budget/candidates) tune on dev ONLY.
  - DS-SCIFACT: no BEIR dev split exists. Matching/tuning subset = a fixed
    seeded sample of TRAIN qids, preregistered below, disjoint from test
    (verified 0 overlap). Target configs selected on this set only; test
    never used for tuning.
- Paired-comparison ordering: seeded per run; the SAME query order is used
  across paired systems/modes (statistical plan).

## 5. Subsample / pilot supersession

C3 pilot used a `SAMPLED-subset` (first min(N,100) of the seeded qrels-test
shuffle) as a bounded pilot slice. C4 confirmed scale runs the FULL frozen
test query sets (300/323) at Section 4; pilot subsets are diagnostic only and
never reported as confirmed-scale conclusions.

## 6. Gating/B3 train/test isolation (C4-G15)

- B3 card `b3-lodo-nf-v2-s20260925-h32-lr0.01.json` trained on NFCorpus TEST
  queries as LODO supervision (disclosed in C1 preregistration + C2
  manifests); SciFact TEST never used for training. C2 INV-L1..L5 leak guards
  PASS. Head-count refusal enforced by engine §15.
- The gating dataset hash (`gating-lodo-nfcorpus.json`) and TrainingMeta are
  recorded on the card; activated card identical for all B3 confirmed cells.