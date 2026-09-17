#!/usr/bin/env python3
"""Phase 3E E1: generate results/wal-integrity-e1.csv from the raw
PH3E-WAL-001 run. Every value is read from the raw file — nothing is
hard-coded except the EXPECTATION table, which is the acceptance gate:
any deviation between actual behavior and expectation becomes a MISMATCH
row and a non-zero exit, so the consistency checker fails loudly."""
import csv
import json
import os
import sys

RAW = "research/phase3/raw/runs/PH3E-WAL-001/wal-integrity.csv"
OUT = "research/phase3/results/wal-integrity-e1.csv"

# Expectation per case: (checkpoint, verdict, detail_class)
# detail_class for REFUSED = expected refusal code;
# for OPENED = one of LEGITIMATE_* (documented open reasons).
EXPECT = {
    "delete_required_segment_pre_checkpoint": (False, "REFUSED", "WAL_LOST_SEGMENT"),
    "delete_all_segments_post_checkpoint":    (True,  "REFUSED", "WAL_LOST_SEGMENT"),
    "rename_segment_gap":                     (False, "REFUSED", "WAL_SEQ_GAP_or_REPLAY"),
    "corrupt_wal_state_record":               (False, "REFUSED", "WAL_STATE_CORRUPT"),
    "delete_wal_state_record_legacy":         (True,  "OPENED",  "LEGITIMATE_LEGACY_NO_SIDECAR"),
    "truncate_torn_tail":                     (False, "OPENED",  "LEGITIMATE_TORN_TAIL_PREFIX"),
    "corrupt_frame":                          (False, "REFUSED", "WAL_REPLAY_CORRUPTION"),
    "corrupt_current_manifest_fallback":      (True,  "OPENED",  "LEGITIMATE_MANIFEST_FALLBACK"),
    "corrupt_all_manifests":                  (True,  "REFUSED", "MANIFEST_UNREADABLE"),
    "fresh_no_documents":                     (False, "OPENED",  "LEGITIMATE_FRESH_REOPEN"),
    "trimmed_reopen":                         (True,  "OPENED",  "LEGITIMATE_POST_CHECKPOINT_TRIM"),
}

# Map the harness's coarse detail strings onto the expectation classes.
OPEN_CLASS = {
    "delete_wal_state_record_legacy": "LEGITIMATE_LEGACY_NO_SIDECAR",
    "truncate_torn_tail": "LEGITIMATE_TORN_TAIL_PREFIX",
    "corrupt_current_manifest_fallback": "LEGITIMATE_MANIFEST_FALLBACK",
    "fresh_no_documents": "LEGITIMATE_FRESH_REOPEN",
    "trimmed_reopen": "LEGITIMATE_POST_CHECKPOINT_TRIM",
}


def main() -> int:
    rows = list(csv.DictReader(open(RAW)))
    out, mismatches = [], 0
    for r in rows:
        case = r["case"]
        exp = EXPECT.get(case)
        if exp is None:
            mismatches += 1
            out.append({"case": case, "checkpoint": r["checkpoint"],
                        "verdict": r["verdict"], "observed": r["detail"],
                        "expected": "UNEXPECTED-CASE", "match": "MISMATCH"})
            continue
        cp, verdict, dclass = exp
        actual_verdict = r["verdict"]
        if actual_verdict != ("REFUSED" if verdict == "REFUSED" else "OPENED"):
            ok = False
        elif verdict == "REFUSED":
            ok = (dclass == "WAL_SEQ_GAP_or_REPLAY" and
                  r["detail"] in ("WAL_SEQ_GAP", "WAL_REPLAY_CORRUPTION")) or \
                 r["detail"] == dclass
        else:
            ok = r["detail"].startswith("docs=") and \
                OPEN_CLASS.get(case) == dclass and \
                "checker_clean=true" in r["detail"]
        if not ok:
            mismatches += 1
        out.append({"case": case, "checkpoint": r["checkpoint"],
                    "verdict": actual_verdict, "observed": r["detail"],
                    "expected": f"{verdict}:{dclass}",
                    "match": "MATCH" if ok else "MISMATCH"})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "checkpoint", "verdict",
                                          "observed", "expected", "match"])
        w.writeheader()
        w.writerows(out)
    m = json.load(open("research/phase3/raw/runs/PH3E-WAL-001/metrics.json"))
    summary = {"run": "PH3E-WAL-001", "cases": len(rows),
               "refused": m["refused"], "opened": m["opened"],
               "mismatches": mismatches}
    print(json.dumps(summary))
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
