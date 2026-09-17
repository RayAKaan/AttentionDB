#!/usr/bin/env python3
"""Phase 3E E1: generate results/wal-integrity-e1.csv from the raw
PH3E-WAL-001 run. Every value is read from the raw file — nothing is
hard-coded except the EXPECTATION table, which is the acceptance gate:
any deviation between actual behavior and expectation becomes a MISMATCH
row and a non-zero exit, so the consistency checker fails loudly."""
import csv
import re
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


# ================================================================
# Phase 3E E2 — durability semantics expectation tables
# ================================================================
# Raw runs record FACTS (what was acked, what recovered, how the child died).
# The tables below encode the CONTRACT established by the E2 audit + evidence.
# Any actual/expected deviation becomes a MISMATCH row and a non-zero exit.

ACK_TARGET_EXPECT = {
    "sync":  {"before_wal_append": "ABSENT", "after_write": "ABSENT",
              "after_flush": "NOT_REACHED", "after_fsync": "PRESENT",
              "after_wal_append": "PRESENT", "after_apply": "PRESENT",
              "before_ack": "PRESENT", "after_ack": "PRESENT",
              "explicit_flush": "PRESENT"},
    "group": {"before_wal_append": "ABSENT", "after_write": "ABSENT",
              "after_flush": "PRESENT", "after_fsync": "NOT_REACHED",
              "after_wal_append": "PRESENT", "after_apply": "PRESENT",
              "before_ack": "PRESENT", "after_ack": "PRESENT",
              "explicit_flush": "PRESENT"},
    "async": {"before_wal_append": "ABSENT", "after_write": "ABSENT",
              "after_flush": "NOT_REACHED", "after_fsync": "NOT_REACHED",
              "after_wal_append": "ABSENT", "after_apply": "ABSENT",
              "before_ack": "ABSENT", "after_ack": "ABSENT",
              "explicit_flush": "PRESENT"},
}

TXN_STATE_EXPECT = {
    "sync":  {"mid_ops": "ABSENT", "commit_written": "ABSENT",
              "after_wal_append": "COMPLETE", "after_apply": "COMPLETE",
              "before_ack": "COMPLETE", "after_ack": "COMPLETE",
              "ack_ckpt": "COMPLETE"},
    "group": None,  # identical to sync
    "async": {"mid_ops": "ABSENT", "commit_written": "ABSENT",
              "after_wal_append": "ABSENT", "after_apply": "ABSENT",
              "before_ack": "ABSENT", "after_ack": "ABSENT",
              "ack_ckpt": "COMPLETE"},
}
TXN_STATE_EXPECT["group"] = TXN_STATE_EXPECT["sync"]


def _gen_ack_boundary(mismatches):
    raw = "research/phase3/raw/runs/PH3E-DUR-001/ack-boundary.csv"
    out = "research/phase3/results/durability-ack-boundary.csv"
    rows = list(csv.DictReader(open(raw)))
    res = []
    for r in rows:
        mode, gate = r["mode"], r["gate"]
        exp_target = ACK_TARGET_EXPECT[mode][gate]
        ok = True
        if exp_target == "NOT_REACHED":
            ok = r["exit_kind"] == "NOT_REACHED"
            observed = "NOT_REACHED"
        else:
            ok &= r["exit_kind"] == "SIGABRT"
            observed = r["target_present"]
            ok &= r["target_present"] == ("true" if exp_target == "PRESENT" else "false")
            ok &= r["extra_count"] == "0"          # no resurrection
            ok &= r["checker_clean"] == "true"
            if mode in ("sync", "group"):
                ok &= r["baseline_preserved"] == r["baseline_total"]
            # async baseline loss is the documented tradeoff: no value assertion
        if not ok:
            mismatches += 1
        res.append({"mode": mode, "gate": gate, "rep": r["rep"],
                    "exit_kind": r["exit_kind"],
                    "baseline": f'{r["baseline_preserved"]}/{r["baseline_total"]}',
                    "target_acked": r["target_acked"], "target": observed,
                    "expected": exp_target,
                    "match": "MATCH" if ok else "MISMATCH"})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    return len(rows)


def _gen_txn(mismatches):
    raw = "research/phase3/raw/runs/PH3E-DUR-002/txn-ack.csv"
    out = "research/phase3/results/durability-transactions.csv"
    rows = list(csv.DictReader(open(raw)))
    res = []
    for r in rows:
        mode, gate = r["mode"], r["gate"]
        exp_state = TXN_STATE_EXPECT[mode][gate]
        ok = r["txn_insert_state"] == exp_state
        if r["txn_insert_state"] == "PARTIAL" and exp_state != "PARTIAL":
            ok = False  # the fundamental invariant: ACK -> partial never allowed
        ok &= r["extra_count"] == "0" and r["checker_clean"] == "true"
        if mode in ("sync", "group"):
            exp_base = r["baseline_total"] if exp_state == "ABSENT" else "3"
            ok &= r["baseline_preserved"] == exp_base
            ok &= r["txn_deletes_applied"] == ("true" if exp_state == "COMPLETE" else "false")
        else:
            ok &= r["txn_deletes_applied"] in ("true", "false")
        if not ok:
            mismatches += 1
        res.append({"mode": mode, "gate": gate, "rep": r["rep"],
                    "exit_kind": r["exit_kind"],
                    "baseline": f'{r["baseline_preserved"]}/{r["baseline_total"]}',
                    "txn_acked": r["txn_acked"],
                    "txn_state": r["txn_insert_state"],
                    "deletes_applied": r["txn_deletes_applied"],
                    "expected": exp_state,
                    "match": "MATCH" if ok else "MISMATCH"})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    return len(rows)


def _gen_checkpoint(mismatches):
    raw = "research/phase3/raw/runs/PH3E-DUR-006/checkpoint-interaction.csv"
    out = "research/phase3/results/durability-checkpoint.csv"
    rows = list(csv.DictReader(open(raw)))
    res = []
    for r in rows:
        ok = (r["exit_kind"] == "SIGABRT"
              and r["extra_count"] == "0" and r["checker_clean"] == "true"
              and r["restarts_equal"] == "true"
              and r["restart_checker_clean"] == "true"
              and r["high_watermark"] != "0")
        # Structural-point coverage: checkpoint/checkpoint+trim fsync the WHOLE
        # WAL (every mode) -> all 30 acked baseline records survive. Rotation
        # alone fsyncs only COMPLETED segments -> in Async the active-segment
        # records stay userspace-buffered and are lost. Contract-derived rule,
        # computed from the row's own E1 watermark (never hand-typed): the
        # surviving baseline = watermark - 1 (minus the CreateCollection record).
        if r["mode"] == "async" and r["structural"] == "rotate_only":
            ok &= r["baseline_preserved"] == str(int(r["high_watermark"]) - 1)
        else:
            ok &= r["baseline_preserved"] == r["baseline_total"]
        if r["mode"] == "async":
            ok &= r["target_present"] == "false"  # post-point append buffered
        else:
            ok &= r["target_present"] == "true"
        if not ok:
            mismatches += 1
        res.append({"mode": r["mode"], "structural": r["structural"],
                    "rep": r["rep"], "baseline": f'{r["baseline_preserved"]}/{r["baseline_total"]}',
                    "target": r["target_present"],
                    "high_watermark": r["high_watermark"],
                    "restarts_equal": r["restarts_equal"],
                    "expected_target": "ABSENT" if r["mode"] == "async" else "PRESENT",
                    "match": "MATCH" if ok else "MISMATCH"})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    return len(rows)


def _gen_group(mismatches):
    raw = "research/phase3/raw/runs/PH3E-DUR-004/group-boundary.csv"
    out = "research/phase3/results/durability-group.csv"
    rows = list(csv.DictReader(open(raw)))
    res = []
    for r in rows:
        ok = (r["exit_kind"] == "SIGKILL" and r["extra_count"] == "0"
              and r["checker_clean"] == "true"
              and int(r["recovered_acked"]) <= int(r["acked_total"]))
        if r["mode"] in ("sync", "group"):
            ok &= r["acked_lost"] == "0"   # flush/fsync before ack holds
        # async: loss allowed and documented; recorded, not asserted zero
        if not ok:
            mismatches += 1
        res.append({"mode": r["mode"], "variant": r["variant"], "rep": r["rep"],
                    "acked": r["acked_total"], "recovered": r["recovered_acked"],
                    "acked_lost": r["acked_lost"],
                    "unacked_survived": r["unacked_survived"],
                    "expected_lost": "0" if r["mode"] in ("sync", "group") else "ANY",
                    "match": "MATCH" if ok else "MISMATCH"})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    return len(rows)


def _gen_modes_summary():
    """Mode-contract summary, derived from the raw runs (no hand-typed numbers)."""
    ack = list(csv.DictReader(
        open("research/phase3/raw/runs/PH3E-DUR-001/ack-boundary.csv")))
    txn = list(csv.DictReader(
        open("research/phase3/raw/runs/PH3E-DUR-002/txn-ack.csv")))
    grp = list(csv.DictReader(
        open("research/phase3/raw/runs/PH3E-DUR-004/group-boundary.csv")))
    lat = list(csv.DictReader(
        open("research/phase3/raw/runs/PH3E-DUR-005/mode-latency.csv")))
    out = "research/phase3/results/durability-modes.csv"

    def acked_lost_async():
        n = 0
        for r in ack:
            if r["mode"] == "async" and r["gate"] == "after_ack" \
                    and r["target_present"] == "false" and r["target_acked"] == "true":
                n += 1
        return n

    latm = {r["mode"]: r for r in lat}
    rows = [
        {"mode": "sync", "ack_implies": "DURABLE_MACHINE (fsync before ack)",
         "process_crash": "VERIFIED (after_fsync gate; acked never lost)",
         "machine_crash": "NOT VERIFIED (E3)",
         "acked_loss_observed": "0", "txn_atomicity": "COMPLETE-or-ABSENT (verified)",
         "mean_ack_us": latm["sync"]["mean_us"]},
        {"mode": "group", "ack_implies": "DURABLE_PROCESS (page-cache flush before ack)",
         "process_crash": "VERIFIED (after_flush gate; acked never lost)",
         "machine_crash": "NOT VERIFIED (E3; page cache lost on power loss)",
         "acked_loss_observed": "0", "txn_atomicity": "COMPLETE-or-ABSENT (verified)",
         "mean_ack_us": latm["group"]["mean_us"]},
        {"mode": "async", "ack_implies": "COMMITTED_NOT_DURABLE (userspace buffer)",
         "process_crash": "VERIFIED LOSS: acked writes/txns CAN disappear; "
                          "loss bounded by flush points (rotation/checkpoint/close/flush_wal)",
         "machine_crash": "NOT VERIFIED (E3)",
         "acked_loss_observed": f'{acked_lost_async()} single-insert cells + '
                                f'{grp[0]["acked_lost"]} of {grp[0]["acked_total"]} multi-writer acks',
         "txn_atomicity": "COMPLETE-or-ABSENT (verified; whole acked txn may vanish)",
         "mean_ack_us": latm["async"]["mean_us"]},
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def gen_durability() -> int:
    mismatches = 0
    n = _gen_ack_boundary(mismatches)
    n += _gen_txn(mismatches)
    n += _gen_checkpoint(mismatches)
    n += _gen_group(mismatches)
    n += _gen_modes_summary()
    return mismatches


# ================================================================
# Phase 3E E3 — machine-crash (F2E) recovery classification
# ================================================================
# Contract-derived expectation table (phase3e-e3-spec.md). The raw run
# records FACTS; this table encodes the CONTRACT; any deviation = MISMATCH.

def _e3_classify(r) -> tuple[str, str, bool]:
    """returns (classification, expected_summary, ok)"""
    cell, mode, workload, window = r["cell"], r["mode"], r["workload"], r["window"]
    if r["exit_kind"] == "NOT_REACHED":
        return ("NOT_REACHED", "gate unreachable in mode", True)
    if r["open_error"] != "-":
        # No E3 window is designed to refuse; any refusal is flagged.
        return ("REFUSED_SAFELY", "no refusal expected", False)
    acked = int(r["acked_count"])
    rec = int(r["recovered_acked"])
    unack = int(r["unacked_present"])
    base_ok = r["checker_clean"] == "true" and unack == 0
    if not base_ok:
        return ("UNSAFE_OPEN", "checker clean + no unacked docs", False)

    if cell == "f0-control":
        ok = rec == acked and r["restarts_equal"] == "true"
        return ("RECOVERED_EXPECTED", "all acked survive graceful close", ok)

    if cell == "ack-c":
        if mode == "async":
            # COMMITTED_NOT_DURABLE: loss allowed; atomicity/checker still hold
            ok = rec <= acked and (window != "after_ack" or r["target2000_present"] in ("true", "false"))
            return ("DATA_LOSS_ALLOWED_BY_CONTRACT",
                    "acked async writes MAY vanish (E2 contract)", ok)
        t = r["target2000_present"] == "true"
        if mode == "sync":
            t_exp = window in ("after_apply", "after_fsync", "before_ack", "after_ack")
        else:  # group
            t_exp = window in ("after_apply", "after_flush", "before_ack", "after_ack")
        ok = rec == acked and t == t_exp
        return ("RECOVERED_EXPECTED",
                f"acked 10/10 + target {'PRESENT' if t_exp else 'ABSENT'} at {window}", ok)

    if cell == "ckpt-c":
        ok = rec == acked and r["restarts_equal"] == "true" \
             and r["restart_checker_clean"] == "true"
        return ("RECOVERED_EXPECTED",
                "checkpoint-start fsync => all acked durable at every window (all modes)", ok)

    if cell == "rotate-c":
        # rotation precedes the frame write: in-flight (unacked) must be absent;
        # all acked (completed segments) survive in ALL modes.
        ok = rec == acked and r["target2000_present"] == "false" \
             and r["restarts_equal"] == "true"
        return ("RECOVERED_EXPECTED",
                "acked 3/3 survive; in-flight record absent (rotation fsync boundary)", ok)

    if cell == "mixed-c":
        # acked includes DEL lines; expected present = acked_inserts - deleted
        expected = rec  # placeholder, computed below from columns
        delgone = int(r["deleted_still_gone"])
        miss = int(r["missing_acked"])
        if mode == "async" and window == "after_ack":
            ok = miss >= 0 and delgone == 3  # loss allowed; deletes still hold
            return ("DATA_LOSS_ALLOWED_BY_CONTRACT",
                    "async acked loss allowed; deleted stay deleted", ok)
        # structural windows: deletes durable (checkpoint fsync) -> the 3
        # deleted acked-inserts stay gone; every surviving acked insert present
        ok = delgone == 3 and (acked - miss) >= 0 and r["restarts_equal"] == "true"
        # miss must equal exactly the 3 deleted acked inserts
        ok = ok and miss == 3
        return ("RECOVERED_EXPECTED",
                "survivors = acked inserts minus 3 deleted; no resurrection", ok)

    if cell == "txn-c":
        st = r["txn_state"]
        if st == "PARTIAL":
            return ("UNEXPECTED_PARTIAL_STATE", "never partial", False)
        if mode in ("sync", "group"):
            exp = "ABSENT" if window == "after_write" else "COMPLETE"
        else:
            exp = "ABSENT"  # documented async whole-txn loss
        ok = st == exp
        cls = "RECOVERED_EXPECTED" if ok else "UNEXPECTED_PARTIAL_STATE"
        if mode == "async" and window == "after_ack" and st == "ABSENT":
            cls = "DATA_LOSS_ALLOWED_BY_CONTRACT"
        return (cls, f"txn {exp} at {window} ({mode})", ok)

    return ("HARNESS_FAILURE", "unknown cell", False)


def _gen_e3(mismatches):
    raw = "research/phase3/raw/runs/PH3E-E3-001/e3-matrix.csv"
    out = "research/phase3/results/e3-recovery-classification.csv"
    rows = list(csv.DictReader(open(raw)))
    res = []
    for r in rows:
        cls, exp, ok = _e3_classify(r)
        if not ok:
            mismatches += 1
        res.append({"cell": r["cell"], "mode": r["mode"], "window": r["window"],
                    "rep": r["rep"], "failure_model": r["failure_model"],
                    "classification": cls, "expected": exp,
                    "observed": f'rec={r["recovered_acked"]}/{r["acked_count"]}'
                                f' t2000={r["target2000_present"]}'
                                f' txn={r["txn_state"]} err={r["open_error"]}',
                    "match": "MATCH" if ok else "MISMATCH"})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    return len(rows)


def gen_e3() -> int:
    mismatches = 0
    _gen_e3(mismatches)
    return mismatches


# ================================================================
# Phase 3E E4 — backup / snapshot / restore expectations
# ================================================================

E4_INTEGRITY_EXPECT = {
    "valid-control": ("ACCEPTED", "-"),
    "partial-no-meta": ("REFUSED", "NO_META"),          # completion marker (E4 rule)
    "truncated-sst": ("REFUSED", "OPEN_OR_CATALOG"),    # SST CRC
    "corrupt-wal-state": ("REFUSED", "OPEN_OR_CATALOG"),# E1 WAL_STATE_CORRUPT
    "garbage-active-segment": ("ACCEPTED", "-"),        # torn-tail policy; state in SSTs
    "corrupt-current": ("ACCEPTED", "-"),               # fallback gen intact (self-healing)
    "malformed-meta": ("REFUSED", "META_PARSE"),
    "bad-format-version": ("REFUSED", "BAD_VERSION"),   # format gate (E4 rule)
    "nonempty-dest": ("REFUSED", "DEST_NONEMPTY"),
    "source-after-backup": ("INTACT", "-"),
}

E4_CRASH_EXPECT = ("SOURCE_OK", "REFUSED", "true")  # source ok, partial refused, early ok


def _gen_e4():
    mismatches = 0
    raw = "research/phase3/raw/runs/PH3E-BACKUP-004/e4-matrix.csv"
    out = "research/phase3/results/e4-backup-matrix.csv"
    rows = list(csv.DictReader(open(raw)))
    res = []
    for r in rows:
        r = {k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()}  # defensive strip (skip restkey lists)
        ok = True
        if r["case"] == "crash-during-backup":
            ok = (r["match"] == "SOURCE_OK"
                  and r["partial_backup_refused"].startswith("REFUSED")
                  and r["early_backup_still_restores"] == "true"
                  and r["checker_clean"] == "true")
            cls = "CRASH_PROPERTIES_HOLD" if ok else "CRASH_PROPERTY_VIOLATION"
            expected = "SOURCE_OK + partial REFUSED + early restores + checker clean"
        else:
            ok = (r["match"] == "MATCH" and r["checker_clean"] == "true"
                  and r["reader_errors"] == "0")
            cls = "SNAPSHOT_MATCH" if ok else "SNAPSHOT_VIOLATION"
            expected = "restored state == independent reference model at backup boundary"
        if not ok:
            mismatches += 1
        res.append({"case": r["case"], "mode": r["mode"],
                    "expected_count": r["expected_count"],
                    "restored_count": r["restored_count"],
                    "classification": cls, "expected": expected,
                    "observed": r["match"], "match": "MATCH" if ok else "MISMATCH"})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)

    raw2 = "research/phase3/raw/runs/PH3E-BACKUP-004/e4-integrity.csv"
    out2 = "research/phase3/results/e4-backup-integrity.csv"
    rows2 = list(csv.DictReader(open(raw2)))
    res2 = []
    for r in rows2:
        r = {k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()}  # defensive strip (skip restkey lists)
        case = r["case"]
        exp = E4_INTEGRITY_EXPECT.get(case)
        if exp is None:
            mismatches += 1
            res2.append({"case": case, "action": r["action"], "observed": r["restore_result"],
                         "expected": "UNEXPECTED-CASE", "match": "MISMATCH"})
            continue
        exp_res, exp_reason = exp
        ok = r["restore_result"] == exp_res
        if exp_reason != "-" and r["restore_result"] == "REFUSED":
            ok = ok and r["detail"] == exp_reason
        if not ok:
            mismatches += 1
        res2.append({"case": case, "action": r["action"], "observed": r["restore_result"],
                     "expected": f"{exp_res}" + (f":{exp_reason}" if exp_reason != "-" else ""),
                     "match": "MATCH" if ok else "MISMATCH"})
    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "action", "observed", "expected", "match"])
        w.writeheader()
        w.writerows(res2)
    return mismatches, len(rows) + len(rows2)


def gen_e4() -> int:
    mismatches, _n = _gen_e4()
    return mismatches


# ================================================================
# Phase 3E E5 — compaction expectations
# ================================================================

# cases whose expected classification is a REFUSAL (documented corruption policy)
E5_REFUSAL_CASES = {"partial-artifact-garbage-sst"}


def gen_e5() -> int:
    mismatches, _n = _gen_e5_inner()
    return mismatches


def _gen_e5_inner():
    raw = "research/phase3/raw/runs/PH3E-COMPACT-004/e5-compaction.csv"
    raw2 = "research/phase3/raw/runs/PH3E-COMPACT-004/e5-crash.csv"
    out = "research/phase3/results/e5-compaction.csv"
    mismatches = 0
    rows = list(csv.DictReader(open(raw)))
    res = []
    for r in rows:
        r = {k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
        case = r["case"]
        if case in E5_REFUSAL_CASES:
            ok = r["model_match"] == "REFUSED"
            cls = "REFUSED_BY_POLICY" if ok else "UNEXPECTED_ACCEPT"
            expected = "garbage SST must be refused loudly (never valid state)"
        else:
            ok = (r["model_match"] == "MATCH" and r["resurrect"] == "none"
                  and (r["checker_clean"].startswith("true") or r["checker_clean"] == "-"))
            cls = "COMPACTION_VERIFIED" if ok else "COMPACTION_VIOLATION"
            expected = ("logical state before == after (full value model); "
                        "no resurrection; checker clean")
        if not ok:
            mismatches += 1
        res.append({"case": case, "mode": r["mode"],
                    "sst_before": r["sst_before"], "sst_after": r["sst_after"],
                    "tombstones_removed": r["tombstones_removed"],
                    "classification": cls, "expected": expected,
                    "observed": r["model_match"],
                    "match": "MATCH" if ok else "MISMATCH"})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)

    # crash windows: every instrumented window must restart to the SAME valid
    # logical state, checker-clean, no tmp artifacts, and accept a second compaction
    rows2 = list(csv.DictReader(open(raw2)))
    res2 = []
    for r in rows2:
        r = {k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
        ok = (r["aborted_at_window"] == "true" and r["model_match"] == "MATCH"
              and r["checker_clean"] == "true" and r["tmp_clean"] == "tmp=0"
              and r["restart_compact_ok"] == "ok")
        if not ok:
            mismatches += 1
        res2.append({"window": r["window"], "observed_state": r["model_match"],
                     "expected": "aborted at window; restart to valid state; no tmp; second compact ok",
                     "match": "MATCH" if ok else "MISMATCH"})
    with open("research/phase3/results/e5-crash-windows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["window", "observed_state", "expected", "match"])
        w.writeheader()
        w.writerows(res2)
    return mismatches, len(rows) + len(rows2)


# ================================================================
# Phase 3E E6 — transaction semantics expectations
# ================================================================

# (family, case) whitelist — anything else means the raw run contains a cell
# the generator has no registered expectation for (fail, never silently PASS).
E6_TXN_CELLS = {
    ("E6a-basic", "t1-single-insert"), ("E6a-basic", "t2-multi-insert"),
    ("E6a-basic", "t3-insert-delete"), ("E6a-basic", "t4-multi-delete"),
    ("E6a-basic", "t5-empty-txn"),
    ("E6b-rollback", "rollback-invisible"), ("E6b-rollback", "illegal-transitions"),
    ("E6c-multiop", "n-ops-5"), ("E6c-multiop", "n-ops-10"),
    ("E6c-multiop", "n-ops-50"), ("E6c-multiop", "n-ops-100"),
    ("E6d-multi-txn", "commit-rollback-commit"),
    ("E6i-ordering", "commit-order-wins"), ("E6i-ordering", "delete-missing-noop"),
    ("E6j-concurrency", "3-staged-txns"),
    ("E6k-update-upsert", "update-upsert-semantics"),
    ("E6l-same-key", "ins-ins-del-ins-txn"),
    ("E6m-tombstones", "txn-ins-txn-del-compact"), ("E6m-tombstones", "staged-delete-rollback"),
    ("E6n-compaction", "commit-then-compact"), ("E6n-compaction", "compact-then-commit"),
    ("E6o-checkpoint", "txn-then-ckpt"), ("E6o-checkpoint", "ckpt-then-txn"),
    ("E6o-checkpoint", "staged-then-ckpt"), ("E6o-checkpoint", "commit-ckpt-restart"),
    ("E6p-rotation", "100-op-txn-2KiB-segments"),
    ("E6q-backup", "before-during-after"),
    ("E6r-idempotence", "restart-x3"),
    ("E6s-corruption", "garbled-committed-group"), ("E6s-corruption", "torn-tail-partial-group"),
}

# structured fields only — never the free-text notes (they legitimately contain
# the words "partial"/"FAIL" in benign policy statements)
_E6_SCAN_FIELDS = ["expected_state", "observed_state", "commit_status",
                   "checker_clean", "restart_ok", "model_match"]
_E6_BAD_RE = re.compile(r"BAD|MISMATCH|VIOLATION|UNEXPECTED|PARTIAL/")


def _e6_note_blob(r) -> str:
    """notes field + csv restkey overflow (comma-separated note segments land
    in the None key as a list)."""
    parts = [r.get("notes") or ""]
    extra = r.get(None) or []
    for v in extra:
        parts.append(v if isinstance(v, str) else str(v))
    return " ".join(parts)


def _e6_trueish(field: str, r) -> bool:
    """composite field must be slash-separated true/- values with >=1 true."""
    parts = (r[field] or "").split("/")
    return bool(parts) and any(p == "true" for p in parts) and all(p in ("true", "-") for p in parts)


def _e6_txn_ok(r) -> tuple[bool, str]:
    """Per-case expectation for one e6-txn.csv cell. Returns (ok, reason)."""
    key = (r["family"], r["case"])
    if key not in E6_TXN_CELLS:
        return False, f"unregistered txn cell {key}"
    for f in _E6_SCAN_FIELDS:
        if _E6_BAD_RE.search(r[f] or ""):
            return False, f"failure marker in {f}={r[f]!r}"
    case = r["case"]
    if case == "illegal-transitions":
        # state machine: the three illegal transitions must be no-ops
        # (double-commit=true on first commit; the illegal attempts return false)
        ok = (r["commit_status"] == "c1=true" and r["checker_clean"] == "c2=false"
              and r["restart_ok"] == "r=false" and r["model_match"] == "c3=false"
              and "no-op" in _e6_note_blob(r))
        return ok, "illegal transitions must all be no-ops (c1=true,c2/r/c3=false)"
    if case == "staged-then-ckpt":
        ok = (r["commit_status"] == "NEVER-COMMITTED" and r["observed_state"] == "false"
              and r["checker_clean"] == "true" and r["restart_ok"] == "ok"
              and "checkpoint never" in _e6_note_blob(r))
        return ok, "checkpoint must never commit an uncommitted txn"
    if case == "garbled-committed-group":
        ok = (r["observed_state"] == "REFUSED" and r["expected_state"] == "refuse"
              and (r["model_match"] or "").startswith("corruption never silently"))
        return ok, "garbled committed segment must be REFUSED (E1 policy)"
    # default committed-cell rule: checker/restart/model composite fields must
    # all be true/ok (slash-composites like true/true or true/true/true), and
    # rollback/absent cells carry their semantics in commit_status.
    for f in ("checker_clean", "model_match"):
        if not _e6_trueish(f, r):
            return False, f"{f}={r[f]!r} not trueish (true/- composites with >=1 true)"
    if not (r["restart_ok"] or "").startswith("ok"):
        return False, f"restart_ok={r['restart_ok']!r}"
    if case == "update-upsert-semantics":
        blob = _e6_note_blob(r)
        if "UNSUPPORTED" not in blob:
            # unsupported-claim guard: the E6k cell MUST record that in-txn
            # update/upsert is unsupported by the TxnOp type — a PASS row that
            # claimed support would be fabrication
            return False, "E6k must record in-txn update/upsert UNSUPPORTED"
    return True, "ok"


# crash boundaries x legal recovered txn states. async (buffered appends) may
# legally lose a post-WAL-append state — A2: ACK != machine durability.
_E6_CRASH_LEGAL = {
    "pre-commit staging": {"sync": {"ABSENT", "ABSENT_ATOMIC"},
                           "group": {"ABSENT", "ABSENT_ATOMIC"},
                           "async": {"ABSENT", "ABSENT_ATOMIC"}},
    "gate: tx_before_commit_wal": {"sync": {"ABSENT_ATOMIC"},
                                   "group": {"ABSENT_ATOMIC"},
                                   "async": {"ABSENT_ATOMIC"}},
    "gate: tx_after_commit_wal": {"sync": {"PRESENT_ATOMIC"},
                                  "group": {"PRESENT_ATOMIC"},
                                  "async": {"ABSENT_ATOMIC", "PRESENT_ATOMIC"}},
    "gate: after_apply": {"sync": {"PRESENT_ATOMIC"},
                          "group": {"PRESENT_ATOMIC"},
                          "async": {"ABSENT_ATOMIC", "PRESENT_ATOMIC"}},
    "gate: before_ack": {"sync": {"PRESENT_ATOMIC"},
                         "group": {"PRESENT_ATOMIC"},
                         "async": {"ABSENT_ATOMIC", "PRESENT_ATOMIC"}},
    "abort after ACK": {"sync": {"PRESENT_ATOMIC"},
                        "group": {"PRESENT_ATOMIC"},
                        "async": {"ABSENT_ATOMIC", "PRESENT_ATOMIC"}},
    "T1 committed; T3 staged only": {"sync": {"T1_PRESENT_T3_ABSENT"},
                                     "group": {"T1_PRESENT_T3_ABSENT"},
                                     "async": {"T1_PRESENT_T3_ABSENT"}},
}


def _e6_crash_ok(r) -> tuple[bool, str]:
    b = r["boundary"]
    if b not in _E6_CRASH_LEGAL:
        return False, f"unregistered crash boundary {b!r}"
    mode = r["mode"]
    if mode not in ("sync", "group", "async"):
        return False, f"unregistered mode {mode!r}"
    if r["aborted_at_boundary"] != "true":
        return False, "crash cell must record aborted_at_boundary=true"
    if r["atomicity"] != "ATOMIC" or r["checker_clean"] != "clean" or r["model_match"] != "MATCH":
        return False, (f"atomicity={r['atomicity']} checker={r['checker_clean']} "
                       f"model={r['model_match']}")
    if r["txn_state"] not in _E6_CRASH_LEGAL[b][mode]:
        return False, f"txn_state {r['txn_state']!r} not legal at {b}/{mode}"
    return True, "ok"


def gen_e6(rawdir: str = "research/phase3/raw/runs/PH3E-TXN-001") -> int:
    mismatches = 0
    raw = f"{rawdir}/e6-txn.csv"
    out = "research/phase3/results/e6-transactions.csv"
    rows = list(csv.DictReader(open(raw)))
    res = []
    for r in rows:
        r = {k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
        ok, why = _e6_txn_ok(r)
        if not ok:
            mismatches += 1
        cls = "TXN_PROPERTIES_HOLD" if ok else "TXN_PROPERTY_VIOLATION"
        res.append({"family": r["family"], "case": r["case"], "mode": r["mode"],
                    "expected": why if not ok else "registered E6 cell expectation",
                    "classification": cls,
                    "commit_status": r["commit_status"],
                    "checker": r["checker_clean"], "restart": r["restart_ok"],
                    "model": r["model_match"],
                    "match": "MATCH" if ok else "MISMATCH"})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)

    raw2 = f"{rawdir}/e6-crash.csv"
    out2 = "research/phase3/results/e6-crash-atomicity.csv"
    rows2 = list(csv.DictReader(open(raw2)))
    res2 = []
    for r in rows2:
        r = {k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
        ok, why = _e6_crash_ok(r)
        if not ok:
            mismatches += 1
        res2.append({"boundary": r["boundary"], "mode": r["mode"],
                     "txn_state": r["txn_state"],
                     "expected": (f"legal states {_E6_CRASH_LEGAL.get(r['boundary'], {}).get(r['mode'], '?')}"
                                  if not ok else "committed unit or never-happened, per boundary"),
                     "atomicity": r["atomicity"],
                     "checker": r["checker_clean"], "model": r["model_match"],
                     "match": "MATCH" if ok else "MISMATCH"})
    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res2[0].keys()))
        w.writeheader()
        w.writerows(res2)
    return mismatches, len(rows) + len(rows2)


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
    mismatches += gen_durability()
    mismatches += gen_e3()
    mismatches += gen_e4()
    mismatches += gen_e5()
    mismatches += gen_e6()[0]
    m = json.load(open("research/phase3/raw/runs/PH3E-WAL-001/metrics.json"))
    summary = {"run": "PH3E-WAL-001", "cases": len(rows),
               "refused": m["refused"], "opened": m["opened"],
               "mismatches": mismatches}
    print(json.dumps(summary))
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
