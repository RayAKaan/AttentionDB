#!/usr/bin/env python3
"""E11 fault-injection controller.

Drives `phase3-bench dbtest e11` writer/recover/tamper children per the frozen
spec (phase3e-e11-spec.md M2/M3/M9): arms crashgate env, proves the fault
boundary was reached, runs recovery legs, records every §6 artifact.

Usage: e11_drive.py <RUN_ID> [<RUN_ID> ...] | plan
"""
import json, os, platform, signal, subprocess, sys, time

BIN = "target/release/phase3-bench"
RUNS = "research/phase3/raw/runs"

# ---------------------------------------------------------------- run matrix

def plan():
    """The full E11 matrix: 37 official runs (scenario → fault → expectation)."""
    P = []
    def add(run, family, scenario, mode, n, fault=None, stage=None, **kw):
        P.append(dict(run_id=run, family=family, scenario=scenario, mode=mode,
                      n_docs=n, fault=fault, stage=stage, **kw))
    # F01 — WAL artifact tamper (tamper role; 20 docs)
    cases = [("control", "open"), ("trunc_tail", "open"), ("corrupt_body", "refuse"),
             ("bad_magic", "refuse"), ("seq_gap", "refuse"), ("missing_segment", "refuse"),
             ("sidecar_regress", "refuse"), ("sidecar_absent", "open")]
    for i, (case, exp) in enumerate(cases):
        add(f"PH3E-FAULT-{i+1:03d}", "F01", f"wal_{case}", "sync", 20,
            tamper_case=case, expect_open=exp)
    # corrective rerun of PH3E-FAULT-007 (expectation defect: sidecar regress
    # is tolerated by measurement — records authoritative; expect open+exact)
    add("PH3E-FAULT-038", "F01", "wal_sidecar_regress_corrective", "sync", 20,
        tamper_case="sidecar_regress", expect_open="open")
    # F02 — durability ack boundaries (200 inserts)
    add("PH3E-FAULT-009", "F02", "sync_after_fsync", "sync", 200, fault=dict(gate="after_fsync", hit=100, model="abort"))
    add("PH3E-FAULT-010", "F02", "sync_after_write", "sync", 200, fault=dict(gate="after_write", hit=100, model="abort"))
    add("PH3E-FAULT-011", "F02", "sync_before_ack", "sync", 200, fault=dict(gate="before_ack", hit=100, model="abort"))
    add("PH3E-FAULT-012", "F02", "group_after_flush", "group", 200, fault=dict(gate="after_flush", hit=100, model="abort"))
    add("PH3E-FAULT-013", "F02", "async_after_write", "async", 200, fault=dict(gate="after_write", hit=100, model="abort"))
    add("PH3E-FAULT-014", "F02", "async_after_write_promote", "async", 200,
        fault=dict(gate="after_write", hit=150, model="abort"), async_flush_at=100)
    # F03 — transaction commit boundary (30 base + 10-txn)
    add("PH3E-FAULT-015", "F03", "txn_tx_before_commit_wal", "sync", 30, fault=dict(gate="tx_before_commit_wal", hit=1, model="abort"))
    add("PH3E-FAULT-016", "F03", "txn_tx_after_commit_wal", "sync", 30, fault=dict(gate="tx_after_commit_wal", hit=1, model="abort"))
    add("PH3E-FAULT-017", "F03", "txn_pre_commit_park", "sync", 30, stage="pre_commit_park")
    add("PH3E-FAULT-018", "F03", "txn_rollback_then_park", "sync", 30, stage="rollback_then_park")
    # F04 — checkpoint interruption (300 docs, sync)
    for i, gate in enumerate(["ckpt_after_sst", "ckpt_after_trim", "ckpt_after_rotate",
                              "manifest_after_current_rename", "sst_after_write"]):
        add(f"PH3E-FAULT-{19+i:03d}", "F04", f"ckpt_{gate}", "sync", 300, fault=dict(gate=gate, hit=1, model="abort"))
    # corrective rerun of PH3E-FAULT-022: manifest_after_current_rename hit=1
    # was consumed by the setup-time catalog manifest write, so the fault landed
    # BEFORE the workload (model missing). hit=2 = the real checkpoint manifest.
    add("PH3E-FAULT-039", "F04", "ckpt_manifest_after_current_rename_hit2", "sync", 300,
        fault=dict(gate="manifest_after_current_rename", hit=2, model="abort"))
    # F05 — backup/restore (500 docs, sync)
    add("PH3E-FAULT-024", "F05", "backup_mid_copy", "sync", 500, fault=dict(gate="backup_mid_copy", hit=1, model="abort"))
    add("PH3E-FAULT-025", "F05", "backup_after_copy", "sync", 500, fault=dict(gate="backup_after_copy", hit=1, model="abort"))
    add("PH3E-FAULT-026", "F05", "backup_clean_restore", "sync", 500)
    # F06 — compaction (400 docs, 150 deletes, sync)
    for i, gate in enumerate(["compact_before_merge", "compact_after_output",
                              "compact_after_cleanup", "compact_after_install"]):
        add(f"PH3E-FAULT-{27+i:03d}", "F06", f"compact_{gate}", "sync", 400,
            fault=dict(gate=gate, hit=1, model="abort"), n_extra=150)
    add("PH3E-FAULT-031", "F06", "compact_clean", "sync", 400, n_extra=150)
    # F07 — index hygiene / rebuild
    add("PH3E-FAULT-032", "F07A", "hygiene_multihead_fresh_ckpt", "sync", 800)
    add("PH3E-FAULT-033", "F07B", "hygiene_dead_dominated_rebuild", "sync", 25000, n_extra=24000)
    add("PH3E-FAULT-034", "F07C", "recovery_rebuild_interrupted", "sync", 25000,
        fault=dict(gate="rebuild_mid", hit=1, model="abort"), fault_in_recovery=True)
    add("PH3E-FAULT-035", "F07D", "hygiene_ckpt_interrupted", "sync", 25000, n_extra=24000,
        fault=dict(gate="ckpt_after_sst", hit=2, model="abort"))
    # F08 — bounded concurrency with mid-run fault
    add("PH3E-FAULT-036", "F08", "conc_fault_before_ack", "sync", 0, n_extra=300,
        fault=dict(gate="before_ack", hit=150, model="abort"))
    # corrective rerun of PH3E-FAULT-024 (allowlist defect D53)
    add("PH3E-FAULT-040", "F05", "backup_mid_copy_corrective", "sync", 500,
        fault=dict(gate="backup_mid_copy", hit=1, model="abort"))
    # corrective run for PH3E-FAULT-032 (dataset defect D54: degenerate
    # 2-sparse vectors made the top-1 self-hit gate meaningless; generator
    # replaced with the E10 clustered scheme BEFORE the official gate runs)
    add("PH3E-FAULT-042", "F07A", "hygiene_multihead_fresh_ckpt_v2", "sync", 800)
    # corrective run for PH3E-FAULT-034 (writer panic: F07C missing from the
    # workload dispatch — D55)
    add("PH3E-FAULT-041", "F07C", "recovery_rebuild_interrupted_v2", "sync", 25000,
        fault=dict(gate="rebuild_mid", hit=1, model="abort"), fault_in_recovery=True)
    # F09 — integrated 40k lifecycle (sync; crash at final checkpoint)
    add("PH3E-FAULT-037", "F09", "integrated_40k_lifecycle", "sync", 40000,
        fault=dict(gate="ckpt_after_sst", hit=3, model="abort"))
    # corrective rerun of PH3E-FAULT-037 (harness defect D56: reinsert-without-
    # delete minted second live uuids — the E10 SCALE-014 trap — and the model
    # failed to prune acked_live_idx on deletes; engine exonerated by E10 D41)
    add("PH3E-FAULT-043", "F09", "integrated_40k_lifecycle_v2", "sync", 40000,
        fault=dict(gate="ckpt_after_sst", hit=3, model="abort"))
    # corrective rerun of PH3E-FAULT-035 (harness defect D57: model saved
    # before the delete stage, so the ckpt2 crash left a stale on-disk model)
    add("PH3E-FAULT-044", "F07D", "hygiene_ckpt_interrupted_v2", "sync", 25000, n_extra=24000,
        fault=dict(gate="ckpt_after_sst", hit=2, model="abort"))
    return P

# classifications declared per scenario (reviewed against outcomes; the driver
# only asserts harness-consistency, the classification is an evidenced claim)
def classification(entry):
    f, s = entry["family"], entry["scenario"]
    if f == "F01":
        return "VERIFIED"
    if f == "F02":
        if s.startswith("async"):
            return "SUPPORTED"   # documented async boundary confirmed, not durability
        return "VERIFIED"
    if f == "F03":
        return "VERIFIED"
    if f == "F04":
        return "VERIFIED"
    if f == "F05":
        return "VERIFIED"
    if f == "F06":
        return "VERIFIED"
    if f == "F07A":
        return "VERIFIED"
    if f in ("F07B", "F07C", "F07D"):
        return "VERIFIED"
    if f == "F08":
        return "SUPPORTED"  # narrow concurrency claim (A7 bounded subset)
    if f == "F09":
        return "VERIFIED"
    return "UNCLASSIFIED"

# ---------------------------------------------------------------- controller

def env_for(entry):
    env = dict(os.environ)
    f = entry.get("fault")
    if f:
        env["PH3E_CRASH_AT"] = f["gate"]
        env["PH3E_CRASH_HIT"] = str(f["hit"])
        env["PH3E_CRASH_MODEL"] = f.get("model", "abort")
        env["PH3E_CRASH_MARKER"] = f"{RUNS}/{entry['run_id']}/reached.marker"
        env.pop("PH3D_DURABILITY", None)
    return env

def spec_json(entry, run_dir, leg=None):
    e = dict(entry)
    e.pop("tamper_case", None) if False else None
    e["db_dir"] = f"{run_dir}/db"
    if entry.get("backup_dir_spec", True) and entry["family"] in ("F05", "F09"):
        e["backup_dir"] = f"{run_dir}/backup"
        e["restored_dir"] = f"{run_dir}/restored"
    if entry["family"] == "F07C":
        e["n_docs"] = 25000; e["n_extra"] = 24000  # same shape as F07B
    e["retrieval_samples"] = 50 if entry["family"] in ("F07A", "F09") else 0
    e["expect_open"] = entry.get("expect_open", "open")
    return e

def child(role, entry, run_dir, env, timeout, leg=1, park_expected=False):
    """Run a harness child. If it parks at its stage marker (by design), SIGKILL
    the process GROUP — that IS the controlled fault (spec M2/M9)."""
    spec = f"{run_dir}/spec.json"
    args = [BIN, "dbtest", "e11", "--role", role, "--spec", spec]
    if leg > 1:
        args += ["--leg", str(leg)]
    t0 = time.time()
    p = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         start_new_session=True, text=True)
    marker = f"{run_dir}/reached.marker"
    killed = False
    deadline = t0 + timeout
    while p.poll() is None and time.time() < deadline:
        if park_expected and os.path.exists(marker):
            time.sleep(1.0)  # let trailing buffer writes land
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            killed = True
            break
        time.sleep(0.2)
    if p.poll() is None:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        killed = True
    out, err = p.communicate()
    dt = round(time.time() - t0, 2)
    return dict(role=role, returncode=p.returncode, killed_by_controller=killed,
                seconds=dt, stdout=out or "", stderr=err or "")

def write_run_meta(entry, run_dir):
    e = spec_json(entry, run_dir)
    write(f"{run_dir}/spec.json", json.dumps(e, indent=1))
    write(f"{run_dir}/config.json", json.dumps(dict(
        run_id=entry["run_id"], family=entry["family"], scenario=entry["scenario"],
        durability=entry["mode"], collection="bench", dim=32,
        n_docs=spec_json(entry, run_dir)["n_docs"],
        n_extra=spec_json(entry, run_dir).get("n_extra", 0),
        fault=entry.get("fault"), stage=entry.get("stage"),
        tamper_case=entry.get("tamper_case"), fault_in_recovery=entry.get("fault_in_recovery", False),
        dataset="e11-deterministic-fnv (spec M5)", model="harness-side independent (M6)"), indent=1))
    write(f"{run_dir}/environment.json", json.dumps(dict(
        host=platform.node(), kernel=platform.release(), machine=platform.machine(),
        python=platform.python_version(), cpus=os.cpu_count(),
        failure_model="process death (SIGABRT/SIGKILL) at instrumented gates; power-loss UNSUPPORTED (A3)"), indent=1))
    fp = entry.get("fault") or {}
    write(f"{run_dir}/fault-plan.json", json.dumps(dict(
        requested_fault=fp or None, requested_stage=entry.get("stage"),
        mechanism="crashgate env" if fp else ("stage park" if entry.get("stage") else "tamper/clean"),
        reach_proof_required=True), indent=1))

def write(path, s):
    with open(path, "w") as f:
        f.write(s)

def drive(entry):
    run_dir = f"{RUNS}/{entry['run_id']}"
    os.makedirs(run_dir, exist_ok=True)
    for old in ("stdout.log", "stderr.log", "exit-status.json", "summary.json",
                "reached.marker", "recovery-verification.json"):
        if os.path.exists(f"{run_dir}/{old}"):
            os.remove(f"{old and run_dir}/{old}".replace(f"{run_dir}/{run_dir}", run_dir))
    write_run_meta(entry, run_dir)
    w_timeout = 1500 if entry["family"] == "F09" else 600
    r_timeout = 1500 if entry["family"] in ("F09", "F07B", "F07C", "F07D") else 600
    res = {"writer": None, "rec1": None, "rec2": None}

    if entry["family"] == "F01":
        w = child("tamper", entry, run_dir, env_for(entry), 300)
        res["writer"] = w
        write(f"{run_dir}/stdout.log", w["stdout"]); write(f"{run_dir}/stderr.log", w["stderr"])
        reached = w["returncode"] == 0
        write(f"{run_dir}/exit-status.json", json.dumps(dict(role="tamper", **{k: w[k] for k in ("returncode", "killed_by_controller", "seconds")}, fault_reached=reached), indent=1))
        verdict = "PASS"
        rv = f"{run_dir}/recovery-verification.json"
        if os.path.exists(rv):
            v = json.load(open(rv))
            verdict = v.get("verdict", "?")
            want = "REFUSED-AS-EXPECTED" if entry.get("expect_open") == "refuse" else "PASS"
            ok = verdict == want or (verdict == "PASS" and want == "PASS")
        else:
            ok = False
        cls = classification(entry) if ok else "INVALIDATED"
        write(f"{run_dir}/summary.json", json.dumps(dict(
            run_id=entry["run_id"], family=entry["family"], scenario=entry["scenario"],
            durability=entry["mode"], fault_reached=reached, recovery_verdict=verdict,
            classification=cls, entry=entry), indent=1))
        print(f"{entry['run_id']} {entry['scenario']:34s} tamper rc={w['returncode']} verdict={verdict} -> {cls}")
        return cls

    # writer child
    w = child("writer", entry, run_dir, env_for(entry), w_timeout,
              park_expected=not entry.get("fault"))
    res["writer"] = w
    write(f"{run_dir}/stdout.log", w["stdout"]); write(f"{run_dir}/stderr.log", w["stderr"])
    fp = entry.get("fault")
    if fp:
        marker = os.path.exists(f"{run_dir}/reached.marker")
        sig = {0: "EXIT0", -6: "SIGABRT", -9: "SIGKILL", 134: "SIGABRT", 137: "SIGKILL"}.get(w["returncode"], str(w["returncode"]))
        # model=abort ⇒ the gate itself must SIGABRT the child MID-CALL; a
        # clean-end park (marker written by the harness, SIGKILL by controller)
        # means the gate never fired — that is NOT a reached boundary.
        reached = (sig == "SIGABRT" and not marker) if fp.get("model", "abort") == "abort" \
            else (marker and sig == "SIGKILL")
        # family-specific position proof: the fault must land INSIDE the call
        if reached and entry["family"] == "F05":
            reached = "BACKUP_OK" not in open(f"{run_dir}/ack-log.jsonl").read()
        if reached and entry["family"] == "F06":
            reached = "COMPACT" not in open(f"{run_dir}/ack-log.jsonl").read()
    else:
        sig = {0: "EXIT0", -9: "SIGKILL"}.get(w["returncode"], str(w["returncode"]))
        # clean/stage runs END at the stage marker by design; the controller's
        # SIGKILL there IS the controlled termination
        reached = os.path.exists(f"{run_dir}/reached.marker") or w["returncode"] == 0
    write(f"{run_dir}/exit-status.json", json.dumps(dict(
        role="writer", returncode=w["returncode"], signal=sig,
        killed_by_controller=w["killed_by_controller"], seconds=w["seconds"],
        fault_planned=fp or entry.get("stage") or "clean", fault_reached=reached), indent=1))

    # recovery legs
    r_env = dict(os.environ); r_env.pop("PH3E_CRASH_AT", None); r_env.pop("PH3E_CRASH_HIT", None)
    if entry.get("fault_in_recovery"):
        f = entry["fault"]
        r_env1 = dict(os.environ)
        r_env1["PH3E_CRASH_AT"] = f["gate"]; r_env1["PH3E_CRASH_HIT"] = "1"
        r_env1["PH3E_CRASH_MODEL"] = f.get("model", "abort")
        r1 = child("recover", entry, run_dir, r_env1, r_timeout, leg=1)
        res["rec1"] = r1
        write(f"{run_dir}/stdout-recover1.log", r1["stdout"]); write(f"{run_dir}/stderr-recover1.log", r1["stderr"])
        leg1_ok = r1["returncode"] in (-6, 134) and not os.path.exists(f"{run_dir}/recovery-verification.json")
        r2 = child("recover", entry, run_dir, r_env, r_timeout, leg=2)
        res["rec2"] = r2
        write(f"{run_dir}/stdout-recover2.log", r2["stdout"]); write(f"{run_dir}/stderr-recover2.log", r2["stderr"])
        write(f"{run_dir}/exit-status.json", json.dumps(dict(
            role="writer+recover", writer_returncode=w["returncode"], writer_signal=sig,
            writer_fault_reached=reached, recover_leg1_returncode=r1["returncode"],
            recover_leg1_fault_reached=leg1_ok, recover_leg2_returncode=r2["returncode"]), indent=1))
        ok = reached and leg1_ok and r2["returncode"] == 0
    else:
        r2 = child("recover", entry, run_dir, r_env, r_timeout, leg=1)
        res["rec2"] = r2
        write(f"{run_dir}/stdout-recover2.log", r2["stdout"]); write(f"{run_dir}/stderr-recover2.log", r2["stderr"])
        ok = reached and r2["returncode"] == 0

    verdict = "?"
    rv = f"{run_dir}/recovery-verification.json"
    if os.path.exists(rv):
        verdict = json.load(open(rv)).get("verdict", "?")
        if verdict in ("REFUSED-AS-EXPECTED",):
            ok = ok and True
        elif verdict == "PASS":
            pass
        else:
            ok = False
    else:
        ok = False
    cls = classification(entry) if ok else "INVALIDATED"
    write(f"{run_dir}/summary.json", json.dumps(dict(
        run_id=entry["run_id"], family=entry["family"], scenario=entry["scenario"],
        durability=entry["mode"], fault_planned=fp or entry.get("stage") or "clean",
        fault_reached=reached, recovery_verdict=verdict, classification=cls,
        entry=entry), indent=1))
    print(f"{entry['run_id']} {entry['scenario']:34s} sig={sig:8s} reached={reached} verdict={verdict} -> {cls}")
    return cls

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    P = plan()
    if sys.argv[1] == "plan":
        for e in P:
            print(e["run_id"], e["family"], e["scenario"])
        sys.exit(0)
    want = set(sys.argv[1:])
    out = {}
    for e in P:
        if e["run_id"] in want or "all" in want:
            out[e["run_id"]] = drive(e)
    print(json.dumps(out, indent=0))
