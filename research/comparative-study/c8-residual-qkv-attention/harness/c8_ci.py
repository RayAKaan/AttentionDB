#!/usr/bin/env python3
"""C8 CI driver — run plan cells in dependency order on a Linux runner.

Dependency order (protocol Sec.2/5):
  1. SAFETY/PROBE  (EFPROBE, per dataset)
  2. TUNE          (per arm; required before SMOKE/TEST)
  3. SAFETY/SMOKE  (uses TUNE artifacts; also runs the determinism re-run)
  4. SUPPORT       (VALIDATION dimension/depth sweep; self-contained training)
  5. TRK-A/TEST    (primary evidence; 5 reps)

Group cells (PROBE/SMOKE/TEST) share one engine per process, so only the group
leader run_id is invoked; it writes every sibling cell. Already-complete cells are
skipped, making the driver safe to re-invoke.

Usage:
  python harness/c8_ci.py                     # all tracks, in order
  python harness/c8_ci.py --tracks TUNE,TEST
  python harness/c8_ci.py --dry-run
  python harness/c8_ci.py --verify --analyze  # then run the gates/analysis
"""

import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(os.path.join(HERE, "..", "..", "raw"))
PLAN = os.path.abspath(os.path.join(HERE, "..", "c8-run-plan.csv"))
TEST_RUN = os.path.join(HERE, "c8_test_run.py")
VERIFY = os.path.join(HERE, "c8_verify.py")
ANALYZE = os.path.join(HERE, "c8_analyze.py")

ORDER = ["SAFETY/PROBE", "TUNE/VALID", "SAFETY/SMOKE", "SUPPORT/VALID",
         "TRK-A/TEST"]
GROUP_SPLITS = {"PROBE", "SMOKE", "TEST"}


def plan_rows():
    with open(PLAN, encoding="utf8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def phase_of(row):
    return f"{row['track']}/{row['split']}"


def cell_complete(run_id, track):
    marker = "artifacts/model.json" if track == "TUNE" else "metrics.json"
    return os.path.exists(os.path.join(RAW, run_id, marker))


def build_invocations(rows):
    """Return ordered list of (phase, leader_run_id, group_run_ids).

    A "group" is a set of plan cells that one engine process writes together
    (same track+split+dataset). TUNE/SUPPORT cells are one cell per process.
    """
    groups = {}
    for r in rows:
        phase = phase_of(r)
        if r["split"] in GROUP_SPLITS:  # PROBE/SMOKE/TEST -> one process per group
            key = (phase, r["dataset"])
            groups.setdefault(key, []).append(r["run_id"])
        else:
            groups[(phase, r["run_id"])] = [r["run_id"]]

    invocations = []
    for phase in ORDER:
        keys = [k for k in groups if k[0] == phase]
        for key in sorted(keys, key=lambda k: k[1]):
            run_ids = groups[key]
            invocations.append((phase, run_ids[0], run_ids))
    return invocations


def main():
    args = sys.argv[1:]
    tracks_filter = None
    dry = False
    do_verify = do_analyze = False
    i = 0
    while i < len(args):
        if args[i] == "--tracks":
            raw = args[i + 1].strip().upper()
            tracks_filter = None if raw in ("ALL", "") else {
                t.strip() for t in raw.split(",")}
            i += 2
        elif args[i] == "--dry-run":
            dry = True; i += 1
        elif args[i] == "--verify":
            do_verify = True; i += 1
        elif args[i] == "--analyze":
            do_analyze = True; i += 1
        else:
            raise SystemExit(f"unknown arg {args[i]}")

    rows = plan_rows()
    track_of = {r["run_id"]: r["track"] for r in rows}
    invocations = build_invocations(rows)
    if tracks_filter:
        invocations = [x for x in invocations
                       if x[0].split("/", 1)[0].upper() in tracks_filter]

    print(f"C8 CI: {len(invocations)} invocations "
          f"(tracks={tracks_filter or 'ALL'}, dry_run={dry})")
    ran = skipped = 0
    for phase, leader, run_ids in invocations:
        track = track_of[leader]
        complete = all(cell_complete(r, track) for r in run_ids)
        partial = any(cell_complete(r, track) for r in run_ids) and not complete
        status = "SKIP" if complete else ("PARTIAL" if partial else "RUN")
        print(f"  [{status:<7}] {phase:<13} leader={leader} cells={len(run_ids)}")
        if dry or complete:
            skipped += (1 if complete else 0)
            continue
        if partial:
            raise SystemExit(
                f"partial group for {leader}: {run_ids} — remove the incomplete "
                f"run dirs or pick new run_ids (immutability)")
        subprocess.run([sys.executable, TEST_RUN, "--run-id", leader], check=True)
        ran += 1

    print(f"C8 CI done: ran={ran} skipped={skipped}")

    if do_verify:
        subprocess.run([sys.executable, VERIFY], check=True)
    if do_analyze:
        subprocess.run([sys.executable, ANALYZE], check=True)


if __name__ == "__main__":
    main()
