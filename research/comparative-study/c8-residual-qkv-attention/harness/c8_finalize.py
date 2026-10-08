#!/usr/bin/env python3
"""C8 finalization gate.

This is intentionally stricter than the execution driver.  It answers one
question only: do the frozen C8 TEST + SUPPORT cells and the required
verification/statistical artifacts exist so that the study can be closed
without accidentally treating validation evidence as final evidence?

The script does not run experiments, tune models, or modify evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT.parent.parent / "raw"
PLAN = ROOT / "c8-run-plan.csv"
ANALYSIS = ROOT / "analysis"


def plan_rows() -> list[dict[str, str]]:
    with PLAN.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def required_rows(rows: list[dict[str, str]], track: str) -> list[dict[str, str]]:
    return [row for row in rows if row["track"] == track]


def cell_complete(row: dict[str, str]) -> bool:
    run_dir = RAW / row["run_id"]
    return (run_dir / "metrics.json").exists()


def test_rep_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for dataset in ("SCI", "NFC"):
        shared = RAW / f"C8-SHARED-{dataset}"
        counts[dataset] = len(
            list(shared.glob("multi-test-rep*.json"))
        ) if shared.exists() else 0
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit machine-readable status")
    args = parser.parse_args()

    rows = plan_rows()
    tests = required_rows(rows, "TRK-A")
    supports = required_rows(rows, "SUPPORT")

    missing_test = [r["run_id"] for r in tests if not cell_complete(r)]
    missing_support = [r["run_id"] for r in supports if not cell_complete(r)]

    verification_path = ANALYSIS / "verification_report.json"
    statistics_path = ANALYSIS / "statistical_results.json"

    verification_present = verification_path.exists()
    verification_pass = False
    verification_errors: list[str] = []
    if verification_present:
        try:
            payload = json.loads(verification_path.read_text(encoding="utf-8"))
            verification_pass = payload.get("all_pass") is True
            verification_errors = [
                str(item)
                for values in (payload.get("checks") or {}).values()
                for item in (values or [])
            ]
        except (OSError, json.JSONDecodeError) as exc:
            verification_errors = [f"invalid verification_report.json: {exc}"]

    statistics_present = statistics_path.exists()
    rep_counts = test_rep_counts()

    status = {
        "phase": "C8",
        "test_cells_required": len(tests),
        "test_cells_complete": len(tests) - len(missing_test),
        "support_cells_required": len(supports),
        "support_cells_complete": len(supports) - len(missing_support),
        "test_repetitions": rep_counts,
        "verification_present": verification_present,
        "verification_all_pass": verification_pass,
        "verification_errors": verification_errors,
        "statistics_present": statistics_present,
        "ready_to_close": (
            not missing_test
            and not missing_support
            and all(rep_counts.get(ds, 0) >= 5 for ds in ("SCI", "NFC"))
            and verification_present
            and verification_pass
            and statistics_present
            and not verification_errors
        ),
        "missing_test": missing_test,
        "missing_support": missing_support,
    }

    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print(f"C8 TEST: {status['test_cells_complete']}/{status['test_cells_required']} cells complete")
        print(f"C8 SUPPORT: {status['support_cells_complete']}/{status['support_cells_required']} cells complete")
        print(f"C8 TEST reps: SCI={rep_counts.get('SCI', 0)}, NFC={rep_counts.get('NFC', 0)}")
        print(f"C8 verification: {'PASS' if verification_pass else 'PENDING/FAIL'}")
        print(f"C8 statistics: {'present' if statistics_present else 'pending'}")
        print(f"C8 closure: {'READY' if status['ready_to_close'] else 'NOT READY'}")
        if missing_test:
            print("Missing TEST cells:")
            for run_id in missing_test:
                print(f"  - {run_id}")
        if missing_support:
            print("Missing SUPPORT cells:")
            for run_id in missing_support:
                print(f"  - {run_id}")
        if verification_errors:
            print("Verification errors:")
            for error in verification_errors:
                print(f"  - {error}")

    return 0 if status["ready_to_close"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
