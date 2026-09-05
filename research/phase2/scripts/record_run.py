#!/usr/bin/env python3
"""Record a completed experiment into the Phase 2 research registry.

Usage (from repo root):
  python3 research/phase2/scripts/record_run.py \
    --index research/phase2/raw/experiment-index.json \
    --manifest research/phase2/raw/run-manifest.json \
    --run-dir research/phase2/raw/runs/PH2C-QK-001 \
    --experiment-id PH2C-QK-001 \
    --results-csv research/phase2/results/qk-sanity.csv \
    [--metrics-arm-seed agg]

Contract (research mandate):
  - appends a registry entry built from the run's OWN artifacts (metrics.json
    preferred, else eval_test.csv); no manual transcription of numbers;
  - refuses to overwrite: duplicate experiment IDs and existing results CSVs
    are errors (raw runs are immutable; a rerun gets a NEW run id);
  - updates the manifest experiment list + `updated` date.
"""

import argparse
import csv
import json
import os
import sys
from datetime import date


def load_metrics(run_dir: str, agg_seed: str):
    """Return {arm: {metric: value}} for the aggregate rows of a run."""
    mj = os.path.join(run_dir, "metrics.json")
    if os.path.exists(mj):
        data = json.load(open(mj))
        arms = data.get("arms")
        if arms and isinstance(arms, list):
            out = {}
            for a in arms:
                if a.get("seed") == agg_seed:
                    out[a["arm"]] = {
                        k: a[k] for k in ("R@1", "R@5", "R@10", "NDCG@10", "MRR") if k in a
                    }
            return out, "metrics.json(agg rows)"
    et = os.path.join(run_dir, "results.csv")
    if not os.path.exists(et):
        et = os.path.join(run_dir, "eval_test.csv")
    if os.path.exists(et):
        out = {}
        for row in csv.DictReader(open(et)):
            if row.get("seed") == agg_seed:
                out[row["arm"]] = {
                    k: row[k] for k in ("R@1", "R@5", "R@10", "NDCG@10", "MRR") if k in row
                }
        return out, "eval_test.csv(agg rows)"
    raise SystemExit(f"no metrics.json or eval_test.csv under {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--results-csv", default=None,
                    help="canonical copy of the run's eval_test.csv (never overwritten)")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--commit", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--seed", default="42,7,1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--objective", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--status", default="COMPLETED")
    ap.add_argument("--metrics-arm-seed", default="agg")
    ap.add_argument("--extra", default=None,
                    help="JSON object merged into the registry entry")
    args = ap.parse_args()

    idx = json.load(open(args.index))
    if any(e["experiment_id"] == args.experiment_id for e in idx["experiments"]):
        raise SystemExit(f"REFUSING: {args.experiment_id} already in index (raw immutability)")

    metrics, src = load_metrics(args.run_dir, args.metrics_arm_seed)
    entry = {
        "experiment_id": args.experiment_id,
        "date": args.date,
        "git_commit": args.commit,
        "corpus": args.corpus,
        "seed": args.seed,
        "model": args.model,
        "objective": args.objective,
        "metrics": metrics,
        "output_files": sorted(
            os.listdir(args.run_dir)
        ),
        "hardware": args.hardware,
        "status": args.status,
    }
    if args.extra:
        entry.update(json.loads(args.extra))
    idx["experiments"].append(entry)
    idx["updated"] = args.date
    json.dump(idx, open(args.index, "w"), indent=1)

    man = json.load(open(args.manifest))
    if args.experiment_id not in man["experiments"]:
        man["experiments"].append(args.experiment_id)
    man["updated"] = args.date
    json.dump(man, open(args.manifest, "w"), indent=1)

    if args.results_csv:
        if os.path.exists(args.results_csv):
            raise SystemExit(f"REFUSING: {args.results_csv} exists (results immutability)")
        src_csv = os.path.join(args.run_dir, "results.csv")
        if not os.path.exists(src_csv):
            src_csv = os.path.join(args.run_dir, "eval_test.csv")
        with open(src_csv) as f_in, open(args.results_csv, "w") as f_out:
            f_out.write(f_in.read())

    print(f"recorded {args.experiment_id}: {len(metrics)} agg arms from {src}")
    if args.results_csv:
        print(f"results copy -> {args.results_csv}")


if __name__ == "__main__":
    main()
