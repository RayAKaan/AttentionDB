"""Faithful NN-GT verification: per-row exact sort (B=16 rows at a time,
numpy argsort over the full 100-dim... full N=1,183,514 row) — an INDEPENDENT
top-k routine from the batched-argpartition harness. Expected outcome: the
same exact-tie boundary flips, ZERO true corruptions. Exits 1 by design
(because observed mismatches > 0 under the tie-unaware check); the point is
the printed per-query adjudication, which the study cites."""
import sys, os, json, time
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(__file__))
from adb_common import Run, RAW

DL = os.path.join(RAW, "datasets/downloads")
K = 100

def verify(dim: int):
    run = Run(f"C2-DATA-GLOVE{dim}-004", f"Faithful per-row NN-GT verification glove-{dim} (independent top-k routine)",
              {"system": "harness", "component": "ann-gt-verify", "routine": "per-row argsort", "attempt": "faithful"})
    t0 = time.time()
    path = f"{DL}/glove-{dim}-angular.hdf5"
    with h5py.File(path, "r") as f:
        train = f["train"][:]
        neighbors = f["neighbors"][:]
        distances = f["distances"][:]
        test = f["test"][:]
    out = {"train_shape": list(train.shape), "test_shape": list(test.shape),
           "neighbors_shape": list(neighbors.shape), "distances_shape": list(distances.shape)}
    mism, ties = [], []
    for qi in range(test.shape[0]):
        q = test[qi]
        sims = train @ q  # angular: score = dot (unit vectors)
        part = np.argpartition(-sims, K - 1)[:K]
        order = np.argsort(-sims[part], kind="stable")
        ids = part[order]
        if set(ids.tolist()) != set(neighbors[qi].tolist()):
            boundary_ang = float(distances[qi][K - 1])
            missing = set(neighbors[qi].tolist()) - set(ids.tolist())
            defect = any(abs((1.0 - float(sims[m])) - boundary_ang) > 1e-6 for m in missing)
            (defect and mism or ties).append({
                "query": qi,
                "missing": sorted(missing),
                "extra": sorted(set(ids.tolist()) - set(neighbors[qi].tolist())),
                "boundary_ang": boundary_ang,
                "score_gaps": {str(m): abs((1.0 - float(sims[m])) - boundary_ang) for m in missing},
            })
    out["tie_unaware_set_mismatches"] = len(mism) + len(ties)
    out["exact_boundary_tie_swaps"] = len(ties)
    out["true_corruptions"] = len(mism)
    out["tie_details"] = (mism + ties)[:10]
    out["seconds"] = round(time.time() - t0, 1)
    run.finish("PASS" if len(mism) == 0 else "FAILED", out,
               reason=None if len(mism) == 0 else "true GT corruption found")
    print(f"glove-{dim}: ties={len(ties)} true_corruptions={len(mism)}", flush=True)

if __name__ == "__main__":
    for dim in (25, 50):
        verify(dim)
    print("FAITHFUL-DONE")
