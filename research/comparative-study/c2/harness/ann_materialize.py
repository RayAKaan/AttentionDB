"""C2 ANN dataset materialization + NN-GT verification (protocol §11).

DS-ANN-GLOVE-25 / DS-ANN-GLOVE-50 — EFFICIENCY TRACK, NN-GROUND-TRUTH ONLY,
never semantic. Verifies official train/test/neighbors/distances, dims,
dtypes, counts; recomputes exact top-100 for ALL test queries with the B0
oracle and compares to the shipped ground truth. Preregistered query split:
seed 20260925 shuffle of the 10k queries; first 2000 = tuning-validation,
remaining 8000 = test.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import h5py
from adb_common import Run, sha256_file, RAW

DL = f"{RAW}/datasets/downloads"
SEED = 20260925

def materialize(dim: int):
    try:
        _materialize(dim)
    except SystemExit:
        raise
    except Exception as e:
        import glob
        d = sorted(glob.glob(f"{RAW}/C2-DATA-GLOVE{dim}-002"))[-1]
        with open(f"{d}/stderr.log", "a") as f:
            import traceback; traceback.print_exc(file=f)
        raise

def _materialize(dim: int):
    ds_id = f"DS-ANN-GLOVE-{dim}"
    path = f"{DL}/glove-{dim}-angular.hdf5"
    run = Run(f"C2-DATA-GLOVE{dim}-003",
              f"materialize + verify + full NN-GT recompute {ds_id} "
              "(NN-GROUND-TRUTH / EFFICIENCY TRACK — not a semantic dataset)",
              {"dataset_id": ds_id, "component": "ann-materialization", "seed": SEED},
              use_sampler=True)
    if not os.path.exists(path):
        run.finish("BLOCKED", {"error": f"source file missing: {path}"}, reason="download missing")
        return
    f = h5py.File(path, "r")
    train, test = f["train"][:], f["test"][:]
    neighbors, distances = f["neighbors"][:], f["distances"][:]
    structure = {
        "file_sha256": sha256_file(path),
        "file_bytes": os.path.getsize(path),
        "train_shape": list(train.shape), "train_dtype": str(train.dtype),
        "test_shape": list(test.shape), "neighbors_shape": list(neighbors.shape),
        "distances_shape": list(distances.shape),
        "dim": int(train.shape[1]), "dim_matches_id": int(train.shape[1]) == dim,
        "track_label": "NN-GROUND-TRUTH / EFFICIENCY TRACK (never semantic)",
        "upstream": "https://ann-benchmarks.com/glove-%d-angular.hdf5 (ann-benchmarks official pre-split HDF5)" % dim,
    }
    run.print(json.dumps(structure, indent=2))

    # preregistered split: seeded shuffle of test queries
    perm = np.random.default_rng(SEED).permutation(test.shape[0])
    tune_idx, test_idx = perm[:2000], perm[2000:]

    # full exact recompute, batched (B0 oracle, numpy) — verify shipped NN-GT
    # memory-bounded: batch 64 x 1.18M float32 score matrix ~= 150 MiB
    Xn = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-30)
    K = neighbors.shape[1]
    mism = 0
    ties = 0
    checked = 0
    B = 32
    import gc; gc.collect()
    for s in range(0, test.shape[0], B):
        Q = test[s:s + B]
        Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-30)
        S = Qn @ Xn.T
        part = np.argpartition(-S, K, axis=1)[:, :K]
        pscores = np.take_along_axis(S, part, axis=1)
        order = np.argsort(-pscores, kind="stable", axis=1)
        ids = np.take_along_axis(part, order, axis=1)
        for r in range(ids.shape[0]):
            if set(ids[r].tolist()) != set(neighbors[s + r].tolist()):
                # tie-aware adjudication: a swap counts as a GT defect ONLY if
                # the swapped doc's exact score differs from the shipped
                # rank-100 boundary distance beyond float32 tolerance
                mine = set(ids[r].tolist())
                theirs = set(neighbors[s + r].tolist())
                boundary_ang = float(distances[s + r][K - 1])
                missing = theirs - mine
                defect = False
                for m in missing:
                    if abs((1.0 - float(S[r][m])) - boundary_ang) > 1e-6:
                        defect = True
                if defect:
                    mism += 1
                else:
                    ties += 1
            checked += 1
        del S, part, pscores, order, ids
    gt_check = {"queries_checked": checked, "top100_set_mismatches": mism,
                "exact_boundary_tie_swaps": ties,
                "match_rate": 1.0 - mism / max(1, checked),
                "method": "full exact recompute (B0 oracle) vs shipped neighbors, set equality per query",
                "tolerance_note": "set equality used; float association may reorder exact ties only"}
    run.print(json.dumps(gt_check, indent=2))

    # persist preregistered split indices as artifacts
    np.save(f"{run.dir}/artifacts/split-tune-idx.npy", tune_idx)
    np.save(f"{run.dir}/artifacts/split-test-idx.npy", test_idx)
    ok = (structure["dim_matches_id"] and test.shape[0] == 10000
          and gt_check["top100_set_mismatches"] == 0)
    metrics = {**structure, "split": {"tune": 2000, "test": 8000, "seed": SEED}, "nn_gt_check": gt_check}
    run.finish("PASS" if ok else "FAILED", metrics)

if __name__ == "__main__":
    for d in sys.argv[1:]:
        materialize(int(d))
