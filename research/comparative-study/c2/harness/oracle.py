"""C2 oracle: numpy exact top-k + validation battery (protocol §12).

B0 exact oracle, independent of the Rust engine and of every ANN system.
Tie order: score DESC, id ASC (matches engine deterministic top-k).
Run: python3 oracle.py  ->  run dir C2-ORACLE-TESTS-002
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from adb_common import Run

def exact_topk(X: np.ndarray, Q: np.ndarray, k: int, metric: str = "cosine") -> tuple[np.ndarray, np.ndarray]:
    """X (n,d) float32, Q (m,d). Returns (ids (m,k), scores (m,k)); score desc, id asc."""
    if metric == "cosine":
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
        Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-30)
        S = Qn @ Xn.T
    elif metric == "dot":
        S = Q @ X.T
    elif metric == "euclidean":
        S = -np.sqrt(((Q[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    else:
        raise ValueError(metric)
    m, n = S.shape
    kk = min(k, n)
    ids = np.argsort(-S, kind="stable", axis=1)[:, :kk]  # stable: ties keep id (row) order
    return ids.astype(np.int64), np.take_along_axis(S, ids, axis=1)

def pure_python_reference(X, Q, k):
    """Independent brute-force (no numpy vectorization) for cross-check."""
    out = []
    for q in Q:
        scored = []
        for i, x in enumerate(X):
            dot = sum(float(a) * float(b) for a, b in zip(q, x))
            na = sum(float(a) * float(a) for a in x) ** 0.5
            nb = sum(float(a) * float(a) for a in q) ** 0.5
            scored.append((dot / (na * nb) if na * nb > 0 else 0.0, i))
        scored.sort(key=lambda t: (-t[0], t[1]))
        out.append([i for _, i in scored[:k]])
    return out

def main():
    run = Run("C2-ORACLE-TESTS-002",
              "B0 exact oracle validation battery (protocol §12)",
              {"component": "oracle", "metric": "cosine", "tie_rule": "score desc, id asc"})
    tests = []

    # 1. tiny hand-checkable dataset
    X = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.7, 0.7]], dtype=np.float32)
    Q = np.array([[1, 0]], dtype=np.float32)
    ids, sc = exact_topk(X, Q, 3)
    t1 = ids[0].tolist() == [0, 1, 3]
    tests.append(("hand-check tiny", t1, {"expected": [0, 1, 3], "got": ids[0].tolist(), "scores": np.round(sc[0], 6).tolist()}))

    # 2. independent implementation cross-check (numpy vs pure python)
    rng = np.random.default_rng(20260925)
    X2 = rng.standard_normal((120, 16)).astype(np.float32)
    Q2 = rng.standard_normal((9, 16)).astype(np.float32)
    ids2, _ = exact_topk(X2, Q2, 10)
    ref = pure_python_reference(X2, Q2, 10)
    t2 = all(ids2[i].tolist() == ref[i] for i in range(len(Q2)))
    tests.append(("independent cross-check numpy vs pure-python", t2, {"queries": len(Q2)}))

    # 3. deterministic repeat
    a = exact_topk(X2, Q2, 10)[0]
    b = exact_topk(X2, Q2, 10)[0]
    tests.append(("deterministic repeat", bool((a == b).all()), {}))

    # 4. tie case: duplicated vectors -> id ascending among equals
    Xt = np.array([[1.0, 0.0]] * 5 + [[0.0, 1.0]], dtype=np.float32)
    Qt = np.array([[1.0, 0.0]], dtype=np.float32)
    idst, _ = exact_topk(Xt, Qt, 6)
    ties = idst[0][:5].tolist() == [0, 1, 2, 3, 4]
    tests.append(("tie order id-ascending", ties, {"got": idst[0].tolist()}))

    # 5. top-k boundary: k > n, k == n, k == 0
    ok_b = True
    try:
        i_big, s_big = exact_topk(X, Q, 99); ok_b &= i_big.shape[1] == 4
        i_eq, _ = exact_topk(X, Q, 4); ok_b &= i_eq.shape[1] == 4
        i_zero, _ = exact_topk(X, Q, 0); ok_b &= i_zero.shape[1] == 0
    except Exception as e:
        ok_b = False
    tests.append(("top-k boundary (0, n, k>n)", ok_b, {}))

    # 6. empty input
    ok_e = True
    try:
        i_empty, _ = exact_topk(np.zeros((0, 16), np.float32), Q2, 10)
        ok_e &= i_empty.shape == (9, 0)
    except Exception:
        ok_e = False
    tests.append(("empty corpus input", ok_e, {}))

    # 7. recall sanity on moderate random set vs scaled check (invariance)
    X3 = rng.standard_normal((5000, 32)).astype(np.float32)
    Q3 = rng.standard_normal((20, 32)).astype(np.float32)
    i3, s3 = exact_topk(X3, Q3, 10)
    ok_s = bool((np.diff(s3, axis=1) <= 1e-6).all())
    tests.append(("scores monotonically non-increasing", ok_s, {"n": 5000}))

    metrics = {"tests": [{"name": n, "status": "PASS" if ok else "FAILED", **d} for n, ok, d in tests],
               "n_pass": sum(1 for _, ok, _ in tests if ok), "n_total": len(tests)}
    status = "PASS" if all(ok for _, ok, _ in tests) else "FAILED"
    run.print(json.dumps(metrics, indent=2))
    run.finish(status, metrics)

if __name__ == "__main__":
    main()
