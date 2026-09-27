import json, os, subprocess, sys

CS = r"H:\Attention-DB\AttentionDB\research\comparative-study"
RAW = os.path.join(CS, "raw")
OUT = os.path.join(RAW, "C4-BINVERIFY-001", "dist")
BIN = os.path.join(CS, "c2", "probe", "target", "release", "c2pilot.exe")
os.makedirs(OUT, exist_ok=True)

CELLS = [
    ("NFC-B1-EF64", "C4-W19-NFC-B1-EF-001__EF64", 0.1351),
    ("NFC-B2-EF16-C50", "C4-W19-NFC-B2-EF-001__EF16-C50", 0.1393),
    ("NFC-B7-EF64", "C4-W19-NFC-B7-EF-001__EF64", 0.1372),
    ("SCI-B1-EF32", "C4-W19-SCI-B1-EF-001__EF32", 0.7775),
]
REPS = 5

import statistics

for name, cfg_id, recorded in CELLS:
    cfg = os.path.join(RAW, "C4-BINVERIFY-001", f"{cfg_id}.cfg.json")
    recs = []
    for i in range(1, REPS + 1):
        out = os.path.join(OUT, f"{name}.rep{i}.json")
        if not os.path.exists(out):
            subprocess.run([BIN, "--config", cfg, "--out", out], check=True,
                           capture_output=True, text=True, timeout=1800)
        a = json.load(open(out))
        pq = a["per_query"]
        r = sum(p["recall10_qrels"] for p in pq) / len(pq)
        recs.append(r)
    m = statistics.mean(recs); sd = statistics.stdev(recs) if REPS > 1 else 0.0
    within = sum(1 for r in recs if abs(r - recorded) <= 0.01 + 1e-9)
    print(f"{name}: recorded_c4_2={recorded:.4f} | new mean={m:.4f} sd={sd:.4f} "
          f"min={min(recs):.4f} max={max(recs):.4f} reps={recs} within_tol={within}/{REPS}")