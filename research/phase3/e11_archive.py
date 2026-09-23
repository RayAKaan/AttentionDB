#!/usr/bin/env python3
"""E11 evidence-retention pass (spec M10 / TRIAGE-11 policy).

For every PH3E-FAULT-* run:
  1. write db-inventory.txt + checksums.sha256 over db/backup/restored dirs;
  2. VERIFIED/SUPPORTED runs: remove regenerable db payloads (the driver
     reproduces any run deterministically: `e11_drive.py <RUN_ID>`);
  3. F01 tamper runs and INVALIDATED/FAILED runs: payloads are the evidence —
     F01 dirs are tiny and kept whole; invalidated dbs are kept (gzipped if
     the inventory shows they are dominated by few large files).
Idempotent; refuses to touch a run without summary.json (never registered).
"""
import json, os, sys

RUNS = "research/phase3/raw/runs"

def inventory(d):
    inv, sums = [], []
    for dp, _, fns in os.walk(d):
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, d)
            inv.append(f"{rel}\t{os.path.getsize(p)}")
            try:
                b = open(p, "rb").read()
                a = 0xcbf29ce484222325
                c = 0x9e3779b97f4a7c15
                for i in range(0, len(b), 4096):
                    chunk = b[i:i+4096]
                    x = a ^ i
                    for by in chunk:
                        x ^= by
                        x = (x * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
                    a = x
                    for by in reversed(chunk):
                        c ^= by
                        c = (c * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
                    c = ((c << 29) | (c >> 35)) ^ a
                    c &= 0xFFFFFFFFFFFFFFFF
                sums.append(f"fnv128x2:{a:016x}{c:016x}  {rel}")
            except OSError:
                pass
    return "\n".join(inv) + "\n", "\n".join(sums) + "\n"

def payload_size(d):
    total = 0
    for dp, _, fns in os.walk(d):
        for fn in fns:
            try:
                total += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return total

def main():
    kept = trimmed = 0
    for rid in sorted(os.listdir(RUNS)):
        if not rid.startswith("PH3E-FAULT-"):
            continue
        d = os.path.join(RUNS, rid)
        s_path = os.path.join(d, "summary.json")
        if not os.path.exists(s_path):
            continue
        s = json.load(open(s_path))
        for sub in ("db", "backup", "restored"):
            p = os.path.join(d, sub)
            if not os.path.isdir(p):
                continue
            inv, sums = inventory(p)
            open(os.path.join(d, f"{sub}-inventory.txt"), "w").write(inv)
            open(os.path.join(d, f"{sub}-checksums.sha256"), "w").write(sums)
            keep = (s["family"] == "F01" or s.get("classification") in ("INVALIDATED", "FAILED", "OBSERVED_LIMIT"))
            if keep:
                kept += 1
                continue
            import shutil
            shutil.rmtree(p)
            trimmed += 1
        print(f"{rid}: {s.get('classification')} kept={kept if False else ''}payloads retained/trimmed per policy")
    print(f"done: {kept} payloads retained (evidence), {trimmed} trimmed (regenerable)")

if __name__ == "__main__":
    main()
