"""C2 B3 orchestration: training grid + card validation + leakage tests.

Runs after embeddings exist (LODO arm needs 384-d views).
Run dirs: C2-B3-TRAIN-001 (synth-concat), -002 (synth-view0), -003 (lodo-nf),
          C2-B3-LEAK-002 (negative tests), C2-B3-VALID-004 (engine activation).
"""
import glob, hashlib, json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import yaml
from adb_common import Run, sha256_file, RAW, guarded_popen, git_commit

C2 = f"{RAW}/../c2"
PROBE = f"{C2}/probe/target/release/c2probe"
B3 = f"{C2}/b3"
DS = f"{B3}/datasets"
CARDS = f"{B3}/modelcards"
CFG = yaml.safe_load(open(f"{B3}/training-config.yaml"))

def best_card(model_id):
    """Grid selection: min best_val_loss across the arm's .meta.json files (primary seed only)."""
    metas = []
    for p in glob.glob(f"{CARDS}/{model_id}-*.meta.json"):
        m = json.load(open(p))
        if m["seed"] == CFG["seeds"]["primary"]:
            cp = m["card_path"]
            if not os.path.isabs(cp):
                cp = os.path.normpath(os.path.join(os.getcwd(), cp))
            if not os.path.exists(cp):
                cp = os.path.join(CARDS, os.path.basename(cp))
            metas.append((m["best_val_loss"], m["lr"], m["hidden"], cp))
    metas.sort()
    return metas[0] if metas else None

def train_arm(arm_id, dataset, run_id, head_names=None):
    ds_path = f"{DS}/{dataset}"
    ds_hash = sha256_file(ds_path)
    run = Run(run_id, f"B3 training arm {arm_id} (grid + primary seed)",
              {"arm": arm_id, "dataset": dataset, "dataset_sha256": ds_hash,
               "config_yaml_sha256": sha256_file(f"{B3}/training-config.yaml"),
               "seed": CFG["seeds"]["primary"], "grid": CFG["grid"],
               "trainer": "learned::gating_v2::train_gating (in-repo)"},
              use_sampler=True)
    t0 = time.time()
    for hidden in CFG["grid"]["hidden"]:
        for lr in CFG["grid"]["lr"]:
            cmd = [PROBE, "train", ds_path, CARDS, f"{arm_id}", str(CFG["seeds"]["primary"]),
                   str(hidden), str(lr)]
            env = {"C2_GIT_COMMIT": git_commit(),
                   "C2_HARDWARE": f"{os.cpu_count()}vCPU-sandbox-2GiB"}
            if head_names:
                env["C2_HEAD_NAMES"] = head_names
            rc, summ, log = guarded_popen(cmd, run, timeout_s=600, env_extra=env)
            run.print(f"train hidden={hidden} lr={lr} rc={rc} peak_rss_kb={summ['peak_rss_kb']}")
    # multiseed variance on the selected config (3 seeds total, C1 rule)
    sel = best_card(arm_id)
    seeds_used = [CFG["seeds"]["primary"]]
    if sel:
        for s in CFG["seeds"]["variance"]:
            cmd = [PROBE, "train", ds_path, CARDS, f"{arm_id}", str(s), str(sel[2]), str(sel[1])]
            rc, summ, log = guarded_popen(cmd, run, timeout_s=600, env_extra=env)
            run.print(f"variance seed={s} rc={rc}")
            seeds_used.append(s)
    run.finish("PASS" if sel else "FAILED", {
        "selected": {"best_val_loss": sel[0], "lr": sel[1], "hidden": sel[2], "card": sel[3]} if sel else None,
        "seeds_trained": seeds_used, "wall_s": round(time.time() - t0, 1)})

def leakage_tests():
    run = Run("C2-B3-LEAK-002", "B3 leakage negative tests (INV-L1..L5)",
              {"component": "leakage-validation"})
    results = {}
    # INV-L1: craft a headline-test-marked dataset and attempt training
    bad = f"{run.dir}/artifacts/headline-test-export.json"
    os.makedirs(os.path.dirname(bad), exist_ok=True)
    ds = json.load(open(f"{DS}/gating-synth-view0.json"))
    ds["corpus_desc"] = "headline-test-synthetic-MUST-BE-REFUSED"
    open(bad, "w").write(json.dumps(ds))
    rc, summ, log = guarded_popen([PROBE, "train", bad, f"{run.dir}/artifacts",
                                   "leak-attempt", "1", "8", "0.01"], run, timeout_s=120)
    results["INV-L1-headline-refused"] = (rc == 2)
    # INV-L1b: test-exports path refusal
    p2 = f"{run.dir}/artifacts/test-exports/gating.json"
    os.makedirs(os.path.dirname(p2), exist_ok=True)
    open(p2, "w").write(json.dumps(ds))
    rc2, _, _ = guarded_popen([PROBE, "train", p2, f"{run.dir}/artifacts",
                               "leak-attempt2", "1", "8", "0.01"], run, timeout_s=120)
    results["INV-L1-test-path-refused"] = (rc2 == 2)
    # INV-L4: card without valid TrainingMeta -> engine/validate refuses (probe b3-validate 011c)
    results["INV-L4"] = "covered by TEST-C2-011c (tampered card rejected by validate())"
    # INV-L5: bitwise reproducibility recorded per train invocation
    repro = []
    for p in glob.glob(f"{CARDS}/*.meta.json"):
        m = json.load(open(p))
        if "reproducible_bitwise" in m:
            repro.append(m["reproducible_bitwise"])
    results["INV-L5-bitwise-reproducible-all"] = all(repro) if repro else None
    # INV-L2: config hash freeze — this run records the hash of training-config.yaml
    results["INV-L2-config-hash"] = sha256_file(f"{B3}/training-config.yaml")
    # INV-L3: datasets carry provenance (corpus_desc + seed); targets computed by
    # probe gating-dataset from exported training corpora only
    prov = {}
    for p in glob.glob(f"{DS}/gating-*.json"):
        d = json.loads(open(p).read().replace("\n", ""))
        prov[os.path.basename(p)] = d.get("corpus_desc", "?")
    results["INV-L3-dataset-provenance"] = prov
    run.print(json.dumps(results, indent=2))
    ok = results["INV-L1-headline-refused"] and results["INV-L1-test-path-refused"] and results.get("INV-L5-bitwise-reproducible-all")
    run.finish("PASS" if ok else "FAILED", results)

def engine_activation(card_path):
    run = Run("C2-B3-VALID-004", "B3 ModelCard engine activation validation",
              {"card": os.path.basename(card_path), "card_sha256": sha256_file(card_path)})
    rc, summ, log = guarded_popen([PROBE, "b3-validate", card_path, f"{run.dir}/artifacts/b3-validate.json"],
                                  run, timeout_s=600)
    try:
        res = json.load(open(f"{run.dir}/artifacts/b3-validate.json"))
        all_pass = all(r["status"] == "PASS" for r in res["results"])
    except Exception:
        res, all_pass = {"raw": log[-2000:]}, False
    run.finish("PASS" if (rc == 0 and all_pass) else "FAILED", res)

if __name__ == "__main__":
    step = sys.argv[1]
    if step == "train-synth":
        train_arm("b3-synth-concat-v2", "gating-synth-concat.json", "C2-B3-TRAIN-009", "semantic,lexical,metadata")
        train_arm("b3-synth-view0-v2", "gating-synth-view0.json", "C2-B3-TRAIN-010", "semantic,lexical,metadata")
    elif step == "activate-synth":
        sel = best_card("b3-synth-view0")
        engine_activation(sel[3])
    elif step == "leak":
        leakage_tests()
    elif step == "train-lodo":
        train_arm("b3-lodo-nf-v2", "gating-lodo-nfcorpus.json", "C2-B3-TRAIN-011", "HEAD-TITLE,HEAD-BODY,HEAD-CITE")
    elif step == "show":
        for arm in ("b3-synth-concat", "b3-synth-view0", "b3-lodo-nf"):
            print(arm, "->", best_card(arm))
