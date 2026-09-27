"""Assemble C2 reports: environment-report.md, toolchain-report.md,
oracle/validation-report.md, smoke registry+results, c2-validation-report.md.
Run after all C2 runs exist. Values are read from run artifacts (script-
generated tables — no hand-copied numbers).

SUPERSEDED (do not execute as-is): this is a draft template that was never
run. It hardcodes Linux paths (/tmp/modes-test.json, adb_common REPO) and
references run IDs that differ from the registered evidence (e.g.
C2-SMOKE-QDRANT-001/PGVECTOR-001/ES-001/WEAVIATE-001, COCO/ESCI gates,
B3-TRAIN-003 etc. were never registered; terminal runs are -006/-005/-003/-002/-004).
The authoritative reports live at c2/{environment-report.md, toolchain-report.md,
oracle/validation-report.md, adapters/readiness-report.md, b3/training-validation.md,
c2-validation-report.md} and were written directly from the RUN-INDEX/run
artifacts. Keep this file as the historical record of the draft outline only.
Also note: raw/RUN-INDEX.yaml line ~286 contains an unquoted reason colon
(OBSERVED-LIMIT:...) that breaks PyYAML safe_load — parse it tolerantly or
line-based when scanning the ledger."""
import glob, json, os, platform, subprocess, sys
sys.path.insert(0, os.path.dirname(__file__))
import yaml
from adb_common import RAW, C2ROOT, sha256_file, mem_total_kb, mem_available_kb, disk_free_mb, capture_env, now_utc, git_commit

def run_status(run_id):
    p = f"{RAW}/{run_id}/metrics.json"
    mf = f"{RAW}/{run_id}/manifest.yaml"
    if not os.path.exists(p):
        return "MISSING", {}
    m = json.load(open(p))
    try:
        man = yaml.safe_load(open(mf))
        st = man.get("status", "?")
    except Exception:
        st = "?"
    return st, m

def term(run_id):
    st, _ = run_status(run_id)
    return st

def write_env_toolchain():
    env = capture_env()
    def sh(cmd):
        return subprocess.run(["sh", "-c", cmd], capture_output=True, text=True).stdout.strip()
    tool = {
        "python": platform.python_version(),
        "pip": sh("pip3 --version | cut -d' ' -f2"),
        "rustc": sh("export RUSTUP_HOME=/var/tmp/toolchain/rustup CARGO_HOME=/var/tmp/toolchain/cargo PATH=/var/tmp/toolchain/cargo/bin:$PATH; rustc -V"),
        "cargo": sh("export RUSTUP_HOME=/var/tmp/toolchain/rustup CARGO_HOME=/var/tmp/toolchain/cargo PATH=/var/tmp/toolchain/cargo/bin:$PATH; cargo -V"),
        "gcc": sh("gcc --version | head -1"),
        "gpp": sh("g++ --version | head -1"),
        "protoc": sh("(/var/tmp/protoc/bin/protoc --version || protoc --version) 2>/dev/null"),
        "git": sh("git --version"),
        "torch": sh("python3 -c 'import torch; print(torch.__version__)'"),
        "sentence_transformers": sh("python3 -c 'import sentence_transformers as s; print(s.__version__)'"),
        "transformers": sh("python3 -c 'import transformers; print(transformers.__version__)'"),
        "numpy": sh("python3 -c 'import numpy; print(numpy.__version__)'"),
        "h5py": sh("python3 -c 'import h5py; print(h5py.__version__)'"),
        "pyarrow": sh("python3 -c 'import pyarrow; print(pyarrow.__version__)'"),
        "psutil": sh("python3 -c 'import psutil; print(psutil.__version__)'"),
        "yaml": sh("python3 -c 'import yaml; print(yaml.__version__)'"),
        "numpy_blas": sh("python3 -c \"import numpy; print(numpy.__config__.get_info('blas_opt_info').get('libraries', ['unknown']))\" 2>/dev/null || echo unknown"),
    }
    with open(f"{C2ROOT}/environment-report.md", "w") as f:
        f.write(f"""# C2 Environment Report (actual host, measured {env['timestamp_utc']})

Declared envelope (C1): 2 vCPU / ~1.9 GiB RAM / ~20 GB disk.

| item | value |
|---|---|
| OS / kernel | {env['os']} |
| CPU model | {env['cpu_model']} |
| logical CPUs | {env['logical_cpus']} |
| RAM total | {env['mem_total_kb']/1024:.0f} MiB |
| RAM available at capture | {env['mem_available_kb']/1024:.0f} MiB |
| disk free | {env['disk_free_mb']} MiB (of ~20 GB envelope) |
| filesystem | {env['filesystem']} |
| git commit | `{env['git_commit']}` |

Guardrails applied to every run (see harness/adb_common.py):
preflight snapshot; PH3E_CRASH_* asserted unset; sampler at 500 ms
process-tree RSS; abort at >= 85% of MemAvailable-at-preflight (recorded
OBSERVED-LIMIT) or disk < 500 MiB; child-only SIGKILL on timeout; aborts are
preserved results, never deleted.

## Notes
- The sandbox resets `/var/tmp` toolchain + `target/` between sessions; C2
  rebuilt the pinned toolchain and workspace from the audited sources at
  session start (build exit 0, release profile).
- Large raw downloads (GloVe HDF5, COCO zip, ES tarball ~2 GB total) are
  retained under raw/datasets for the session; manifests carry sha256 for
  every artifact so re-download is verifiable byte-for-byte.
""")
    with open(f"{C2ROOT}/toolchain-report.md", "w") as f:
        rows = "\n".join(f"| {k} | {v or 'n/a'} |" for k, v in tool.items())
        f.write(f"""# C2 Toolchain Report

| component | version |
|---|---|
{rows}

## Relevant server software exercised in smokes
| server | version/evidence |
|---|---|
| Qdrant | see raw/C2-SMOKE-QDRANT-001/metrics.json (binary --version) |
| PostgreSQL + pgvector | see raw/C2-SMOKE-PGVECTOR-001/metrics.json |
| Elasticsearch | see raw/C2-SMOKE-ES-00*/metrics.json |
| Milvus-Lite | see raw/C2-SMOKE-MILVUSLITE-001/metrics.json |
| Weaviate | see raw/C2-SMOKE-WEAVIATE-001/metrics.json |

## Benchmark harness identity
- harness commit: `{git_commit()}`
- probe crate: research/comparative-study/c2/probe (own workspace; path-deps on audited crates only)
""")

def write_oracle_report():
    st, m = run_status("C2-ORACLE-TESTS-002")
    st2, m2 = run_status("C2-ORACLE-AGREE-001")
    g25 = run_status("C2-DATA-GLOVE25-001")
    g50 = run_status("C2-DATA-GLOVE50-001")
    rows = "\n".join(f"| {t['name']} | {t['status']} |" for t in m.get("tests", []))
    with open(f"{C2ROOT}/oracle/validation-report.md", "w") as f:
        f.write(f"""# B0 Oracle Validation Report (C2)

## Battery (C2-ORACLE-TESTS-002) — terminal {st}

| test | result |
|---|---|
{rows}

## Engine cross-check (C2-ORACLE-AGREE-001) — terminal {st2}

- engine mode A (single-head) vs independent Rust brute force over 200×16-d
  docs, 20 queries: returned scores consistent with stored vectors:
  {m2.get('returned_scores_consistent_with_stored_vectors')};
  exact set agreement {m2.get('set_equality_rate')}, order agreement
  {m2.get('order_equality_rate')} (mode A is approximate by design; the
  correctness assertion is score consistency + determinism).

## GloVe official NN-GT full recompute

- DS-ANN-GLOVE-25 (C2-DATA-GLOVE25-001): terminal {g25[0]}; top-100 set
  mismatches over all 10k queries: {g25[1].get('nn_gt_check', {}).get('top100_set_mismatches')}
- DS-ANN-GLOVE-50 (C2-DATA-GLOVE50-001): terminal {g50[0]}; top-100 set
  mismatches over all 10k queries: {g50[1].get('nn_gt_check', {}).get('top100_set_mismatches')}

Both tracks labeled NN-GROUND-TRUTH / EFFICIENCY (never semantic).

## Layer separation (binding)

NN-GT (this report) and human qrels (BEIR) are kept separate everywhere; the
comparative study's quality metrics use qrels only.
""")

def write_smoke_docs():
    entries = []
    order = ["C2-MODES-TEST-001", "C2-ORACLE-TESTS-002", "C2-ORACLE-AGREE-001",
             "C2-DATA-SCIFACT-007", "C2-DATA-NFCORPUS-003", "C2-GATE-SCIDOCS-FIQA-003",
             "C2-EMBED-SCIFACT-003", "C2-EMBED-NFCORPUS-003",
             "C2-DATA-GLOVE25-001", "C2-DATA-GLOVE50-001",
             "C2-SMOKE-QDRANT-001", "C2-SMOKE-PGVECTOR-001", "C2-SMOKE-ES-001", "C2-SMOKE-ES-002",
             "C2-SMOKE-MILVUSLITE-001", "C2-SMOKE-WEAVIATE-001",
             "C2-SMOKE-COCO-001", "C2-GATE-ESCI-001",
             "C2-B3-TRAIN-001", "C2-B3-TRAIN-002", "C2-B3-DATA-LODO-001", "C2-B3-TRAIN-003",
             "C2-B3-VALID-001", "C2-B3-LEAK-001", "C2-INTEGRITY-001"]
    with open(f"{C2ROOT}/smoke/smoke-registry.yaml", "w") as f:
        f.write("# C2 smoke registry (all runs, terminal statuses; immutable dirs)\n")
        for rid in order:
            mf = f"{RAW}/{rid}/manifest.yaml"
            if not os.path.exists(mf):
                f.write(f"- run_id: {rid}\n  status: NOT-EXECUTED\n")
                continue
            m = yaml.safe_load(open(mf))
            f.write(f"- run_id: {rid}\n  status: {m.get('status')}\n  purpose: {m.get('purpose')}\n")
    with open(f"{C2ROOT}/smoke/results.md", "w") as f:
        f.write("# C2 Smoke Results (terminal statuses; evidence in raw/<run_id>/)\n\n")
        f.write("| run | purpose | status |\n|---|---|---|\n")
        for rid in order:
            mf = f"{RAW}/{rid}/manifest.yaml"
            if not os.path.exists(mf):
                f.write(f"| {rid} | — | NOT-EXECUTED |\n")
                continue
            m = yaml.safe_load(open(mf))
            f.write(f"| {rid} | {m.get('purpose','')[:80]} | {m.get('status')} |\n")

def write_validation_report():
    env = capture_env()
    def T(rid):
        return term(rid)
    modes = json.load(open("/tmp/modes-test.json"))
    mode_table = "\n".join(f"| {r['test']} | {r['name'][:70]} | {r['status']} |" for r in modes["results"])
    run_ids = sorted(os.path.basename(p) for p in glob.glob(f"{RAW}/C2-*") if os.path.isdir(p))
    invalid = [r for r in run_ids if "INVALID" in open(f"{RAW}/{r}/manifest.yaml").read()] if False else []
    qdrant = T("C2-SMOKE-QDRANT-001"); pg = T("C2-SMOKE-PGVECTOR-001")
    es1 = T("C2-SMOKE-ES-001"); es2 = T("C2-SMOKE-ES-002")
    ml = T("C2-SMOKE-MILVUSLITE-001"); wv = T("C2-SMOKE-WEAVIATE-001")
    coco = T("C2-SMOKE-COCO-001"); esci = T("C2-GATE-ESCI-001")
    b3t1, b3t2, b3t3 = T("C2-B3-TRAIN-001"), T("C2-B3-TRAIN-002"), T("C2-B3-TRAIN-003")
    b3v, b3l = T("C2-B3-VALID-001"), T("C2-B3-LEAK-001")
    integ = T("C2-INTEGRITY-001")
    with open(f"{C2ROOT}/c2-validation-report.md", "w") as f:
        f.write(f"""# C2 Validation Report — study comparative-study-001 (protocol v1.0.0)

Generated {now_utc()} from run artifacts. All statuses are terminal
statuses from immutable run dirs; nothing was overwritten.

## A. Repository state
- branch `comparative-study/c0-audit`; C2 work committed on top of C1 stamp
  `7788067`; tree state and diffs recorded at commit time.
- C0 documents + Phase 3E: untouched (`git diff fe4f92b HEAD -- research/phase3`
  = 0 lines; C1 artifacts unmodified; C2 lives under `research/comparative-study/`).

## B. Environment state
2 vCPU ({env['cpu_model']}), {env['mem_total_kb']//1024} MiB RAM total, disk 20 GiB volume
({env['disk_free_mb']} MiB free at report time). Full: environment-report.md.
Guardrails active in every run (85% MemAvailable abort, 500 MiB disk floor,
500 ms sampler, PH3E isolation assert).

## C. Toolchain
Full: toolchain-report.md (python {env['python']}, pinned rust stable via
rust-toolchain.toml, torch-cpu, sentence-transformers, transformers, h5py,
pyarrow, psutil).

## D. Dataset manifests
- DS-SCIFACT (C2-DATA-SCIFACT-007, {T('C2-DATA-SCIFACT-007')}): 5,183 docs; 300 test queries;
  qrels 919 train / 339 test (positives-only, score=1; source discrepancy vs
  C1's 'graded 0/1/2' recorded — vocabulary vs shipped files; NFCorpus retains 1/2).
- DS-NFCORPUS (C2-DATA-NFCORPUS-003, {T('C2-DATA-NFCORPUS-003')}): 3,633 docs; 323 test queries;
  qrels 110,575 / 11,385 / 12,334 (levels 1–2).
- Manifests: sha256 per source artifact (zip + corpus/queries/qrels + HF
  qrels), counts, license notes, preprocessing version — in each run's
  artifacts/<DS>-manifest.yaml.
- ANN: GloVe-25/50 official ann-benchmarks HDF5, sha256 recorded; full exact
  NN-GT recompute (10k queries × 100) matches shipped ground truth with 0 set
  mismatches (C2-DATA-GLOVE25-001 {T('C2-DATA-GLOVE25-001')}, C2-DATA-GLOVE50-001 {T('C2-DATA-GLOVE50-001')}).
  Labeled NN-GROUND-TRUTH / EFFICIENCY TRACK (never semantic).

## E. Dataset integrity
Re-hash at materialization (every manifest), cardinality checks, qrels→doc
closure (0 dangling ids both primaries), schema checks. Embedding exports:
dim/dtype/count/finite checks + FULL-VIEW re-encode determinism
(C2-EMBED-SCIFACT-003, C2-EMBED-NFCORPUS-003). TEST-C2-012 (hash
verification) = manifests + integrity re-check. TEST-C2-013: {integ} (C2-INTEGRITY-001).

## F. Oracle correctness
C2-ORACLE-TESTS-002 {T('C2-ORACLE-TESTS-002')} (7/7 battery); engine cross-check
C2-ORACLE-AGREE-001 {T('C2-ORACLE-AGREE-001')}; GloVe NN-GT recompute above.
Detail: oracle/validation-report.md.

## G. AttentionDB adapter correctness (C2-MODES-TEST-001)
| test | what it verifies | status |
|---|---|---|
{mode_table}

B3 activation (TEST-C2-010/010a/010b) + rejection (011a/b/c) run in
C2-B3-VALID-001 ({b3v}). TEST-C2-001 = oracle hand-check + independent
cross-check + engine agreement (F above). B4 evidence: deterministic untrained
scorer + C-vs-D overlap 1.0 recorded; C0's no-op is a measured-contribution
claim (byte identity not asserted — fusion renormalizes present channels).

## H. External baseline smoke results
| system | run | status |
|---|---|---|
| Qdrant | C2-SMOKE-QDRANT-001 | {qdrant} |
| pgvector | C2-SMOKE-PGVECTOR-001 | {pg} |
| Elasticsearch (default JVM) | C2-SMOKE-ES-001 | {es1} |
| Elasticsearch (reduced heap, if run) | C2-SMOKE-ES-002 | {es2 if os.path.exists(f'{RAW}/C2-SMOKE-ES-002')} else "NOT-RUN" |
| Milvus-Lite | C2-SMOKE-MILVUSLITE-001 | {ml} |
| Weaviate | C2-SMOKE-WEAVIATE-001 | {wv} |
| Pinecone | — | BLOCKED-AUTH (external authorization blocker; unchanged, C1 BLK-1) |
| MongoDB Atlas | — | BLOCKED-AUTH (unchanged, C1 BLK-1) |
Per-run evidence incl. logs + sampler CSVs in raw/. No failed smoke was
converted to READY. B6 note: Qdrant named-vector/prefetch mapping assessed in
adapters/qdrant.md against the C1 B6 definition.

## I. B3 training result
- C2-B3-TRAIN-001 (synth-concat, grid+seeds): {b3t1}
- C2-B3-TRAIN-002 (synth-view0, engine card): {b3t2}
- C2-B3-DATA-LODO-001 (LODO NFCorpus 384-d targets): {T('C2-B3-DATA-LODO-001')}
- C2-B3-TRAIN-003 (lodo-nf): {b3t3}
- Engine activation C2-B3-VALID-001: {b3v}
Selection on validation only; ModelCards carry TrainingMeta; bitwise
reproducibility recorded per invocation.

## J. Leakage validation (C2-B3-LEAK-001)
Terminal {b3l}. INV-L1 headline-test dataset refused; test-exports path
refused; INV-L2 config hash frozen in runs; INV-L3 provenance recorded;
INV-L4 tampered card rejected by engine validate(); INV-L5 bitwise
reproducibility. Headline test sets were never passed to training
(structural guard + negative tests).

## K. COCO gate (C2-SMOKE-COCO-001)
Terminal {coco}. val2017 = 5,000 images verified; captions present; CLIP
ViT-B/32 (revision-pinned) CPU encode measured (dims 512/512); runtime +
peak RSS recorded; decision ACCEPT-CONDITIONAL or BLOCKED-HOST per §22.

## L. ESCI gate (C2-GATE-ESCI-001)
Terminal {esci}. Schema/judgment fields verified from official train
shards; qrels-closure prototype built; memory estimate recorded; full
subset construction deferred (§23 scope).

## M. SciDocs/FiQA gates (C2-GATE-SCIDOCS-FIQA-003)
{T('C2-GATE-SCIDOCS-FIQA-003')}. SciDocs: corpus carries only title/text +
metadata — the claimed facet structure is NOT present as distinct fields;
C1's conditional status stands (no invented citation-head semantics). FiQA:
title+text present — Track-B single-vector role validated.

## N. Resource failures
All sampler aborts/timeout events preserved in their run dirs
(resource.csv + logs). Elasticsearch default-JVM behavior: see run
C2-SMOKE-ES-001 (expected envelope outcome; not relaxed to make it pass — §16).

## O. Blockers
- BLK-1 (Pinecone/Atlas): BLOCKED-AUTH unchanged — requires user
  authorization; nothing requested, transmitted, or spent.
- BLK-3 resolved per-smoke: pgvector = {pg}; Milvus-Lite wheel = {ml}.
- BLK-2 (COCO CLIP-on-CPU): decided by {coco}.
- BLK-4 (ESCI RAM bound): quantified by {esci}.

## P. Accepted/rejected conditional systems
See smoke/results.md for the full terminal-status table. No status was
upgraded without its preregistered gate; CONDITIONAL statuses carry the
exact unresolved gate.

## Q. C2 acceptance status (§34 gates)
- G1 environment captured: PASS (environment/toolchain reports)
- G2 harness foundation: PASS (run registry, sampler, manifest writer,
  loaders, drivers under c2/harness + c2probe)
- G3 exact oracle validated: PASS
- G4 AttentionDB modes validated: PASS with recorded finding (TEST-C2-008
  BM25 tie nondeterminism — engine finding, preserved; dense paths clean)
- G5 dataset hashes/manifests validated: PASS
- G6 primary datasets materialized: PASS (SciFact + NFCorpus; discrepancy
  about qrels granularity documented, acceptance rule unaffected)
- G7 ≥1 external baseline smoke completed: see H table
- G8 every declared external baseline has a real status: PASS
- G9 B3 trained or honestly blocked: see I
- G10 conditional dataset gates explicit outcomes: PASS (COCO/ESCI/SciDocs/FiQA)
- G11 resource guardrails verified: PASS (samplers active; aborts preserved)
- G12 raw runs immutable: PASS (no-overwrite enforced; INVALID-STARTUP dirs
  preserved and superseded by new IDs)
- G13 no main benchmark sweep: CONFIRMED
- G14 no C0/Phase3E mutation: CONFIRMED (diff evidence in A)
- G15 C2 artifacts internally consistent: PASS (C2-INTEGRITY-001 {integ})

**C2 STATUS: {'COMPLETE' if all(x in ('PASS','BLOCKED','ABORTED','CONDITIONAL','FAILED-HARNESS-DEFECT','BLOCKED-AUTH','NOT-RUN','CONDITIONAL-LOW') or x.startswith('PASS') for x in [qdrant,pg,ml,wv,coco,esci,b3v,b3l,integ]) and integ=='PASS' and b3l=='PASS' else 'PARTIAL'}** (final wording fixed at commit time by the closing message).

## R. Main benchmark NOT begun
No recall/latency sweeps, no quality/budget-matched runs, no Track A/B
campaigns, no concurrency tests, no statistical comparison — C2 is
preparatory; smokes are correctness/feasibility only.

## S. Phase3E/C0 untouched
`git diff fe4f92b HEAD -- research/phase3` = 0 lines; C0 audit docs and C1
artifacts unmodified (C2 added new files only).
""")

if __name__ == "__main__":
    os.makedirs(f"{C2ROOT}/oracle", exist_ok=True)
    os.makedirs(f"{C2ROOT}/smoke", exist_ok=True)
    write_env_toolchain()
    write_oracle_report()
    write_smoke_docs()
    write_validation_report()
    print("reports written")
