"""C2 external baseline feasibility smokes (protocol §14–§19).

One external server at a time (C1 host etiquette). Every server smoke runs the
SERVER ITSELF under guarded_popen (500ms tree-RSS sampler, 85% MemAvailable
abort, disk floor, timeout, logs preserved). Classification is data-driven:
sampler abort => ABORTED/OBSERVED-LIMIT; stage timeout => ABORTED; clean ops
=> PASS; server died => FAILED; prereq missing => BLOCKED. Never reclassified.
Per preregistered plan: ES default-JVM abort is an expected legitimate result —
NO reduced-heap retry (the -002 metrics note announcing one is RETRACTED).
"""
import json, os, subprocess, sys, time, threading, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
from adb_common import Run, RAW, guarded_popen

DL = f"{RAW}/datasets/downloads"

def wait_http(url, timeout_s, run=None):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status < 500:
                    return True, round(time.time() - t0, 2)
        except Exception:
            time.sleep(0.5)
    return False, round(time.time() - t0, 2)

def rest(method, url, body=None, timeout=10):
    req = urllib.request.Request(url, method=method)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data, timeout=timeout) as r:
            return r.status, r.read().decode()[:2000]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:2000]

def guarded_server(cmd, run, timeout_s, env_extra=None):
    """Run a long-lived server under guarded_popen in a helper thread.
    Returns (box, thread); box['result'] = (rc, summary, log) when it ends."""
    box = {}
    def _t():
        box["result"] = guarded_popen(cmd, run, timeout_s=timeout_s, env_extra=env_extra)
    th = threading.Thread(target=_t, daemon=True)
    th.start()
    return box, th

def kill_tree_by_dir(workdir):
    """Kill the server started from this unique mkdtemp workdir (exact path)."""
    import psutil
    me = os.getpid()
    killed = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            if p.pid == me:
                continue
            cl = " ".join(p.info["cmdline"] or [])
            if workdir in cl:
                for ch in p.children(recursive=True):
                    ch.kill()
                p.kill()
                killed.append(p.pid)
        except Exception:
            pass
    return killed

def server_verdict(run, out, box, th, workdir, ok, startup_s, ops_ok, join_after_fail=75):
    """Join the guarded server thread and classify data-driven."""
    if not ok:
        th.join(join_after_fail)  # guarded timeout is 75s past wait_http's
        kill_tree_by_dir(workdir)  # never leave an orphan on failed startup
    else:
        kill_tree_by_dir(workdir)
        th.join(30)
    rc, summ, log = box.get("result", (None, {"peak_rss_kb": None, "abort_reason": None, "n_samples": 0, "timed_out": False, "exit_code": None}, ""))
    out["server_peak_rss_kb"] = summ.get("peak_rss_kb")
    out["server_n_samples"] = summ.get("n_samples")
    out["server_abort_reason"] = summ.get("abort_reason")
    out["server_timed_out"] = summ.get("timed_out")
    out["server_exit_code"] = summ.get("exit_code")
    out["server_log_tail"] = log[-1500:]
    if summ.get("abort_reason"):
        run.finish("ABORTED", out, reason=summ["abort_reason"])
    elif summ.get("timed_out"):
        run.finish("ABORTED", out, reason="startup stage timeout (guarded)")
    elif not ok:
        run.finish("FAILED", out, reason=f"server not healthy; exit_code={summ.get('exit_code')}; log tail in metrics.server_log_tail")
    else:
        run.finish("PASS" if ops_ok else "FAILED", out,
                   reason=None if ops_ok else "health OK but minimal op set failed (see metrics)")

def qdrant():
    run = Run("C2-SMOKE-QDRANT-006", "Qdrant local feasibility smoke under guarded sampler, incl. named-vector validation (protocol §14)",
              {"system": "qdrant", "component": "baseline-smoke", "attempt": "guarded-sampler"})
    import tarfile, tempfile, shutil
    out = {}
    workdir = None
    try:
        workdir = tempfile.mkdtemp(prefix="c2-qdrant-", dir="/var/tmp")
        with tarfile.open(f"{DL}/bin/qdrant.tar.gz") as t:
            t.extractall(workdir)
        binp = f"{workdir}/qdrant"
        out["binary_version"] = subprocess.run([binp, "--version"], capture_output=True, text=True).stdout.strip()
        env = {"QDRANT__STORAGE__STORAGE_PATH": f"{workdir}/storage"}
        box, th = guarded_server([binp], run, 300, env)
        ok, startup_s = wait_http("http://127.0.0.1:6333/healthz", 120)
        out["startup_healthy"] = ok
        out["startup_s"] = startup_s
        ops_ok = False
        if ok:
            import random
            random.seed(20260925)
            v = [[random.random() for _ in range(384)] for _ in range(100)]
            st, _ = rest("PUT", "http://127.0.0.1:6333/collections/c2smoke",
                         {"vectors": {"size": 384, "distance": "Cosine"}})
            out["create_collection"] = st
            st, _ = rest("PUT", "http://127.0.0.1:6333/collections/c2smoke/points?wait=true",
                         {"points": [{"id": i, "vector": v[i]} for i in range(100)]})
            out["upsert_100"] = st
            st, body = rest("POST", "http://127.0.0.1:6333/collections/c2smoke/points/search",
                            {"vector": v[0], "top": 5})
            hits = json.loads(body).get("result", [])
            out["search_5"] = st
            out["search_returns_exact_self_top1"] = bool(hits and hits[0]["id"] == 0)
            # named-vector validation (B6 proposed-config feasibility, not an adapter)
            st, _ = rest("PUT", "http://127.0.0.1:6333/collections/c2named",
                         {"vectors": {"HEAD-TITLE": {"size": 384, "distance": "Cosine"},
                                      "HEAD-BODY": {"size": 384, "distance": "Cosine"}}})
            out["create_named_collection"] = st
            st, _ = rest("PUT", "http://127.0.0.1:6333/collections/c2named/points?wait=true",
                         {"points": [{"id": i, "vector": {"HEAD-TITLE": v[i], "HEAD-BODY": v[99 - i]}} for i in range(100)]})
            out["upsert_named_100"] = st
            st, body = rest("POST", "http://127.0.0.1:6333/collections/c2named/points/search",
                            {"vector": {"name": "HEAD-TITLE", "vector": v[7]}, "top": 5})
            out["named_search_body"] = body[:400]
            nh = json.loads(body).get("result", []) if st == 200 else []
            out["named_search_5"] = st
            out["named_search_self_top1"] = bool(nh and nh[0]["id"] == 7)
            ops_ok = (out.get("create_collection") == 200 and out.get("upsert_100") == 200
                      and out.get("search_5") == 200 and out.get("search_returns_exact_self_top1")
                      and out.get("create_named_collection") == 200 and out.get("upsert_named_100") == 200
                      and out.get("named_search_5") == 200 and out.get("named_search_self_top1"))
        time.sleep(1)
        shutil.rmtree(workdir, ignore_errors=True)
        server_verdict(run, out, box, th, workdir or "c2-qdrant-", ok, startup_s, ops_ok)
    except Exception as e:
        if workdir:
            kill_tree_by_dir(workdir)
        run.finish("FAILED", out, reason=repr(e))
        raise

def pgvector():
    run = Run("C2-SMOKE-PGVECTOR-005", "PostgreSQL+pgvector feasibility smoke (protocol §15)",
              {"system": "pgvector", "component": "baseline-smoke"})
    out = {}
    try:
        apt = subprocess.run(["sudo", "-n", "apt-get", "update", "-qq"], capture_output=True, text=True, timeout=300)
        out["apt_update_rc"] = apt.returncode
        search = subprocess.run(["apt-cache", "search", "pgvector"], capture_output=True, text=True)
        out["apt_pgvector_packages"] = [l for l in search.stdout.splitlines() if l.strip()]
        pkg = next((p.split()[0] for p in out["apt_pgvector_packages"] if p.startswith("postgresql-")), None)
        out["pgvector_pkg"] = pkg
        if not pkg:
            run.finish("BLOCKED", out, reason="no pgvector package in distro repos (BLK-3 resolution: blocked on this host)")
            return
        inst = subprocess.run(["sudo", "-n", "env", "DEBIAN_FRONTEND=noninteractive",
                              "apt-get", "install", "-y", "-qq", "postgresql", pkg],
                              capture_output=True, text=True, timeout=900)
        out["install_rc"] = inst.returncode
        import glob as g
        vers = sorted(os.path.basename(p) for p in g.glob("/usr/lib/postgresql/*"))
        out["pg_versions_present"] = vers
        if not vers:
            run.finish("FAILED", out, reason="no postgres cluster after install")
            return
        v = vers[-1]
        ctl = subprocess.run(["sudo", "-n", "pg_ctlcluster", v, "main", "start"], capture_output=True, text=True)
        out["cluster_start_rc"] = ctl.returncode
        if ctl.returncode != 0:
            out["cluster_start_err"] = ctl.stderr[-500:]
        def psql(db, sql, timeout=300):
            if len(sql) > 60_000:
                p = f"/var/tmp/c2_pgsmoke_{abs(hash(sql)) % 10**8}.sql"
                with open(p, "w") as f:
                    f.write(sql)
                os.chmod(p, 0o644)
                return subprocess.run(["sudo", "-n", "-u", "postgres", "psql", db, "-f", p],
                                      capture_output=True, text=True, timeout=timeout)
            return subprocess.run(["sudo", "-n", "-u", "postgres", "psql", db, "-c", sql],
                                  capture_output=True, text=True, timeout=timeout)
        for s in ("DROP DATABASE IF EXISTS c2smoke", "CREATE DATABASE c2smoke"):
            psql("postgres", s, timeout=60)
        cmds = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            "CREATE TABLE t (id int primary key, v vector(384))",
        ]
        import random
        random.seed(20260925)
        vecs = [[random.random() for _ in range(384)] for _ in range(100)]
        vals = ",".join(f"({i}, '[{','.join(f'{x:.5f}' for x in vecs[i])}]')" for i in range(100))
        cmds.append(f"INSERT INTO t (id, v) VALUES {vals}")
        cmds.append("CREATE INDEX ON t USING hnsw (v vector_cosine_ops)")
        cmds.append("CREATE INDEX ON t USING ivfflat (v vector_cosine_ops) WITH (lists = 4)")
        ok_all = True
        for s in cmds:
            r = psql("c2smoke", s)
            if r.returncode != 0:
                ok_all = False
                out.setdefault("sql_errors", []).append(s[:40] + " => " + r.stderr[-200:])
        out["hnsw_and_ivfflat_created"] = ok_all
        q = "SELECT id, v <=> '[%s]' AS d FROM t ORDER BY v <=> '[%s]' LIMIT 5" % (
            ",".join(f"{x:.5f}" for x in vecs[0]), ",".join(f"{x:.5f}" for x in vecs[0]))
        r = psql("c2smoke", q, timeout=120)
        out["ann_query_rc"] = r.returncode
        out["ann_top1_is_self"] = " 0 |" in (r.stdout.splitlines()[2] if len(r.stdout.splitlines()) > 2 else "")
        out["pgvector_version"] = (psql("c2smoke", "SELECT extversion FROM pg_extension WHERE extname='vector'").stdout.splitlines()[2:3] or ["?"])[0].strip()
        stp = subprocess.run(["sudo", "-n", "pg_ctlcluster", v, "main", "stop"], capture_output=True, text=True)
        out["cluster_stop_rc"] = stp.returncode
        # evidence limitation: cluster managed by pg_ctlcluster (system service);
        # per-process tree sampler not applicable; preflight env + disk recorded.
        out["sampler_note"] = "pg_ctlcluster-managed; no tree sampler; preflight-only resources"
        feasible = inst.returncode == 0 and ctl.returncode == 0 and ok_all and r.returncode == 0
        run.finish("PASS" if feasible else "FAILED", out)
    except Exception as e:
        run.finish("FAILED", out, reason=repr(e))
        raise

def elasticsearch():
    run = Run("C2-SMOKE-ES-003", "Elasticsearch feasibility smoke at default JVM settings, server under guarded sampler (protocol §16)",
              {"system": "elasticsearch", "component": "baseline-smoke", "attempt": "default-heap",
               "note": "per prereg: abort at this envelope is a legitimate OBSERVED-LIMIT; no reduced-heap retry"})
    import tarfile, tempfile, shutil
    out = {}
    workdir = None
    try:
        workdir = tempfile.mkdtemp(prefix="c2-es-", dir="/var/tmp")
        with tarfile.open(f"{DL}/bin/elasticsearch.tar.gz") as t:
            t.extractall(workdir)
        esh = f"{workdir}/elasticsearch-8.15.2"
        try:
            mmc = int(open("/proc/sys/vm/max_map_count").read().strip())
            out["vm_max_map_count"] = mmc
        except Exception:
            out["vm_max_map_count"] = "unreadable"
        env = {"ES_JAVA_HOME": f"{esh}/jdk", "JAVA_HOME": f"{esh}/jdk", "HOME": workdir}
        box, th = guarded_server([f"{esh}/bin/elasticsearch"], run, 300, env)
        ok, startup_s = wait_http("http://127.0.0.1:9200", 240)
        out["startup_healthy_default_jvm"] = ok
        out["startup_s"] = startup_s
        ops_ok = False
        if ok:
            import random
            random.seed(20260925)
            v0 = [random.random() for _ in range(384)]
            st, _ = rest("PUT", "http://127.0.0.1:9200/c2smoke",
                         {"mappings": {"properties": {"v": {"type": "dense_vector", "dims": 384, "index": True, "similarity": "cosine"}}}})
            out["create_index"] = st
            st, _ = rest("PUT", "http://127.0.0.1:9200/c2smoke/_doc/1", {"v": v0})
            out["index_doc"] = st
            st, body = rest("POST", "http://127.0.0.1:9200/c2smoke/_search",
                            {"knn": {"field": "v", "query_vector": v0, "k": 5, "num_candidates": 16}})
            out["knn_search"] = st
            try:
                out["knn_top1_is_self"] = json.loads(body)["hits"]["hits"][0]["_id"] == "1"
            except Exception:
                out["knn_top1_is_self"] = False
            ops_ok = out.get("create_index") == 200 and out.get("index_doc") in (200, 201) and out.get("knn_search") == 200 and out.get("knn_top1_is_self")
        for ln in ("logs/elasticsearch.log", "logs/stdout.err", "logs/stderr.err"):
            p = f"{esh}/{ln}"
            if os.path.exists(p):
                run.artifact(os.path.basename(p), p)
        server_verdict(run, out, box, th, workdir, ok, startup_s, ops_ok)
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception as e:
        if workdir:
            kill_tree_by_dir(workdir)
        run.finish("FAILED", out, reason=repr(e))
        raise

def milvus_lite():
    run = Run("C2-SMOKE-MILVUSLITE-002", "Milvus-Lite wheel feasibility smoke (protocol §17)",
              {"system": "milvus-lite", "component": "baseline-smoke"})
    out = {}
    pip = subprocess.run([sys.executable, "-m", "pip", "install", "pymilvus[milvus-lite]"], capture_output=True, text=True, timeout=600)
    out["pip_rc"] = pip.returncode
    out["pip_tail"] = (pip.stdout + pip.stderr)[-800:]
    if pip.returncode != 0:
        run.finish("BLOCKED", out, reason="milvus-lite wheel install failed (sandbox has no PyPI egress; BLK-3-style evidence in metrics.pip_tail)")
        return
    try:
        from pymilvus import MilvusClient
        import tempfile
        db = tempfile.mktemp(suffix=".db", dir="/var/tmp")
        mc = MilvusClient(f"{db}")
        mc.create_collection("c2smoke", dimension=384)
        import random
        random.seed(1)
        vecs = [[random.random() for _ in range(384)] for _ in range(50)]
        mc.insert("c2smoke", [{"id": i, "vector": vecs[i]} for i in range(50)])
        r = mc.search("c2smoke", data=[vecs[0]], limit=5)
        out["search_hits"] = len(list(r)[0]) if r else 0
        out["status_detail"] = "import+init+create+insert+search OK"
        run.finish("PASS", out)
    except Exception as e:
        run.finish("FAILED", out, reason=repr(e))

def weaviate():
    run = Run("C2-SMOKE-WEAVIATE-004", "Weaviate standalone binary feasibility smoke under guarded sampler (protocol §18)",
              {"system": "weaviate", "component": "baseline-smoke", "attempt": "guarded-sampler"})
    import tarfile, tempfile, shutil
    out = {}
    workdir = None
    try:
        workdir = tempfile.mkdtemp(prefix="c2-weav-", dir="/var/tmp")
        with tarfile.open(f"{DL}/bin/weaviate.tar.gz") as t:
            t.extractall(workdir)
        binp = f"{workdir}/weaviate"
        os.chmod(binp, 0o755)
        datadir = f"{workdir}/data"
        os.makedirs(datadir, exist_ok=True)
        # v1.39 raft: cluster comms need an explicit loopback advertise addr in
        # this sandbox (no private IP); REST listener forced to http:8080
        # (default swagger spec would otherwise demand TLS certs / random port).
        env = {"AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true",
               "PERSISTENCE_DATA_PATH": datadir,
               "DEFAULT_VECTORIZER_MODULE": "none",
               "CLUSTER_ADVERTISE_ADDR": "127.0.0.1",
               "CLUSTER_GOSSIP_BIND_PORT": "7001",
               "CLUSTER_DATA_BIND_PORT": "7002"}
        box, th = guarded_server([binp, "--scheme", "http", "--port", "8080"], run, 300, env)
        ok, startup_s = wait_http("http://127.0.0.1:8080/v1/.well-known/ready", 120)
        out["startup_healthy"] = ok
        out["startup_s"] = startup_s
        ops_ok = False
        if ok:
            st, meta = rest("GET", "http://127.0.0.1:8080/v1/meta")
            import re
            mm = re.search(r'"version"\s*:\s*"([^"]+)"', meta)
            out["version"] = mm.group(1) if mm else None
            st, _ = rest("POST", "http://127.0.0.1:8080/v1/schema",
                         {"class": "C2Smoke", "vectorizer": "none", "properties": []})
            out["create_schema"] = st
            import random
            random.seed(2)
            v = [random.random() for _ in range(384)]
            st, _ = rest("POST", "http://127.0.0.1:8080/v1/objects",
                         {"class": "C2Smoke", "vector": v, "properties": {}})
            out["insert_object"] = st
            st, body = rest("POST", "http://127.0.0.1:8080/v1/graphql",
                            {"query": "{Get {C2Smoke(nearVector: {vector: [%s], limit: 2}) {id _additional {id}}}}" % ",".join(f"{x:.4f}" for x in v)})
            out["vector_query"] = st
            ops_ok = out.get("create_schema") in (200, 201) and out.get("insert_object") in (200, 201) and out.get("vector_query") == 200
        server_verdict(run, out, box, th, workdir, ok, startup_s, ops_ok)
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception as e:
        if workdir:
            kill_tree_by_dir(workdir)
        run.finish("FAILED", out, reason=repr(e))
        raise

if __name__ == "__main__":
    {"qdrant": qdrant, "pgvector": pgvector, "es": elasticsearch,
     "milvus": milvus_lite, "weaviate": weaviate}[sys.argv[1]]()
