"""C2 harness common: env capture, guardrails, run registry, sampler.

Study comparative-study-001. Implements C1 environment-guardrails.md:
- preflight snapshot into every run dir (environment.yaml)
- PH3E_CRASH_* must be unset (hard abort otherwise)
- run dirs under research/comparative-study/raw/<run_id>/ are never overwritten
- resource sampler: 500ms process-tree RSS, 85% MemAvailable abort (OBSERVED-LIMIT)
- disk abort < 500 MiB free
"""
from __future__ import annotations
import hashlib, json, os, platform, shutil, subprocess, sys, threading, time
from datetime import datetime, timezone

REPO = "/home/user/AttentionDB"
RAW = f"{REPO}/research/comparative-study/raw"
C2ROOT = f"{REPO}/research/comparative-study/c2"

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()

def mem_available_kb() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    return 0

def mem_total_kb() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    return 0

def disk_free_mb(path: str = "/home/user") -> int:
    st = shutil.disk_usage(path)
    return st.free // (1024 * 1024)

def assert_phase3e_isolation() -> None:
    bad = [k for k in ("PH3E_CRASH_AT", "PH3E_CRASH_HIT", "PH3E_CRASH_MODEL", "PH3E_CRASH_MARKER")
           if k in os.environ]
    if bad:
        raise RuntimeError(f"PH3E crashgate env set, hard abort per guardrails: {bad}")

def capture_env() -> dict:
    cpu = subprocess.run(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo || true"],
                         capture_output=True, text=True).stdout.strip()
    return {
        "timestamp_utc": now_utc(),
        "os": platform.system() + " " + platform.release(),
        "kernel": platform.release(),
        "cpu_model": cpu.split(":", 1)[-1].strip(),
        "logical_cpus": os.cpu_count(),
        "mem_total_kb": mem_total_kb(),
        "mem_available_kb": mem_available_kb(),
        "disk_free_mb": disk_free_mb(),
        "filesystem": "ext4",
        "python": platform.python_version(),
        "git_commit": git_commit(),
    }

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

class Sampler:
    """Samples the process tree of one child every ~500ms; aborts at threshold."""
    def __init__(self, proc, mem_budget_kb: int, disk_floor_mb: int = 500):
        self.proc = proc
        self.budget = mem_budget_kb
        self.disk_floor = disk_floor_mb
        self.samples: list[tuple] = []
        self.peak_rss_kb = 0
        self.abort_reason = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _tree_rss(self):
        try:
            import psutil
            p = psutil.Process(self.proc.pid)
            procs = [p] + p.children(recursive=True)
            return sum(pp.memory_info().rss for pp in procs) // 1024
        except Exception:
            return 0

    def _loop(self):
        t0 = time.time()
        while not self._stop.is_set():
            rss = self._tree_rss()
            self.peak_rss_kb = max(self.peak_rss_kb, rss)
            self.samples.append((round(time.time() - t0, 3), rss, mem_available_kb(), disk_free_mb()))
            if rss >= self.budget:
                self.abort_reason = f"OBSERVED-LIMIT: tree RSS {rss}KB >= 85% MemAvailable budget {self.budget}KB"
                try:
                    import psutil
                    for ch in psutil.Process(self.proc.pid).children(recursive=True):
                        ch.kill()
                    self.proc.kill()
                except Exception:
                    pass
                self._stop.set()
                return
            if disk_free_mb() < self.disk_floor:
                self.abort_reason = f"DISK-FLOOR: {disk_free_mb()}MB < {self.disk_floor}MB"
                try:
                    self.proc.kill()
                except Exception:
                    pass
                self._stop.set()
                return
            time.sleep(0.5)

    def start(self):
        self._thread.start()

    def stop(self, resource_csv_path: str) -> dict:
        self._stop.set()
        self._thread.join(timeout=2)
        with open(resource_csv_path, "w") as f:
            f.write("t_sec,tree_rss_kb,mem_available_kb,disk_free_mb\n")
            for row in self.samples:
                f.write(",".join(str(x) for x in row) + "\n")
        return {"peak_rss_kb": self.peak_rss_kb, "n_samples": len(self.samples),
                "abort_reason": self.abort_reason}

class Run:
    """One immutable run directory under raw/<run_id>."""
    def __init__(self, run_id: str, purpose: str, config: dict, use_sampler: bool = False,
                 mem_fraction: float = 0.85):
        assert_phase3e_isolation()
        self.dir = os.path.join(RAW, run_id)
        if os.path.exists(self.dir):
            raise RuntimeError(f"run dir exists, never overwrite: {self.dir}")
        os.makedirs(os.path.join(self.dir, "artifacts"))
        self.id = run_id
        self.config = config
        self.use_sampler = use_sampler
        self.mem_budget_kb = int(mem_available_kb() * mem_fraction)
        env = capture_env()
        env["mem_abort_budget_kb"] = self.mem_budget_kb
        env["phase3e_crash_env_unset"] = True
        _dump_yaml(os.path.join(self.dir, "environment.yaml"), env)
        _dump_yaml(os.path.join(self.dir, "config.yaml"), {
            "run_id": run_id, "study_id": "comparative-study-001",
            "protocol_version": "1.0.0", "git_commit": git_commit(),
            "purpose": purpose, **config})
        self.log = open(os.path.join(self.dir, "stdout.log"), "w")
        self.tee = Tee(sys.stdout, self.log)
        self.manifest: dict = {"run_id": run_id, "purpose": purpose,
                               "started_utc": now_utc(), "artifacts": {}}
        self.sampler = None

    def print(self, *a):
        print(*a, file=self.tee)

    def artifact(self, name: str, path: str):
        """Register a file as run artifact (hashed into manifest)."""
        dst = os.path.join(self.dir, "artifacts", os.path.basename(path))
        if os.path.abspath(path) != os.path.abspath(dst):
            shutil.copy(path, dst)
        h = sha256_file(os.path.join(self.dir, "artifacts", os.path.basename(path)))
        self.manifest["artifacts"][os.path.basename(path)] = {
            "sha256": h, "bytes": os.path.getsize(os.path.join(self.dir, "artifacts", os.path.basename(path)))}

    def finish(self, status: str, metrics: dict | None = None, reason: str | None = None):
        self.manifest.update({
            "finished_utc": now_utc(), "status": status,
            "study_id": "comparative-study-001", "protocol_version": "1.0.0",
            "git_commit": git_commit(),
            "host": {"logical_cpus": os.cpu_count(), "mem_total_kb": mem_total_kb()},
            "guardrails": {"mem_budget_kb_85pct_available": self.mem_budget_kb,
                           "disk_floor_mb": 500,
                           "phase3e_crash_env_asserted_unset": True},
            "seed": self.config.get("seed"),
        })
        if reason:
            self.manifest["reason"] = reason
        _dump_yaml(os.path.join(self.dir, "manifest.yaml"), self.manifest)
        with open(os.path.join(self.dir, "metrics.json"), "w") as f:
            json.dump(metrics or {"status": status}, f, indent=2)
        with open(os.path.join(RAW, "RUN-INDEX.yaml"), "a") as f:
            f.write(f"- run_id: {self.id}\n  status: {status}\n  purpose: {self.manifest['purpose']}\n"
                    f"  started: {self.manifest['started_utc']}\n  finished: {self.manifest['finished_utc']}\n"
                    + (f"  reason: {reason}\n" if reason else ""))
        self.print(f"RUN {self.id} TERMINAL STATUS: {status}")
        self.log.close()

def _dump_yaml(path: str, obj):
    try:
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(obj, f, sort_keys=False)
    except ImportError:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

def guarded_popen(cmd: list[str], run: Run, timeout_s: int, env_extra: dict | None = None):
    """Run a command under the sampler; returns (returncode, summary). Timeout kills child only."""
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    s = Sampler(proc, run.mem_budget_kb)
    s.start()
    try:
        out, _ = proc.communicate(timeout=timeout_s)
        rc = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            import psutil
            for ch in psutil.Process(proc.pid).children(recursive=True):
                ch.kill()
        except Exception:
            pass
        proc.kill()
        out, _ = proc.communicate()
        rc = proc.returncode
    summary = s.stop(os.path.join(run.dir, "resource.csv"))
    summary["exit_code"] = rc
    summary["timed_out"] = timed_out
    log = out.decode(errors="replace") if out else ""
    with open(os.path.join(run.dir, "stderr.log"), "a") as f:
        f.write(log[-200000:])
    if s.abort_reason and summary["abort_reason"] is None:
        summary["abort_reason"] = s.abort_reason
    return rc, summary, log
