#!/usr/bin/env python
"""Resumable, parallel driver for the PRICE model's scenario runs.

Runs main.py once per run listed in scenarios/runs.json, each in its own
process (scenarios/run_one.py) and its own directory under --out-dir, at most
--jobs at a time. Each run's settings reach main.py and config.py as PRICE_*
environment variables, so neither file is edited per run. See scenarios/README.md.
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN_ONE = os.path.join(HERE, "run_one.py")

# Environment forced on every child: one thread per child (parallelism comes from
# the process pool), unbuffered output so a killed child's log is complete, no
# bytecode files written into the source tree, and a non-interactive plot backend.
CHILD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "MPLBACKEND": "Agg",
}

# Setting keys of one run, and the PRICE_* variable each one is passed in.
SETTING_ENV = {
    "num_steps": "PRICE_NUM_STEPS",
    "historical_cutoff": "PRICE_HISTORICAL_CUTOFF",
    "output_name": "PRICE_OUTPUT_NAME",
    "efficiency_scenario": "PRICE_EFFICIENCY_SCENARIO",
    "forecast_model": "PRICE_FORECAST_MODEL",
    "forecast_target_date": "PRICE_FORECAST_TARGET_DATE",
    "forecast_target_price": "PRICE_FORECAST_TARGET_PRICE",
    "block_pace_data": "PRICE_BLOCK_PACE_DATA",
    "price_data": "PRICE_PRICE_DATA",
}
DATA_SETTINGS = ("block_pace_data", "price_data")
OPTIONAL_KEYS = {"figure", "original", "default", "expect_config"}
FORECAST_MODELS = ("powerlaw", "fixed", "constant", "linear", "logistic")
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")
OUTPUT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")  # same rule as config.parse_output_name
DATA_CSV_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\.csv")  # same rule as config.parse_data_csv
# config.py names that hold input paths; each must have the form ../data/<file>
INPUT_NAMES = ("BLOCK_PACE_DATA", "TX_BLOCK_DATA", "PRICE_DATA", "MACHINE_DATA_FILE")
# Files main.py writes into ../data/ besides the result CSV
OTHER_OUTPUTS = ("hashrate_comparison.png", "logs")

_stop_signal = None  # set by the signal handler


# ------------------------------------------------------------------ helpers
def write_json_atomic(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix="." + os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def naive_iso(text, what):
    try:
        v = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        raise SystemExit(f"ERROR: {what}={text!r} is not an ISO date or date-time")
    if v.tzinfo is not None:
        raise SystemExit(f"ERROR: {what}={text!r} must not carry a UTC offset")
    return text


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def inside(path, parent):
    path, parent = os.path.realpath(path), os.path.realpath(parent)
    return path == parent or path.startswith(parent + os.sep)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_runs(path):
    """Read and validate the run table. Returns the list of run dicts."""
    doc = read_json(path)
    if doc is None or not isinstance(doc.get("runs"), list):
        raise SystemExit(f"ERROR: {path} is not a JSON object with a 'runs' list")
    runs, names, outputs = [], set(), set()
    for i, r in enumerate(doc["runs"]):
        where = f"{path} run #{i}"
        if not isinstance(r, dict):
            raise SystemExit(f"ERROR: {where} is not an object")
        missing = {"name", *SETTING_ENV} - r.keys()
        unknown = r.keys() - {"name", *SETTING_ENV, *OPTIONAL_KEYS}
        if missing or unknown:
            raise SystemExit(f"ERROR: {where}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
        n = r["name"]
        if not isinstance(n, str) or not NAME_RE.fullmatch(n) or n in names:
            raise SystemExit(f"ERROR: {where}: name {n!r} is invalid or repeated")
        names.add(n)
        where = f"{path} run {n}"
        if not isinstance(r["num_steps"], int) or isinstance(r["num_steps"], bool) or r["num_steps"] <= 0:
            raise SystemExit(f"ERROR: {where}: num_steps must be a positive integer")
        naive_iso(r["historical_cutoff"], f"{where}: historical_cutoff")
        naive_iso(r["forecast_target_date"], f"{where}: forecast_target_date")
        if r["efficiency_scenario"] not in ("frozen", "frontier"):
            raise SystemExit(f"ERROR: {where}: efficiency_scenario must be frozen or frontier")
        if r["forecast_model"] not in FORECAST_MODELS:
            raise SystemExit(f"ERROR: {where}: forecast_model must be one of {FORECAST_MODELS}")
        for k in DATA_SETTINGS:
            data_csv(r[k], f"{where}: {k}")
        p = r["forecast_target_price"]
        if p is not None and not (is_number(p) and 0 < p < float("inf")):
            raise SystemExit(f"ERROR: {where}: forecast_target_price must be null or a positive number")
        o = r["output_name"]
        if not isinstance(o, str) or not OUTPUT_RE.fullmatch(o) or o.endswith(".csv") or o in outputs:
            raise SystemExit(f"ERROR: {where}: output_name {o!r} is invalid or repeated")
        if r.get("original") is not None and o + ".csv" == r["original"]:
            raise SystemExit(f"ERROR: {where}: output_name must differ from the original's file name")
        outputs.add(o)
        ec = r.get("expect_config")
        if ec is not None and not (isinstance(ec, dict) and all(isinstance(k, str) for k in ec)):
            raise SystemExit(f"ERROR: {where}: expect_config must be an object")
        runs.append(r)
    return runs


def data_csv(text, what):
    """Normalise a bare <name>.csv or ../data/<name>.csv to ../data/<name>.csv (as config.py does)."""
    if not isinstance(text, str):
        raise SystemExit(f"ERROR: {what} must be a string")
    name = text[len("../data/"):] if text.startswith("../data/") else text
    if not DATA_CSV_RE.fullmatch(name) or ".." in name:
        raise SystemExit(f"ERROR: {what}={text!r} must be <name>.csv or ../data/<name>.csv")
    return "../data/" + name


def settings_of(run, num_steps_override=0):
    """The settings that define a run's output (compared on resume)."""
    s = {k: run[k] for k in SETTING_ENV}
    for k in DATA_SETTINGS:
        s[k] = data_csv(s[k], k)
    if num_steps_override:
        s["num_steps"] = num_steps_override
    s["expect_config"] = run.get("expect_config") or {}
    return s


def env_of(settings):
    env = {}
    for key, var in SETTING_ENV.items():
        v = settings[key]
        if v is None:
            env[var] = "none"
        elif isinstance(v, float):
            env[var] = repr(v)
        else:
            env[var] = str(v)
    return env


def clean_environ():
    """os.environ without PRICE_* variables: a child receives only its own run's."""
    return {k: v for k, v in os.environ.items() if not k.startswith("PRICE_")}


# --------------------------------------------------------------- orchestrator
class Orchestrator:
    def __init__(self, args):
        self.a = args
        self.code_root = os.path.abspath(args.code_root)
        self.data_dir = os.path.abspath(args.data_dir or os.path.join(self.code_root, "data"))
        self.out = os.path.abspath(args.out_dir)
        for guarded in (self.data_dir, os.path.join(self.code_root, "data"), os.path.join(ROOT, "data")):
            if inside(self.out, guarded):
                raise SystemExit(f"ERROR: --out-dir {self.out} lies inside {guarded}; the runs must "
                                 f"never write under a data directory. Choose a directory elsewhere.")
        self.runs_root = os.path.join(self.out, "runs")
        os.makedirs(self.runs_root, exist_ok=True)
        self.progress_path = os.path.join(self.out, "progress.log")
        self.lock_f = open(os.path.join(self.out, ".orchestrator.lock"), "w")
        try:
            fcntl.flock(self.lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"ERROR: another run_scenarios.py is using {self.out}")
        self.children = {}  # Popen -> run name
        self.attempts = {}  # run name -> attempts launched in this invocation
        self.inputs = {}  # run name -> input manifest
        self._probe_cache = {}  # input-file PRICE_* values -> {config name: file name}
        self._hash_cache = {}  # realpath -> size, mtime, sha256

    # ---------------------------------------------------------------- logging
    def log(self, msg):
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        with open(self.progress_path, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    # ------------------------------------------------------------ provenance
    def git_state(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_rgp", os.path.join(ROOT, "calibration", "run_grid_point.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.git_state(self.code_root)

    def config_inputs(self, settings):
        """{config name: file name} of the inputs config.py resolves for this run.

        config.py is imported in a subprocess with the run's input-file PRICE_* values,
        so the result is what the child will read (code without the overrides ignores
        them and reports its hardcoded paths; run_one.py then refuses a mismatch)."""
        data_env = {SETTING_ENV[k]: settings[k] for k in DATA_SETTINGS}
        key = tuple(sorted(data_env.items()))
        if key not in self._probe_cache:
            probe = ("import json, config; print(json.dumps({k: getattr(config, k) for k in %r}))"
                     % (INPUT_NAMES,))
            env = clean_environ()
            env.update(CHILD_ENV)
            env.update(data_env)
            r = subprocess.run([self.a.python, "-c", probe], cwd=self.code_root, env=env,
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit(f"ERROR: could not import config.py from {self.code_root}:\n{r.stderr}")
            paths, files = json.loads(r.stdout.strip().splitlines()[-1]), {}
            for name, rel in paths.items():
                parts = rel.split("/")
                if len(parts) != 3 or parts[:2] != ["..", "data"] or not parts[2]:
                    raise SystemExit(f"ERROR: config.{name} = {rel!r}; expected the form ../data/<file>")
                if parts[2] in files.values():
                    raise SystemExit(f"ERROR: config.{name} names {parts[2]}, already used by another input")
                files[name] = parts[2]
            self._probe_cache[key] = files
        return self._probe_cache[key]

    def input_manifest(self, files):
        """Manifest (source path, size, mtime, SHA-256) of the given inputs in --data-dir."""
        manifest = {}
        for name, fname in files.items():
            real = os.path.realpath(os.path.join(self.data_dir, fname))
            if real not in self._hash_cache:
                st = os.stat(real)
                self.log(f"hashing input {fname} ({st.st_size} bytes)")
                self._hash_cache[real] = {
                    "source": real, "bytes": st.st_size, "sha256": sha256(real),
                    "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}
            manifest[fname] = {"config_name": name, **self._hash_cache[real]}
        return manifest

    # ------------------------------------------------------------ run dirs
    def run_dir(self, name):
        return os.path.join(self.runs_root, name)

    def done_path(self, name):
        return os.path.join(self.run_dir(name), "DONE.json")

    def csv_path(self, name, settings):
        return os.path.join(self.run_dir(name), "data", settings["output_name"] + ".csv")

    def completed(self, name, settings):
        """True if the run has a valid completion marker matching the requested settings.

        Raises SystemExit if a marker exists but describes a different run (other settings,
        inputs or commit) or no longer matches the CSV on disk: nothing is silently reused
        or silently re-run over."""
        done = read_json(self.done_path(name))
        if done is None:
            return False
        rd = self.run_dir(name)
        if done.get("settings") != settings:
            raise SystemExit(f"ERROR: {rd} holds a completed run with other settings "
                             f"({done.get('settings')}); requested {settings}. Use a fresh --out-dir "
                             f"or delete {rd}.")
        def ident(manifest):  # what identifies an input; mtime is informational only
            return {f: (i.get("source"), i.get("bytes"), i.get("sha256"))
                    for f, i in (manifest or {}).items()}
        if ident(done.get("inputs")) != ident(self.inputs[name]):
            raise SystemExit(f"ERROR: {rd} was completed with other input files (sha256 or path "
                             f"differ). Use a fresh --out-dir or delete {rd}.")
        if done.get("commit") != self.commit:
            raise SystemExit(f"ERROR: {rd} was completed by code commit {done.get('commit')}, but "
                             f"the code root is at {self.commit}. Use a fresh --out-dir or delete {rd}.")
        csv = self.csv_path(name, settings)
        if not os.path.isfile(csv) or os.path.getsize(csv) != done.get("csv_bytes"):
            raise SystemExit(f"ERROR: {csv} is missing or its size differs from {self.done_path(name)}; "
                             f"inspect it, then delete {rd} to re-run.")
        return True

    def prepare(self, name, settings):
        """Create the run directory: work/ (the child's cwd) and data/ with input symlinks."""
        rd = self.run_dir(name)
        data = os.path.join(rd, "data")
        for d in (rd, os.path.join(rd, "work"), data, os.path.join(data, "logs")):
            os.makedirs(d, exist_ok=True)
        manifest = self.inputs[name]
        for entry in os.listdir(data):  # input links left by an attempt with other inputs
            link = os.path.join(data, entry)
            if entry not in manifest and os.path.islink(link):
                os.unlink(link)
                self.log(f"[{name}] removed stale input link {entry}")
        for fname, info in manifest.items():
            link = os.path.join(data, fname)
            if os.path.islink(link):
                if os.path.realpath(link) != info["source"]:
                    raise SystemExit(f"ERROR: {link} points to {os.path.realpath(link)}, "
                                     f"not {info['source']}")
            elif os.path.exists(link):
                raise SystemExit(f"ERROR: {link} exists and is not a symlink to the input")
            else:
                os.symlink(info["source"], link)
        csv = self.csv_path(name, settings)
        if os.path.lexists(csv):
            if os.path.islink(csv):
                raise SystemExit(f"ERROR: {csv} is a symlink; refusing to let main.py write through it")
            os.unlink(csv)  # partial output of an earlier, unfinished attempt (no DONE.json)
            self.log(f"[{name}] removed partial output {csv}")

    # ----------------------------------------------------------------- pool
    def launch(self, name, settings):
        self.prepare(name, settings)
        rd = self.run_dir(name)
        self.attempts[name] = self.attempts.get(name, 0) + 1
        spec = {"name": name, "settings": settings, "env": env_of(settings), "inputs": self.inputs[name],
                "code_root": self.code_root, "commit": self.commit, "dirty": self.dirty,
                "attempt_in_invocation": self.attempts[name],
                "launched_at": datetime.now().isoformat(timespec="seconds")}
        settings_path = os.path.join(rd, "settings.json")
        write_json_atomic(settings_path, spec)
        cmd = [self.a.python, RUN_ONE, "--code-root", self.code_root, "--settings", settings_path,
               "--done", self.done_path(name), "--rss-interval", str(self.a.rss_interval)]
        env = clean_environ()
        env.update(CHILD_ENV)
        env.update(spec["env"])
        logf = open(os.path.join(rd, "stdout.log"), "a")  # append: earlier attempts are kept
        logf.write(f"\n===== attempt {self.attempts[name]} (this invocation) started "
                   f"{spec['launched_at']} =====\n")
        logf.flush()
        p = subprocess.Popen(cmd, cwd=os.path.join(rd, "work"), stdout=logf, stderr=subprocess.STDOUT,
                             env=env, start_new_session=True)  # signals reach children only via us
        logf.close()  # the child holds its own copy of the descriptor
        self.children[p] = name
        return p

    def run_pool(self, todo):
        """Run the given (name, settings) pairs with at most --jobs children at a time."""
        queue, running, done = list(todo), {}, 0
        while (queue or running) and _stop_signal is None:
            while queue and len(running) < self.a.jobs and _stop_signal is None:
                name, settings = queue.pop(0)
                running[self.launch(name, settings)] = (name, settings, time.time())
                self.log(f"[{name}] START (running={len(running)}, queued={len(queue)})")
            time.sleep(1.0)
            for p in [p for p in running if p.poll() is not None]:
                name, settings, t_start = running.pop(p)
                self.children.pop(p, None)
                done += 1
                el = time.time() - t_start
                d = read_json(self.done_path(name))
                tail = f"| done {done}/{len(todo)}, remaining {len(queue) + len(running)}"
                if p.returncode == 0 and d is not None:
                    self.log(f"[{name}] FINISH elapsed={el:.0f}s peak_rss={d.get('peak_rss_mb', 0):.0f} MB "
                             f"rows={d.get('rows')} {tail}")
                else:
                    self.log(f"[{name}] FAILED rc={p.returncode} elapsed={el:.0f}s (see "
                             f"runs/{name}/stdout.log) {tail}")
        if _stop_signal is not None:
            self.terminate_children()
            raise SystemExit(128 + _stop_signal)

    def terminate_children(self):
        """Terminate any live children (idempotent). Completed runs stay on disk."""
        live = [p for p in self.children if p.poll() is None]
        for p in live:
            try:
                p.terminate()
            except ProcessLookupError:
                pass
        deadline = time.time() + 10
        for p in live:
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        self.children.clear()
        if live:
            self.log(f"terminated {len(live)} running child process(es); completed runs kept")

    # ------------------------------------------------------------------ main
    def run(self):
        all_runs = load_runs(self.a.runs_file)
        by_name = {r["name"]: r for r in all_runs}
        if self.a.runs == "default":
            selected = [r for r in all_runs if r.get("default", True)]
        elif self.a.runs == "all":
            selected = list(all_runs)
        else:
            names = [n.strip() for n in self.a.runs.split(",") if n.strip()]
            unknown = [n for n in names if n not in by_name]
            if unknown:
                raise SystemExit(f"ERROR: unknown run name(s) {unknown}; known: {sorted(by_name)}")
            selected = [by_name[n] for n in names]
        plan = [(r["name"], settings_of(r, self.a.child_num_steps)) for r in selected]

        self.commit, self.dirty = self.git_state()
        self.log(f"run_scenarios start: out={self.out} code_root={self.code_root} commit={self.commit} "
                 f"dirty={self.dirty} data_dir={self.data_dir} jobs={self.a.jobs} python={self.a.python}")
        if self.a.child_num_steps:
            self.log(f"TEST MODE: num_steps={self.a.child_num_steps} for every run")
        # Every input file of every selected run must exist before anything starts.
        files_of, missing = {}, []
        for name, s in plan:
            files_of[name] = self.config_inputs(s)
            for cname, fname in files_of[name].items():
                if not os.path.isfile(os.path.join(self.data_dir, fname)):
                    missing.append(f"{fname} (config.{cname}, run {name})")
        if missing:
            raise SystemExit("ERROR: input file(s) not found in --data-dir " + self.data_dir + ":\n  "
                             + "\n  ".join(missing) + "\nNothing was started.")
        for name, s in plan:
            self.inputs[name] = self.input_manifest(files_of[name])
            for fname, info in self.inputs[name].items():
                self.log(f"[{name}] input {fname} ({info['config_name']}) -> {info['source']} "
                         f"{info['bytes']} bytes sha256={info['sha256'][:16]}...")
            if s["output_name"] + ".csv" in set(self.inputs[name]) | set(OTHER_OUTPUTS):
                raise SystemExit(f"ERROR: run {name}: output {s['output_name']}.csv would collide "
                                 f"with an input or another output of main.py")

        todo, skipped = [], []
        for name, s in plan:
            if self.completed(name, s):
                skipped.append(name)
            else:
                todo.append((name, s))
        self.log(f"{len(plan)} runs selected: {len(skipped)} already complete (skipped"
                 f"{': ' + ', '.join(skipped) if skipped else ''}), {len(todo)} to run")

        self.run_pool(todo)
        failed = [(n, s) for n, s in todo if not self.completed(n, s)]
        if failed and _stop_signal is None:
            for n, _ in failed:
                self.log(f"[{n}] RETRY (no completion marker after the first attempt)")
            self.run_pool(failed)
        return self.summarize(plan)

    def summarize(self, plan):
        rows, ok = [], True
        for name, s in plan:
            d = read_json(self.done_path(name)) if self.completed(name, s) else None
            if d is None:
                ok = False
                rows.append({"name": name, "status": "FAILED", "settings": s,
                             "log": os.path.join(self.run_dir(name), "stdout.log")})
                continue
            rows.append({"name": name, "status": "ok", "csv": d["csv"], "rows": d["rows"],
                         "last_timestamp": d["last_timestamp"], "elapsed_s": d["elapsed_s"],
                         "peak_rss_mb": d["peak_rss_mb"], "commit": d["commit"],
                         "forecaster_priced_rows": d["forecaster_priced_rows"],
                         "first_forecast_mode_block": d["first_forecast_mode_block"],
                         "settings": s})
        write_json_atomic(os.path.join(self.out, "summary.json"),
                          {"written_at": datetime.now().isoformat(timespec="seconds"),
                           "code_root": self.code_root, "commit": self.commit, "runs": rows})
        self.log("summary (also in summary.json):")
        for r in rows:
            if r["status"] == "ok":
                self.log(f"  {r['name']:<34} ok      rows={r['rows']} last={r['last_timestamp']} "
                         f"forecast_from_block={r['first_forecast_mode_block']} "
                         f"forecaster_priced_rows={r['forecaster_priced_rows']} "
                         f"elapsed={r['elapsed_s']:.0f}s peak_rss={r['peak_rss_mb']:.0f} MB")
            else:
                self.log(f"  {r['name']:<34} FAILED  see {r['log']}")
        self.log("run_scenarios finished: " + ("all runs complete" if ok else "SOME RUNS FAILED"))
        return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jobs", type=int, default=1, help="parallel child processes")
    ap.add_argument("--python", default=sys.executable, help="interpreter for children")
    ap.add_argument("--runs", default="default",
                    help="'default' (runs with default=true), 'all', or comma-separated run names")
    ap.add_argument("--runs-file", default=os.path.join(HERE, "runs.json"))
    ap.add_argument("--code-root", default=ROOT, help="tree with main.py and config.py (default: this repo)")
    ap.add_argument("--data-dir", default=None, help="directory with the input files (default: <code-root>/data)")
    ap.add_argument("--rss-interval", type=float, default=120.0, help="seconds between child memory reports")
    # testing only
    ap.add_argument("--child-num-steps", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.jobs < 1:
        ap.error("--jobs must be >= 1")
    if args.child_num_steps < 0:
        ap.error("--child-num-steps must be >= 0")

    orch = Orchestrator(args)

    def handler(signum, frame):
        global _stop_signal
        _stop_signal = signum

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGHUP, handler)
    try:
        return orch.run()
    except SystemExit as e:
        orch.terminate_children()
        if _stop_signal is None:
            orch.log(str(e.code) if isinstance(e.code, str) else f"exit {e.code}")
            return e.code if isinstance(e.code, int) else 2
        return e.code
    except BaseException:
        orch.terminate_children()  # do not leave children running after an unexpected error
        raise


if __name__ == "__main__":
    sys.exit(main())
