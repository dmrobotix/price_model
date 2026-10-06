#!/usr/bin/env python
"""Run main.py once, for one scenario, in its own process.

run_scenarios.py launches this script with the working directory set to
``<run dir>/work``, so main.py's relative ``../data/`` paths resolve to
``<run dir>/data``: symlinks to the shared input files, plus this run's own
outputs (the result CSV, ``hashrate_comparison.png`` and ``logs/model_debug.log``).
The run's settings arrive as PRICE_* environment variables (see config.py).

Before the simulation starts, the values of config.py actually in effect are
printed and compared with the requested settings. After main.py returns, the
values main.py resolved (num_steps, historical_cutoff, file_name) are compared
too, the CSV on disk is checked for its last block, and a completion marker
(DONE.json) is written atomically. The marker is what run_scenarios.py uses to
decide that a run is complete.

The same checks apply to code without the PRICE_* overrides (the April 2026
code, patched by prepare_april_reproduction.py): its hardcoded values must
then equal the requested settings.

Exit status: 0 success; 1 main.py raised; 3 config.py values differ from the
requested settings (nothing run); 4 main.py resolved values differ, or the CSV
is missing or incomplete; 5 another process holds this run's lock; 2 bad
arguments.
"""
import argparse
import ast
import fcntl
import importlib.util
import json
import os
import resource
import runpy
import socket
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# config.py names printed and recorded for every run
CONFIG_NAMES = ("S", "C_ELEC", "S_0", "C_ELEC_0", "T_STAR", "EFFICIENCY_SCENARIO",
                "FORECAST_MODEL", "FORECAST_TARGET_DATE", "FORECAST_TARGET_PRICE",
                "CALIBRATION_MODE", "BLOCK_PACE_DATA", "TX_BLOCK_DATA", "PRICE_DATA",
                "MACHINE_DATA_FILE")


def say(msg):
    print(f"[run_one] {msg}", flush=True)


def load_git_state():
    """git_state() from calibration/run_grid_point.py, loaded by file path so that the
    code root of the run (which may be another tree) is not shadowed on sys.path."""
    path = os.path.join(ROOT, "calibration", "run_grid_point.py")
    spec = importlib.util.spec_from_file_location("_run_grid_point_for_git_state", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.git_state


def jsonable(v):
    """Convert config/main values to JSON-safe values (datetimes as ISO text)."""
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if hasattr(v, "item"):  # numpy scalar
        return v.item()
    return v


def same_number(a, b):
    if a is None or b is None:
        return a is None and b is None
    return float(a) == float(b)


def config_mismatches(config, s):
    """Requested settings vs config.py values in effect; returns a list of messages."""
    want = {
        "EFFICIENCY_SCENARIO": s["efficiency_scenario"],
        "FORECAST_MODEL": s["forecast_model"],
        "FORECAST_TARGET_DATE": datetime.fromisoformat(s["forecast_target_date"]),
        "FORECAST_TARGET_PRICE": s["forecast_target_price"],
        "BLOCK_PACE_DATA": s["block_pace_data"],
        "PRICE_DATA": s["price_data"],
    }
    want.update(s.get("expect_config") or {})
    bad = []
    for name, expected in want.items():
        actual = getattr(config, name, "<missing>")
        ok = (same_number(actual, expected) if name == "FORECAST_TARGET_PRICE"
              else actual == expected)
        if not ok:
            bad.append(f"config.{name} = {actual!r}, requested {expected!r}")
    return bad


def main_mismatches(g, s):
    want = {"num_steps": s["num_steps"],
            "historical_cutoff": datetime.fromisoformat(s["historical_cutoff"]),
            "file_name": s["output_name"]}
    got = {"num_steps": g.get("num_steps"),
           "historical_cutoff": g.get("params", {}).get("historical_cutoff"),
           "file_name": g.get("file_name")}
    return [f"main.py {k} = {got[k]!r}, requested {want[k]!r}" for k in want if got[k] != want[k]], got


def hardcoded_main_values(path):
    """num_steps, file_name and params['historical_cutoff'] as hardcoded in main.py.

    Used for code without the PRICE_* overrides, where the hardcoded lines decide the
    run. Takes the last top-level assignment of each name; a value that is not a
    literal (or a datetime(...) of literals) is reported as such."""
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    found = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if name in ("num_steps", "file_name"):
            try:
                found[name] = ast.literal_eval(node.value)
            except ValueError:
                found[name] = "<not a literal>"
        elif name == "params" and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == "historical_cutoff":
                    try:
                        if not (isinstance(v, ast.Call) and getattr(v.func, "id", None) == "datetime"
                                and not v.keywords):
                            raise ValueError
                        found["historical_cutoff"] = datetime(*[ast.literal_eval(a) for a in v.args])
                    except ValueError:
                        found["historical_cutoff"] = "<not a datetime literal>"
    return found


def csv_last_block(path):
    """block_height of the last data row of a CSV, read from the file's tail."""
    with open(path, "rb") as f:
        header = f.readline().decode().rstrip("\r\n").split(",")
        col = header.index("block_height")
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 65536))
        lines = [ln for ln in f.read().decode(errors="replace").splitlines() if ln.strip()]
    return int(float(lines[-1].split(",")[col])), header


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


def rss_sampler(interval, t0):
    """Print resident and peak resident memory of this process every ``interval`` s."""
    def read():
        vals = {}
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    k, v = line.split(":", 1)
                    vals[k] = int(v.split()[0]) / 1024.0  # kB -> MB
        return vals
    while True:
        time.sleep(interval)
        try:
            v = read()
            say(f"memory t={time.time() - t0:.0f}s rss={v.get('VmRSS', float('nan')):.0f} MB "
                f"peak={v.get('VmHWM', float('nan')):.0f} MB")
        except OSError:
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--code-root", required=True, help="tree holding main.py, config.py, modules/")
    ap.add_argument("--settings", required=True, help="settings.json written by run_scenarios.py")
    ap.add_argument("--done", required=True, help="completion marker to write on success")
    ap.add_argument("--rss-interval", type=float, default=120.0,
                    help="seconds between memory reports (0 = off)")
    args = ap.parse_args()

    t0 = time.time()
    code_root = os.path.abspath(args.code_root)
    with open(args.settings) as f:
        spec = json.load(f)
    s = spec["settings"]
    name = spec["name"]

    # One process per run directory: an orphan from an earlier orchestrator holds the lock.
    lock_f = open(os.path.join(os.path.dirname(os.path.abspath(args.done)), ".run.lock"), "w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        say(f"another process is already running {name} (lock held); not starting")
        return 5

    commit, dirty = load_git_state()(code_root)
    say(f"=== run {name} on {socket.gethostname()} at {datetime.now().isoformat(timespec='seconds')}")
    say(f"code_root={code_root} commit={commit} dirty={dirty}")
    provenance_file = None
    try:
        with open(os.path.join(code_root, "PROVENANCE")) as f:
            provenance_file = f.read()
        say("PROVENANCE: " + " | ".join(provenance_file.strip().splitlines()))
    except OSError:
        pass
    say(f"python={sys.executable} cwd={os.getcwd()}")
    say("environment: " + " ".join(f"{k}={os.environ[k]}" for k in sorted(os.environ)
                                   if k.startswith("PRICE_")))
    say("requested settings: " + json.dumps(s, sort_keys=True))

    sys.path.insert(0, code_root)
    import config  # the same module object main.py's "from config import ..." will use
    in_effect = {k: getattr(config, k, None) for k in CONFIG_NAMES}
    say("config.py in effect: " + json.dumps({k: jsonable(v) for k, v in in_effect.items()}))
    bad = config_mismatches(config, s)
    if bad:
        for b in bad:
            say("MISMATCH " + b)
        say("config.py does not carry the requested settings; simulation NOT started")
        return 3

    if s.get("end_boundary_utc"):
        # The run must end at the boundary block of a UTC instant (modules/boundaries.py):
        # the first block stamped later than the instant, which is the last block of the
        # period before it. Checked against the run's block data before the long run.
        try:
            from modules.boundaries import boundary_block, load_block_times
            end_block = boundary_block(load_block_times(config.BLOCK_PACE_DATA),
                                       datetime.fromisoformat(s["end_boundary_utc"]))
        except Exception as exc:
            say(f"cannot find the boundary block of end_boundary_utc={s['end_boundary_utc']}: {exc!r}; "
                "simulation NOT started")
            return 3
        say(f"end_boundary_utc {s['end_boundary_utc']}: boundary block {end_block}")
        if s["num_steps"] != end_block:
            say(f"MISMATCH num_steps {s['num_steps']} is not the boundary block {end_block} of "
                f"{s['end_boundary_utc']} UTC; simulation NOT started")
            return 3

    main_py = os.path.join(code_root, "main.py")
    if not hasattr(config, "RUN_OVERRIDE_VARS"):
        # Code without the PRICE_* overrides ignores them: its hardcoded main.py
        # values must already be the requested ones. Check before the long run.
        hard = hardcoded_main_values(main_py)
        say("code without PRICE_* overrides; main.py hardcodes: "
            + json.dumps({k: jsonable(v) for k, v in hard.items()}))
        bad, _ = main_mismatches({"num_steps": hard.get("num_steps"), "file_name": hard.get("file_name"),
                                  "params": {"historical_cutoff": hard.get("historical_cutoff")}}, s)
        if bad:
            for b in bad:
                say("MISMATCH " + b)
            say("main.py does not hardcode the requested settings; simulation NOT started")
            return 3

    if args.rss_interval > 0:
        threading.Thread(target=rss_sampler, args=(args.rss_interval, t0), daemon=True).start()

    say(f"running {main_py}")
    try:
        g = runpy.run_path(main_py, run_name="__main__")
    except BaseException:
        traceback.print_exc()
        say("main.py raised; no completion marker written")
        return 1
    elapsed = time.time() - t0

    bad, main_values = main_mismatches(g, s)
    if bad:
        for b in bad:
            say("MISMATCH " + b)
        say("main.py ran with values other than the requested ones; no completion marker written")
        return 4

    csv_path = os.path.abspath(os.path.join("..", "data", s["output_name"] + ".csv"))
    if not os.path.isfile(csv_path):
        say(f"output CSV {csv_path} not found; no completion marker written")
        return 4
    last_block, header = csv_last_block(csv_path)
    df = g["simulation_df"]
    if last_block != s["num_steps"] or len(df) != s["num_steps"] + 1:
        say(f"output incomplete: last block in CSV {last_block}, rows in memory {len(df)}, "
            f"expected last block {s['num_steps']} and {s['num_steps'] + 1} rows")
        return 4

    # Where the model left history. Forecast mode = rows with mean_bt set; forecaster
    # prices = rows whose pre-step time is past last_hist_time (simulation.py).
    import pandas as pd
    last_hist_time = pd.Timestamp(g["last_hist_time"])
    pre_step = df["Timestamp"].shift(1)
    priced = (pre_step > last_hist_time).to_numpy()
    fmode = df["mean_bt"].notna().to_numpy() if "mean_bt" in df.columns else None
    mp = g["market_prices"].dropna(subset=["Time", "Price (USD)"]).sort_values("Time")
    diag = {
        "rows": int(len(df)),
        "last_block": last_block,
        "last_timestamp": str(df["Timestamp"].iloc[-1]),
        "first_forecast_mode_block": (int(df["block_height"].to_numpy()[fmode.argmax()])
                                      if fmode is not None and fmode.any() else None),
        "last_hist_time": str(last_hist_time),
        "forecaster_priced_rows": int(priced.sum()),
        "first_forecaster_priced_block": (int(df["block_height"].to_numpy()[priced.argmax()])
                                          if priced.any() else None),
        "price_data_last_time": str(mp["Time"].iloc[-1]),
        "price_data_last_price": float(mp["Price (USD)"].iloc[-1]),
    }
    say("diagnostics: " + json.dumps(diag))
    if diag["forecaster_priced_rows"] == 0:
        say("no step was priced by the forecaster: FORECAST_* settings had no effect on this run")

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KiB
    done = {
        "name": name,
        "settings": s,
        "inputs": spec.get("inputs"),
        "config_in_effect": {k: jsonable(v) for k, v in in_effect.items()},
        "main_in_effect": {k: jsonable(v) for k, v in main_values.items()},
        "code_root": code_root,
        "commit": commit,
        "dirty": dirty,
        "provenance_file": provenance_file,
        "python": sys.executable,
        "host": socket.gethostname(),
        "csv": csv_path,
        "csv_bytes": os.path.getsize(csv_path),
        "csv_columns": header,
        **diag,
        "elapsed_s": elapsed,
        "peak_rss_mb": peak_rss_mb,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json_atomic(args.done, done)
    say(f"complete: {csv_path} ({done['csv_bytes']} bytes) elapsed={elapsed:.0f}s "
        f"peak_rss={peak_rss_mb:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
