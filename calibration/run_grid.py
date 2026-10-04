#!/usr/bin/env python
"""Resumable, parallel driver for the two calibration grids.

Phase 1 evaluates every early-era point (S_0, C_elec_0) with the modern-era
parameters fixed from config.py and picks the minimum-RMSLE point.  Phase 2
evaluates every modern-era point (S, C_elec) with the early-era parameters set
to the phase-1 optimum.  Each point runs in its own process
(calibration/run_grid_point.py).  See calibration/README.md.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
POINT_SCRIPT = os.path.join(HERE, "run_grid_point.py")
CRASH_SENTINEL = 1e9

# name -> (merged file, optimum file, coordinate columns)
GRIDS = {
    "pre2018": dict(merged="grid_pre2018.csv", optimum="phase1_optimum.json", cols=("S_0", "C_elec_0")),
    "modern": dict(merged="grid_2018_2024.csv", optimum="phase2_optimum.json", cols=("S", "C_elec")),
}

# Environment forced on every child: one thread per child (parallelism comes from
# the process pool), unbuffered output so a killed child's log is complete, and no
# bytecode files written into the source tree.
CHILD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}

_stop_signal = None  # set by the signal handler


class Orchestrator:
    def __init__(self, args):
        self.a = args
        self.out = os.path.abspath(args.out_dir)
        self.points_dir = os.path.join(self.out, "points")
        self.logs_dir = os.path.join(self.out, "logs")
        for d in (self.out, self.points_dir, self.logs_dir):
            os.makedirs(d, exist_ok=True)
        self.progress_path = os.path.join(self.out, "progress.log")
        self.children = {}  # Popen -> label
        self.any_crash = False

    # ---------------------------------------------------------------- logging
    def log(self, msg):
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        with open(self.progress_path, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    # ------------------------------------------------------------------- grid
    def points(self):
        """(S, C) pairs, computed exactly as the scripts do (np.linspace)."""
        s_vals = np.linspace(0.0, 0.1, 11)
        c_vals = np.linspace(10, 150, 15)
        if not self.a.include_s0:
            s_vals = s_vals[1:]  # drop S = 0
        pts = [(float(s), float(c)) for s in s_vals for c in c_vals]
        if self.a.max_points:
            pts = pts[: self.a.max_points]
        return pts

    @staticmethod
    def stem(grid, s, c):
        return f"{grid}_S{s:.2f}_C{c:.0f}"

    def point_path(self, grid, s, c):
        return os.path.join(self.points_dir, self.stem(grid, s, c) + ".csv")

    def read_point(self, path):
        """Return the one-row result as a dict, or None if unreadable."""
        try:
            df = pd.read_csv(path, float_precision="round_trip")
            return df.iloc[0].to_dict() if len(df) == 1 else None
        except Exception:
            return None

    @staticmethod
    def is_crashed(row):
        r = row.get("RMSLE")
        return r is None or not np.isfinite(r) or r >= CRASH_SENTINEL

    # ------------------------------------------------------------- one phase
    def run_phase(self, grid, fixed=None):
        """Run all points of one grid (with one in-invocation retry); return status lists."""
        pts = self.points()
        todo, skipped = [], 0
        for s, c in pts:
            path = self.point_path(grid, s, c)
            if os.path.exists(path):
                row = self.read_point(path)
                if row is not None and grid == "modern" and fixed is not None:
                    if not (np.isclose(row.get("fixed_S_0", np.nan), fixed["S_0"], rtol=1e-12, atol=0)
                            and np.isclose(row.get("fixed_C_elec_0", np.nan), fixed["C_elec_0"],
                                           rtol=1e-12, atol=0)):
                        raise SystemExit(
                            f"ERROR: {path} was computed with S_0={row.get('fixed_S_0')}, "
                            f"C_elec_0={row.get('fixed_C_elec_0')}, but the phase-1 optimum is "
                            f"S_0={fixed['S_0']}, C_elec_0={fixed['C_elec_0']}. The phase-1 optimum "
                            f"changed after these points were computed. Use a fresh --out-dir or "
                            f"delete points/modern_*.csv.")
                if (row is None or (self.a.retry_crashed and self.is_crashed(row))):
                    why = "unreadable" if row is None else "crashed"
                    self.log(f"[{grid}] re-running {self.stem(grid, s, c)} (existing result {why})")
                    os.unlink(path)
                else:
                    skipped += 1
                    continue
            todo.append((s, c))
        self.log(f"[{grid}] {len(pts)} points: {skipped} already done (skipped), {len(todo)} to run, "
                 f"jobs={self.a.jobs}")

        attempted = list(todo)
        self.run_pool(grid, todo, fixed)

        # In-invocation retry: every point attempted above that has no result or a
        # crashed result is run once more through the same pool.
        failed = [(s, c) for s, c in attempted if self.point_failed(grid, s, c)]
        if failed and _stop_signal is None:
            for s, c in failed:
                path = self.point_path(grid, s, c)
                row = self.read_point(path)
                self.log(f"[{grid}] RETRY {self.stem(grid, s, c)} "
                         f"({'no result' if row is None else 'crashed'})")
                if os.path.exists(path):
                    os.unlink(path)
            self.run_pool(grid, failed, fixed)
        return self.status(grid, pts)

    def point_failed(self, grid, s, c):
        row = self.read_point(self.point_path(grid, s, c))
        return row is None or self.is_crashed(row)

    def run_pool(self, grid, todo, fixed):
        """Run the given points with at most --jobs children at a time."""
        done = 0
        queue = list(todo)
        running = {}  # Popen -> (s, c, t_start)
        while (queue or running) and _stop_signal is None:
            while queue and len(running) < self.a.jobs and _stop_signal is None:
                s, c = queue.pop(0)
                running[self.launch(grid, s, c, fixed)] = (s, c, time.time())
                self.log(f"[{grid}] START {self.stem(grid, s, c)} (running={len(running)}, "
                         f"queued={len(queue)})")
            time.sleep(1.0)
            for p in [p for p in running if p.poll() is not None]:
                s, c, t_start = running.pop(p)
                self.children.pop(p, None)
                done += 1
                self.report_finish(grid, s, c, p.returncode, time.time() - t_start,
                                   done, len(todo), len(queue) + len(running))
        if _stop_signal is not None:
            self.terminate_children()
            raise SystemExit(128 + _stop_signal)

    def launch(self, grid, s, c, fixed):
        stem = self.stem(grid, s, c)
        cmd = [self.a.python, POINT_SCRIPT, "--grid", grid,
               # repr() round-trips the float exactly
               "--s", repr(s), "--c", repr(c), "--out", self.point_path(grid, s, c)]
        if grid == "modern":
            cmd += ["--s0", repr(float(fixed["S_0"])), "--c0", repr(float(fixed["C_elec_0"]))]
        if self.a.child_num_steps:
            cmd += ["--num-steps", str(self.a.child_num_steps)]
        env = os.environ.copy()
        env.update(CHILD_ENV)
        logf = open(os.path.join(self.logs_dir, stem + ".log"), "w")
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                             start_new_session=True)  # signals reach children only via us
        logf.close()  # child holds its own copy of the descriptor
        self.children[p] = stem
        return p

    def report_finish(self, grid, s, c, rc, elapsed, done, n_todo, remaining):
        stem = self.stem(grid, s, c)
        row = self.read_point(self.point_path(grid, s, c))
        if row is None:
            self.log(f"[{grid}] FAILED {stem} rc={rc} elapsed={elapsed:.0f}s (no result written; "
                     f"see logs/{stem}.log) | done {done}/{n_todo}, remaining {remaining}")
        elif rc != 0 or self.is_crashed(row):
            self.log(f"[{grid}] CRASHED {stem} rc={rc} elapsed={elapsed:.0f}s RMSLE={row['RMSLE']:.4e} "
                     f"| done {done}/{n_todo}, remaining {remaining}")
        else:
            self.log(f"[{grid}] FINISH {stem} elapsed={elapsed:.0f}s RMSLE={row['RMSLE']:.6e} "
                     f"| done {done}/{n_todo}, remaining {remaining}")

    def terminate_children(self):
        """Terminate any live children (idempotent). Completed points stay on disk."""
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
            self.log(f"terminated {len(live)} running child process(es); completed points kept")

    # ----------------------------------------------------- merge and optimum
    def status(self, grid, pts):
        """Return (ok, crashed, missing) as lists of point names."""
        ok, crashed, missing = [], [], []
        for s, c in pts:
            row = self.read_point(self.point_path(grid, s, c))
            name = self.stem(grid, s, c)
            if row is None:
                missing.append(name)
            elif self.is_crashed(row):
                crashed.append(name)
            else:
                ok.append(name)
        return ok, crashed, missing

    def merge(self, grid, pts):
        rows = []
        for s, c in pts:
            row = self.read_point(self.point_path(grid, s, c))
            if row is not None:
                rows.append(row)
        if not rows:
            return None
        df = pd.DataFrame(rows).sort_values(list(GRIDS[grid]["cols"])).reset_index(drop=True)
        merged_path = os.path.join(self.out, GRIDS[grid]["merged"])
        df.to_csv(merged_path, index=False)
        self.log(f"[{grid}] merged {len(df)} rows -> {merged_path}")
        return df

    def optimum(self, grid, df, pts, n_crashed, n_missing, fixed=None):
        valid = df[np.isfinite(df["RMSLE"]) & (df["RMSLE"] < CRASH_SENTINEL)]
        if valid.empty:
            raise SystemExit(f"ERROR: [{grid}] no valid (non-crashed) rows; cannot pick an optimum")
        best = valid.loc[valid["RMSLE"].idxmin()]
        xc, yc = GRIDS[grid]["cols"]
        opt = {xc: float(best[xc]), yc: float(best[yc]), "RMSLE": float(best["RMSLE"]),
               "n_points_valid": int(len(valid)), "n_points_crashed": n_crashed,
               "n_points_expected": len(pts), "n_points_missing": n_missing,
               "include_s0": bool(self.a.include_s0)}
        edge_axes = []
        if opt[xc] in (min(p[0] for p in pts), max(p[0] for p in pts)):
            edge_axes.append(xc)
        if opt[yc] in (min(p[1] for p in pts), max(p[1] for p in pts)):
            edge_axes.append(yc)
        opt["on_grid_edge"] = bool(edge_axes)
        if fixed is not None:
            opt["fixed_S_0"], opt["fixed_C_elec_0"] = fixed["S_0"], fixed["C_elec_0"]
        path = os.path.join(self.out, GRIDS[grid]["optimum"])
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(opt, f, indent=2)
        os.replace(tmp, path)
        self.log(f"[{grid}] optimum {xc}={opt[xc]!r} {yc}={opt[yc]!r} RMSLE={opt['RMSLE']:.6e} "
                 f"-> {path}")
        if n_crashed or n_missing:
            self.log(f"[{grid}] WARNING optimum chosen with {n_crashed} crashed and {n_missing} missing points")
        if edge_axes:
            self.log(f"[{grid}] WARNING optimum is on the grid edge ({', '.join(edge_axes)}); "
                     f"the true optimum may lie outside the grid")
        return opt

    def finish_phase(self, grid, fixed=None):
        """Run, merge and report one phase. Returns (optimum dict, complete flag)."""
        ok, crashed, missing = self.run_phase(grid, fixed)
        pts = self.points()
        df = self.merge(grid, pts)
        self.log(f"[{grid}] complete: {len(ok)} ok, {len(crashed)} crashed, {len(missing)} missing "
                 f"of {len(pts)}")
        complete = not crashed and not missing
        if not complete:
            self.any_crash = True
            self.log(f"ERROR [{grid}] incomplete after retry. crashed: {crashed or 'none'}; "
                     f"missing (no result): {missing or 'none'}")
        if df is None:
            raise SystemExit(f"ERROR: [{grid}] no results to merge")
        return self.optimum(grid, df, pts, len(crashed), len(missing), fixed), complete

    def run(self):
        self.log(f"run_grid start: out={self.out} phase={self.a.phase} jobs={self.a.jobs} "
                 f"include_s0={self.a.include_s0} python={self.a.python}")
        opt1_path = os.path.join(self.out, GRIDS["pre2018"]["optimum"])
        if self.a.phase in ("1", "both"):
            _, complete = self.finish_phase("pre2018")
            if not complete and self.a.phase == "both":
                self.log("ERROR phase 2 NOT started: phase 1 is incomplete. Fix or re-run the "
                         "listed points (use --retry-crashed for crashed ones), then resume.")
                return 1
        if self.a.phase in ("2", "both"):
            if not os.path.exists(opt1_path):
                raise SystemExit(f"ERROR: {opt1_path} not found; run phase 1 first")
            with open(opt1_path) as f:
                o1 = json.load(f)
            if o1.get("n_points_crashed") or o1.get("n_points_missing"):
                raise SystemExit(
                    f"ERROR: phase 2 NOT started: {opt1_path} was written from an incomplete phase 1 "
                    f"({o1.get('n_points_crashed')} crashed, {o1.get('n_points_missing')} missing). "
                    f"Re-run phase 1 until it is complete.")
            fixed = {"S_0": o1["S_0"], "C_elec_0": o1["C_elec_0"]}
            self.log(f"phase 2 uses phase-1 optimum S_0={fixed['S_0']!r}, C_elec_0={fixed['C_elec_0']!r}")
            self.finish_phase("modern", fixed)
        self.log("run_grid finished with " + ("CRASHED/MISSING POINTS" if self.any_crash else "no crashes"))
        return 1 if self.any_crash else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jobs", type=int, default=1, help="parallel child processes")
    ap.add_argument("--python", default=sys.executable, help="interpreter for children")
    ap.add_argument("--include-s0", action="store_true", help="also run S = 0 (default off)")
    ap.add_argument("--phase", choices=["1", "2", "both"], default="both")
    ap.add_argument("--retry-crashed", action="store_true",
                    help="re-run points whose existing result is a crash (RMSLE = 1e9)")
    # testing only
    ap.add_argument("--max-points", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--child-num-steps", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.jobs < 1:
        ap.error("--jobs must be >= 1")

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
