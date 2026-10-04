#!/usr/bin/env python
"""Evaluate ONE calibration grid point in its own process.

Imports one of the two grid scripts in ``notebooks/`` (their grid loops sit
under ``if __name__ == "__main__"``, so importing runs only the data setup),
then calls the module's own ``objective_function`` with the module's own
``initial_state``, ``params``, ``num_steps`` and ``H_hist_series``, exactly as
the serial loop in that script does.  The result is one CSV row, written
atomically.

Exit status: 0 on success; 1 if the objective returned the 1e9 crash sentinel
(the row is still written) or raised; 2 on bad arguments.
"""
import argparse
import importlib.util
import os
import resource
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = os.path.join(ROOT, "notebooks")

SCRIPTS = {
    "pre2018": "sensitivity_analysis_pre2018_blocks.py",
    "modern": "sensitivity_analysis_2018-2024_rmsle_blocks.py",
}
# Column names of the two coordinates, as in each script's own results CSV.
COLUMNS = {
    "pre2018": ("S_0", "C_elec_0"),
    "modern": ("S", "C_elec"),
}
CRASH_SENTINEL = 1e9


def git_state(root=ROOT):
    """Return (commit, dirty) for the code under ``root``.

    1. If ``root`` is itself the top level of a git work tree: HEAD hash and a
       dirty flag (tracked files differ from HEAD; untracked files are ignored).
       A ``root`` nested inside some other repository does NOT count, so a parent
       repository's HEAD is never reported.
    2. Otherwise, if ``root/PROVENANCE`` exists (written when the code was exported
       with ``git archive``): its ``commit=<sha>`` line, with dirty = "archive".
    3. Otherwise ("unknown", "unknown").
    """
    def run(*args):
        return subprocess.run(["git", "-C", root, *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        top = run("rev-parse", "--show-toplevel")
        if os.path.realpath(top) == os.path.realpath(root):
            return run("rev-parse", "HEAD"), str(bool(run("status", "--porcelain",
                                                          "--untracked-files=no"))).lower()
    except Exception:
        pass  # no git, not a repository, or git failed: fall through to PROVENANCE
    prov = os.path.join(root, "PROVENANCE")
    try:
        with open(prov) as f:
            fields = dict(line.strip().split("=", 1) for line in f if "=" in line)
        if fields.get("commit"):
            return fields["commit"], "archive"
    except OSError:
        pass
    print("[run_grid_point] no git work tree at the code root and no usable PROVENANCE file",
          flush=True)
    return "unknown", "unknown"


def write_row_atomic(path, header, values):
    """Write a one-row CSV via a temp file in the same directory + os.replace."""
    import csv
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix="." + os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerow([repr(v) if isinstance(v, float) else v for v in values])
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grid", required=True, choices=sorted(SCRIPTS))
    ap.add_argument("--s", type=float, required=True)
    ap.add_argument("--c", type=float, required=True)
    ap.add_argument("--out", required=True, help="one-row result CSV")
    ap.add_argument("--s0", type=float, help="modern grid only: early-era S_0")
    ap.add_argument("--c0", type=float, help="modern grid only: early-era C_elec_0")
    ap.add_argument("--num-steps", type=int,
                    help="testing only; default is the script's own num_steps")
    args = ap.parse_args()

    if args.grid == "pre2018" and (args.s0 is not None or args.c0 is not None):
        ap.error("--s0/--c0 are only valid with --grid modern")
    if (args.s0 is None) != (args.c0 is None):
        ap.error("--s0 and --c0 must be given together")

    out_path = os.path.abspath(args.out)  # resolve before chdir
    git_head, git_dirty = git_state()

    os.chdir(NOTEBOOKS)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    script_path = os.path.join(NOTEBOOKS, SCRIPTS[args.grid])
    t_setup = time.perf_counter()
    spec = importlib.util.spec_from_file_location("grid_script_" + args.grid, script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    setup_s = time.perf_counter() - t_setup
    print(f"[run_grid_point] setup (import of {SCRIPTS[args.grid]}) took {setup_s:.1f}s", flush=True)

    params = mod.params
    if args.grid == "modern":
        if args.s0 is not None:
            params["S_0"] = args.s0
            params["C_elec_0"] = args.c0
        fixed_s0, fixed_c0 = float(params["S_0"]), float(params["C_elec_0"])
    num_steps = args.num_steps if args.num_steps is not None else mod.num_steps

    t0 = time.perf_counter()
    try:
        rmsle = mod.objective_function(
            [args.s, args.c], mod.initial_state, params, num_steps,
            H_hist_series_input=mod.H_hist_series,
        )
    except Exception:
        traceback.print_exc()
        print("[run_grid_point] objective_function raised; no row written", flush=True)
        return 1
    elapsed_s = time.perf_counter() - t0
    rmsle = float(rmsle)

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KiB
    x_name, y_name = COLUMNS[args.grid]
    header = ["grid", x_name, y_name, "RMSLE"]
    values = [args.grid, float(args.s), float(args.c), rmsle]
    if args.grid == "modern":
        header += ["fixed_S_0", "fixed_C_elec_0"]
        values += [fixed_s0, fixed_c0]
    header += ["num_steps", "elapsed_s", "setup_s", "peak_rss_mb", "git_head", "git_dirty"]
    values += [int(num_steps), elapsed_s, setup_s, peak_rss_mb, git_head, git_dirty]
    write_row_atomic(out_path, header, values)
    print(f"[run_grid_point] wrote {out_path}: RMSLE={rmsle:.6e} elapsed={elapsed_s:.1f}s "
          f"peak_rss={peak_rss_mb:.0f} MB", flush=True)

    if rmsle >= CRASH_SENTINEL:
        print("[run_grid_point] objective_function returned the crash sentinel (1e9)", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
