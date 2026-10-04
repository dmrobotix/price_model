#!/usr/bin/env python
"""Export the April 2026 code and patch it to re-run one original scenario.

The April code (git tag paper3-april-2026) has no PRICE_* overrides: its run
settings are hardcoded lines in main.py and config.py. This script exports the
tag with ``git archive`` (code only: main.py, config.py, modules/) into a new directory, writes a PROVENANCE file, and
replaces exactly those lines with the settings of the saved original
``data/results/simulation_results_efficiency_frozen_price_fixed.csv``.

Every replaced line must match its expected April text exactly, and exactly
once, or the script stops without patching. The unified diff is written to
``<export-dir>/PATCH.diff`` and printed.

The patched tree is then run by run_scenarios.py with
``--code-root <export-dir> --runs repro_april_frozen_price_fixed``; run_one.py
checks that the patched values are the ones in effect. See scenarios/README.md.
"""
import argparse
import difflib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

OUTPUT_NAME = "simulation_results_efficiency_frozen_price_fixed_repro-april"
# Only the code main.py needs. The tag also tracks data/ (about 5.9 GB), which is
# not exported: the inputs come from run_scenarios.py --data-dir.
CODE_PATHS = ["main.py", "config.py", "modules"]

# (file, expected April line without trailing whitespace, replacement line).
# A replacement of None asserts the line is present and leaves it unchanged.
PATCHES = [
    ("main.py",
     "num_steps = 921_683 # October 31, 2025",
     "num_steps = 1_260_000 # ~ 2032  [patched by prepare_april_reproduction.py]"),
    ("main.py",
     "    'historical_cutoff': datetime(2025, 1, 1, 0, 0, 0),",
     "    'historical_cutoff': datetime(2025, 11, 8, 17, 58, 32), # last transaction fee data  [patched]"),
    ("main.py",
     'file_name = "simulation_results_hindcasting"',
     f'file_name = "{OUTPUT_NAME}"  # [patched]'),
    ("config.py",
     'FORECAST_MODEL = "powerlaw"          # Options: "linear", "logistic", or "powerlaw", "fixed"',
     'FORECAST_MODEL = "fixed"  # [patched]'),
    ("config.py",
     "FORECAST_TARGET_DATE = datetime(2032, 1, 1)  # Target date for the forecast (e.g., January 1, 2040)",
     None),
    ("config.py",
     "FORECAST_TARGET_PRICE = 1_000_000       # Target BTC price (in USD) for the forecast use None is fixed",
     "FORECAST_TARGET_PRICE = None  # [patched] fixed model: last historical price"),
    ("config.py",
     'EFFICIENCY_SCENARIO = "frontier"   # options: "frozen", "frontier"',
     'EFFICIENCY_SCENARIO = "frozen"  # [patched]'),
    # The original was produced from the pre-"latest" data files: its forecast price
    # (91821.06) is the last row of market_price_min.csv, and its H_hist column ends at
    # block 925,641, the last block of combined_block_data.csv.
    ("config.py",
     "BLOCK_PACE_DATA = '../data/combined_block_data_latest.csv'",
     "BLOCK_PACE_DATA = '../data/combined_block_data.csv'  # [patched]"),
    ("config.py",
     "TX_BLOCK_DATA = '../data/txfee_data.csv'",
     None),
    ("config.py",
     "PRICE_DATA = '../data/market_price_min_latest.csv'",
     "PRICE_DATA = '../data/market_price_min.csv'  # [patched]"),
    ("config.py",
     "MACHINE_DATA_FILE = '../data/cbeci_machines_090325.csv'",
     None),
]


def git(repo, *args, **kw):
    return subprocess.run(["git", "-C", repo, *args], check=True, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--export-dir", required=True, help="new directory for the patched export")
    ap.add_argument("--tag", default="paper3-april-2026")
    ap.add_argument("--repo", default=ROOT, help="git repository holding the tag")
    # testing only: a short patched run (the output name gets a _steps<N> suffix)
    ap.add_argument("--num-steps", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()

    patches = list(PATCHES)
    if args.num_steps:
        if args.num_steps <= 0:
            ap.error("--num-steps must be positive")
        patches = [(f, old, f"num_steps = {args.num_steps}  # [patched: TEST LENGTH]")
                   if old.startswith("num_steps =") else
                   (f, old, f'file_name = "{OUTPUT_NAME}_steps{args.num_steps}"  # [patched: TEST]')
                   if old.startswith("file_name =") else (f, old, new)
                   for f, old, new in PATCHES]

    out = os.path.abspath(args.export_dir)
    if os.path.exists(out):
        print(f"ERROR: {out} already exists; give a new directory", file=sys.stderr)
        return 2
    commit = git(args.repo, "rev-parse", f"{args.tag}^{{commit}}",
                 capture_output=True, text=True).stdout.strip()
    os.makedirs(out)
    archive = subprocess.Popen(["git", "-C", args.repo, "archive", "--format=tar", commit, "--", *CODE_PATHS],
                               stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", out], stdin=archive.stdout, check=True)
    archive.stdout.close()
    if archive.wait() != 0:
        print("ERROR: git archive failed", file=sys.stderr)
        return 2

    originals, patched = {}, {}
    for fname in sorted({p[0] for p in patches}):
        with open(os.path.join(out, fname)) as f:
            originals[fname] = f.read().splitlines(keepends=True)
        patched[fname] = list(originals[fname])
    for fname, old, new in patches:
        lines = patched[fname]
        hits = [i for i, ln in enumerate(lines) if ln.rstrip() == old]
        if len(hits) != 1:
            print(f"ERROR: {fname}: expected exactly one line {old!r}, found {len(hits)}; "
                  f"nothing patched (remove {out} before retrying)", file=sys.stderr)
            return 2
        if new is not None:
            lines[hits[0]] = new + "\n"

    diff = []
    for fname in sorted(patched):
        diff += difflib.unified_diff(originals[fname], patched[fname],
                                     fromfile=f"a/{fname}", tofile=f"b/{fname}")
        with open(os.path.join(out, fname), "w") as f:
            f.writelines(patched[fname])
    with open(os.path.join(out, "PATCH.diff"), "w") as f:
        f.writelines(diff)
    with open(os.path.join(out, "PROVENANCE"), "w") as f:
        f.write(f"commit={commit}\nsource=git-archive\ntag={args.tag}\n"
                f"patched=scenarios/prepare_april_reproduction.py (see PATCH.diff): "
                f"settings of simulation_results_efficiency_frozen_price_fixed.csv\n")
        if args.num_steps:
            f.write(f"test_num_steps={args.num_steps} (TEST: not the reproduction length)\n")
    sys.stdout.writelines(diff)
    print(f"\nexported {args.tag} ({commit}) to {out}, patched; PROVENANCE and PATCH.diff written.")
    print("Input files this patched code reads from --data-dir: combined_block_data.csv, "
          "txfee_data.csv, market_price_min.csv, cbeci_machines_090325.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
