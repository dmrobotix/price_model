#!/usr/bin/env python
"""Statistics and Figures 4-9 of the Paper 3 manuscript, from the saved run CSVs.

This script replaces the parts of data/results/scenario_results.ipynb that the paper
uses. It reads the ten run CSVs of one set (the out-of-sample test, the hindcast, and
the eight forecast scenarios), computes the statistics the paper reports, and draws
Figures 4 to 9. The two sets are:

  original  the CSVs in data/results/ that the paper's numbers and figures came from;
  new       the re-runs in data/results/scenarios_2026-10/runs/<name>/data/
            (scenarios/runs.json gives each run's output name).

Which notebook cell drew which manuscript figure. The images embedded in the
manuscript are byte-identical (md5) to the PNGs the notebook saved in data/results/:

  Figure 4  cell 18  figure_hashrate_hindcasting_windows_with_monthly_ratio.png
  Figure 5  cell 20  figure_hashrate_testing_2025_with_monthly_ratio.png
  Figure 6  cell 3   figure_efficiency_over_time_2026plus.png
  Figure 7  cell 23  figure1_hashrate_EHs__.png
  Figure 8  cell 1   figure2_annual_energy_TWh.png (the second figure of the cell)
  Figure 9  cell 2   figure3_Rpe_line.png (draws with cell 1's data and time axis)

Statistics and the cells they follow:

  out-of-sample (testing run)  cells 11-13: RMSLE, mean and median log residual over
      block_height >= 877,260; cell 20: monthly Sim/Hist ratio over 2025-01-01 to
      2025-10-31 23:59:59;
  hindcast                     cell 18: monthly Sim/Hist ratio in the two windows of
      Figure 4; cells 14 and 16: block-level, weekly-mean and monthly-mean log metrics
      over a 2018-2024 block-height window (see --hindcast-window);
  scenarios                    annual energy as in cell 1 (sum of E_sim, Wh per block,
      by Timestamp year, / 1e12); fleet efficiency as the annual mean of the
      `efficiency` column (W/TH, numerically J/TH); peak R_pe and the share of blocks
      with R_pe below break-even over forecast rows (rows with a non-null mean_bt, never
      a date filter); the 2028 halving date (block 1,050,000) as in cell 24; peak
      hashrate within 2026-01-01..2031-12-31 as in cell 21; the linearity check of
      cell 22.

Clock (--clock). Comparisons with history (the out-of-sample and hindcast monthly
Sim/Hist ratios, their date windows, the weekly and monthly hindcast metrics, and
Figures 4 and 5) bin and plot both hashrate series by one time column:
  historical (default)  Hist_Timestamp, the real time of each block;
  simulated             Timestamp, the model's clock, as the notebook did.
The forecast scenario statistics and Figures 6-9 always use Timestamp (model time).

2018-2024 hindcast metric window (--hindcast-window):
  calibration (default)  blocks [501,995, 877,280), hashrate in TH/s, as the objective
                         function of the 2018-2024 calibration grid
                         (notebooks/sensitivity_analysis_2018-2024_rmsle_blocks.py);
  notebook               blocks [505,227, 877,280), hashrate in H/s, as cells 14-16.
The weekly and monthly metrics use the same block window and units.

Cell 22 used FORECAST_START from an earlier kernel state that the notebook does not
show. Any start from 2025-11-08 00:00 to 2025-11-09 00:00 reproduces its printed
output (N = 30/29 months pre-halving); this script uses 2025-11-08 17:58:33, the
notebook's TIMESTAMP_END_HIST. Cell 22 also used H_COL, which no cell defines; this
script uses H_sim.

Differences from the notebook's figures:
  * the font is Liberation Serif with STIX mathtext (the notebook used DejaVu Sans);
  * each figure keeps the notebook's width (14.5 in) and takes its height from the
    aspect ratio of its drawing in the manuscript (wp:extent of the picture beside the
    caption "Figure N."), or from --size-in; it is saved without bbox_inches="tight",
    so figure-level legends are placed inside the canvas and the axes are laid out
    below or above them;
  * Figure 9's y label prints "$/MWh". The notebook's label "($/MWh$)" was parsed as
    mathtext, so the published figure shows an italic "/MWh" without the dollar sign;
  * Figure 9's break-even line is drawn at the set's electricity cost C_elec. For the
    original set that is $50/MWh, the notebook's hard-coded value; the new runs use
    $40/MWh. "Break-even" everywhere in the outputs means C_elec; the share of blocks
    below $50/MWh is also reported, under that name;
  * Figure 7 has no "Historical" legend entry. The notebook's mask for that line selects
    no rows, so the published figure has the entry and no line;
  * Figures 4 and 5 use the --clock time column (historical by default).

Electricity cost of each set: for `new`, C_ELEC from each run's DONE.json
(config_in_effect); for `original`, C_ELEC in config.py at git tag paper3-april-2026.
--c-elec overrides both.

Every CSV is read in chunks with only the needed columns.

Outputs (in --out-dir, which must not exist; the run writes into a temporary sibling
directory and renames it at the end, so a failed run leaves no --out-dir): stats.json (everything), CSV tables,
verification.csv (computed values against the paper's and the notebook's printed
values), summary.txt, figure4.png .. figure9.png, and run.log-style lines on stdout.

--compare ORIG_DIR NEW_DIR writes comparison.csv and comparison.txt (quantity, original,
new, change) from two stats.json files into --out-dir.

Usage:
    python scenarios/analyze_results.py --set original --out-dir data/results/analysis_2026-10/original
    python scenarios/analyze_results.py --set new --out-dir data/results/analysis_2026-10/new
    python scenarios/analyze_results.py --compare data/results/analysis_2026-10/original \\
        data/results/analysis_2026-10/new --out-dir data/results/analysis_2026-10/compare
"""

import argparse
import importlib.util
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "data", "results")
NEW_RUNS_DIR = os.path.join(RESULTS, "scenarios_2026-10", "runs")
RUNS_JSON = os.path.join(HERE, "runs.json")
APRIL_TAG = "paper3-april-2026"
CHUNK = 200_000
TS_FORMAT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------- notebook constants
OOS_START_HEIGHT = 877_260                                    # cells 11, 12
TEST_T_START = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")  # cell 20
TEST_T_END = pd.Timestamp("2025-10-31 23:59:59", tz="UTC")
HIND_WINDOWS = [                                              # cell 18
    ("2010-07-17 to 2017-12-31", pd.Timestamp("2010-07-17 00:00:00", tz="UTC"),
     pd.Timestamp("2017-12-31 23:59:59", tz="UTC")),
    ("2018-01-01 to 2024-12-31", pd.Timestamp("2018-01-01 00:00:00", tz="UTC"),
     pd.Timestamp("2024-12-31 23:59:59", tz="UTC")),
]
# 2018-2024 hindcast metrics: block-height window [start, end) and the unit scale applied to
# both hashrate series before log1p. "notebook" is cells 14-16 (H/s). "calibration" is the
# window and units of the 2018-2024 calibration grid's objective function
# (notebooks/sensitivity_analysis_2018-2024_rmsle_blocks.py: blocks [501,995, 877,280), TH/s).
HIND_METRIC_WINDOWS = {
    "notebook": (505_227, 877_280, 1.0),
    "calibration": (501_995, 877_280, 1e-12),
}
ORIG_GRID_CSV = os.path.join(RESULTS, "sensitivity_grid_results_rmsle_2018_2024_blocks.csv")
NEW_GRID_OPTIMUM = os.path.join(RESULTS, "grid_2026-10-01", "phase2_optimum.json")
CLOCK_COLUMN = {"historical": "Hist_Timestamp", "simulated": "Timestamp"}
MONTH_RULE = "MS"
WEEK_RULE = "W-SUN"
RATIO_ROLL_MONTHS = 3
TIMESTAMP_END_HIST = pd.Timestamp("2025-11-08 17:58:33")      # cells 1, 2, 23
RPE_START = pd.Timestamp("2025-11-08 17:58:33")               # cells 1, 2
EFF_START = pd.Timestamp("2026-01-01")                        # cell 3
PLOT_START = pd.Timestamp("2026-01-01")                       # cell 23
ENERGY_YEARS = list(range(2026, 2032))                        # cell 1
PEAK_START = pd.Timestamp("2026-01-01")                       # cell 21
PEAK_END = pd.Timestamp("2031-12-31")
HALVING_BLOCK = 1_050_000                                     # cells 22, 24
LIN_START = pd.Timestamp("2025-11-08 17:58:33")               # cell 22 (see docstring)
LIN_END = pd.Timestamp("2031-12-31")
KEEP_FROM = pd.Timestamp("2025-11-01")  # scenario rows kept in memory for the figures
NOTEBOOK_WIDTH_IN = 14.5                # width of every figure 4-9 in the notebook
DPI = 300

EFFS = ["frozen", "frontier"]
PRICES = ["fixed", "200k", "500k", "1M"]
SCENARIOS = [f"{e}_price_{p}" for e in EFFS for p in PRICES]   # runs.json names
COLORS = {"hist": "#111111", "fixed": "#1f77b4", "200k": "#f2c300",
          "500k": "#1b9e77", "1M": "#7b6fd0"}
LABELS = {"fixed": "Price: fixed", "200k": "Price: $200k",
          "500k": "Price: $500k", "1M": "Price: $1M"}
COL_RATIO = "#1f77b4"
COL_RATIO_ROLL = "#FFD700"

# --------------------------------------------- reference values (the claims under test)
# (statistic key, reference value, tolerance, source[, condition]). Keys index the
# flattened stats.json. Tolerances are half a unit in the last printed digit unless noted.
# A condition is a dict of run settings (clock, hindcast_window, set); a reference with a
# condition applies only to a run with those settings, because the notebook computed it
# that way. verify() lists the others as not applicable.
SIM_CLOCK = {"clock": "simulated"}
NB_WINDOW = {"hindcast_window": "notebook"}
NB_WINDOW_SIM_CLOCK = {"hindcast_window": "notebook", "clock": "simulated"}
REFERENCES = [
    ("oos.RMSLE", 0.05188, 5e-6, "paper / CODE_AUDIT F (0.05188)"),
    ("oos.mean_log_residual", 0.03861, 5e-6, "paper / CODE_AUDIT F"),
    ("oos.median_log_residual", 0.03875, 5e-6, "paper / CODE_AUDIT F"),
    ("oos.expm1_RMSLE", 0.053, 5e-4, "paper: 5.3%"),
    ("oos.expm1_mean_log_residual", 0.039, 5e-4, "paper: 3.9%"),
    ("oos.n", 44424, 0, "notebook cell 11"),
    ("oos.RMSLE", 0.05188033650038668, 1e-12, "notebook cell 11 (full precision)"),
    ("oos.mean_residual_rel", 0.03999311116417236, 1e-12, "notebook cell 12"),
    ("oos.median_residual_rel", 0.039508802417095, 1e-12, "notebook cell 12"),
    ("oos.std_log_residual", 0.03465150357168411, 1e-12, "notebook cell 12"),
    ("oos.monthly_ratio_min", 1.005, 5e-4, "CODE_AUDIT F (1.005-1.087)", SIM_CLOCK),
    ("oos.monthly_ratio_max", 1.087, 5e-4, "CODE_AUDIT F (1.005-1.087)", SIM_CLOCK),
    ("oos.first_forecast_block", 877260, 0, "CODE_AUDIT (forecast from 877,260)"),
    ("hindcast.metrics_2018_2024.block_level.n", 372053, 0, "notebook cell 16", NB_WINDOW),
    ("hindcast.metrics_2018_2024.block_level.RMSLE", 0.212490211006604, 1e-12, "notebook cell 16",
     NB_WINDOW),
    ("hindcast.metrics_2018_2024.block_level.mean_log_residual", -0.05343946799710758, 1e-12,
     "notebook cell 16", NB_WINDOW),
    ("hindcast.metrics_2018_2024.weekly_mean.n", 363, 0, "notebook cell 16", NB_WINDOW_SIM_CLOCK),
    ("hindcast.metrics_2018_2024.weekly_mean.RMSLE", 0.21193272972423635, 1e-12, "notebook cell 16",
     NB_WINDOW_SIM_CLOCK),
    ("hindcast.metrics_2018_2024.monthly_mean.n", 84, 0, "notebook cell 16", NB_WINDOW_SIM_CLOCK),
    ("hindcast.metrics_2018_2024.monthly_mean.RMSLE", 0.20924615344344996, 1e-12, "notebook cell 16",
     NB_WINDOW_SIM_CLOCK),
    ("hindcast.metrics_2018_2024.block_level.RMSLE", 0.2133, 5e-5,
     "paper / CODE_AUDIT F: 2018-2024 grid optimum RMSLE 0.2133",
     {"hindcast_window": "calibration", "set": "original"}),
    ("scenarios.frozen_price_1M.annual_energy_TWh.2031", 824.8, 0.05, "CODE_AUDIT F (824.8 TWh)"),
    ("scenarios.frontier_price_1M.peak_Rpe_forecast", 221, 0.5, "CODE_AUDIT F ($221/MWh)"),
    ("scenarios.frozen_price_fixed.share_Rpe_below_c_elec", 0.623, 5e-4,
     "CODE_AUDIT F (62.3% below the $50/MWh break-even, which is C_elec)"),
    ("halving.earliest_date", "2028-03-28", None, "CODE_AUDIT F / paper (28 Mar)"),
    ("halving.latest_date", "2028-04-03", None, "CODE_AUDIT F / paper (3 Apr)"),
    ("checks.subsidy_halves_at_1050000_all", True, None, "CODE_AUDIT F"),
    ("checks.frontier_gt_frozen_peak_every_price", True, None, "CODE_AUDIT F"),
    ("checks.frontier_gt_frozen_mean2031_every_price", True, None, "CODE_AUDIT F"),
    ("checks.peak_monotone_in_price_both_regimes", True, None, "CODE_AUDIT F"),
    ("checks.mean2031_monotone_in_price_both_regimes", True, None, "CODE_AUDIT F"),
    ("scenarios.frozen_price_fixed.first_forecast_block", 922757, 0, "CODE_AUDIT"),
    ("scenarios.frozen_price_fixed.first_forecast_timestamp", "2025-11-08 18:14:23", None,
     "CODE_AUDIT"),
    ("scenarios.frozen_price_fixed.peak_H_EHs_2026_2031", 1991, 0.5,
     "notebook cell 21 (paper: plateau near 2,000 EH/s)"),
]
# CODE_AUDIT B1: annual mean fleet efficiency, J/TH
_B1 = {"frozen_price_fixed": [17.75, 15.81, 10.96, 9.50, 9.50, 9.50],
       "frozen_price_1M": [18.58, 17.08, 14.70, 13.32, 11.19, 10.83],
       "frontier_price_fixed": [17.81, 15.63, 10.57, 7.51, 5.65, 3.63],
       "frontier_price_1M": [18.61, 16.87, 13.89, 11.46, 7.58, 3.90]}
for _s, _vals in _B1.items():
    for _y, _v in zip(ENERGY_YEARS, _vals):
        REFERENCES.append((f"scenarios.{_s}.efficiency_mean_by_year.{_y}", _v, 0.005,
                           "CODE_AUDIT B1"))
# notebook cell 21 peak hashrate (EH/s, printed to 0 decimals)
for _s, _v in zip(SCENARIOS, [1991, 3076, 5896, 9444, 3568, 6082, 10662, 17073]):
    REFERENCES.append((f"scenarios.{_s}.peak_H_EHs_2026_2031", _v, 0.5, "notebook cell 21"))
# notebook cell 24 halving timestamps
for _s, _v in zip(SCENARIOS, ["2028-04-03 02:21:45", "2028-04-01 14:31:57",
                              "2028-03-30 12:01:08", "2028-03-28 18:52:20",
                              "2028-04-02 23:28:17", "2028-04-01 11:31:48",
                              "2028-03-30 11:23:13", "2028-03-28 13:09:04"]):
    REFERENCES.append((f"scenarios.{_s}.halving_timestamp", _v, None, "notebook cell 24"))
# notebook cell 22 linearity check: (N, linear R2, quadratic R2, delta) pre and post
_LIN = {"frozen_price_fixed": [(30, .9998, .9999, .0001), (45, .9999, 1.0000, .0000)],
        "frozen_price_200k": [(30, .9965, .9999, .0034), (45, .9975, 1.0000, .0025)],
        "frozen_price_500k": [(29, .9818, .9998, .0180), (46, .9602, .9994, .0391)],
        "frozen_price_1M": [(29, .9717, .9997, .0280), (46, .9361, .9997, .0636)],
        "frontier_price_fixed": [(30, .9999, .9999, .0000), (45, .8958, .9951, .0993)],
        "frontier_price_200k": [(30, .9946, .9999, .0053), (45, .8925, .9950, .1025)],
        "frontier_price_500k": [(29, .9811, .9998, .0187), (46, .8661, .9859, .1198)],
        "frontier_price_1M": [(29, .9680, .9996, .0316), (46, .8322, .9827, .1504)]}
for _s, _pp in _LIN.items():
    for _per, (_n, _l, _q, _d) in zip(["pre", "post"], _pp):
        REFERENCES += [
            (f"scenarios.{_s}.linearity.{_per}.n_months", _n, 0, "notebook cell 22"),
            (f"scenarios.{_s}.linearity.{_per}.r2_linear", _l, 5e-5, "notebook cell 22"),
            (f"scenarios.{_s}.linearity.{_per}.r2_quadratic", _q, 5e-5, "notebook cell 22"),
            (f"scenarios.{_s}.linearity.{_per}.delta_r2", _d, 5e-5, "notebook cell 22"),
        ]


def say(msg):
    print(f"[analyze] {msg}", flush=True)


def warn(msg):
    print(f"[analyze] WARNING: {msg}", file=sys.stderr, flush=True)


def make_tmp(out):
    """Temporary sibling directory of out; the parent of out is created if missing."""
    parent = os.path.dirname(out)
    os.makedirs(parent, exist_ok=True)
    return tempfile.mkdtemp(prefix="." + os.path.basename(out) + ".partial-", dir=parent)


def finalize(tmp, out):
    """Give tmp the mode a plain mkdir would (0777 & ~umask; mkdtemp makes it 0700, which
    would also mask a default ACL inherited from the parent), then rename it to out."""
    old = os.umask(0)
    os.umask(old)
    os.chmod(tmp, 0o777 & ~old)
    if os.path.exists(out):
        sys.exit(f"{out} appeared while running; results left in {tmp}")
    os.rename(tmp, out)


def file_references():
    """References read from the calibration grid outputs: the minimum RMSLE of the
    2018-2024 grid that calibrated each set. They apply under the calibration window."""
    refs = []
    if os.path.exists(ORIG_GRID_CSV):
        g = pd.read_csv(ORIG_GRID_CSV)
        refs.append(("hindcast.metrics_2018_2024.block_level.RMSLE", float(g["RMSLE"].min()), 1e-9,
                     f"minimum RMSLE in {os.path.relpath(ORIG_GRID_CSV, ROOT)}",
                     {"hindcast_window": "calibration", "set": "original"}))
    else:
        warn(f"grid file {ORIG_GRID_CSV} not found; its RMSLE reference is not checked")
    if os.path.exists(NEW_GRID_OPTIMUM):
        with open(NEW_GRID_OPTIMUM) as f:
            opt = json.load(f)
        refs.append(("hindcast.metrics_2018_2024.block_level.RMSLE", float(opt["RMSLE"]), 1e-12,
                     f"RMSLE in {os.path.relpath(NEW_GRID_OPTIMUM, ROOT)}",
                     {"hindcast_window": "calibration", "set": "new"}))
    else:
        warn(f"grid file {NEW_GRID_OPTIMUM} not found; its RMSLE reference is not checked")
    return refs


# ------------------------------------------------------------------- run resolution
def load_runs():
    with open(RUNS_JSON) as f:
        runs = {r["name"]: r for r in json.load(f)["runs"] if r.get("default")}
    need = ["testing", "hindcasting"] + SCENARIOS
    missing = [n for n in need if n not in runs]
    if missing:
        sys.exit(f"runs.json lacks default runs {missing}")
    return runs, need


def april_c_elec():
    """C_ELEC of config.py at the April 2026 tag (the code of the original runs)."""
    try:
        text = subprocess.run(["git", "-C", ROOT, "show", f"{APRIL_TAG}:config.py"],
                              check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        sys.exit(f"cannot read config.py at {APRIL_TAG} ({exc}); pass --c-elec")
    m = re.findall(r"^C_ELEC\s*=\s*([0-9.]+)", text, flags=re.M)
    if len(m) != 1:
        sys.exit(f"expected one C_ELEC line in config.py at {APRIL_TAG}, found {len(m)}")
    return float(m[0])


def resolve(args):
    """Return ({run name: csv path}, C_elec, provenance dict)."""
    runs, need = load_runs()
    paths, prov = {}, {}
    if args.set == "original":
        for n in need:
            paths[n] = os.path.join(RESULTS, runs[n]["original"])
        c_elec = args.c_elec if args.c_elec is not None else april_c_elec()
        prov["c_elec_source"] = ("--c-elec" if args.c_elec is not None
                                 else f"config.py at git tag {APRIL_TAG}")
    else:
        c_values, done_bytes = {}, {}
        for n in need:
            run_dir = os.path.join(args.runs_dir, n)
            paths[n] = os.path.join(run_dir, "data", runs[n]["output_name"] + ".csv")
            done = os.path.join(run_dir, "DONE.json")
            if not os.path.exists(done):
                sys.exit(f"{done} missing: run {n} is not complete")
            with open(done) as f:
                d = json.load(f)
            c_values[n] = float(d["config_in_effect"]["C_ELEC"])
            done_bytes[n] = d.get("csv_bytes")
        if len(set(c_values.values())) != 1:
            sys.exit(f"runs disagree on C_ELEC: {c_values}")
        c_elec = args.c_elec if args.c_elec is not None else next(iter(c_values.values()))
        prov["c_elec_source"] = ("--c-elec" if args.c_elec is not None
                                 else "config_in_effect.C_ELEC in each run's DONE.json")
    overridden = set()
    for item in args.csv or []:
        name, _, path = item.partition("=")
        if name not in paths or not path:
            sys.exit(f"--csv expects NAME=PATH with NAME in {need}; got {item!r}")
        paths[name] = os.path.abspath(path)
        overridden.add(name)
    for n, p in paths.items():
        if not os.path.isfile(p):
            sys.exit(f"missing CSV for {n}: {p}")
    prov["inputs"] = {n: {"path": p, "bytes": os.path.getsize(p)} for n, p in paths.items()}
    # For the new set, each run's CSV must have the size its DONE.json records. A CSV given
    # with --csv is not that run's output, so the check is skipped for it and recorded.
    if args.set == "new":
        for n in need:
            if n in overridden:
                prov["inputs"][n]["done_size_check"] = "skipped (--csv)"
            elif done_bytes[n] is None:
                prov["inputs"][n]["done_size_check"] = "skipped (no csv_bytes in DONE.json)"
            elif prov["inputs"][n]["bytes"] != done_bytes[n]:
                sys.exit(f"{paths[n]} size differs from DONE.json csv_bytes")
            else:
                prov["inputs"][n]["done_size_check"] = "passed"
    prov["c_elec"] = c_elec
    return paths, c_elec, prov


# ------------------------------------------------------------------------- reading
def parse_ts(s, utc=False):
    """Parse a timestamp column; stop if any non-empty string fails to parse."""
    out = pd.to_datetime(s, format=TS_FORMAT, errors="coerce", utc=utc)
    bad = out.isna() & s.notna() & (s.astype(str).str.len() > 0)
    if bad.any():
        sys.exit(f"unparseable timestamps, e.g. {s[bad].iloc[0]!r}")
    return out


def iter_chunks(path, cols):
    for ch in pd.read_csv(path, usecols=cols, chunksize=CHUNK, low_memory=False):
        yield ch


def read_hashrate_run(path):
    """block_height, Timestamp and Hist_Timestamp (UTC), H_hist, H_sim, mean_bt-is-set."""
    parts = []
    for ch in iter_chunks(path, ["block_height", "Timestamp", "Hist_Timestamp", "H_hist", "H_sim",
                                 "mean_bt"]):
        parts.append(pd.DataFrame({
            "block_height": pd.to_numeric(ch["block_height"], errors="coerce"),
            "Timestamp": parse_ts(ch["Timestamp"], utc=True),
            "Hist_Timestamp": parse_ts(ch["Hist_Timestamp"], utc=True),
            "H_hist": pd.to_numeric(ch["H_hist"], errors="coerce"),
            "H_sim": pd.to_numeric(ch["H_sim"], errors="coerce"),
            "is_fc": pd.to_numeric(ch["mean_bt"], errors="coerce").notna(),
        }))
    return pd.concat(parts, ignore_index=True)


class YearAcc:
    """Sum and count of a column by calendar year, accumulated over chunks."""

    def __init__(self):
        self.s = {}
        self.c = {}

    def add(self, years, values):
        g = pd.DataFrame({"y": years, "v": values}).groupby("y")["v"].agg(["sum", "count"])
        for y, row in g.iterrows():
            self.s[int(y)] = self.s.get(int(y), 0.0) + float(row["sum"])
            self.c[int(y)] = self.c.get(int(y), 0) + int(row["count"])

    def sums(self):
        return dict(sorted(self.s.items()))

    def means(self):
        return {y: self.s[y] / self.c[y] for y in sorted(self.s) if self.c[y]}


def scan_scenario(path, c_elec):
    """One chunked pass over a scenario CSV. Returns (stats, kept rows)."""
    cols = ["block_height", "Timestamp", "H_sim", "E_sim", "R_pe", "efficiency",
            "mean_bt", "R_block"]
    energy, eff, hsim = YearAcc(), YearAcc(), YearAcc()
    n_fc = n_rpe = n_below_50 = n_below_c = 0
    rpe_max, rpe_max_ts, rpe_max_block = -np.inf, None, None
    first_fc = None
    halving_ts = None
    r_block = {}
    last_block, last_ts = None, None
    kept = []
    for ch in iter_chunks(path, cols):
        b = pd.to_numeric(ch["block_height"], errors="coerce")
        ts = parse_ts(ch["Timestamp"])
        e_sim = pd.to_numeric(ch["E_sim"], errors="coerce")
        rpe = pd.to_numeric(ch["R_pe"], errors="coerce")
        effc = pd.to_numeric(ch["efficiency"], errors="coerce")
        h = pd.to_numeric(ch["H_sim"], errors="coerce")
        fc = pd.to_numeric(ch["mean_bt"], errors="coerce").notna()
        m = ts.notna() & e_sim.notna()                      # cell 1: dropna on [ts, E_sim]
        energy.add(ts[m].dt.year.values, e_sim[m].values)
        m = ts.notna() & effc.notna()
        eff.add(ts[m].dt.year.values, effc[m].values)
        m = ts.notna() & h.notna()
        hsim.add(ts[m].dt.year.values, h[m].values)
        n_fc += int(fc.sum())
        if fc.any() and first_fc is None:
            i = fc.idxmax()
            first_fc = (int(b[i]), str(ts[i]))
        r = rpe[fc & rpe.notna()]
        n_rpe += len(r)
        n_below_50 += int((r < 50.0).sum())
        n_below_c += int((r < c_elec).sum())
        if len(r) and r.max() > rpe_max:
            i = r.idxmax()
            rpe_max, rpe_max_ts, rpe_max_block = float(r[i]), str(ts[i]), int(b[i])
        for blk in (HALVING_BLOCK - 1, HALVING_BLOCK):
            hit = b == blk
            if hit.any():
                r_block[blk] = float(pd.to_numeric(ch.loc[hit, "R_block"]).iloc[0])
                if blk == HALVING_BLOCK:
                    halving_ts = ts[hit].iloc[0]
        last_block, last_ts = int(b.iloc[-1]), str(ts.iloc[-1])
        k = ts >= KEEP_FROM
        if k.any():
            kept.append(pd.DataFrame({"block_height": b[k].astype("int64"), "Timestamp": ts[k],
                                      "H_sim": h[k], "R_pe": rpe[k], "efficiency": effc[k],
                                      "is_fc": fc[k]}))
    kept = pd.concat(kept, ignore_index=True).sort_values("Timestamp", kind="mergesort")

    years_e = energy.sums()
    st = {
        "first_forecast_block": first_fc[0] if first_fc else None,
        "first_forecast_timestamp": first_fc[1] if first_fc else None,
        "last_block": last_block,
        "last_timestamp": last_ts,
        "n_forecast_rows": n_fc,
        "n_forecast_rows_with_Rpe": n_rpe,
        "annual_energy_TWh": {str(y): v / 1e12 for y, v in years_e.items() if y >= 2025},
        "efficiency_mean_by_year": {str(y): v for y, v in eff.means().items() if y >= 2025},
        "H_EHs_mean_by_year": {str(y): v / 1e18 for y, v in hsim.means().items() if y >= 2025},
        "peak_Rpe_forecast": rpe_max,
        "peak_Rpe_timestamp": rpe_max_ts,
        "peak_Rpe_block": rpe_max_block,
        "share_Rpe_below_50": n_below_50 / n_rpe if n_rpe else None,
        "share_Rpe_below_c_elec": n_below_c / n_rpe if n_rpe else None,
        "c_elec": c_elec,
        "halving_timestamp": str(halving_ts) if halving_ts is not None else None,
        "R_block_before_halving": r_block.get(HALVING_BLOCK - 1),
        "R_block_at_halving": r_block.get(HALVING_BLOCK),
    }
    fcrows = kept[kept["is_fc"]]
    st["peak_H_EHs_forecast"] = float(fcrows["H_sim"].max() / 1e18)
    win = kept[(kept["Timestamp"] >= PEAK_START) & (kept["Timestamp"] <= PEAK_END)]
    st["peak_H_EHs_2026_2031"] = float(win["H_sim"].max() / 1e18)              # cell 21
    st["H_EHs_last_block"] = float(
        kept.loc[kept["block_height"] == last_block, "H_sim"].iloc[0] / 1e18)
    if halving_ts is not None:
        at = kept[kept["block_height"] == HALVING_BLOCK]
        st["H_EHs_at_halving"] = float(at["H_sim"].iloc[0] / 1e18)
    st["linearity"] = linearity(kept, halving_ts)
    return st, kept


def linearity(kept, halving_ts):
    """Cell 22: linear and quadratic R2 of monthly mean hashrate on time, split at the
    halving timestamp. The halving month is partial in both periods."""
    if halving_ts is None:
        return None
    d = kept[(kept["Timestamp"] >= LIN_START) & (kept["Timestamp"] <= LIN_END)]
    d = d[["Timestamp", "H_sim"]].dropna().copy()
    d["H_EHs"] = d["H_sim"] / 1e18
    out = {}
    for per, start, end in [("pre", LIN_START, halving_ts), ("post", halving_ts, LIN_END)]:
        p = d[(d["Timestamp"] >= start) & (d["Timestamp"] < end)].copy()
        p["t_days"] = (p["Timestamp"] - p["Timestamp"].min()).dt.total_seconds() / 86400
        mo = p.set_index("Timestamp").resample("MS")[["t_days", "H_EHs"]].mean().dropna()
        t, hh = mo["t_days"].values, mo["H_EHs"].values
        if len(t) < 4:
            out[per] = None
            continue
        r2l = stats.linregress(t, hh).rvalue ** 2
        c2 = np.polyfit(t, hh, deg=2)
        ss_tot = np.sum((hh - hh.mean()) ** 2)
        r2q = 1 - np.sum((hh - np.polyval(c2, t)) ** 2) / ss_tot if ss_tot > 0 else np.nan
        out[per] = {"n_months": int(len(t)), "r2_linear": float(r2l),
                    "r2_quadratic": float(r2q), "delta_r2": float(r2q - r2l)}
    return out


# ------------------------------------------------------------------ hashrate stats
def log_metrics(sim, hist):
    """Cells 12-16: clip at 0, log1p, residual = log1p(sim) - log1p(hist)."""
    s, h = sim.clip(lower=0), hist.clip(lower=0)
    rl = np.log1p(s) - np.log1p(h)
    rmsle = float(np.sqrt((rl ** 2).mean()))
    mean_l = float(rl.mean())
    rel = (s - h) / h.replace(0, np.nan)
    return {"n": int(len(rl)), "RMSLE": rmsle, "mean_log_residual": mean_l,
            "median_log_residual": float(rl.median()),
            "std_log_residual": float(rl.std(ddof=1)),
            "expm1_RMSLE": float(np.expm1(rmsle)),
            "expm1_mean_log_residual": float(np.expm1(mean_l)),
            "mean_residual_rel": float(rel.mean(skipna=True)),
            "median_residual_rel": float(rel.median(skipna=True))}


def monthly_ratio_df(df, tcol):
    """Cells 18 and 20: monthly mean H_hist and H_sim, ratio, centred rolling mean, binned
    by the clock column tcol."""
    tmp = df[[tcol, "H_hist", "H_sim"]].dropna().set_index(tcol).sort_index()
    m = tmp.resample(MONTH_RULE).mean(numeric_only=True).dropna()
    m["ratio"] = m["H_sim"] / m["H_hist"]
    m["ratio_roll"] = m["ratio"].rolling(window=RATIO_ROLL_MONTHS,
                                         min_periods=max(1, RATIO_ROLL_MONTHS // 2),
                                         center=True).mean()
    return m.reset_index()


def oos_stats(df, tcol):
    first_fc = int(df.loc[df["is_fc"], "block_height"].min())
    d = df[df["block_height"] >= OOS_START_HEIGHT].dropna(subset=["H_hist", "H_sim"])
    st = log_metrics(d["H_sim"], d["H_hist"])
    st["oos_start_height"] = OOS_START_HEIGHT
    st["first_forecast_block"] = first_fc
    st["last_block"] = int(df["block_height"].max())
    st["last_timestamp"] = str(df.loc[df["block_height"].idxmax(), "Timestamp"])
    w = df.sort_values(tcol, kind="mergesort")
    w = w[(w[tcol] >= TEST_T_START) & (w[tcol] <= TEST_T_END)]
    mr = monthly_ratio_df(w, tcol)
    st["monthly_ratio_min"] = float(mr["ratio"].min())
    st["monthly_ratio_max"] = float(mr["ratio"].max())
    st["monthly_ratio_roll_min"] = float(mr["ratio_roll"].min())
    st["monthly_ratio_roll_max"] = float(mr["ratio_roll"].max())
    st["monthly_ratio"] = {str(t.date()): float(r) for t, r in zip(mr[tcol], mr["ratio"])}
    return st, w, mr


def hindcast_stats(df, tcol, window):
    st = {"first_forecast_block": int(df.loc[df["is_fc"], "block_height"].min()),
          "last_block": int(df["block_height"].max())}
    w = df.sort_values(tcol, kind="mergesort")
    frames, wins = [], []
    for j, (title, t0, t1) in enumerate(HIND_WINDOWS):
        m = w[tcol].notna() & (w[tcol] >= t0) & (w[tcol] <= t1)
        sub = w.loc[m, [tcol, "H_hist", "H_sim"]]
        mr = monthly_ratio_df(sub, tcol)
        mr.insert(0, "window", title)
        frames.append(mr)
        wins.append(sub)
        st[f"era{j + 1}"] = {
            "window": title,
            "n_months": int(len(mr)),
            "monthly_ratio_min": float(mr["ratio"].min()),
            "monthly_ratio_min_month": str(mr.loc[mr["ratio"].idxmin(), tcol].date()),
            "monthly_ratio_max": float(mr["ratio"].max()),
            "monthly_ratio_max_month": str(mr.loc[mr["ratio"].idxmax(), tcol].date()),
            "monthly_ratio_roll_min": float(mr["ratio_roll"].min()),
            "monthly_ratio_roll_max": float(mr["ratio_roll"].max()),
            "monthly_ratio_last": float(mr["ratio"].iloc[-1]),
        }
    # cells 14-16: block-level, weekly and monthly metrics over a height window; the weekly
    # and monthly means are binned by the clock column
    h0, h1, scale = HIND_METRIC_WINDOWS[window]
    d = df[(df["block_height"] >= h0) & (df["block_height"] < h1)]
    d = d.dropna(subset=[tcol, "H_hist", "H_sim"]).sort_values("block_height")
    d = d.assign(H_hist=d["H_hist"] * scale, H_sim=d["H_sim"] * scale)
    idx = d.set_index(tcol)[["H_hist", "H_sim"]]
    dw = idx.resample(WEEK_RULE).mean(numeric_only=True).dropna()
    dm = idx.resample(MONTH_RULE).mean(numeric_only=True).dropna()
    st["metrics_2018_2024"] = {
        "window": window,
        "height_window": [h0, h1],
        "unit_scale": scale,
        "block_level": log_metrics(d["H_sim"], d["H_hist"]),
        "weekly_mean": log_metrics(dw["H_sim"], dw["H_hist"]),
        "monthly_mean": log_metrics(dm["H_sim"], dm["H_hist"]),
    }
    return st, wins, frames


# ----------------------------------------------------------------- cross-run checks
def monthly_fc_mean(df):
    """Mean H_sim by calendar month over forecast rows, for months that are all
    forecast (the first forecast month, which starts with historical rows, and the
    last month of the run, which is partial, are dropped)."""
    f = df[df["is_fc"]].set_index("Timestamp")["H_sim"]
    m = f.resample(MONTH_RULE).mean()
    return m.iloc[1:-1]


def cross_checks(sc, kept):
    out = {}
    out["subsidy_halves_at_1050000_all"] = all(
        sc[s]["R_block_before_halving"] is not None and sc[s]["R_block_at_halving"] is not None
        and math.isclose(sc[s]["R_block_at_halving"], sc[s]["R_block_before_halving"] / 2)
        for s in SCENARIOS)
    out["subsidy_before_after"] = {s: [sc[s]["R_block_before_halving"], sc[s]["R_block_at_halving"]]
                                   for s in SCENARIOS}
    # frontier versus frozen at each price
    fg = {}
    for p in PRICES:
        a, b = sc[f"frozen_price_{p}"], sc[f"frontier_price_{p}"]
        j = pd.concat({"frozen": monthly_fc_mean(kept[f"frozen_price_{p}"]),
                       "frontier": monthly_fc_mean(kept[f"frontier_price_{p}"])},
                      axis=1, join="inner")
        fg[p] = {
            "peak_frontier_gt_frozen": b["peak_H_EHs_forecast"] > a["peak_H_EHs_forecast"],
            "mean_by_year_frontier_gt_frozen": {
                y: b["H_EHs_mean_by_year"][str(y)] > a["H_EHs_mean_by_year"][str(y)]
                for y in ENERGY_YEARS},
            "months_compared": int(len(j)),
            "months_frontier_gt_frozen": int((j["frontier"] > j["frozen"]).sum()),
            "months_frontier_eq_frozen": int((j["frontier"] == j["frozen"]).sum()),
            "months_frontier_lt_frozen": int((j["frontier"] < j["frozen"]).sum()),
            "first_month_frontier_gt_frozen_from_then_on": first_true_run(
                j.index, j["frontier"].values > j["frozen"].values),
        }
    out["frontier_vs_frozen"] = fg
    out["frontier_gt_frozen_peak_every_price"] = all(fg[p]["peak_frontier_gt_frozen"] for p in PRICES)
    out["frontier_gt_frozen_mean2031_every_price"] = all(
        fg[p]["mean_by_year_frontier_gt_frozen"][2031] for p in PRICES)
    # monotone in price within each efficiency regime
    mono = {}
    for e in EFFS:
        names = [f"{e}_price_{p}" for p in PRICES]
        peaks = [sc[n]["peak_H_EHs_forecast"] for n in names]
        by_year = {y: [sc[n]["H_EHs_mean_by_year"][str(y)] for n in names] for y in ENERGY_YEARS}
        j = pd.concat({p: monthly_fc_mean(kept[n]) for n, p in zip(names, PRICES)},
                      axis=1, join="inner")
        inc = np.all(np.diff(j[PRICES].values, axis=1) > 0, axis=1)
        mono[e] = {
            "peak_increasing": bool(np.all(np.diff(peaks) > 0)),
            "mean_by_year_increasing": {y: bool(np.all(np.diff(v) > 0)) for y, v in by_year.items()},
            "months_compared": int(len(j)),
            "months_strictly_increasing": int(inc.sum()),
            "first_month_increasing_from_then_on": first_true_run(j.index, inc),
        }
    out["monotone_in_price"] = mono
    out["peak_monotone_in_price_both_regimes"] = all(mono[e]["peak_increasing"] for e in EFFS)
    out["mean2031_monotone_in_price_both_regimes"] = all(
        mono[e]["mean_by_year_increasing"][2031] for e in EFFS)
    return out


def first_true_run(index, flags):
    """First index label from which every later flag is True; None if the last is False."""
    flags = np.asarray(flags, dtype=bool)
    if len(flags) == 0 or not flags[-1]:
        return None
    false = np.where(~flags)[0]
    i = 0 if len(false) == 0 else false.max() + 1
    return str(pd.Timestamp(index[i]).date())


# ------------------------------------------------------------------------- figures
def load_fig3_helpers():
    path = os.path.join(ROOT, "calibration", "plot_grid_figure3.py")
    spec = importlib.util.spec_from_file_location("_plot_grid_figure3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extent_from_docx(path, number):
    """(cx, cy, media part) of the first inline picture (wp:inline) in the paragraph
    whose text starts 'Figure <number>.'; the same rule as
    calibration/plot_grid_figure3.py. The caption paragraph sits inside a text box,
    which is itself a wp:anchor whose extent is the box's, so anchors are not read."""
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
          "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
          "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
          "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    z = zipfile.ZipFile(path)
    doc = ET.fromstring(z.read("word/document.xml"))
    rels = ET.fromstring(z.read("word/_rels/document.xml.rels"))
    target = {r.get("Id"): r.get("Target") for r in rels}
    w = "{%s}" % ns["w"]
    cap = re.compile(r"\s*Figure\s*%d\." % number)
    for para in doc.iter(w + "p"):
        text = "".join(t.text or "" for t in para.iter(w + "t"))
        if not cap.match(text):
            continue
        for el in para.iter("{%s}inline" % ns["wp"]):
            blip = el.find(".//a:blip", ns)
            ext = el.find("wp:extent", ns)
            if blip is None or ext is None:
                continue
            rid = blip.get("{%s}embed" % ns["r"])
            return int(ext.get("cx")), int(ext.get("cy")), "word/" + target[rid]
    sys.exit(f"no picture beside a caption starting 'Figure {number}.' in {path}")


def set_pub_style(font):
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": DPI, "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 13, "legend.fontsize": 10, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "axes.linewidth": 1.0, "axes.spines.top": False,
        "axes.spines.right": False, "grid.alpha": 0.25, "grid.linewidth": 0.8,
        "lines.linewidth": 2.0, "font.family": "serif", "font.serif": [font],
        "mathtext.fontset": "stix",
    })


def layout_with_fig_legends(fig, top=None, bottom=None, pad=0.01):
    """tight_layout leaving room for figure-level legends at the top and bottom edge.
    The notebook put these legends outside the canvas and saved with a tight bbox; here
    the canvas size is fixed, so the legends sit inside it."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    hpx = fig.get_figheight() * fig.dpi
    t = top.get_window_extent(r).height / hpx + pad if top is not None else 0.0
    b = bottom.get_window_extent(r).height / hpx + pad if bottom is not None else 0.0
    fig.tight_layout(rect=[0, b, 1, 1 - t])


def year_axis(ax, base):
    ax.xaxis.set_major_locator(mdates.YearLocator(base=base))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, which="major", axis="both")
    ax.margins(x=0.01)


def figure4(size, hind_windows, hind_ratio, tcol):
    """Cell 18, with both series on the clock column tcol."""
    fig, axes = plt.subplots(2, 2, figsize=size, sharex="col",
                             gridspec_kw={"height_ratios": [2.0, 1.0]})
    for j, (sub, mr) in enumerate(zip(hind_windows, hind_ratio)):
        top, bot = axes[0, j], axes[1, j]
        mh = sub["H_hist"].notna()
        top.plot(sub.loc[mh, tcol], sub.loc[mh, "H_hist"] / 1e18, color=COLORS["hist"],
                 label="Historical", zorder=3)
        ms = sub["H_sim"].notna()
        top.plot(sub.loc[ms, tcol], sub.loc[ms, "H_sim"] / 1e18, color=COLORS["fixed"],
                 label="Simulated", zorder=2)
        top.set_ylabel("Hashrate (EH/s)")
        year_axis(top, 1)
        bot.plot(mr[tcol], mr["ratio"], color=COL_RATIO,
                 label="Monthly ratio (Sim / Hist)", zorder=2)
        bot.plot(mr[tcol], mr["ratio_roll"], color=COL_RATIO_ROLL,
                 label=f"{RATIO_ROLL_MONTHS}-month rolling mean", zorder=3)
        bot.axhline(1.0, linestyle="--", linewidth=1.5, color=COLORS["hist"], alpha=0.9)
        bot.set_ylabel("Sim / Hist")
        bot.set_xlabel("Time")
        year_axis(bot, 1)
    allr = pd.concat(hind_ratio, ignore_index=True)
    vals = pd.concat([allr["ratio"], allr["ratio_roll"]], ignore_index=True).dropna()
    if not vals.empty:
        lo, hi = float(vals.min()), float(vals.max())
        pad = 0.03 * (hi - lo) if hi > lo else 0.05
        axes[1, 0].set_ylim(lo - pad, hi + pad)
        axes[1, 1].set_ylim(lo - pad, hi + pad)
    h, lab = axes[0, 0].get_legend_handles_labels()
    lt = fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    h, lab = axes[1, 0].get_legend_handles_labels()
    lb = fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    layout_with_fig_legends(fig, lt, lb)
    return fig


def figure5(size, win, mr, tcol):
    """Cell 20, with both series on the clock column tcol."""
    fig, (top, bot) = plt.subplots(2, 1, figsize=size, sharex=True,
                                   gridspec_kw={"height_ratios": [2.0, 1.0]})
    mh = win["H_hist"].notna()
    top.plot(win.loc[mh, tcol], win.loc[mh, "H_hist"] / 1e18, color=COLORS["hist"],
             label="Historical", zorder=3)
    ms = win["H_sim"].notna()
    top.plot(win.loc[ms, tcol], win.loc[ms, "H_sim"] / 1e18, color=COLORS["fixed"],
             label="Simulated", zorder=2)
    top.set_ylabel("Hashrate (EH/s)")
    top.grid(True, which="major", axis="both")
    top.legend(loc="upper left", frameon=False)
    bot.plot(mr[tcol], mr["ratio"], color=COL_RATIO, label="Monthly ratio (Sim / Hist)",
             zorder=2)
    bot.plot(mr[tcol], mr["ratio_roll"], color=COL_RATIO_ROLL,
             label=f"{RATIO_ROLL_MONTHS}-month rolling mean", zorder=3)
    bot.axhline(1.0, linestyle="--", linewidth=1.5, color=COLORS["hist"], alpha=0.9)
    bot.set_ylabel("Sim / Hist")
    bot.set_xlabel("Time")
    bot.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    bot.grid(True, which="major", axis="both")
    bot.margins(x=0.01)
    bot.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


def two_panel(size):
    return plt.subplots(1, 2, figsize=size, sharey=True)


def top_legend(fig, axes, ncol):
    h, lab = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, lab, loc="upper center", ncol=ncol, frameon=False,
                     bbox_to_anchor=(0.5, 1.0))
    layout_with_fig_legends(fig, top=leg)


def figure6(size, kept):
    """Cell 3."""
    fig, axes = two_panel(size)
    for ax, (title, e) in zip(axes, [("Frozen efficiency", "frozen"),
                                     ("Frontier efficiency", "frontier")]):
        for p in PRICES:
            df = kept[f"{e}_price_{p}"]
            m = df["Timestamp"].notna() & df["efficiency"].notna() & (df["Timestamp"] >= EFF_START)
            ax.plot(df.loc[m, "Timestamp"], df.loc[m, "efficiency"], color=COLORS[p],
                    label=LABELS[p])
        ax.set_title(title)
        ax.set_xlabel("Time")
        year_axis(ax, 2)
    axes[0].set_ylabel("Efficiency (J/TH)")
    top_legend(fig, axes, 4)
    return fig


def figure7(size, kept):
    """Cell 23, without its historical trace. The notebook selected historical rows with
    PLOT_START <= Hist_Timestamp <= TIMESTAMP_END_HIST, which no row satisfies because
    PLOT_START is later; the published figure therefore has a "Historical" legend entry
    with no line. Here neither the empty line nor its legend entry is drawn, so the
    legend has four entries (ncol 4)."""
    fig, axes = two_panel(size)
    for ax, (title, e) in zip(axes, [("Frozen efficiency", "frozen"),
                                     ("Frontier efficiency", "frontier")]):
        for p in PRICES:
            df = kept[f"{e}_price_{p}"]
            m = df["Timestamp"].notna() & (df["Timestamp"] >= TIMESTAMP_END_HIST)
            ax.plot(df.loc[m, "Timestamp"], df.loc[m, "H_sim"] / 1e18, color=COLORS[p],
                    label=f"Simulated, price: {LABELS[p].split(': ', 1)[1]}", zorder=2)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_xlim(left=PLOT_START)
        year_axis(ax, 2)
    axes[0].set_ylabel("Hashrate (EH/s)")
    top_legend(fig, axes, 4)
    return fig


def figure8(size, sc):
    """Cell 1, second figure."""
    fig, axes = two_panel(size)
    years = np.array(ENERGY_YEARS, dtype=int)
    x = np.arange(len(years))
    bar_w = 0.20
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_w
    for ax, (title, e) in zip(axes, [("Frozen efficiency", "frozen"),
                                     ("Frontier efficiency", "frontier")]):
        for off, p in zip(offsets, PRICES):
            ann = sc[f"{e}_price_{p}"]["annual_energy_TWh"]
            vals = [ann.get(str(y), 0.0) for y in years]
            ax.bar(x + off, vals, width=bar_w, color=COLORS[p], label=LABELS[p])
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in years])
        ax.grid(True, axis="y")
    axes[0].set_ylabel("Annual energy consumption (TWh)")
    top_legend(fig, axes, 4)
    return fig


def figure9(size, kept, c_elec):
    """Cell 2 (with cell 1's data and yearly time axis)."""
    fig, axes = two_panel(size)
    for ax, (title, e) in zip(axes, [("Frozen efficiency", "frozen"),
                                     ("Frontier efficiency", "frontier")]):
        for p in PRICES:
            df = kept[f"{e}_price_{p}"]
            m = df["Timestamp"].notna() & (df["Timestamp"] >= RPE_START) & df["R_pe"].notna()
            ax.plot(df.loc[m, "Timestamp"], df.loc[m, "R_pe"], color=COLORS[p], label=LABELS[p])
        ax.axhline(c_elec, color="black", linestyle="--", linewidth=1.5, alpha=0.9,
                   label=f"Break-even (${c_elec:g}/MWh)" if ax is axes[0] else None)
        ax.set_title(title)
        ax.set_xlabel("Time")
        year_axis(ax, 1)
    axes[0].set_ylabel(r"Energy-adjusted expected revenue, $R_{pe}$ (\$/MWh)")
    top_legend(fig, axes, 5)
    return fig


# ------------------------------------------------------------------ verification
def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def verify(flat, ctx):
    """ctx: the run's settings (set, clock, hindcast_window). A reference whose condition
    does not hold for ctx is listed with applies=False and matches_reference empty."""
    rows = []
    for entry in REFERENCES + file_references():
        key, ref, tol, src = entry[:4]
        cond = entry[4] if len(entry) > 4 else {}
        got = flat.get(key)
        applies = all(ctx.get(k) == v for k, v in cond.items())
        cond_txt = ", ".join(f"{k}={v}" for k, v in cond.items())
        if not applies:
            rows.append({"statistic": key, "reference": ref, "tolerance": tol, "computed": got,
                         "applies": False, "matches_reference": None, "source": src,
                         "condition": cond_txt})
            continue
        if got is None:
            ok = False
        elif tol is None:
            ok = (got == ref) if not isinstance(ref, str) else str(got).startswith(ref)
        else:
            ok = abs(float(got) - float(ref)) <= tol
        rows.append({"statistic": key, "reference": ref, "tolerance": tol, "computed": got,
                     "applies": True, "matches_reference": bool(ok), "source": src,
                     "condition": cond_txt})
    return pd.DataFrame(rows)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    return o


# ------------------------------------------------------------------------- outputs
def write_tables(out, st):
    sc = st["scenarios"]
    yrs = sorted({y for s in SCENARIOS for y in sc[s]["annual_energy_TWh"]})
    pd.DataFrame({s: [sc[s]["annual_energy_TWh"].get(y) for y in yrs] for s in SCENARIOS},
                 index=pd.Index(yrs, name="year")).to_csv(os.path.join(out, "annual_energy_TWh.csv"))
    yrs = sorted({y for s in SCENARIOS for y in sc[s]["efficiency_mean_by_year"]})
    pd.DataFrame({s: [sc[s]["efficiency_mean_by_year"].get(y) for y in yrs] for s in SCENARIOS},
                 index=pd.Index(yrs, name="year")).to_csv(
        os.path.join(out, "fleet_efficiency_mean_by_year_W_per_TH.csv"))
    yrs = sorted({y for s in SCENARIOS for y in sc[s]["H_EHs_mean_by_year"]})
    pd.DataFrame({s: [sc[s]["H_EHs_mean_by_year"].get(y) for y in yrs] for s in SCENARIOS},
                 index=pd.Index(yrs, name="year")).to_csv(
        os.path.join(out, "hashrate_mean_by_year_EHs.csv"))
    keys = ["first_forecast_block", "first_forecast_timestamp", "last_block", "last_timestamp",
            "n_forecast_rows", "peak_Rpe_forecast", "peak_Rpe_timestamp", "share_Rpe_below_50",
            "share_Rpe_below_c_elec", "c_elec", "halving_timestamp", "R_block_before_halving",
            "R_block_at_halving", "peak_H_EHs_forecast", "peak_H_EHs_2026_2031",
            "H_EHs_at_halving", "H_EHs_last_block"]
    pd.DataFrame([{"scenario": s, **{k: sc[s].get(k) for k in keys}} for s in SCENARIOS]) \
        .to_csv(os.path.join(out, "scenario_summary.csv"), index=False)
    rows = []
    for s in SCENARIOS:
        for per in ("pre", "post"):
            v = (sc[s]["linearity"] or {}).get(per) or {}
            rows.append({"scenario": s, "period": per, **v})
    pd.DataFrame(rows).to_csv(os.path.join(out, "linearity_cell22.csv"), index=False)


def summary_text(st, ver):
    o, h, sc, ck = st["oos"], st["hindcast"], st["scenarios"], st["checks"]
    L = [f"Set: {st['set']}    C_elec (break-even): ${st['provenance']['c_elec']:g}/MWh "
         f"({st['provenance']['c_elec_source']})",
         f"Clock for comparisons with history: {st['clock']} ({CLOCK_COLUMN[st['clock']]})    "
         f"2018-2024 hindcast metric window: {st['hindcast_window']}", ""]
    L.append("Out-of-sample test (testing run, block_height >= %d)" % o["oos_start_height"])
    L.append(f"  first forecast block {o['first_forecast_block']}, last block {o['last_block']} "
             f"at {o['last_timestamp']}")
    L.append(f"  N = {o['n']}, RMSLE = {o['RMSLE']:.5f} (exp-1 = {o['expm1_RMSLE']:.2%})")
    L.append(f"  mean log residual = {o['mean_log_residual']:.5f} (exp-1 = "
             f"{o['expm1_mean_log_residual']:.2%}), median = {o['median_log_residual']:.5f}")
    L.append(f"  monthly Sim/Hist ratio 2025-01..2025-10 ({CLOCK_COLUMN[st['clock']]}): "
             f"{o['monthly_ratio_min']:.3f} to {o['monthly_ratio_max']:.3f}")
    L.append("")
    L.append(f"Hindcast (first economic block {h['first_forecast_block']})")
    for e in ("era1", "era2"):
        x = h[e]
        L.append(f"  {x['window']}: monthly Sim/Hist {x['monthly_ratio_min']:.3f} "
                 f"({x['monthly_ratio_min_month']}) to {x['monthly_ratio_max']:.3f} "
                 f"({x['monthly_ratio_max_month']}); 3-month mean {x['monthly_ratio_roll_min']:.3f}"
                 f" to {x['monthly_ratio_roll_max']:.3f}")
    m = h["metrics_2018_2024"]
    L.append(f"  2018-2024 metrics over blocks [{m['height_window'][0]}, {m['height_window'][1]}) "
             f"({m['window']} window, hashrate x {m['unit_scale']:g})")
    for lvl in ("block_level", "weekly_mean", "monthly_mean"):
        L.append(f"  2018-2024 {lvl}: N={m[lvl]['n']} RMSLE={m[lvl]['RMSLE']:.6f} "
                 f"mean log resid={m[lvl]['mean_log_residual']:.4f}")
    L.append("")
    L.append("Annual energy (TWh); 2025 and 2032 are partial years")
    L.append("  " + "scenario".ljust(22) + "".join(f"{y:>9}" for y in ENERGY_YEARS))
    for s in SCENARIOS:
        L.append("  " + s.ljust(22) + "".join(f"{sc[s]['annual_energy_TWh'][str(y)]:9.1f}"
                                              for y in ENERGY_YEARS))
    L.append("")
    L.append("Fleet efficiency, annual mean (W/TH = J/TH)")
    for s in SCENARIOS:
        L.append("  " + s.ljust(22) + "".join(f"{sc[s]['efficiency_mean_by_year'][str(y)]:9.2f}"
                                              for y in ENERGY_YEARS))
    L.append("")
    L.append(f"R_pe over forecast rows: share below break-even (C_elec = "
             f"${st['provenance']['c_elec']:g}/MWh), and share below $50/MWh")
    for s in SCENARIOS:
        x = sc[s]
        L.append(f"  {s.ljust(22)} peak {x['peak_Rpe_forecast']:7.2f} at {x['peak_Rpe_timestamp']}"
                 f"  below break-even: {x['share_Rpe_below_c_elec']:.1%}  below $50/MWh: "
                 f"{x['share_Rpe_below_50']:.1%}")
    L.append("")
    L.append(f"2028 halving (block {HALVING_BLOCK}): {st['halving']['earliest_date']} to "
             f"{st['halving']['latest_date']}")
    for s in SCENARIOS:
        L.append(f"  {s.ljust(22)} {sc[s]['halving_timestamp']}")
    L.append("")
    L.append("Hashrate (EH/s): peak 2026-2031 (cell 21), at the halving, at the last block")
    for s in SCENARIOS:
        x = sc[s]
        L.append(f"  {s.ljust(22)} {x['peak_H_EHs_2026_2031']:9.0f} {x['H_EHs_at_halving']:9.0f} "
                 f"{x['H_EHs_last_block']:9.0f}")
    L.append("")
    L.append("Linearity check (cell 22): pre and post halving, N / linear R2 / delta R2")
    for s in SCENARIOS:
        lin = sc[s]["linearity"]
        L.append(f"  {s.ljust(22)} " + "   ".join(
            f"{per}: {lin[per]['n_months']} / {lin[per]['r2_linear']:.4f} / {lin[per]['delta_r2']:.4f}"
            for per in ("pre", "post")))
    L.append("")
    L.append("Checks")
    for k in ("subsidy_halves_at_1050000_all", "frontier_gt_frozen_peak_every_price",
              "frontier_gt_frozen_mean2031_every_price", "peak_monotone_in_price_both_regimes",
              "mean2031_monotone_in_price_both_regimes"):
        L.append(f"  {k}: {ck[k]}")
    for p in PRICES:
        x = ck["frontier_vs_frozen"][p]
        L.append(f"  price {p}: monthly mean hashrate, frontier > frozen in "
                 f"{x['months_frontier_gt_frozen']} of {x['months_compared']} months, equal in "
                 f"{x['months_frontier_eq_frozen']}, lower in {x['months_frontier_lt_frozen']}; "
                 f"higher in every month from {x['first_month_frontier_gt_frozen_from_then_on']}")
    for e in EFFS:
        x = ck["monotone_in_price"][e]
        L.append(f"  {e}: monthly mean hashrate strictly increasing in price in "
                 f"{x['months_strictly_increasing']} of {x['months_compared']} months; in every "
                 f"month from {x['first_month_increasing_from_then_on']}")
    L.append("")
    app = ver[ver["applies"]]
    n_ok = int(app["matches_reference"].astype(bool).sum())
    L.append(f"Reference values matched: {n_ok} of {len(app)} that apply to this clock and "
             f"window ({len(ver) - len(app)} others not applicable; see verification.csv)")
    for _, r in app[~app["matches_reference"].astype(bool)].iterrows():
        L.append(f"  NOT MATCHED {r['statistic']}: reference {r['reference']} ({r['source']}), "
                 f"computed {r['computed']}")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------- compare
DEFINITION_KEYS = ("clock", "hindcast_window")


def definition_mismatch(orig_dir, new_dir):
    """[(key, value in orig, value in new)] for the definition settings that differ. A
    stats.json written before these settings existed has no such key (value None)."""
    vals = []
    for d in (orig_dir, new_dir):
        with open(os.path.join(d, "stats.json")) as f:
            st = json.load(f)
        vals.append({k: st.get(k) for k in DEFINITION_KEYS})
    return [(k, vals[0][k], vals[1][k]) for k in DEFINITION_KEYS if vals[0][k] != vals[1][k]]


def compare(orig_dir, new_dir, out, mismatch=()):
    with open(os.path.join(orig_dir, "stats.json")) as f:
        a = flatten(json.load(f))
    with open(os.path.join(new_dir, "stats.json")) as f:
        b = flatten(json.load(f))
    rows = []
    for k in sorted(set(a) | set(b)):
        if k.startswith("provenance.inputs"):
            continue
        va, vb = a.get(k), b.get(k)
        change = rel = None
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) \
                and not isinstance(va, bool) and not isinstance(vb, bool):
            change = vb - va
            rel = change / va if va else None
        elif isinstance(va, str) and isinstance(vb, str) and re.match(r"\d{4}-\d\d-\d\d", va) \
                and re.match(r"\d{4}-\d\d-\d\d", vb):
            try:
                change = str(pd.Timestamp(vb) - pd.Timestamp(va))
            except ValueError:
                change = None
        rows.append({"quantity": k, "original": va, "new": vb, "change": change,
                     "relative_change": rel})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "comparison.csv"), index=False)
    with open(os.path.join(out, "comparison.txt"), "w") as f:
        if mismatch:
            f.write("WARNING: the two outputs use different definitions "
                    "(--allow-mixed-definitions):\n")
            for k, va, vb in mismatch:
                f.write(f"  {k}: {va} in {orig_dir}, {vb} in {new_dir}\n")
            f.write("Changes in quantities that depend on these settings mix the effect of the "
                    "runs with the effect of the definitions.\n\n")
        f.write(df.to_string(index=False, max_colwidth=60))
    say(f"comparison.csv: {len(df)} quantities")


# ---------------------------------------------------------------------------- main
def parse_size(items):
    sizes = {}
    for fig, w, h in items or []:
        if int(fig) not in range(4, 10):
            sys.exit("--size-in FIG must be 4..9")
        sizes[int(fig)] = (float(w), float(h))
    return sizes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", choices=["original", "new"])
    ap.add_argument("--out-dir", required=True, help="new directory for outputs (must not exist)")
    ap.add_argument("--runs-dir", default=NEW_RUNS_DIR,
                    help="for --set new: directory holding <name>/DONE.json and <name>/data/ (default %(default)s)")
    ap.add_argument("--csv", action="append", metavar="NAME=PATH",
                    help="use PATH for run NAME (runs.json name, e.g. testing)")
    ap.add_argument("--c-elec", type=float, help="electricity cost $/MWh (overrides the set's)")
    ap.add_argument("--docx", default=None, help="manuscript whose drawings fix the aspect ratios")
    ap.add_argument("--size-in", nargs=3, action="append", metavar=("FIG", "W", "H"),
                    help="size of figure FIG (4..9) in inches; skips the docx for it")
    ap.add_argument("--no-figures", action="store_true", help="statistics only")
    ap.add_argument("--clock", choices=["historical", "simulated"], default="historical",
                    help="time axis for comparisons with history (out-of-sample and hindcast "
                         "monthly ratios, their windows, Figures 4 and 5): historical = "
                         "Hist_Timestamp, simulated = Timestamp as in the notebook "
                         "(default %(default)s)")
    ap.add_argument("--hindcast-window", choices=["calibration", "notebook"],
                    default="calibration",
                    help="2018-2024 hindcast metrics: calibration = blocks [501,995, 877,280) in "
                         "TH/s as the calibration grid; notebook = [505,227, 877,280) in H/s as "
                         "cells 14-16 (default %(default)s)")
    ap.add_argument("--compare", nargs=2, metavar=("ORIG_DIR", "NEW_DIR"),
                    help="compare two output directories instead of analysing a set")
    ap.add_argument("--allow-mixed-definitions", action="store_true",
                    help="with --compare: compare outputs whose clock or hindcast window "
                         "differ (refused otherwise); the mismatch is recorded at the top of "
                         "comparison.txt")
    args = ap.parse_args()

    out = os.path.abspath(args.out_dir)
    if os.path.exists(out):
        sys.exit(f"refusing to write into existing {out}")
    if args.compare:
        mismatch = definition_mismatch(*args.compare)
        if mismatch:
            text = "; ".join(f"{k}: {va} vs {vb}" for k, va, vb in mismatch)
            if not args.allow_mixed_definitions:
                print(f"[analyze] refusing to compare outputs with different definitions "
                      f"({text}); pass --allow-mixed-definitions to compare them anyway",
                      file=sys.stderr, flush=True)
                sys.exit(2)
            warn(f"comparing outputs with different definitions ({text}); recorded at the "
                 f"top of comparison.txt")
        tmp = make_tmp(out)
        ok = False
        try:
            compare(*args.compare, tmp, mismatch)
            ok = True
        finally:
            if not ok:
                shutil.rmtree(tmp, ignore_errors=True)
        finalize(tmp, out)
        say(f"output in {out}")
        return
    if not args.set:
        sys.exit("--set is required unless --compare is given")
    paths, c_elec, prov = resolve(args)
    sizes = parse_size(args.size_in)
    fig3 = load_fig3_helpers()
    docx = args.docx or fig3.DEFAULT_DOCX
    # Everything is written into a temporary sibling directory, renamed to --out-dir only
    # when the run has finished; a failed run leaves no --out-dir behind.
    tmp = make_tmp(out)
    ok = False
    try:
        run_set(args, out, tmp, paths, c_elec, prov, sizes, fig3, docx)
        ok = True
    finally:
        if not ok:
            shutil.rmtree(tmp, ignore_errors=True)
    finalize(tmp, out)
    say(f"output in {out}")
    say("peak RSS %.0f MB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))


def run_set(args, final_out, out, paths, c_elec, prov, sizes, fig3, docx):
    tcol = CLOCK_COLUMN[args.clock]
    say(f"set {args.set}, C_elec {c_elec:g} ({prov['c_elec_source']}), clock {args.clock} "
        f"({tcol}), hindcast window {args.hindcast_window}, out {final_out}")
    st = {"set": args.set, "clock": args.clock, "clock_column": tcol,
          "hindcast_window": args.hindcast_window, "provenance": prov}
    say(f"reading {paths['testing']}")
    test = read_hashrate_run(paths["testing"])
    st["oos"], test_win, test_mr = oos_stats(test, tcol)
    test_mr.to_csv(os.path.join(out, "oos_monthly_ratio_2025.csv"), index=False)
    del test
    say(f"reading {paths['hindcasting']}")
    hind = read_hashrate_run(paths["hindcasting"])
    st["hindcast"], hind_wins, hind_ratio = hindcast_stats(hind, tcol, args.hindcast_window)
    pd.concat(hind_ratio, ignore_index=True).to_csv(
        os.path.join(out, "hindcast_monthly_ratio.csv"), index=False)
    del hind

    sc, kept = {}, {}
    for s in SCENARIOS:
        say(f"reading {paths[s]}")
        sc[s], kept[s] = scan_scenario(paths[s], c_elec)
    st["scenarios"] = sc
    hts = [pd.Timestamp(sc[s]["halving_timestamp"]) for s in SCENARIOS]
    st["halving"] = {"earliest": str(min(hts)), "latest": str(max(hts)),
                     "earliest_date": str(min(hts).date()), "latest_date": str(max(hts).date())}
    st["checks"] = cross_checks(sc, kept)

    st = jsonable(st)
    ver = verify(flatten(st), {"set": args.set, "clock": args.clock,
                               "hindcast_window": args.hindcast_window})
    ver.to_csv(os.path.join(out, "verification.csv"), index=False)
    write_tables(out, st)

    if not args.no_figures:
        fig3.ensure_font()
        set_pub_style(fig3.FONT)
        figs = {}
        for n in range(4, 10):
            if n in sizes:
                figs[n] = {"size_in": list(sizes[n]), "source": "--size-in"}
            else:
                cx, cy, media = extent_from_docx(docx, n)
                figs[n] = {"size_in": [NOTEBOOK_WIDTH_IN, NOTEBOOK_WIDTH_IN * cy / cx],
                           "source": f"{os.path.basename(docx)} {media} extent {cx}x{cy} EMU"}
        builders = {
            4: lambda z: figure4(z, hind_wins, hind_ratio, tcol),
            5: lambda z: figure5(z, test_win, test_mr, tcol),
            6: lambda z: figure6(z, kept),
            7: lambda z: figure7(z, kept),
            8: lambda z: figure8(z, sc),
            9: lambda z: figure9(z, kept, c_elec),
        }
        for n, build in builders.items():
            path = os.path.join(out, f"figure{n}.png")
            fig = build(tuple(figs[n]["size_in"]))
            fig.savefig(path, dpi=DPI)
            plt.close(fig)
            figs[n]["path"] = os.path.join(final_out, f"figure{n}.png")
            say(f"wrote {figs[n]['path']} ({figs[n]['size_in'][0]:.3f} x {figs[n]['size_in'][1]:.3f} in; "
                f"{figs[n]['source']})")
        st["figures"] = figs

    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(st, f, indent=1)
    text = summary_text(st, ver)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
