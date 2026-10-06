"""
CBECI replication (validation of the model's CBECI efficiency implementation)

What this validates
-------------------
The model's fleet efficiency follows the Cambridge Bitcoin Electricity Consumption
Index (CBECI) methodology (modules/efficiency_cbeci.py). This script checks that,
when run with CBECI's own assumptions, the model's efficiency series reproduces
CBECI's published annual electricity-consumption estimates for 2011-2023.

Assumptions that differ from the model's calibrated run
-------------------------------------------------------
* Electricity price: CBECI Assumption 1 uses $50/MWh for ALL years. This script
  sets both params['C_elec'] and params['C_elec_0'] to 50, so the profitability
  threshold uses $50/MWh before and after T_STAR (2018-01-01). The model's
  calibrated run uses $100/MWh before 2018 (config.C_ELEC_0) and $40/MWh after
  (config.C_ELEC).
* Hardware set: the CBECI manufacturer filter (Bitmain, MicroBT, Canaan for
  hardware released on or after July 2014) is applied by the loader,
  modules/efficiency_cbeci._load_machines, for every run of the model.
* PUE is 1.10, CBECI's value.
* config.py is not modified.

Source of CBECI methodology: https://ccaf.io/cbnsi/cbeci/methodology

Method
------
1. Run the simulation over the historical blocks up to END_DATE.
2. Daily efficiency psi_d (J/TH) = daily mean of the simulated per-block
   efficiency (J/hash) x 1e12, indexed by UTC date.
3. Two historical daily hashrates (TH/s) from the block data:
     H_2016 : Bitcoin-Core style rolling 2016-block hashrate per block, daily mean
              (the definition main.py uses).
     H_1day : per UTC day, sum(difficulty x 2^32) over the day's blocks / 86,400
              (CBECI's "mean hashrate on a given day").
4. Annual energy (TWh) = sum over days of H_d x psi_d x PUE x 24 / 1e12.
5. Percent difference vs the CBECI annual TWh, plus MAPE and WAPE over 2011-2023,
   2015-2023, 2018-2023 and 2020-2023.

How to run (about 40-50 minutes)
--------------------------------
    cd <repo root>/validation && MPLBACKEND=Agg <python> cbeci_replication.py [--out-dir DIR]

Outputs go to DIR (default <repo root>/data/results/cbeci_replication/; a relative
DIR is resolved against the repo root). The script raises instead of overwriting
any existing output file:
    annual_comparison.csv, error_summary.csv, daily_efficiency_J_per_TH.csv,
    model_debug.log
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Repo root on sys.path so `import config` and `import modules...` work.
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (
    R_BLOCK, P_BTC, MACHINE_DATA_FILE, PERIOD_BLOCKS, HALVING_INTERVAL,
    EXPECTED_BLOCK_TIME, DEFAULT_TARGET, PARETO_TOP_WEIGHT, PARETO_BOTTOM_WEIGHT,
    REGRESSION_WEIGHT, FORECAST_MODEL, FORECAST_TARGET_DATE, FORECAST_TARGET_PRICE,
    INITIAL_HASHRATE, INITIAL_DIFFICULTY, BLOCK_PACE_DATA, S, CALIBRATION_MODE,
    TX_BLOCK_DATA, PRICE_DATA, S_0, DIFFICULTY_BASE,
)
from modules.simulation import run_simulation
from modules.boundaries import GENESIS_TIME_S
from modules.debug import setup_logging
from modules.price import load_market_prices, build_price_forecaster, build_price_lookup
from modules.data_processing import load_block_paces, get_max_block_height, get_last_tx_fee_time
from modules.network import calc_core_hashrate
from modules.economics import build_shock_profile

# ----------------------------------------------------------------------
# CBECI assumptions
# ----------------------------------------------------------------------
CBECI_PRICE_USD_PER_MWH = 50.0      # CBECI Assumption 1: same price in every year
PUE_CBECI = 1.10
END_DATE = datetime(2023, 12, 31, 23, 59, 59)

# CBECI annual TWh; the values used in main.py.
CBECI_TWH = {
    2011: 0.14,
    2012: 0.10,
    2013: 1.06,
    2014: 4.73,
    2015: 3.62,
    2016: 5.73,
    2017: 12.93,
    2018: 43.32,
    2019: 54.63,
    2020: 67.14,
    2021: 89.00,
    2022: 95.53,
    2023: 121.13,
}
WINDOWS = [(2011, 2023), (2015, 2023), (2018, 2023), (2020, 2023)]

DEFAULT_OUT_DIR = os.path.join("data", "results", "cbeci_replication")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="output directory; a relative path is resolved against the repo root "
                             f"(default: {DEFAULT_OUT_DIR})")
    args = parser.parse_args()
    OUT_DIR = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    OUT_DIR = os.path.abspath(OUT_DIR)
    OUT_FILES = {
        "annual": os.path.join(OUT_DIR, "annual_comparison.csv"),
        "errors": os.path.join(OUT_DIR, "error_summary.csv"),
        "daily": os.path.join(OUT_DIR, "daily_efficiency_J_per_TH.csv"),
        "log": os.path.join(OUT_DIR, "model_debug.log"),
    }

    # config.py data paths are relative ('../data/...'); require the documented cwd.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Refuse to overwrite earlier outputs (checked before the long simulation).
    existing = [p for p in OUT_FILES.values() if os.path.exists(p)]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing output files: {existing}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # The log goes into the new output directory; the default log file
    # (data/logs/model_debug.log) would be truncated by setup_logging.
    setup_logging(OUT_FILES["log"])

    # ------------------------------------------------------------------
    # Inputs, as in main.py
    # ------------------------------------------------------------------
    market_prices = load_market_prices(PRICE_DATA)
    real_paces = load_block_paces(BLOCK_PACE_DATA, TX_BLOCK_DATA)
    max_hist_block = get_max_block_height(BLOCK_PACE_DATA)

    # Genesis block time, 2009-01-03 18:15:05 UTC (was 1230988505 before 2026-10-05,
    # the Eastern clock reading 13:15:05 taken as UTC; run_simulation now rejects it).
    genesis_time_s = GENESIS_TIME_S
    start_dt = pd.to_datetime(genesis_time_s, unit="s", utc=True).tz_convert(None)

    initial_state = {
        'R_block': R_BLOCK,
        'target': DEFAULT_TARGET,
        'D': INITIAL_DIFFICULTY,
        'H': INITIAL_HASHRATE,
        'sim_timestamp': start_dt,
        'sim_timestamp_s': genesis_time_s,
        'block_height': 0,
        'difficulty_adjusted': False,
        'R_pe': 0.0,
        'efficiency': 1636007 * 1e-12,
        'R_t': 1.1641354547009541e-08,
        'P': INITIAL_HASHRATE * 1636007 * 1e-12,
        'E': 0,
        'P_USD': 0,
        'TX_fee': 0.0,
        'T_block': 0.0,
    }

    # Height of the last historical block at or before END_DATE.
    hist_times = real_paces["Time"]
    num_steps = int(real_paces.index[(hist_times <= END_DATE).to_numpy()].max())
    print(f"num_steps (height of last block at or before {END_DATE}): {num_steps}")

    initial_block_height = initial_state['block_height']
    last_hist_time = get_last_tx_fee_time()
    price_lookup = build_price_lookup(market_prices)
    shock_profile = build_shock_profile(max_hist_block, initial_block_height, num_steps,
                                        shock_duration=4 * PERIOD_BLOCKS, L=0.40)

    params = {
        'P_USD': P_BTC,
        'C_elec': CBECI_PRICE_USD_PER_MWH,     # CBECI: $50/MWh ...
        'C_elec_0': CBECI_PRICE_USD_PER_MWH,   # ... in every year, including before T_STAR
        'initial_block_subsidy': R_BLOCK,
        'halving_interval': HALVING_INTERVAL,
        'period_blocks': PERIOD_BLOCKS,
        'expected_block_time': EXPECTED_BLOCK_TIME,
        'default_target': DEFAULT_TARGET,
        'machine_data_file': MACHINE_DATA_FILE,
        'pareto_top_weight': PARETO_TOP_WEIGHT,
        'pareto_bottom_weight': PARETO_BOTTOM_WEIGHT,
        'regression_weight': REGRESSION_WEIGHT,
        'market_prices': market_prices,
        'block_paces': real_paces,
        'historical_cutoff': datetime(2025, 1, 1, 0, 0, 0),
        'last_hist_time': last_hist_time,
        'price_lookup': price_lookup,
        'shock_profile': shock_profile,
        'shock_enabled': False,
        'price_forecaster': build_price_forecaster(
            market_prices,
            model_type=FORECAST_MODEL,
            anchor_date=FORECAST_TARGET_DATE,
            anchor_price=FORECAST_TARGET_PRICE,
        ),
        'fix_efficiency': False,
        'fixed_efficiency': None,
        'total_operating': True,
        'calibration_mode': CALIBRATION_MODE,
        'S': S,
        'S_0': S_0,
        'daily_window_start_idx': 0,
        'daily_window_day': None,
    }

    # ------------------------------------------------------------------
    # Simulation and daily efficiency psi_d (J/TH)
    # ------------------------------------------------------------------
    print("Simulation started.")
    history = run_simulation(initial_state, params, num_steps, initial_block_height)
    sim_df = pd.DataFrame(history)
    sim_df["sim_timestamp"] = pd.to_datetime(sim_df["sim_timestamp"])

    psi = (sim_df.set_index("sim_timestamp")["efficiency"] * 1e12).resample("D").mean()
    psi.index = psi.index.normalize()
    psi.name = "psi_J_per_TH"

    # ------------------------------------------------------------------
    # Historical daily hashrates (TH/s)
    # ------------------------------------------------------------------
    hist = real_paces.dropna(subset=["Target", "Inter_Block_Interval_Seconds"]).sort_index()
    hist = hist.loc[hist["Time"] <= END_DATE].copy()
    hist["Target"] = hist["Target"].astype(float)
    d = DEFAULT_TARGET / hist["Target"].to_numpy()                       # difficulty
    t = hist["Inter_Block_Interval_Seconds"].to_numpy()
    hist["day"] = hist["Time"].dt.normalize()

    # (a) rolling 2016-block Core-style hashrate (H/s -> TH/s), daily mean
    hist["H_2016"] = calc_core_hashrate(d, t, window=2016) / 1e12
    H_2016 = hist.groupby("day")["H_2016"].mean()

    # (b) CBECI daily mean hashrate: sum of work in the day's blocks / 86,400 s
    hist["work"] = d * DIFFICULTY_BASE
    H_1day = hist.groupby("day")["work"].sum() / 86_400.0 / 1e12

    # ------------------------------------------------------------------
    # Annual energy (TWh)
    # ------------------------------------------------------------------
    def annual_twh(H_daily: pd.Series) -> pd.Series:
        full_idx = pd.date_range(H_daily.index.min(), H_daily.index.max(), freq="D")
        H_full = H_daily.reindex(full_idx)
        psi_d = psi.reindex(full_idx).ffill()
        e_twh = H_full * psi_d * PUE_CBECI * 24.0 / 1e12
        return e_twh.groupby(e_twh.index.year).sum(min_count=1)

    annual = {"H_2016": annual_twh(H_2016), "H_1day": annual_twh(H_1day)}

    cbeci = pd.Series(CBECI_TWH, name="CBECI_TWh")
    comp = pd.DataFrame({"CBECI_TWh": cbeci})
    comp.index.name = "year"
    for name, ser in annual.items():
        comp[f"TWh_{name}"] = ser.reindex(comp.index)
        comp[f"PctDiff_{name}"] = 100.0 * (comp[f"TWh_{name}"] - comp["CBECI_TWh"]) / comp["CBECI_TWh"]

    # ------------------------------------------------------------------
    # MAPE and WAPE (formulas as in main.py)
    # ------------------------------------------------------------------
    rows = []
    for name in annual:
        for lo, hi in WINDOWS:
            sub = comp.loc[lo:hi]
            mape = sub[f"PctDiff_{name}"].abs().mean()
            wape = (sub[f"TWh_{name}"] - sub["CBECI_TWh"]).abs().sum() / sub["CBECI_TWh"].sum() * 100.0
            rows.append({"window": f"{lo}-{hi}", "statistic": "MAPE", "hashrate": name, "value_pct": mape})
            rows.append({"window": f"{lo}-{hi}", "statistic": "WAPE", "hashrate": name, "value_pct": wape})
    err = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Write (never overwrite) and print
    # ------------------------------------------------------------------
    for key in ("annual", "errors", "daily"):
        if os.path.exists(OUT_FILES[key]):
            raise FileExistsError(OUT_FILES[key])
    comp.to_csv(OUT_FILES["annual"])
    err.to_csv(OUT_FILES["errors"], index=False)
    psi.to_frame().rename_axis("date_utc").to_csv(OUT_FILES["daily"])

    pd.set_option("display.width", 200)
    print("\nAnnual comparison vs CBECI (TWh; percent difference):")
    print(comp.to_string(float_format=lambda x: f"{x:.2f}"))
    print("\nError summary (percent):")
    print(err.pivot_table(index=["hashrate", "statistic"], columns="window", values="value_pct",
                          sort=False).to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
