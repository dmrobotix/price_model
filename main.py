"""
Main Script for Miner Model Simulation

This script sets up the simulation by loading configuration values, initializing state,
and then running the simulation. The simulation now uses an economic-driven,
stochastic block time model where the observed block time is used to update
the network hashrate and drive difficulty retargeting.
"""

from datetime import datetime, timedelta
import os
import time
import pandas as pd
import numpy as np
from config import (
    SHA256_MAX, DIFFICULTY_BASE, EXPECTED_BLOCK_TIME, R_BLOCK, P_BTC, C_ELEC,
    MACHINE_DATA_FILE, PERIOD_BLOCKS, INITIAL_BLOCK_SUBSIDY, HALVING_INTERVAL,
    DEFAULT_TARGET, PARETO_TOP_WEIGHT, PARETO_BOTTOM_WEIGHT, REGRESSION_WEIGHT,
    FORECAST_MODEL, FORECAST_TARGET_DATE, FORECAST_TARGET_PRICE, INITIAL_HASHRATE, 
    INITIAL_DIFFICULTY, BLOCK_PACE_DATA, S, CALIBRATION_MODE, TX_BLOCK_DATA, PRICE_DATA,
    S_0, C_ELEC_0,
    env_override, parse_positive_int, parse_naive_datetime, parse_output_name
)
from modules.simulation import run_simulation
from modules.visualization import (
    plot_simulation_overview, plot_hashrate, plot_block_times,
    plot_energy_consumption, plot_block_time_distribution,
    plot_dynamic_efficiency, plot_annual_energy, plot_price_usd,
    plot_cost_per_btc, plot_margin_index,
    plot_time_varying_power_price_ceiling,
    compute_and_plot_annual_energy_from_hashrate
)
from modules.debug import setup_logging
from modules.price import load_market_prices, build_price_forecaster, build_price_lookup
from modules.data_processing import (load_block_paces, get_max_block_height, calculate_hashrate_rmse, calculate_hashrate_rmse_block_level, 
    get_last_tx_fee_time, calculate_hashrate_rmsle_block_level)
from modules.network import calc_core_hashrate
from modules.energy import calc_power_demand, calc_energy_consumption
from modules.economics import build_shock_profile
from modules.boundaries import GENESIS_TIME_S, check_boundary_blocks
from modules.energy import calc_power_demand, calc_energy_consumption

# Set up logging.
setup_logging()

# Load historical market price data from CSV.
market_prices = load_market_prices(PRICE_DATA)
real_paces = load_block_paces(BLOCK_PACE_DATA, TX_BLOCK_DATA)
max_hist_block = get_max_block_height(BLOCK_PACE_DATA)
# Period boundaries are UTC instants; their block heights come from the UTC block
# times (modules/boundaries.py). This checks them, and the genesis time, against the data.
boundary_blocks = check_boundary_blocks(real_paces)
print(f"Boundary blocks (last block of the earlier period): {boundary_blocks}")

# Define Initial State
# NOTE: 'sim_timestamp_s' starts at the genesis block's UTC time, 1231006505
# (2009-01-03 18:15:05 UTC). Until 2026-10-05 it was 1230988505, the Eastern clock
# reading 13:15:05 taken as UTC.
genesis_time_s = GENESIS_TIME_S
start_dt = pd.to_datetime(genesis_time_s, unit='s', utc=True).tz_convert(None)

initial_state = {
    'R_block': R_BLOCK,
    'target': DEFAULT_TARGET,
    'D': INITIAL_DIFFICULTY,
    'H': INITIAL_HASHRATE,
    'sim_timestamp': start_dt,      # Datetime object (UTC-naive)
    'sim_timestamp_s': genesis_time_s, # Integer seconds (Source of Truth)
    'block_height': 0,
    'difficulty_adjusted': False,
    'R_pe': 0.0,
    'efficiency': 1636007*1e-12,
    'R_t': 1.1641354547009541e-08,
    'P': INITIAL_HASHRATE*1636007*1e-12,
    'E': 0,
    'P_USD': 0,
    'TX_fee': 0.0,
    'T_block': 0.0
}
# Set the number of simulation steps.
# num_steps = 888_300
num_steps = 921_683 # boundary block of 2025-11-01 00:00 UTC (stamped 00:00:27 UTC); the testing run ends here
# num_steps = 1_260_000 # ~ 2032
# num_steps = 877_259 # boundary block of 2025-01-01 UTC (877,280 was the Eastern-midnight block)
# PRICE_NUM_STEPS in the environment overrides the value above (see config.py).
num_steps = env_override("PRICE_NUM_STEPS", parse_positive_int, num_steps)

initial_block_height = initial_state['block_height']

# ... after loading market_prices and real_paces, and defining num_steps & initial_block_height:
#last_hist_time = market_prices['Time'].max
last_hist_time = get_last_tx_fee_time() # transaction feed data is the limiting timestamp
price_lookup = build_price_lookup(market_prices)
shock_profile = build_shock_profile(max_hist_block, initial_block_height, num_steps, shock_duration=4*PERIOD_BLOCKS, L=0.40)
# Define simulation parameters.
params = {
    'P_USD': P_BTC,                      # Fallback Bitcoin price if needed.
    'C_elec': C_ELEC,                    # Electricity cost ($/MWh).
    'C_elec_0': C_ELEC_0,
    'initial_block_subsidy': R_BLOCK,    # Initial block reward (before halving).
    'halving_interval': HALVING_INTERVAL,  # Blocks between subsidy halvings.
    'period_blocks': PERIOD_BLOCKS,        # Blocks per difficulty adjustment period.
    'expected_block_time': EXPECTED_BLOCK_TIME,  # Base expected block time (e.g., 600 seconds).
    'default_target': DEFAULT_TARGET,          # Baseline target corresponding to difficulty 1.
    'machine_data_file': MACHINE_DATA_FILE,      # Path to machine efficiency data file.
    'pareto_top_weight': PARETO_TOP_WEIGHT,      # Weight for the top machines in efficiency computation.
    'pareto_bottom_weight': PARETO_BOTTOM_WEIGHT,  # Weight for the bottom machines.
    'regression_weight': REGRESSION_WEIGHT,        # Blending weight for regression-based efficiency.
    'market_prices': market_prices,                # Historical BTC-USD price data.
    'block_paces': real_paces,
    #'historical_cutoff': datetime(2010, 7, 17, 0, 0, 0), # start of minute price data
    # Cutoff, a UTC instant. With 2025-01-01: block 877,259 (00:21:15 UTC, the boundary
    # block) is the last historical block; 877,260 is the first block whose time the
    # economic model sets (modules/boundaries.py).
    'historical_cutoff': datetime(2025, 1, 1, 0, 0, 0), 
    #'historical_cutoff': datetime(2025, 11, 8, 17, 58, 32), # last transaction fee data
    #'historical_cutoff': datetime(2025, 12, 31, 23, 59, 59),
    # 'historical_cutoff': datetime(2024, 12, 31, 23, 59, 59),
    'last_hist_time': last_hist_time,                   # The last historical date in the simulation
    'price_lookup': price_lookup,
    'shock_profile': shock_profile,
    'shock_enabled': False,
    'price_forecaster': build_price_forecaster(         # Prefit the forecast function once
                            market_prices,
                            model_type    = FORECAST_MODEL,
                            anchor_date   = FORECAST_TARGET_DATE,
                            anchor_price  = FORECAST_TARGET_PRICE
                        ),
    'fix_efficiency': False,        # flip this to False to run dynamic efficiency as before
    'fixed_efficiency': None,      # this will hold the last historical value once we cross the cutoff
    'total_operating': True,       # in the revenue calculation compute the total operationg cost not just electricity cost (based on coinshares esimate)
    'calibration_mode': CALIBRATION_MODE,
    'S': S,
    'S_0': S_0,
    'daily_window_start_idx': 0,
    'daily_window_day': None,
}
# PRICE_HISTORICAL_CUTOFF in the environment overrides the cutoff above (see config.py).
params['historical_cutoff'] = env_override("PRICE_HISTORICAL_CUTOFF", parse_naive_datetime,
                                           params['historical_cutoff'])

# file_name = "simulation_results_efficiency_frozen_price_fixed" DONE
# file_name = "simulation_results_efficiency_frozen_price_200k" DONE
# file_name = "simulation_results_efficiency_frozen_price_500k" DONE
# file_name = "simulation_results_efficiency_frontier_price_fixed" DONE
# file_name = "simulation_results_efficiency_frontier_price_200k" DONE
# file_name = "simulation_results_efficiency_frontier_price_500k" DONE
# file_name = "simulation_results_efficiency_frontier_price_1M" DONE
# file_name = "simulation_results_efficiency_frozen_price_1M" DONE
# file_name = "simulation_results_hindcasting" DONE (num_steps 877_259, cutoff 2010-07-17)
# file_name = "simulation_results_calibration" DONE
# The default settings above (num_steps 921,683, cutoff 2025-01-01) are the out-of-sample
# test, the `testing` run of scenarios/runs.json, so the default output is named after it.
# Until 2026-10-06 it was named simulation_results_hindcasting.
file_name = "simulation_results_testing"
# PRICE_OUTPUT_NAME in the environment overrides file_name (see config.py).
file_name = env_override("PRICE_OUTPUT_NAME", parse_output_name, file_name)
if any(v in os.environ for v in ("PRICE_NUM_STEPS", "PRICE_HISTORICAL_CUTOFF", "PRICE_OUTPUT_NAME")):
    print(f"[run overrides] num_steps={num_steps} "
          f"historical_cutoff={params['historical_cutoff'].isoformat(sep=' ')} file_name={file_name}")
# Run the simulation with exception handling.
print("Simulation started. Detailed logs available at: data/logs/model_debug.log")
try:
    # Record the start time.
    start_wall_time = time.time()
    history = run_simulation(initial_state, params, num_steps, initial_block_height)
    # Record the end time.
    end_wall_time = time.time()
    execution_time = end_wall_time - start_wall_time
    print(f"Wall-clock execution time for simulation: {execution_time:.2f} seconds")
except Exception as e:
    print(f"Simulation failed: {e}")
    raise

# Convert simulation history to a DataFrame.
simulation_df = pd.DataFrame(history)

# The 'sim_timestamp' column from the simulation is the correct one.
# We just need to rename it to 'Timestamp' for the plotting functions.
simulation_df.rename(columns={'sim_timestamp': 'Timestamp'}, inplace=True)

# Ensure 'Timestamp' is a datetime object (it should be, but good to be sure)
simulation_df['Timestamp'] = pd.to_datetime(simulation_df['Timestamp'])

# Ensure 'T_block' (which is the interval) is present and fill any NaNs
# (like in the initial state) with 0.
if 'T_block' not in simulation_df.columns:
    simulation_df['T_block'] = 0.0
simulation_df['T_block'] = simulation_df['T_block'].fillna(0)

# Rename 'T_block' for consistency in plots
simulation_df.rename(columns={'T_block': 'Block_Time_Seconds'}, inplace=True)

# Print out whhen difficulty adjustment(s) occurred.
adjustments = simulation_df[simulation_df['difficulty_adjusted'] == True]
print("Difficulty adjustments occurred at the following timestamps and block heights:")
print(adjustments[['Timestamp', 'block_height', 'D']])

# ------------------------------------------------------------------
# Hashrate & energy: simulated vs. historical (from block_paces)
# ------------------------------------------------------------------

# 1) Historical hashrate from historical target + inter-block intervals
hist_df = real_paces.copy()

# Drop rows missing target or interval
hist_df = hist_df.dropna(subset=["Target", "Inter_Block_Interval_Seconds"])

# Ensure we’re ordered by block height (index is Height)
hist_df = hist_df.sort_index()

# Make sure Target is numeric
hist_df["Target"] = hist_df["Target"].astype(float)

# Difficulty and Core-style hashrate (H/s) from historical data
d_hist = (DEFAULT_TARGET / hist_df["Target"].to_numpy())
t_hist = hist_df["Inter_Block_Interval_Seconds"].to_numpy()
H_hist = calc_core_hashrate(d_hist, t_hist, window=2016)  # H/s

# Put both hashrate and block interval into a DataFrame indexed by Height
hist_df["H_hist"] = H_hist
hist_for_merge = hist_df[["Time", "H_hist", "Inter_Block_Interval_Seconds"]].rename(
    columns={
        "Time": "Hist_Timestamp",
        "Inter_Block_Interval_Seconds": "T_hist_seconds",
    }
)

# 2) Map historical series onto the simulation by block_height
simulation_df = simulation_df.merge(
    hist_for_merge,
    left_on="block_height",
    right_index=True,
    how="left"
)

# 3) Simulated hashrate from simulated target + simulated block times
d_sim = (DEFAULT_TARGET / simulation_df["target"].to_numpy())
t_sim = simulation_df["Block_Time_Seconds"].to_numpy()
H_sim = calc_core_hashrate(d_sim, t_sim, window=2016)  # H/s

simulation_df["H_sim"] = H_sim
simulation_df["H"] = H_sim  # keep for backward compatibility

# 4) Power & energy using both hashrates (where applicable)
if CALIBRATION_MODE == False:
    # efficiency is in J/hash from the simulation state
    eta_arr = simulation_df["efficiency"].to_numpy()  # J/hash

    # --- Using simulated hashrate ---
    P_sim = calc_power_demand(H_sim, eta_arr, 1.10)          # W
    E_sim = calc_energy_consumption(P_sim, t_sim / 3600.0)   # Wh

    simulation_df["P_sim"] = P_sim
    simulation_df["E_sim"] = E_sim
    simulation_df["P"] = P_sim
    simulation_df["E"] = E_sim

    # --- Using historical hashrate (from block_paces) ---
    H_hist_arr = simulation_df["H_hist"].to_numpy()
    T_hist_arr = simulation_df["T_hist_seconds"].to_numpy()  # same length as simulation_df

    P_hist = calc_power_demand(H_hist_arr, eta_arr, 1.10)                 # W
    E_hist = calc_energy_consumption(P_hist, T_hist_arr / 3600.0)         # Wh

    simulation_df["P_hist"] = P_hist
    simulation_df["E_hist"] = E_hist

    # Convert efficiency to W/TH for plotting (J/hash * 1e12)
    simulation_df["efficiency"] = simulation_df["efficiency"] * 1e12


# Save simulation results to CSV.
simulation_df.to_csv(f"../data/{file_name}.csv", index=False)

# Find the RMSE of hashrate results
rmse = calculate_hashrate_rmse_block_level(simulation_df)
rmsle = calculate_hashrate_rmsle_block_level(simulation_df)
# Generate plots.
# plot_simulation_overview(simulation_df)
plot_hashrate(simulation_df, use_moving_avg=False)
# plot_block_times(simulation_df)
if CALIBRATION_MODE == False:
    plot_energy_consumption(simulation_df)
    # plot_block_time_distribution(simulation_df)  # New plot of block time distribution.
    plot_annual_energy(simulation_df)
    plot_price_usd(simulation_df)
    plot_cost_per_btc(simulation_df)
    plot_margin_index(simulation_df)
    plot_time_varying_power_price_ceiling(simulation_df)
    plot_annual_energy(simulation_df, energy_col="E_hist", timestamp_col="Hist_Timestamp")
    H_hist_series = simulation_df.set_index("Hist_Timestamp")["H_hist"]
    eta_series    = simulation_df.set_index("Hist_Timestamp")["efficiency"] / 1e12
    annual_energy = compute_and_plot_annual_energy_from_hashrate(hashrate=H_hist_series, efficiency=eta_series, pue=1.10, energy_unit="TWh")
    
    # --- 1. Your annual energy series (TWh) ---
    # annual_energy_TWh = compute_and_plot_annual_energy_from_hashrate(...)
    
    # Convert index from year-end timestamps to plain year integers
    my_annual_TWh = annual_energy.copy()
    my_annual_TWh.index = my_annual_TWh.index.year
    
    # --- 2. CBECI annual TWh data ---
    cbeci_data = {
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
    cbeci_TWh = pd.Series(cbeci_data, name="CBECI_TWh")
    
    # --- 3. Align years and compute percent difference ---
    common_years = sorted(set(my_annual_TWh.index) & set(cbeci_TWh.index))
    
    comparison = pd.DataFrame({
        "My_TWh": my_annual_TWh.loc[common_years],
        "CBECI_TWh": cbeci_TWh.loc[common_years],
    })
    
    # Percent difference = (mine - CBECI) / CBECI * 100
    comparison["Pct_Diff"] = 100.0 * (
        comparison["My_TWh"] - comparison["CBECI_TWh"]
    ) / comparison["CBECI_TWh"]
    
    # --- 4. Print nicely ---
    print("Comparison vs CBECI (TWh and % difference):")
    print(comparison.to_string(float_format=lambda x: f"{x:.2f}"))

    # ---------------------------------------------------------
    # 4. Helper to compute mean absolute percent difference
    # ---------------------------------------------------------
    def mean_abs_pct_diff(df: pd.DataFrame, start_year: int, end_year: int) -> float:
        """Compute mean absolute percent difference over a year range."""
        mask = (df.index >= start_year) & (df.index <= end_year)
        if not mask.any():
            return float("nan")
        return df.loc[mask, "Pct_Diff"].abs().mean()

    def weighted_abs_pct_error(df: pd.DataFrame, start_year: int, end_year: int) -> float:
        """
        Compute Weighted Absolute Percent Error (WAPE) over a year range.
        Formula: Σ|Model - Actual| / Σ|Actual|
        """
        mask = (df.index >= start_year) & (df.index <= end_year)
        subset = df.loc[mask]
    
        if subset.empty:
            return float("nan")
    
        # Calculate WAPE components
        total_abs_error = (subset["My_TWh"] - subset["CBECI_TWh"]).abs().sum()
        total_actual = subset["CBECI_TWh"].sum()
    
        if total_actual == 0:
            return float("nan")

        # Returns a decimal (e.g., 0.072 for 7.2%). 
        # Multiply by 100 if you want the result in percentage points.
        return total_abs_error / total_actual*100
    
    overall_mape     = mean_abs_pct_diff(comparison, 2011, 2023)
    mape_2015_2023   = mean_abs_pct_diff(comparison, 2015, 2023)
    mape_2018_2023   = mean_abs_pct_diff(comparison, 2018, 2023)
    mape_2020_2023   = mean_abs_pct_diff(comparison, 2020, 2023)
    
    print("\nMean absolute percent difference (MAPE):")
    print(f"2011–2023: {overall_mape:.2f}%")
    print(f"2015–2023: {mape_2015_2023:.2f}%")
    print(f"2018–2023: {mape_2018_2023:.2f}%")
    print(f"2020–2023: {mape_2020_2023:.2f}%")

    overall_wape     = weighted_abs_pct_error(comparison, 2011, 2023)
    wape_2015_2023   = weighted_abs_pct_error(comparison, 2015, 2023)
    wape_2018_2023   = weighted_abs_pct_error(comparison, 2018, 2023)
    wape_2020_2023   = weighted_abs_pct_error(comparison, 2020, 2023)
    
    print("\nWeighted mean absolute percent difference (WAPE):")
    print(f"2011–2023: {overall_wape:.2f}%")
    print(f"2015–2023: {wape_2015_2023:.2f}%")
    print(f"2018–2023: {wape_2018_2023:.2f}%")
    print(f"2020–2023: {wape_2020_2023:.2f}%")

df = simulation_df.copy()
df["Year"] = df["Timestamp"].dt.year

annual_hashrate = df.groupby("Year")["H_sim"].mean()       # TH/s if you convert
annual_eff      = df.groupby("Year")["efficiency"].mean()  # W/TH (after *1e12 in main.py)
annual_TWh      = df.groupby("Year")["E_sim"].sum() / 1e12

print(annual_hashrate.loc[2020:2023])
print(annual_eff.loc[2020:2023])
print(annual_TWh.loc[2020:2023])
    
print(f"Simulation complete. Results saved to '../data/{file_name}.csv'.")

