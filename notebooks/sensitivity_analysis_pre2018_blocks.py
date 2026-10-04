import pandas as pd
import numpy as np
import os
import sys
import traceback
from datetime import datetime
from scipy.optimize import minimize

PROJECT_ROOT = os.path.abspath("..")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from modules.simulation import run_simulation
from modules.network import calc_core_hashrate
from modules.price import build_price_lookup
from config import DEFAULT_TARGET, BLOCK_PACE_DATA

# Import efficiency module to clear caches
import modules.efficiency_cbeci as eff_module 

# ---------------------------------------------------------
# Evaluation window: ONLY use pre-2018 era
# 2010-07-17 (start of minute price data) through 2017-12-31
# ---------------------------------------------------------
EVAL_START = datetime(2010, 7, 17)
EVAL_END   = datetime(2018, 1, 1)   # exclusive upper bound (pre-2018)

# Track best results globally (for S_0, C_elec_0)
best = {'rmsle': np.inf, 'S_0': None, 'C_elec_0': None, 'n': 0}

def objective_function(S_candidate, initial_state, params, num_blocks_to_fit, H_hist_series_input=None):
    """
    Runs the simulation with a given S_0 and C_elec_0 (early-era parameters),
    computes hashrate error vs historical.

    RMSLE is evaluated only over blocks whose simulated timestamps
    are in the pre-2018 window [EVAL_START, EVAL_END).
    """
    # --- 1. RESET GLOBAL CACHES (Efficiency Module) ---
    try:
        eff_module._daily_revenue_ma.clear()
        eff_module._daily_threshold_ma.clear()
        eff_module._last_rev_day = None
        eff_module._prev_nonempty_set_mask = None
        eff_module._model_forecast_cache.clear()
    except AttributeError as e:
        print(f"Warning: Could not clear efficiency caches: {e}")

    # --- 2. Unpack Parameters ---
    # S_candidate is [S_0, C_elec_0]
    S0_val = float(S_candidate[0])
    C_elec0_val = float(S_candidate[1])

    # --- 3. Prepare Params (Rebuild Stateful Objects) ---
    cal_params = params.copy()
    # Keep modern-era S and C_elec fixed from params;
    # override ONLY the early-era parameters used before T_STAR.
    cal_params['S_0'] = S0_val
    cal_params['C_elec_0'] = C_elec0_val
    cal_params['calibration_mode'] = True
    
    # CRITICAL: Rebuild price_lookup so 'idx' resets to 0 for every run
    from modules.price import build_price_lookup  # re-import safe here
    cal_params['price_lookup'] = build_price_lookup(params['market_prices'])

    # DEBUG: Print progress
    print(f"Running sim #{best['n']+1} with S_0={S0_val:.5f}, C_elec_0={C_elec0_val:.2f}...", flush=True)

    try:
        # Use deepcopy-like behavior for state to avoid mutation leaking
        history = run_simulation(
            initial_state=initial_state.copy(), 
            params=cal_params,
            num_steps=num_blocks_to_fit,
            initial_block_height=initial_state.get('block_height', 0)
        )
        sim_df = pd.DataFrame(history)
        print(" Done.")
        
    except Exception as e:
        print(f"\n[!] Simulation CRASHED: {e}")
        # traceback.print_exc() # Uncomment for deep debugging
        return 1e9

    # --- 4. Process Results & Calculate RMSLE ---
    # Standardize column names
    if 'sim_timestamp' in sim_df.columns and 'Timestamp' not in sim_df.columns:
        sim_df = sim_df.rename(columns={'sim_timestamp': 'Timestamp'})
    if 'T_block' in sim_df.columns and 'Block_Time_Seconds' not in sim_df.columns:
        sim_df.rename(columns={'T_block': 'Block_Time_Seconds'}, inplace=True)
        
    # Calculate simulated hashrate (Core-style)
    d_arr   = DEFAULT_TARGET / sim_df['target'].to_numpy()
    t_arr   = sim_df['Block_Time_Seconds'].to_numpy()
    H_arr   = calc_core_hashrate(d_arr, t_arr, window=2016)
    
    # Convert to TH/s to match historical units (assuming H_hist_series_input is TH/s)
    # H_arr is in H/s. 
    sim_th = pd.Series(H_arr * 1e-12, index=sim_df['block_height'].astype(int))

    # Restrict to pre-2018 evaluation window based on simulated timestamps
    # time_mask = (sim_df['Timestamp'] >= EVAL_START) & (sim_df['Timestamp'] < EVAL_END)
    # eval_heights = sim_df.loc[time_mask, 'block_height'].astype(int)
    # sim_th = sim_th.loc[eval_heights]
    # sim_th = sim_th[time_mask]
    # Define the exact height that started 2018 historically
    # Block 501951 was the first block mined after 2018-01-01 00:00:00 UTC
    MAX_PRE2018_HEIGHT = 501995 
    MIN_PRE2018_HEIGHT = 68633
    # Filter by Height (Deterministic) instead of Time (Variable)
    height_mask = (sim_df['block_height'] >= MIN_PRE2018_HEIGHT) & (sim_df['block_height'] < MAX_PRE2018_HEIGHT)
    sim_th = sim_th[height_mask]

    # Get Historical Data
    if H_hist_series_input is not None:
        # Use the pre-calculated series passed from outside (matches main.py)
        # We'll assume the input is in H/s and convert it here
        hist_th = H_hist_series_input * 1e-12 
    else:
        # Fallback: Load file fresh (Risk of "warm-up" data loss mismatch)
        try:
            hist_df_temp = pd.read_csv(BLOCK_PACE_DATA)
            print("Warning: Using fallback historical loader inside objective function.")
            return 1e9 
        except:
            return 1e9

    # Align comparison by block height (index)
    combined = pd.concat([sim_th, hist_th], axis=1, join='inner')
    combined.columns = ['sim', 'hist']
    combined = combined.dropna()

    if combined.empty:
        print(" [!] RMSLE failed: No overlapping blocks in evaluation window.")
        return 1e9

    # --- COMPUTE RMSLE ---
    # 1. Clip negative values (just in case, though hashrate should be > 0)
    sim_vals = combined['sim'].clip(lower=0)
    hist_vals = combined['hist'].clip(lower=0)

    # 2. Log Transform (log1p adds 1 automatically to handle zeros safely)
    log_sim = np.log1p(sim_vals)
    log_hist = np.log1p(hist_vals)

    # 3. Mean Squared Error of Logs
    log_diff = log_sim - log_hist
    msle = (log_diff ** 2).mean()
    
    # 4. Root
    rmsle = np.sqrt(msle)

    best['n'] += 1

    # Track Best (Minimize RMSLE)
    if rmsle < best['rmsle']:
        best['rmsle'] = rmsle
        best['S_0'] = S0_val
        best['C_elec_0'] = C_elec0_val
        print(f" -> NEW BEST: RMSLE={rmsle:.4e}")
    else:
        print(f" -> RMSLE={rmsle:.4e} (Best: {best['rmsle']:.4e})")

    return rmsle


import gc

from config import (
    SHA256_MAX, DIFFICULTY_BASE, EXPECTED_BLOCK_TIME, R_BLOCK, P_BTC, C_ELEC,
    MACHINE_DATA_FILE, PERIOD_BLOCKS, INITIAL_BLOCK_SUBSIDY, HALVING_INTERVAL,
    DEFAULT_TARGET, PARETO_TOP_WEIGHT, PARETO_BOTTOM_WEIGHT, REGRESSION_WEIGHT,
    FORECAST_MODEL, FORECAST_TARGET_DATE, FORECAST_TARGET_PRICE, INITIAL_HASHRATE, 
    INITIAL_DIFFICULTY, BLOCK_PACE_DATA, S, CALIBRATION_MODE, TX_BLOCK_DATA,
    C_ELEC_0, S_0
)

from modules.price import load_market_prices, build_price_forecaster, build_price_lookup
from modules.data_processing import load_block_paces, get_max_block_height, get_last_tx_fee_time
from modules.economics import build_shock_profile
from modules.network import calc_core_hashrate

# -------------------------------------------------------------------
# 1. Load historical data & PRE-CALCULATE HASHRATE
# -------------------------------------------------------------------
market_prices = load_market_prices("../data/market_price_min.csv") 
real_paces = load_block_paces(BLOCK_PACE_DATA, TX_BLOCK_DATA)
max_hist_block = get_max_block_height(BLOCK_PACE_DATA)

# --- CRITICAL FIX: Pre-calculate H_hist exactly like main.py ---
print("Pre-calculating historical hashrate baseline...")
hist_df = real_paces.copy()
hist_df = hist_df.dropna(subset=["Target", "Inter_Block_Interval_Seconds"])
hist_df = hist_df.sort_index()
hist_df["Target"] = hist_df["Target"].astype(float)

d_hist = (DEFAULT_TARGET / hist_df["Target"].to_numpy())
t_hist = hist_df["Inter_Block_Interval_Seconds"].to_numpy()

# Calculate Core-style hashrate over the full history (in H/s)
H_hist_full = calc_core_hashrate(d_hist, t_hist, window=2016)

# Create a Series indexed by height (integers) containing RAW H/s
# We will convert to TH/s inside objective_function to match the sim output
H_hist_series = pd.Series(H_hist_full, index=hist_df.index.astype(int), name="hist_hashrate")
print("Baseline calculated.")

# -------------------------------------------------------------------
# 2. Simulation Controls
# -------------------------------------------------------------------
num_steps = 888_300  # or your preferred step count

# -------------------------------------------------------------------
# 3. Initial simulation state
# -------------------------------------------------------------------
initial_state = {
    'R_block': R_BLOCK,
    'target': DEFAULT_TARGET,
    'D': INITIAL_DIFFICULTY,
    'H': INITIAL_HASHRATE,
    'sim_timestamp': datetime(2009, 1, 3, 13, 15, 5),
    'block_height': 0,
    'Cumulative_Seconds': 0,
    'difficulty_adjusted': False,
    'R_pe': 0.0,
    'efficiency': 1636007 * 1e-12,
    'R_t': 1.1641354547009541e-08,
    'P': INITIAL_HASHRATE * 1636007 * 1e-12,
    'E': 0.0,
    'P_USD': 0.0,
    'TX_fee': 0.0,
}

initial_block_height = initial_state['block_height']

# -------------------------------------------------------------------
# 4. Shock profile & Price Lookup
# -------------------------------------------------------------------
last_hist_time = get_last_tx_fee_time() # transaction feed data is the limiting timestamp
price_lookup = build_price_lookup(market_prices)

shock_profile = build_shock_profile(
    max_hist_block=max_hist_block,
    initial_block_height=initial_block_height,
    num_steps=num_steps,
    shock_duration=4 * PERIOD_BLOCKS,
    L=0.40
)

# -------------------------------------------------------------------
# 5. Simulation parameters dict
# -------------------------------------------------------------------
# NOTE: 'S' and 'C_elec' are the MODERN-era parameters (post-T_STAR),
#       'S_0' and 'C_elec_0' are the EARLY-era parameters (pre-T_STAR).
params = {
    'P_USD': P_BTC,
    'C_elec': C_ELEC,          # modern-era electricity cost ($/MWh)
    'C_elec_0': C_ELEC_0,      # early-era electricity cost ($/MWh) - overridden in objective_function
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
    'historical_cutoff': datetime(2010, 7, 17, 0, 0, 0), # start of minute price data
    'last_hist_time': last_hist_time,
    'price_lookup': price_lookup,
    'shock_profile': shock_profile,
    'shock_enabled': False,
    'price_forecaster': build_price_forecaster(
        market_prices,
        model_type    = FORECAST_MODEL,
        anchor_date   = FORECAST_TARGET_DATE,
        anchor_price  = FORECAST_TARGET_PRICE
    ),
    'fix_efficiency': False,
    'fixed_efficiency': None,
    'total_operating': True,
    'calibration_mode': CALIBRATION_MODE,
    'S': S,              # modern-era elasticity
    'S_0': S_0,          # early-era elasticity - overridden in objective_function
}

# -------------------------------------------------------------------
# 6. Run Sensitivity Grid Search (over S_0, C_elec_0)
# -------------------------------------------------------------------
if __name__ == "__main__":

    # Define the grid for EARLY-era parameters
    S0_values = np.linspace(0.0, 0.1, 11)   # 11 points from 0.0 to 0.1
    C0_values = np.linspace(10, 150, 15)    # 15 points from 10 to 150
    grid_results = []

    print(f"Starting Grid Search with {len(S0_values)*len(C0_values)} iterations...")

    for s0_val in S0_values:
        for c0_val in C0_values:
            # Pass the PRE-CALCULATED H_hist_series
            rmsle = objective_function(
                [s0_val, c0_val], 
                initial_state, 
                params, 
                num_steps,
                H_hist_series_input=H_hist_series  # <--- Passed here
            )
        
            grid_results.append({
                'S_0': s0_val, 
                'C_elec_0': c0_val, 
                'RMSLE': rmsle
            })
            print(f"S_0={s0_val:.4f}, C_elec_0={c0_val:.1f} -> RMSLE={rmsle:.4e}")
            gc.collect()

    # Save to analyze later
    df_grid = pd.DataFrame(grid_results)
    output_file = "sensitivity_grid_results_rmsle_pre2018_blocks.csv"
    df_grid.to_csv(output_file, index=False)
    print(f"Grid search complete. Results saved to {output_file}")