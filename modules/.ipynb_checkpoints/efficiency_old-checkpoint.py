# modules/efficiency.py

"""
Dynamic Efficiency Module

This module computes a dynamic mining hardware efficiency (in Joules per hash)
based on real-world miner data from a machine data file with updated column names.
The calculation blends:
  - A Pareto weighted average computed from profitable machines.
  - A regression-based prediction computed using all machines.
  
Enhancements in this version:
  - Caching: Machine data is loaded once per file and reused in subsequent calls.
  - Robustness: Additional error checks around file I/O, column existence, log transformation,
    and regression are implemented to handle edge cases.
  
Required columns in the CSV:
  - "Release": Release date (formatted as "%b %Y")
  - "Power (W)": Power consumption (in Watts)
  - "Hashrate (TH/s)": Hashrate in TH/s
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from config import SHA256_MAX, ALPHA, PARETO_TOP_WEIGHT, PARETO_BOTTOM_WEIGHT, MACHINE_DATA_FILE, ELEC_FRACTION, MOVING_AVERAGE  # For 2**256
from collections import deque

# Global cache to store machine data keyed by file path to avoid repeated I/O.
_machine_data_cache = {}

# Global cache for regression model and t0
_cached_reg_model = None
_cached_reg_t0 = None

# Cache to hold the ratio between last blended efficiency and pure regression
_fallback_scale: float | None = None

# Module‐level one‐time load & sort

# 1) Read once
_df = pd.read_csv(MACHINE_DATA_FILE, parse_dates=['Release'])
_df["Release"] = pd.to_datetime(
    _df["Release"],
    format="%b %Y",     # three‑letter month + space + 4‑digit year
    errors="coerce"     # anything that doesn’t match becomes NaT
)

# 2) Extract arrays
power_W            = _df['Power (W)'].values                          # Watts
hashrate_th        = _df['Hashrate (TH/s)'].values                    # TH/s
release_months     = _df['Release'].values.astype('datetime64[M]')     # month resolution

# 3) Precompute constants
hashrate_hashes    = hashrate_th * 1e12                                # hash/s
cost_per_unit      = (power_W * 24.0 / 1e6)                            # $/day per W→MWh→$
eff_w_per_th       = power_W / hashrate_th                             # W per TH/s
eff_J_per_hash     = eff_w_per_th / 1e12                               # J per hash

# 4) Sort by efficiency (best → worst)
_order              = np.argsort(eff_J_per_hash)
hashrate_hashes    = hashrate_hashes[_order]
cost_per_unit      = cost_per_unit[_order]
efficiency_sorted  = eff_J_per_hash[_order]
release_sorted     = release_months[_order]

# Global cache to hold the last 14 days of daily revenue
_daily_rev_cache = deque(maxlen=14)


def update_daily_revenue_cache(daily_rev_USD: float) -> None:
    """
    Append the latest daily revenue to the global 14-day cache.
    """
    _daily_rev_cache.append(daily_rev_USD)


def get_14d_ma_revenue() -> float | np.ndarray:
    """
    Compute and return the 14-day moving average of daily revenues per machine.
    If fewer than 14 data points exist, average over available points.

    Returns:
        - 0.0 (float) if no cache data is available.
        - NumPy array of average daily revenue per machine otherwise.
    """
    if len(_daily_rev_cache) == 0:
        # No history yet → zero revenue
        return 0.0

    # Sum element-wise over all the arrays in the deque, then divide by count
    # This yields an array of shape (n_machines,)
    return sum(_daily_rev_cache) / len(_daily_rev_cache)

def _load_machine_data(file_path: str) -> pd.DataFrame:
    """
    Load machine data from a CSV file using caching.
    
    If the file has been loaded previously, returns a copy from the cache.
    
    Args:
        file_path (str): Path to the machine data CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing machine data.
        
    Raises:
        ValueError: If reading the CSV file fails.
    """
    global _machine_data_cache
    if file_path in _machine_data_cache:
        return _machine_data_cache[file_path].copy()
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read machine data file '{file_path}': {e}")
    
    # Parse the "Release" column as datetime; invalid dates become NaT.
    df['Release'] = pd.to_datetime(df['Release'])
    # Drop rows with invalid or missing release dates.
    df = df.dropna(subset=['Release'])
    _machine_data_cache[file_path] = df.copy()
    return df.copy()

def fit_and_cache_regression(eff_df: pd.DataFrame):
    """
    Fit a regression model for efficiency prediction using normalized time.
    Uses years elapsed since the earliest release (plus an offset of 1 to avoid log(0)).
    Caches the regression model and t0 for later use.
    """
    global _cached_reg_model, _cached_reg_t0
    df = eff_df.copy()
    df['Release'] = pd.to_datetime(df['Release'], errors='coerce')
    # Remove rows with missing or invalid release dates or efficiency values.
    df = df.dropna(subset=['Release', 'Power (W)', 'Hashrate (TH/s)'])
    # Compute computed efficiency (in W/TH/s)
    df['computed_efficiency'] = df['Power (W)'] / df['Hashrate (TH/s)']
    df = df[df['computed_efficiency'] > 0]
    
    # t₀ is the earliest release date
    t0 = df['Release'].min()
    seconds_per_year = 365.25 * 24 * 3600
    # Normalize time: t = (release_date - t0) / seconds_per_year + 1
    df['t_norm'] = ((df['Release'] - t0).dt.total_seconds() / seconds_per_year) + 1
    # Fit regression on log-transformed normalized time and computed efficiency
    X = np.log(df['t_norm'].values).reshape(-1, 1)
    y = np.log(df['computed_efficiency'].values)
    model = LinearRegression().fit(X, y)
    
    # Cache the model and t0
    _cached_reg_model = model
    _cached_reg_t0 = t0
    return model, t0
    
def compute_pareto_efficiency_for_timestamp(eff_df: pd.DataFrame, target_date: datetime,
                                            R_block: float, target: float, C_elec: float, P_BTC: float, calibrate = False, total_operating = False) -> float:
    """
    Compute the Pareto‐weighted hardware efficiency (in Joules per hash) for a given simulation timestamp.

    This uses module‐level NumPy arrays (loaded and sorted once) to:
      1. Calculate each machine's daily profit margin.
      2. Mask for machines released on or before `target_date` and profitable.
      3. Compute a Pareto average of their efficiencies (best machines weighted by PARETO_TOP_WEIGHT).
      4. Fallback to all released machines if none are profitable.

    Args:
        eff_df: Unused placeholder (retained for signature compatibility).
        target_date (datetime): Simulation timestamp to select machines released ≤ this date.
        R_block (float): Current block subsidy in BTC.
        target (float): Current network target (dimensionless).
        C_elec (float): Electricity cost in USD per MWh.
        P_BTC (float): Bitcoin price in USD.

    Returns:
        float: Pareto‐weighted efficiency in Joules per hash.

    Raises:
        ValueError: If no machines have been released by `target_date` (i.e., no data to compute efficiency).
    """
    # Scalar expected revenue per hash (BTC/hash)
    exp_rev_hash = (R_block * target) / 2**256

    # Vectorized profit margin ($/day) per machine
    daily_rev_USD      = exp_rev_hash * hashrate_hashes * 86400 * P_BTC

    if MOVING_AVERAGE:
        # Update the revenue cache with today's revenue
        update_daily_revenue_cache(daily_rev_USD)
        # Compute the 14-day moving average revenue
        daily_rev_USD = get_14d_ma_revenue()
        
    operating_cost_USD = cost_per_unit * C_elec
    
    if total_operating:
        operating_cost_USD = operating_cost_USD / ELEC_FRACTION # provides the total daily operating cost not just electricity cost
        
    profit_margin      = daily_rev_USD - operating_cost_USD

    # Mask for machines released by this month and profitable
    month_key   = np.datetime64(target_date, 'M')
    ten_years_ago = month_key - np.timedelta64(120, 'M')  # 120 months = 10 years
    alive_mask     = (release_sorted >= ten_years_ago) & (release_sorted <= month_key) # only machines released in [ten_years_ago, month_key]
    profit_mask = (profit_margin > 0)
    mask        = alive_mask & profit_mask

    
    # Fallback if P_BTC = 0 (early discovery years)
    if P_BTC == 0:
        mask = alive_mask
    # Fallback if no profitable machines
    if not mask.any() and calibrate:
        mask = alive_mask
    if not mask.any() and not calibrate:
        # raise ValueError(f"No machines released on or before {target_date}")
        # No machines within the last 10 years → signal via NaN
        return np.nan

    # Select sorted efficiencies via mask
    effs = efficiency_sorted[mask]
    n    = len(effs)
    n_top = int(n * PARETO_TOP_WEIGHT)

    if n_top == 0:
        pareto = effs.mean()
    else:
        top_mean    = effs[:n_top].mean()
        bottom_mean = effs[n_top:].mean() if n > n_top else top_mean
        pareto      = PARETO_TOP_WEIGHT * top_mean + PARETO_BOTTOM_WEIGHT * bottom_mean
    return pareto

def compute_dynamic_efficiency(file_path: str, target_date: datetime,
                               R_block: float, target: float, C_elec: float, P_BTC: float, total_operating: bool) -> float:
    """
    Compute dynamic machine efficiency (Joules per hash) by blending Pareto and regression estimates.

    Steps:
      1. Call `compute_pareto_efficiency_for_timestamp` for the Pareto component.
      2. Fit (once) or reuse a log‐log regression model of efficiency vs. release date.
      3. Predict regression efficiency for `target_date`.
      4. Blend: `alpha*pareto + (1-alpha)*regression`, where alpha=ALPHA.

    Args:
        file_path (str): Path to machine data CSV (unused; module‐level arrays used).
        target_date (datetime): Simulation timestamp for efficiency estimation.
        R_block (float): Current block subsidy in BTC.
        target (float): Current network target (dimensionless).
        C_elec (float): Electricity cost in USD per MWh.
        P_BTC (float): Bitcoin price in USD.

    Returns:
        float: Combined dynamic efficiency in Joules per hash.

    Raises:
        ValueError: If the regression model cannot be fit or `compute_pareto_efficiency_for_timestamp` fails.
    """
    global _cached_reg_model, _cached_reg_t0, _fallback_scale

    # Pareto component
    pareto_eff = compute_pareto_efficiency_for_timestamp(
        None, target_date, R_block, target, C_elec, P_BTC, False, total_operating
    )

    # Regression component caching
    if _cached_reg_model is None or _cached_reg_t0 is None:
        try:
            df_full = pd.read_csv(MACHINE_DATA_FILE, parse_dates=['Release'])
            df_full["Release"] = pd.to_datetime(df_full["Release"], format="%b %Y", errors="coerce")
            _cached_reg_model, _cached_reg_t0 = fit_and_cache_regression(df_full)
        except Exception as e:
            raise ValueError(f"Failed to fit regression model: {e}")

    # Normalize time in years since first release
    seconds_per_year = 365.25 * 24 * 3600
    delta_secs = (pd.to_datetime(target_date) - _cached_reg_t0).total_seconds()
    t_norm = (delta_secs / seconds_per_year) + 1

    # Predict log‐efficiency and convert to J/hash
    log_pred = _cached_reg_model.predict(np.array([[np.log(t_norm)]]))
    reg_eff  = np.exp(log_pred[0]) / 1e12

    # If no machines are available (i.e. pareto_eff is NaN), revert to regression‐only
    if np.isnan(pareto_eff):
        if _fallback_scale is None:
            # First time past cutoff: just use regression
            return reg_eff
            
        # Subsequent times: apply the cached scale for continuity
        return reg_eff * _fallback_scale

    # Blend
    blended = ALPHA * pareto_eff + (1 - ALPHA) * reg_eff
    
    # Cache ratio so future regression calls start from this blended value
    _fallback_scale = blended / reg_eff
    
    return blended