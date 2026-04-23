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
  - Performance: Efficiency is recomputed only once every 14 days (aligned to a fixed epoch)
    and reused for all blocks within that 14-day window. The 14-day revenue moving average
    is advanced once per calendar day.

Required columns in the CSV:
  - "Release": Release date (formatted as "%b %Y")
  - "Power (W)": Power consumption (in Watts)
  - "Hashrate (TH/s)": Hashrate in TH/s
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
from config import (
    SHA256_MAX, ALPHA, PARETO_TOP_WEIGHT, PARETO_BOTTOM_WEIGHT,
    MACHINE_DATA_FILE, ELEC_FRACTION, MOVING_AVERAGE
)
from collections import deque

# -----------------------------------------------------------------------------
# Module-level caches and constants
# -----------------------------------------------------------------------------

# Global cache to store machine data keyed by file path to avoid repeated I/O.
_machine_data_cache = {}

# Global cache for regression model and t0
_cached_reg_model = None
_cached_reg_t0 = None

# Cache to hold the ratio between last blended efficiency and pure regression
_fallback_scale: float | None = None

# Efficiency cache: map from 14-day period index -> blended efficiency (J/hash)
_eff_cache_14d: dict[int, float] = {}

# Track the last calendar day for which we pushed daily revenue into the 14-day deque
_last_rev_day: date | None = None

# Define a fixed epoch to align 14-day windows deterministically (Bitcoin genesis)
_EFF_EPOCH_DAY: date = date(2009, 1, 3)

# -----------------------------------------------------------------------------
# One-time machine table load & preprocessing
# -----------------------------------------------------------------------------

# 1) Read once
_df = pd.read_csv(MACHINE_DATA_FILE, parse_dates=['Release'])
_df["Release"] = pd.to_datetime(
    _df["Release"],
    format="%b %Y",     # three-letter month + space + 4-digit year
    errors="coerce"     # anything that doesn’t match becomes NaT
)

# 2) Extract arrays
power_W            = _df['Power (W)'].values                           # Watts
hashrate_th        = _df['Hashrate (TH/s)'].values                     # TH/s
release_months     = _df['Release'].values.astype('datetime64[M]')     # month resolution

# 3) Precompute constants
hashrate_hashes    = hashrate_th * 1e12                                # hash/s
# electricity cost units: we’ll compute $/day from power (W) using 24h and convert to MWh
cost_per_unit      = (power_W * 24.0 / 1e6)                            # MWh/day-equivalent factor (multiply by $/MWh)
eff_w_per_th       = power_W / hashrate_th                             # W per TH/s
eff_J_per_hash     = eff_w_per_th / 1e12                               # J per hash

# 4) Sort by efficiency (best → worst)
_order              = np.argsort(eff_J_per_hash)
hashrate_hashes    = hashrate_hashes[_order]
cost_per_unit      = cost_per_unit[_order]
efficiency_sorted  = eff_J_per_hash[_order]
release_sorted     = release_months[_order]

# Global cache to hold the last 14 days of daily revenue (per machine, vectorized)
_daily_rev_cache = deque(maxlen=14)


# -----------------------------------------------------------------------------
# Helpers for 14-day MA and period indexing
# -----------------------------------------------------------------------------
def update_daily_revenue_cache(daily_rev_USD: np.ndarray) -> None:
    """
    Append the latest (vectorized per-machine) daily revenue to the global 14-day cache.
    """
    _daily_rev_cache.append(daily_rev_USD)


def get_14d_ma_revenue() -> float | np.ndarray:
    """
    Compute and return the 14-day moving average of daily revenues per machine.
    If fewer than 14 data points exist, average over available points.

    Returns:
        - 0.0 (float) if no cache data is available (early periods).
        - NumPy array of average daily revenue per machine otherwise.
    """
    if len(_daily_rev_cache) == 0:
        return 0.0
    return sum(_daily_rev_cache) / len(_daily_rev_cache)


def _period_index_14d(dt: datetime) -> int:
    """
    Map a datetime to an integer 14-day period index, aligned to _EFF_EPOCH_DAY.
    All dates in the same 14-day window share the same index.
    """
    d = pd.to_datetime(dt).date()
    days_since_epoch = (d.toordinal() - _EFF_EPOCH_DAY.toordinal())
    return days_since_epoch // 14


# -----------------------------------------------------------------------------
# Optional: load machine data via file cache (signature kept for compatibility)
# -----------------------------------------------------------------------------
def _load_machine_data(file_path: str) -> pd.DataFrame:
    """
    Load machine data from a CSV file using caching.
    """
    global _machine_data_cache
    if file_path in _machine_data_cache:
        return _machine_data_cache[file_path].copy()
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read machine data file '{file_path}': {e}")
    
    df['Release'] = pd.to_datetime(df['Release'])
    df = df.dropna(subset=['Release'])
    _machine_data_cache[file_path] = df.copy()
    return df.copy()


# -----------------------------------------------------------------------------
# Regression fit (cached)
# -----------------------------------------------------------------------------
def fit_and_cache_regression(eff_df: pd.DataFrame):
    """
    Fit a regression model for efficiency prediction using normalized time.
    Uses years elapsed since the earliest release (plus an offset of 1 to avoid log(0)).
    Caches the regression model and t0 for later use.
    """
    global _cached_reg_model, _cached_reg_t0
    df = eff_df.copy()
    df['Release'] = pd.to_datetime(df['Release'], errors='coerce')
    df = df.dropna(subset=['Release', 'Power (W)', 'Hashrate (TH/s)'])

    # Computed efficiency in W/TH/s
    df['computed_efficiency'] = df['Power (W)'] / df['Hashrate (TH/s)']
    df = df[df['computed_efficiency'] > 0]
    
    # t₀ is the earliest release date
    t0 = df['Release'].min()
    seconds_per_year = 365.25 * 24 * 3600

    # Normalize time: t = (release_date - t0)/year + 1
    df['t_norm'] = ((df['Release'] - t0).dt.total_seconds() / seconds_per_year) + 1

    # Fit regression on log-transformed normalized time and computed efficiency
    X = np.log(df['t_norm'].values).reshape(-1, 1)
    y = np.log(df['computed_efficiency'].values)
    model = LinearRegression().fit(X, y)
    
    _cached_reg_model = model
    _cached_reg_t0 = t0
    return model, t0


# -----------------------------------------------------------------------------
# Pareto efficiency at a timestamp (vectorized across machines)
# -----------------------------------------------------------------------------
def compute_pareto_efficiency_for_timestamp(
    eff_df: pd.DataFrame,
    target_date: datetime,
    R_block: float,
    target: float,
    C_elec: float,
    P_BTC: float,
    calibrate: bool = False,
    total_operating: bool = False
) -> float:
    """
    Compute the Pareto‐weighted hardware efficiency (in Joules per hash) for a given simulation timestamp.

    Steps:
      1) Calculate each machine's daily revenue ($/day).
      2) If MOVING_AVERAGE: use 14-day MA of daily revenue (updated once per *calendar day*).
      3) Subtract operating cost to get daily profit margin.
      4) Filter to released & profitable machines (or fallback).
      5) Weighted average (PARETO_TOP_WEIGHT / PARETO_BOTTOM_WEIGHT) of efficiencies.

    Returns: Pareto‐weighted efficiency in J/hash (float).
    """
    # Expected revenue per hash (BTC/hash); target is dimensionless
    exp_rev_hash = (R_block * target) / 2**256

    # Vectorized daily revenue per machine ($/day)
    daily_rev_USD = exp_rev_hash * hashrate_hashes * 86400.0 * P_BTC

    # Update the revenue deque *once per calendar day* and then use the 14-day MA
    if MOVING_AVERAGE:
        day = pd.to_datetime(target_date).date()
        global _last_rev_day
        if _last_rev_day != day:
            update_daily_revenue_cache(daily_rev_USD)
            _last_rev_day = day
        daily_rev_USD = get_14d_ma_revenue()

    # Operating cost ($/day) per machine (electricity only by default)
    operating_cost_USD = cost_per_unit * C_elec
    if total_operating:
        # Scale to approximate total operating cost (not just electricity)
        operating_cost_USD = operating_cost_USD / ELEC_FRACTION

    profit_margin = daily_rev_USD - operating_cost_USD

    # Keep only machines released in the last 10 years and no later than this month
    month_key      = np.datetime64(target_date, 'M')
    ten_years_ago  = month_key - np.timedelta64(120, 'M')
    alive_mask     = (release_sorted >= ten_years_ago) & (release_sorted <= month_key)

    # Profitability mask
    profit_mask = (profit_margin > 0)
    mask        = alive_mask & profit_mask

    # Fallback if P_BTC = 0 (early discovery years): ignore profitability, only release gate
    if P_BTC == 0:
        mask = alive_mask

    # If still nothing and calibrate=True: fallback to all released machines
    if not mask.any() and calibrate:
        mask = alive_mask

    # If still nothing and not calibrating: return NaN (signals regression fallback)
    if not mask.any() and not calibrate:
        return np.nan

    # Pareto blend among selected machines (arrays are already sorted by efficiency ascending)
    effs  = efficiency_sorted[mask]
    n     = len(effs)
    n_top = int(n * PARETO_TOP_WEIGHT)

    if n_top == 0:
        pareto = effs.mean()
    else:
        top_mean    = effs[:n_top].mean()
        bottom_mean = effs[n_top:].mean() if n > n_top else top_mean
        pareto      = PARETO_TOP_WEIGHT * top_mean + PARETO_BOTTOM_WEIGHT * bottom_mean
    return pareto


# -----------------------------------------------------------------------------
# Main entry: dynamic efficiency with 14-day recompute cadence
# -----------------------------------------------------------------------------
def compute_dynamic_efficiency(
    file_path: str,
    target_date: datetime,
    R_block: float,
    target: float,
    C_elec: float,
    P_BTC: float,
    total_operating: bool
) -> float:
    """
    Compute dynamic machine efficiency (Joules per hash) by blending Pareto and regression estimates.

    Performance policy:
      - Update the 14-day revenue deque once per *calendar day* (for MA correctness).
      - Recompute efficiency only when we enter a *new 14-day period* (epoch-aligned).
      - Otherwise, return the cached efficiency for the current 14-day index.

    Steps at recompute:
      1. Call `compute_pareto_efficiency_for_timestamp` for the Pareto component.
      2. Fit (once) or reuse a log-log regression model of efficiency vs. release date.
      3. Predict regression efficiency for `target_date`.
      4. Blend: `ALPHA*pareto + (1-ALPHA)*regression`.

    Returns:
        float: Combined dynamic efficiency in Joules per hash.

    Raises:
        ValueError: If the regression model cannot be fit.
    """
    global _cached_reg_model, _cached_reg_t0, _fallback_scale

    # Always determine the current 14-day period index
    idx_14d = _period_index_14d(target_date)

    # If we already have efficiency for this 14-day window, return it immediately.
    if idx_14d in _eff_cache_14d:
        return _eff_cache_14d[idx_14d]

    # --- Recompute path (first call within this 14-day window) ---

    # Pareto component (uses 14-day MA revenue which is advanced once/day)
    pareto_eff = compute_pareto_efficiency_for_timestamp(
        None, target_date, R_block, target, C_elec, P_BTC, False, total_operating
    )

    # Regression component (fit once, reuse thereafter)
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

    # Predict log-efficiency and convert to J/hash
    log_pred = _cached_reg_model.predict(np.array([[np.log(t_norm)]]))
    reg_eff  = np.exp(log_pred[0]) / 1e12

    # If no machines are available (i.e., pareto_eff NaN), use regression-only with continuity
    if np.isnan(pareto_eff):
        if _fallback_scale is None:
            blended = reg_eff
        else:
            blended = reg_eff * _fallback_scale
        _eff_cache_14d[idx_14d] = blended
        return blended

    # Blend the two
    blended = ALPHA * pareto_eff + (1 - ALPHA) * reg_eff

    # Cache scale so future regression-only calls remain continuous
    _fallback_scale = blended / reg_eff

    # Store result for this 14-day period
    _eff_cache_14d[idx_14d] = blended
    return blended
