# modules/efficiency_cbeci.py
"""
CBECI-style Efficiency Module (Paper 3 -- CBECI Efficiency)

Implements the instructions in "Paper 3 -- CBECI Efficiency.pdf".

Inputs
------
• Machine table: cbeci_machines_090325.csv
    Required columns (case-sensitive):
        - "Power (W)"
        - "Hashing Power (TH/s)"
        - "Efficiency (J/Gh)"
        - "Miner_name"
        - "Type"
        - "Date of release"    (DD/MM/YYYY; used where UNIX_date_of_release is blank)
        - "UNIX_date_of_release"
        - "Weight in kg"
    Hardware released on or after July 2014 is kept only for Bitmain, MicroBT and
    Canaan (CBECI v1.2.0+); earlier hardware is kept regardless of manufacturer.

Core ideas from the PDF
-----------------------
1) Daily revenue uses a 14-day moving average.
2) Define months-since-deployment for each machine with a fixed deployment lag
   (Release + LAG_MONTHS).
3) Use a *piecewise* age weight w_i(M_i):
        1.0 if   0 <= M < 12
        0.8 if  12 <= M < 24
        0.6 if  24 <= M < 36
        0.4 if  36 <= M < 48
        0.2 if  48 <= M < 60
        0.0 if M < 0 or M > 60
4) Profitability threshold Θ_d(P) = R_d / P (per the PDF text).
   - R_d is in $ / (TH·day).
   - P (electricity price) is converted to a unit that makes Θ have units J/TH.
5) Select the profitable set S_d(P) = { machines i : η_i <= Θ_d(P) }.
   If S_d(P) is empty for day d, use the most recent *non-empty* set from a prior day.
6) Weighted average efficiency ψ_d = Σ_i (η_i * w_i,d) where w_i,d are normalized weights.
7) For timestamps > historical cutoff (forecast), include an *exponential model* machine
   that represents the most recent release efficiency. Always give this model-entry weight=1.

The function exported here matches the existing simulation signature
and returns efficiency in J / hash.
"""

from __future__ import annotations

import math
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, date
from collections import deque
from modules.network import calc_core_hashrate
import math
from datetime import datetime
from scipy.optimize import curve_fit

from config import EFFICIENCY_SCENARIO, T_STAR, ELEC_FRACTION, USE_FRACTION


# -------------------------
# Tunables (config-friendly)
# -------------------------

# Two-month deployment lag per PDF (can be overridden via config import if desired)
DEPLOYMENT_LAG_MONTHS = 2

# CBECI v1.2.0+: only Bitmain, MicroBT and Canaan hardware released on or after
# July 2014 is considered. Earlier hardware (CPU/GPU/FPGA/early ASIC) is kept.
CBECI_MANUFACTURERS = ("Bitmain", "MicroBT", "Canaan")
CBECI_MANUFACTURER_FILTER_FROM = pd.Timestamp("2014-07-01")

# Age-weight buckets (months, weight)
# Implements the piecewise function shown in the PDF's image [cite: 41-54]
AGE_WEIGHTS = [
    (0,   12, 1.0),
    (12,  24, 0.8),
    (24,  36, 0.6),
    (36,  48, 0.4),
    (48,  60, 0.2),
]

# 14-day daily-revenue moving average [cite: 20]
REVENUE_MA_DAYS = 14

# Fixed epoch to align day tracking (any stable date works)
EPOCH_DAY = date(2009, 1, 3)

# --------------------------------
# Module-level caches / state
# --------------------------------

_machine_df: pd.DataFrame | None = None
_prev_nonempty_set_mask: np.ndarray | None = None  # last non-empty S_d(P) mask
_daily_revenue_ma: deque = deque(maxlen=REVENUE_MA_DAYS)
## MODIFICATION: Added a new deque to store the 14-day MA of the *threshold*
_daily_threshold_ma: deque = deque(maxlen=REVENUE_MA_DAYS)
_last_rev_day: date | None = None

# Exponential efficiency model cache
_model_fitted: bool = False
_model_params: tuple[float, float] | None = None    # (A, b) for η_TH(t) = A * exp(b * t_years)
_model_t0: pd.Timestamp | None = None               # earliest release
_model_forecast_cache: dict[int, float] = {}        # day_index -> eta_TH

# -------------------------
# Utility helpers
# -------------------------

# def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
#     """Fractional months between two timestamps."""
#     days = (end - start).days
#     return days / (365.25 / 12.0)
def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Whole calendar months between two dates; negative if end < start."""
    if pd.isna(start) or pd.isna(end):
        return float("nan")
    y = end.year - start.year
    m = end.month - start.month
    months = y * 12 + m
    # If day-of-month hasn't been reached yet, subtract one month
    if end.day < start.day:
        months -= 1
    return float(months)

# def _deployment_date(release_ts: pd.Timestamp) -> pd.Timestamp:
#     """Release + lag months."""
#     days = int(round(DEPLOYMENT_LAG_MONTHS * (365.25 / 12.0)))
#     return release_ts + pd.Timedelta(days=days)

def _deployment_date(release_ts: pd.Timestamp) -> pd.Timestamp:
    """Release + lag months (true calendar months)."""
    return release_ts + pd.DateOffset(months=DEPLOYMENT_LAG_MONTHS)    

def _age_weight(M_months: float) -> float:
    """Piecewise weight w_i(M_i) from the PDF [cite: 41-54]."""
    if M_months < 0:
        return 0.0
    for lo, hi, w in AGE_WEIGHTS:
        if lo <= M_months < hi:
            return w
    return 0.0

def _day_index(dt: datetime) -> int:
    """Integer day index from fixed epoch (for simple caches)."""
    d = pd.to_datetime(dt).date()
    return (d.toordinal() - EPOCH_DAY.toordinal())

# --------------------------------
# Machine table loading & prep
# --------------------------------

def _load_machines(csv_path: str) -> pd.DataFrame:
    """
    Load and normalize the CBECI machine table.
    Ensures columns exist, parses dates, and builds convenience fields.
    """
    df = pd.read_csv(csv_path)
    # **Correction 1: Added missing columns to the required list per PDF.**
    required = ["Power (W)", "Hashing Power (TH/s)", "Efficiency (J/Gh)",
                "Miner_name", "Type", "Date of release", "UNIX_date_of_release", "Weight in kg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    # Parse dates. The UNIX date is used where present because it is unambiguous.
    # The fallback is day-first because the file is DD/MM/YYYY, and day-first
    # agrees with the UNIX date on every row that has both.
    unix_txt = df["UNIX_date_of_release"].astype("string").str.strip()
    unix_txt = unix_txt.mask(unix_txt == "")
    unix_num = pd.to_numeric(unix_txt.str.replace(",", "", regex=False),
                             errors="coerce").astype(float)
    bad = unix_txt.notna().to_numpy() & ~np.isfinite(unix_num.to_numpy())
    if bad.any():
        offenders = list(zip(df.loc[bad, "Miner_name"], unix_txt[bad]))
        raise ValueError(f"Unparseable UNIX_date_of_release in {csv_path} "
                         f"(Miner_name, value): {offenders}")
    unix_dt = pd.to_datetime(unix_num, unit="s")

    dmy_txt = df["Date of release"].astype("string").str.strip()
    dmy_txt = dmy_txt.mask(dmy_txt == "")
    dmy_dt = pd.to_datetime(dmy_txt, format="%d/%m/%Y", errors="coerce")
    bad = (dmy_txt.notna() & dmy_dt.isna()).to_numpy()
    if bad.any():
        offenders = list(zip(df.loc[bad, "Miner_name"], dmy_txt[bad]))
        raise ValueError(f"Date of release not DD/MM/YYYY in {csv_path} "
                         f"(Miner_name, value): {offenders}")

    both = (unix_dt.notna() & dmy_dt.notna()).to_numpy()
    disagree = both & ((unix_dt - dmy_dt).abs() > pd.Timedelta(days=1)).to_numpy()
    if disagree.any():
        offenders = [(n, d.date(), u) for n, d, u in
                     zip(df.loc[disagree, "Miner_name"], dmy_dt[disagree], unix_dt[disagree])]
        raise ValueError(f"Day-first date and UNIX date differ by more than 1 day in "
                         f"{csv_path} (Miner_name, day-first date, UNIX date): {offenders}")

    df["Date of release"] = unix_dt.fillna(dmy_dt)
    df = df.dropna(subset=["Date of release"])

    # CBECI v1.2.0+ manufacturer filter (maker = first token of Miner_name).
    maker = df["Miner_name"].astype("string").str.strip().str.split().str[0]
    drop = (df["Date of release"] >= CBECI_MANUFACTURER_FILTER_FROM) & ~maker.isin(CBECI_MANUFACTURERS)
    df = df.loc[~drop.to_numpy()]

    # Normalize fields
    df = df.rename(columns={
        "Power (W)": "power_W",
        "Hashing Power (TH/s)": "hashrate_THs",
        "Efficiency (J/Gh)": "eff_J_per_Gh",
        "Date of release": "release_ts",
        "Miner_name": "miner_name",
        "Type": "type",
    })

    # Efficiency conversions
    df["eta_TH_J_per_TH"] = df["eff_J_per_Gh"].astype(float) * 1000.0
    df["eta_hash_J_per_hash"] = df["eff_J_per_Gh"].astype(float) / 1e9

    # Sanity fallback: if "eff_J_per_Gh" is missing/zero, derive from power/hashrate
    bad_mask = ~np.isfinite(df["eff_J_per_Gh"]) | (df["eff_J_per_Gh"] <= 0)
    if bad_mask.any():
        eta_j_per_hash = (df.loc[bad_mask, "power_W"].astype(float) /
                          (df.loc[bad_mask, "hashrate_THs"].astype(float) * 1e12))
        df.loc[bad_mask, "eta_hash_J_per_hash"] = eta_j_per_hash
        df.loc[bad_mask, "eta_TH_J_per_TH"] = eta_j_per_hash * 1e12
        df.loc[bad_mask, "eff_J_per_Gh"] = eta_j_per_hash * 1e9

    # Deployment date = release + lag [cite: 39]
    df["deployment_ts"] = df["release_ts"].apply(_deployment_date)
    df = df.sort_values("release_ts").reset_index(drop=True)
    return df


# # --------------------------------
# # Logistic model for newest tech
# # --------------------------------

# def _logistic_eff_model(t_years: np.ndarray, eta_low: float, eta_high: float, k: float, t_mid: float) -> np.ndarray:
#     """
#     Logistic frontier efficiency model (J/TH) as a *decreasing* function of time:

#         η_frontier(t) = η_low + (η_high - η_low) / (1 + exp(k * (t - t_mid)))

#     - eta_high: asymptotic *past* (inefficient) level
#     - eta_low : asymptotic *future* (efficient) level
#     - k       : >0, controls speed of transition
#     - t_mid   : inflection point (in years since t0)
#     """
#     return eta_low + (eta_high - eta_low) / (1.0 + np.exp(k * (t_years - t_mid)))


# def _fit_exponential_model(df: pd.DataFrame):
#     """
#     Fit a *logistic* frontier efficiency curve η_TH(t) in J/TH across historical
#     machine release dates.

#     We keep the old name for backwards compatibility, but the model is:

#         η_frontier(t) = η_low + (η_high - η_low) / (1 + exp(k * (t - t_mid)))

#     with t in years since the first release.
#     """
#     global _model_fitted, _model_params, _model_t0
#     if _model_fitted:
#         return

#     # Time axis in years since first release
#     t0 = df["release_ts"].min()
#     sec_per_year = 365.25 * 24 * 3600.0
#     t_years = ((df["release_ts"] - t0).dt.total_seconds()) / sec_per_year
#     y = np.clip(df["eta_TH_J_per_TH"].astype(float).to_numpy(), 1e-9, np.inf)

#     X = t_years.to_numpy()

#     # Initial guesses for parameters
#     eta_high0 = float(y.max())
#     eta_low0  = float(y.min())
#     k0        = 0.5
#     t_mid0    = float(X.mean())

#     p0 = [eta_low0, eta_high0, k0, t_mid0]

#     # Bounds: keep everything positive and k reasonably sized
#     bounds = (
#         [0.0, 0.0, 0.0, X.min() - 5.0],   # lower
#         [np.inf, np.inf, 10.0, X.max() + 5.0]  # upper
#     )

#     try:
#         params, _ = curve_fit(
#             _logistic_eff_model,
#             X,
#             y,
#             p0=p0,
#             bounds=bounds,
#             maxfev=20000,
#         )
#         eta_low, eta_high, k, t_mid = params

#         # Enforce a *decreasing* frontier: eta_high should represent the past (worse),
#         # eta_low should represent the future (better).
#         # curve_fit does not enforce eta_low < eta_high, so swap if necessary.
#         if eta_low > eta_high:
#             eta_low, eta_high = eta_high, eta_low

#     except Exception as e:
#         # Fallback: no fit; mark model as unavailable
#         _model_fitted = False
#         _model_params = None
#         _model_t0 = None
#         return

#     _model_params = (float(eta_low), float(eta_high), float(k), float(t_mid))
#     _model_t0 = pd.to_datetime(t0)
#     _model_fitted = True

# def _model_eta_TH_on_date(dt: datetime) -> float:
#     """
#     Predict J/TH at datetime dt using cached *logistic* frontier model.
#     """
#     global _model_params, _model_t0, _model_forecast_cache
#     if not _model_fitted or _model_params is None or _model_t0 is None:
#         raise RuntimeError("Frontier model not fitted yet. Call _fit_exponential_model first.")
    
#     idx = _day_index(dt)
#     if idx in _model_forecast_cache:
#         return _model_forecast_cache[idx]
    
#     eta_low, eta_high, k, t_mid = _model_params
#     sec_per_year = 365.25 * 24 * 3600.0
#     t_years = (pd.to_datetime(dt) - _model_t0).total_seconds() / sec_per_year
    
#     eta_TH = float(_logistic_eff_model(np.array([t_years], dtype=float), eta_low, eta_high, k, t_mid)[0])
#     _model_forecast_cache[idx] = eta_TH
#     return eta_TH

# --------------------------------
# Exponential model for newest tech
# --------------------------------

# --------------------------------
# Power-law model for newest tech
# --------------------------------

# --------------------------------
# Power-law model for newest tech
# --------------------------------

def _powerlaw_eff_model(
    t_years: np.ndarray,
    eta_floor: float,
    A: float,
    p: float,
    t_shift: float,
) -> np.ndarray:
    """
    Power-law frontier efficiency model (J/TH) as a *decreasing* function of time:

        η_frontier(t) = η_floor + A * (t_years + t_shift)^(-p)

    Constraints for a decreasing, well-behaved curve:
      - eta_floor >= 0  : asymptotic *future* (best) efficiency floor (J/TH)
      - A >= 0          : positive scale
      - p > 0           : decay exponent
      - t_shift > 0     : avoids singularity at t=0

    t_years is years since the first machine release date used for fitting.
    """
    t = np.asarray(t_years, dtype=float)
    z = np.maximum(t + float(t_shift), 1e-12)
    return float(eta_floor) + float(A) * np.power(z, -float(p))


def _log_powerlaw_eff_model(t_years, eta_floor, A, p, t_shift):
    """log of _powerlaw_eff_model; the fit target is log(eta) (see _fit_exponential_model)."""
    return np.log(_powerlaw_eff_model(t_years, eta_floor, A, p, t_shift))


def _fit_exponential_model(df: pd.DataFrame):
    """
    Fit a *power-law* frontier efficiency curve η_TH(t) (J/TH) across historical
    machine release dates.

    NOTE: We keep the function name for backwards compatibility.

    Key change vs naive fit:
      - Fit ONLY to the observed frontier (lower envelope) of machine efficiencies.
      - Up-weight recent frontier points so the fit matches modern efficiencies.
      - Fit in LOG space: the residual is log(model) - log(frontier), so every
        frontier point counts in relative terms. The frontier spans about seven
        orders of magnitude (14,313,725 J/TH for a 2009 CPU to 9.5 J/TH in 2026);
        a least-squares fit in J/TH lets the 2009-2011 points supply almost all of
        the error and leaves the modern end of the curve far too high.

        η_frontier(t) = η_floor + A * (t_years + t_shift)^(-p)
    """
    global _model_fitted, _model_params, _model_t0, _model_forecast_cache
    if _model_fitted:
        return

    # ---- Clean / extract ----
    df2 = df.copy()
    df2["release_ts"] = pd.to_datetime(df2["release_ts"], errors="coerce")
    df2["eta_TH_J_per_TH"] = pd.to_numeric(df2["eta_TH_J_per_TH"], errors="coerce")
    df2 = df2.dropna(subset=["release_ts", "eta_TH_J_per_TH"]).copy()

    if df2.empty:
        _model_fitted = False
        _model_params = None
        _model_t0 = None
        return

    # ---- Time axis in years since first release ----
    t0 = df2["release_ts"].min()
    sec_per_year = 365.25 * 24 * 3600.0
    df2["t_years"] = ((df2["release_ts"] - t0).dt.total_seconds()) / sec_per_year

    # ---- Build observed frontier (lower envelope) ----
    # Step 1: compress multiple releases on same day (or timestamp) to the best (min eta)
    # Using date (day) is typically sufficient and avoids over-weighting days with many rows.
    df2["release_day"] = df2["release_ts"].dt.floor("D")
    by_day = (
        df2.groupby("release_day", as_index=False)
           .agg(t_years=("t_years", "min"), eta=("eta_TH_J_per_TH", "min"))
           .sort_values("t_years")
    )

    X = by_day["t_years"].to_numpy(dtype=float)
    y = by_day["eta"].to_numpy(dtype=float)

    # Ensure strictly positive
    y = np.clip(y, 1e-9, np.inf)

    # Step 2: enforce non-increasing frontier over time (cumulative minimum)
    # (Frontier should never get worse as time advances.)
    y_frontier = np.minimum.accumulate(y)

    # If the last part of the frontier is flat due to data quirks, that's fine; the weights handle it.

    if X.size < 6:
        _model_fitted = False
        _model_params = None
        _model_t0 = None
        return

    # ---- Recency weighting (force fit to modern efficiencies) ----
    # Larger weights for later times. curve_fit uses sigma: smaller sigma => higher weight.
    # beta controls how strongly recent points dominate (try 4–8 if still not matching modern end).
    beta = 6.0
    t_norm = (X - X.min()) / max(X.max() - X.min(), 1e-12)  # in [0,1]
    w = np.exp(beta * t_norm)                                # in [1, e^beta]
    sigma = 1.0 / np.sqrt(w)                                 # inverse-sqrt weighting

    # ---- Initial guesses (from frontier only) ----
    # floor 1 J/TH, A at the first frontier value, p = 2, t_shift = 0.25 years.
    eta_floor0, A0, p0, t_shift0 = 1.0, float(y_frontier[0]), 2.0, 0.25
    p0_vec = [eta_floor0, A0, p0, t_shift0]

    # ---- Bounds (allow steeper decay than before) ----
    # If you still see a curve that stays too high in the modern era, increasing the p upper bound
    # and/or beta above typically fixes it.
    bounds = (
        [0.0,   0.0,   1e-6, 1e-6],   # lower
        [np.inf, np.inf, 40.0, 5.0],  # upper
    )

    try:
        params, _ = curve_fit(
            _log_powerlaw_eff_model,
            X,
            np.log(y_frontier),
            p0=p0_vec,
            bounds=bounds,
            sigma=sigma,
            absolute_sigma=False,
            maxfev=200000,
        )
        eta_floor, A, p, t_shift = params

        # Numerical guards
        eta_floor = max(float(eta_floor), 0.0)
        A = max(float(A), 0.0)
        p = max(float(p), 1e-6)
        t_shift = max(float(t_shift), 1e-6)

        if not np.all(np.isfinite(params)):
            raise ValueError(f"non-finite fitted parameters {params.tolist()}")

    except Exception as exc:
        # No fallback to the initial guess: it is not a fitted curve (it gives
        # ~47,000 J/TH in 2026). Leave the model unavailable instead, so that a
        # frontier-scenario run fails loudly (see compute_dynamic_efficiency).
        warnings.warn(
            f"Frontier efficiency fit failed ({type(exc).__name__}: {exc}); "
            "the frontier model is unavailable.",
            RuntimeWarning,
        )
        _model_fitted = False
        _model_params = None
        _model_t0 = None
        _model_forecast_cache = {}
        return

    _model_params = (float(eta_floor), float(A), float(p), float(t_shift))
    _model_t0 = pd.to_datetime(t0)
    _model_fitted = True
    _model_forecast_cache = {}  # clear cached forecasts in case of re-fit



def _model_eta_TH_on_date(dt: datetime) -> float:
    """
    Predict J/TH at datetime dt using cached *power-law* frontier model.
    """
    global _model_params, _model_t0, _model_forecast_cache, _model_fitted
    if (not _model_fitted) or (_model_params is None) or (_model_t0 is None):
        raise RuntimeError("Frontier model not fitted yet. Call _fit_exponential_model first.")

    idx = _day_index(dt)
    if idx in _model_forecast_cache:
        return _model_forecast_cache[idx]

    eta_floor, A, p, t_shift = _model_params

    sec_per_year = 365.25 * 24 * 3600.0
    t_years = (pd.to_datetime(dt) - _model_t0).total_seconds() / sec_per_year

    # If a caller asks for dates before t0, clamp to t=0 (keeps model finite)
    if t_years < 0.0:
        t_years = 0.0

    eta_TH = float(_powerlaw_eff_model(np.array([t_years], dtype=float), eta_floor, A, p, t_shift)[0])
    _model_forecast_cache[idx] = eta_TH
    return eta_TH


# --------------------------------
# Daily revenue 14-day moving average
# --------------------------------

def _update_daily_revenue_MA_if_new_day(revenue_per_TH_day: float, current_date: datetime):
    """Push revenue for the day into the 14-day MA cache once per calendar day."""
    ## NOTE: This function appears unused in the main logic, but is left for compatibility.
    ## The main logic is inside compute_dynamic_efficiency.
    global _last_rev_day, _daily_revenue_ma
    day = pd.to_datetime(current_date).date()
    if _last_rev_day != day:
        _daily_revenue_ma.append(revenue_per_TH_day)
        _last_rev_day = day

def _revenue_14d_ma() -> float:
    if not _daily_revenue_ma:
        return 0.0
    return float(sum(_daily_revenue_ma) / len(_daily_revenue_ma))

## MODIFICATION: Added a helper for the threshold MA
def _threshold_14d_ma() -> float:
    if not _daily_threshold_ma:
        return 0.0
    return float(sum(_daily_threshold_ma) / len(_daily_threshold_ma))


# --------------------------------
# Public API (drop-in)
# --------------------------------

@dataclass
class _Env:
    csv_path: str
    hist_cutoff: pd.Timestamp
    initialized: bool = False

_env = _Env(csv_path="", hist_cutoff=pd.Timestamp("1970-01-01 00:00:00"))

def reset_run_state():
    """
    Reset the per-run state: the 14-day revenue and threshold moving averages, the
    last-day marker and the last non-empty profitable-set mask.

    run_simulation calls this once before its loop, so that a second run in the same
    process (for example in a calibration loop) starts from empty state rather than
    from the previous run's last day. The machine table, `_env` and the fitted
    frontier model and its cache are not touched; they depend only on the input file.
    """
    global _last_rev_day, _prev_nonempty_set_mask
    _last_rev_day = None
    _daily_revenue_ma.clear()
    _daily_threshold_ma.clear()
    _prev_nonempty_set_mask = None


def _ensure_initialized(file_path: str):
    """Lazily load machine table & fit frontier model once."""
    global _machine_df, _env
    if not _env.initialized:
        _env.csv_path = file_path
        mdf = _load_machines(file_path)
        _machine_df = mdf
        _fit_exponential_model(mdf)  # now logistic under the hood
        _env.initialized = True


def compute_dynamic_efficiency(
    file_path: str,
    target_date: datetime,
    R_block: float,
    # target: float,  <-- This argument is no longer used
    C_elec: float,
    C_elec_0: float,        # hobby-era electricity cost ($/MWh)
    P_BTC: float,
    TX_FEE_btc: float,
    daily_difficulties: np.ndarray, # <-- NEW ARGUMENT
    daily_block_times: np.ndarray, # <-- NEW ARGUMENT
    total_operating: bool,
    *,
    historical_cutoff: datetime | None = None
) -> float:
    """
    CBECI-style dynamic efficiency (J / hash) for the simulation timestamp.

    Parameters (annotated with corrections)
    ----------
    ...
    TX_FEE_btc : float
        Daily aggregate transaction fees for a given block.
    ...
    """
    global _prev_nonempty_set_mask

    _ensure_initialized(file_path)
    assert _machine_df is not None, "Machine DataFrame was not initialized."
    df = _machine_df

    # ---- 1) Daily revenue (14d MA) -----------------------------------------
    # Per the PDF, we calculate R_d = (R_block + TX_FEE) / (H_d * 86400).
    # This is calculated ONCE per day and pushed into a 14-day MA.
    
    ## MODIFICATION: Need to update both MAs now.
    global _last_rev_day, _daily_revenue_ma, _daily_threshold_ma
    day = pd.to_datetime(target_date).date()

    # Only update the daily revenue value once per calendar day. Block timestamps
    # are not monotonic, so a block can be stamped with an earlier day than the
    # previous call; `!=` would append a second entry for that day. Appending only
    # on a later day keeps one moving-average entry per calendar day.
    if _last_rev_day is None or day > _last_rev_day:
        
        # 1. Calculate H_d (daily average hashrate) in H/s
        H_array = calc_core_hashrate(daily_difficulties, daily_block_times, window=None)
        H_d_hashes_s = H_array[0] # The avg hashrate in H/s
        
        # 2. Convert H_d to TH/s
        H_d_THs = H_d_hashes_s / 1e12
        
        if H_d_THs > 0:
            # 3. Calculate total reward in $
            # TX_FEE_btc = R_block * TX_FEE_PCT
            # total_reward_btc = R_block + TX_FEE_btc
            total_reward_btc = R_block    
            total_reward_usd = total_reward_btc * P_BTC

            # Get the *actual* number of blocks from your input array
            num_blocks_day_d = len(daily_difficulties)
            total_daily_reward_usd = total_reward_usd * num_blocks_day_d + TX_FEE_btc*P_BTC # Here we passed the aggregate transaction fees for the 
            #total_daily_reward_usd = total_reward_usd * 144 + TX_FEE_btc*P_BTC #
        
            # 4. Calculate total TH produced in the day
            total_TH_per_day = H_d_THs * 86400.0
            
            # 5. Calculate R_d in $/TH
            revenue_per_TH = total_daily_reward_usd / total_TH_per_day
            
            ## MODIFICATION: Calculate daily threshold (Theta_d = Rd / Pd)
            ## and append to *both* MAs.
            
            # Choose era-specific electricity cost based on target_date
            if target_date < T_STAR:
                C_used = C_elec_0
            else:
                C_used = C_elec

            if USE_FRACTION:
                C_used = C_used*ELEC_FRACTION

            # Get daily electricity price Pd in $/J (C_used is in $/MWh)
            P_per_J_daily = float(C_used) / 3.6e9  # $/MWh -> $/J
            
            if P_per_J_daily <= 0:
                daily_theta_J_per_TH = 0.0
            else:
                # Unit check: ($/TH) / ($/J) -> J/TH. This is Rd / Pd.
                daily_theta_J_per_TH = float(revenue_per_TH) / P_per_J_daily
            
            # Append to both MAs
            _daily_revenue_ma.append(revenue_per_TH)
            _daily_threshold_ma.append(daily_theta_J_per_TH)

        else:
            # If hashrate is zero, revenue and threshold are zero.
            ## MODIFICATION: Append 0 to both MAs
            _daily_revenue_ma.append(0.0)
            _daily_threshold_ma.append(0.0)

        _last_rev_day = day

    # Get the 14-day moving average revenue. Unit is $/TH.
    # This is kept to preserve the function's return signature.
    R_d = _revenue_14d_ma()

    # ---- 2) Profitability threshold Θ_d(P) = R_d / P -----------------------
    
    ## MODIFICATION: This section is now simplified.
    ## We just get the 14-day MA of the *threshold* directly.
    theta_J_per_TH = _threshold_14d_ma()
    
    # (The old, incorrect logic is removed)
    # P_per_J = float(C_elec) / 3.6e9  # $/MWh -> $/J
    # if P_per_J <= 0: ...
    # else: theta_J_per_TH = float(R_d) / P_per_J 

    eta_TH = df["eta_TH_J_per_TH"].to_numpy(copy=False)

    # ---- 3) Build S_d(P): profitable set mask ------------------------------
    mask_profitable = eta_TH <= theta_J_per_TH

    # ---- 3) Deployment filtering (N_d) -------------------------------------
    # Compute months-since-deployment for each machine *before* profitability test
    now_ts = pd.to_datetime(target_date)
    dep_ts = df["deployment_ts"].to_numpy()
    M_months = np.array([_months_between(pd.Timestamp(d), now_ts) for d in dep_ts], dtype=float)
    #mask_deployed = M_months >= 0
    mask_deployed = (M_months >= 0) & (M_months < 60)


    # ---- 4) Build S_d(P): profitable *and* deployed set mask ---------------
    mask_profitable = (eta_TH <= theta_J_per_TH) & mask_deployed

    if not mask_profitable.any():
        if _prev_nonempty_set_mask is not None and _prev_nonempty_set_mask.any():
            mask_profitable = _prev_nonempty_set_mask.copy()
        else:
            # On first days or with zero revenue, fallback to worst available machine
            best_eta_hash = float(df["eta_hash_J_per_hash"].max())
            return best_eta_hash, float(R_d)
    else:
        _prev_nonempty_set_mask = mask_profitable.copy()

    # ---- 6) Forecast: include exponential-model machine with weight=1 -------
    weights = np.zeros(len(df), dtype=float)
    for i in np.where(mask_profitable)[0]:
        weights[i] = _age_weight(M_months[i])

    # ---- 7) Scenario: frozen vs frontier logistic --------------------------
    # By default (frozen tech), we only use the historical machine table.
    include_model = False

    # if EFFICIENCY_SCENARIO == "frontier" and historical_cutoff is not None:
    #     # Only include the synthetic frontier machine once we are:
    #     #   (i) past the historical cutoff (i.e., in the forecast region), AND
    #     #  (ii) out of eligible machines due to the age-based inclusion cutoff
    #     #       (your 5-year / 60-month deployment window).
    #     in_forecast_region = pd.to_datetime(target_date) > pd.to_datetime(historical_cutoff)
    #     inventory_exhausted = not mask_deployed.any()
    #     include_model = in_forecast_region and inventory_exhausted

    if EFFICIENCY_SCENARIO == "frontier" and historical_cutoff is not None:
        # Only include the synthetic frontier machine once we are:
        #   (i) past the historical cutoff (i.e., in the forecast region), AND
        #  (ii) past the deployment window of the *most recently released* machine
        #       (i.e., last_release + DEPLOYMENT_WINDOW months).
        in_forecast_region = pd.to_datetime(target_date) > pd.to_datetime(historical_cutoff)
    
        # Most recent machine release in the dataset
        last_machine_deploy_ts = pd.to_datetime(df["deployment_ts"].max())
        past_last_deploy = pd.to_datetime(target_date) >= last_machine_deploy_ts
        
        include_model = in_forecast_region and past_last_deploy


    eta_TH_vec = eta_TH.copy()
    w_vec = weights.copy()

    if include_model:
        try:
            eta_TH_model = _model_eta_TH_on_date(target_date)
            eta_TH_vec = np.append(eta_TH_vec, [eta_TH_model])
            # Give the frontier machine weight 1.0 so it acts like a young,
            # heavily-used rig; you can tune this if you want.
            w_vec = np.append(w_vec, [1.0])
        except RuntimeError as exc:
            # The frontier scenario was requested; running on without the model
            # would silently produce frozen-technology results.
            raise RuntimeError(
                "frontier scenario requested but frontier fit unavailable"
            ) from exc

    # ---- 8) Weighted average efficiency ψ_d ---------------------------------
    valid = w_vec > 0
    if not np.any(valid):
        # Fallback if all profitable machines are too old (weight=0)
        # Use a simple average of the profitable-but-zero-weight set
        eta_subset = eta_TH[mask_profitable]
        if len(eta_subset) > 0:
            eta_TH_avg = float(np.mean(eta_subset))
        else: # Should not be reached due to earlier checks, but as a safeguard
            eta_TH_avg = float(np.min(eta_TH))
    else:
        # The formula below correctly implements the weighted average per the PDF.
        wsum = float(np.sum(w_vec[valid]))
        eta_TH_avg = float(np.sum(eta_TH_vec[valid] * w_vec[valid]) / wsum)

    # Convert final result to J/hash for the simulator
    eta_J_per_hash = eta_TH_avg / 1e12
    
    # Return the final efficiency and the 14-day MA of *revenue* (to match original signature)
    return eta_J_per_hash, float(R_d)