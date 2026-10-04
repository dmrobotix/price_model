# config.py
"""
Global configuration for the miner model simulation.
This file contains global constants, economic parameters,
network parameters, and simulation controls.
"""
import logging
from datetime import datetime
import pandas as pd

# -------------------------------
# Price Forecasting Parameters
# -------------------------------
FORECAST_MODEL = "powerlaw"          # Options: "linear", "logistic", or "powerlaw", "fixed"
FORECAST_TARGET_DATE = datetime(2032, 1, 1)  # Target date for the forecast (e.g., January 1, 2040)
FORECAST_TARGET_PRICE = 1_000_000       # Target BTC price (in USD) for the forecast use None is fixed

# -------------------------------
# Blockchain and Network Constants
# -------------------------------
SHA256_MAX = 2**256                # Maximum number of SHA-256 outcomes
DIFFICULTY_BASE = 2**32            # Expected number of hashes at difficulty 1
EXPECTED_BLOCK_TIME = 600          # Bitcoin target block time in seconds (10 minutes)

# -------------------------------
# Initial Network Parameters
# -------------------------------
DIFF1_EXPONENT = 0x1d              # 29 in decimal
DIFF1_COEFFICIENT = 0x00ffff       # 65535 in decimal
#DEFAULT_TARGET = DIFF1_COEFFICIENT * (2 ** (8 * (DIFF1_EXPONENT - 3))) # Baseline target corresponding to difficulty 1 
DEFAULT_TARGET = 65535 * 2**208
INITIAL_DIFFICULTY = 1.0           # Normalized difficulty (can be derived from target)
#INITIAL_HASHRATE = (INITIAL_DIFFICULTY * 2**32) / 600.0  # implied hashrate for 600 s blocks
INITIAL_HASHRATE = 1.0
INITIAL_BLOCK_HEIGHT = 0           # Initial block height
# BLOCK_PACE_DATA = '../data/combined_block_data.csv'
# TX_BLOCK_DATA = '../data/txfee_data.csv'
# PRICE_DATA = '../data/market_price_min.csv'
# SANGHA DATA
BLOCK_PACE_DATA = '../data/combined_block_data_latest.csv'
TX_BLOCK_DATA = '../data/txfee_data.csv'
PRICE_DATA = '../data/market_price_min_latest.csv'

# -------------------------------
# Economic Parameters
# -------------------------------
R_BLOCK = 50                       # Block subsidy
TX_FEE_PCT = 0.04                 # Transaction fee as percentage of block reward (was 0.015)
P_BTC = 0                          # Bitcoin price in USD (fallback value)
ELEC_FRACTION = 0.7
USE_FRACTION = False
LOWER_BLOCK_MULTIPLIER = 0.5       # Clamping values for block time multipliers
UPPER_BLOCK_MULTIPLIER = 1.5       # Clamping values for block time multipliers
DEFAULT_MULTIPLIER = 1.64          # When BTC price is zero
# Calibration: grid run data/results/grid_2026-10-01 (commit 09730b6); RMSLE 0.7649 (pre-2018 era) and 0.2475 (modern era)
C_ELEC = 40                        # Electricity cost in $/MWh for modern era
S = 0.02
T_STAR = pd.Timestamp("2018-01-01") 
C_ELEC_0 = 110
S_0 = 0.09
# Previous calibration (month-first release dates, unfiltered machine table): S = 0.02, C = 50 modern era; S = 0.07, C = 100 hobby era
# S = 0.026238, C=50
# S=0.0280, C=54.0 -> RMSE=2.887e+07 TH/s from grid search for full historical
# S=0.03, C=60 -> RMSE=2.894e+07 TH/s
BLOCKTIME_SHAPE_K = 0.4            # calibrate on pre-ban; start with 0.3–0.6

# -------------------------------
# Simulation Controls
# -------------------------------
CALIBRATION_MODE = False         # True -> deterministic blocks + use params['S'] for calibrating S
PERIOD_BLOCKS = 2016             # Number of blocks per difficulty adjustment period (Bitcoin default)
SIMULATE_PERIODS = 10            # Number of difficulty periods to simulate
TOTAL_BLOCKS = PERIOD_BLOCKS * SIMULATE_PERIODS
INITIAL_BLOCK_SUBSIDY = R_BLOCK  # BTC, as per Bitcoin protocol
HALVING_INTERVAL = 210000        # Blocks per halving cycle
# Exogenous Miner Entry Parameters (Non-Economic)
# EXO_POWER_SCALE = 1e10           # Baseline scaling factor for exogenous hashrate
# EXO_POWER_EXPONENT = 0.5         # Exponent for the power law growth
# USE_NON_ECONOMIC_HASHRATE = False # Toggle using the non-economic power law in modeling hashrate
# EXO_MODEL = 'additive'           # Default mode: choose between 'additive' or 'multiplicative'
PARETO_TOP_WEIGHT = 0.8          # Used in combining machine efficiencies
PARETO_BOTTOM_WEIGHT = 0.2       # Used in combining machine efficiencies
REGRESSION_WEIGHT = 0.5          # Used for estimating historical/projected machine efficiency

#--------------------------------
# Energy Factors
#--------------------------------
PUE = 1.10                        # Based on CBECI best estimate

# -------------------------------
# Conversion Factors
# -------------------------------
JOULES_PER_MWH = 3.6e9             # 1 MWh = 3.6e9 Joules
HASHES_PER_TH = 1e12                # 1 TH = 1e12 hashes

# -------------------------------
# Economic Sensitivity Parameters
# -------------------------------
LAMBDA_PARAM = 0.001                 # Responsiveness parameter for capacity expansion
ECONOMIC_BLOCK_TIME_SCALING = 0.001  # Tunable scaling factor for economic block time adjustment

# -------------------------------
# Mining Machine Efficiency (Joules per TH)
# -------------------------------
ETA_FLEX = 20                      # Efficiency for on-grid (flexible) mining, in J/TH
ETA_OFF = 20                       # Efficiency for off-grid mining, in J/TH
ETA_MINING = 20                    # Efficiency used for lost capacity (mining), in J/TH
# MACHINE_DATA_FILE = '../data/all_asics_updated03242025_clean.csv' # Machine efficiencies going back to CPU era.
MACHINE_DATA_FILE = '../data/cbeci_machines_090325.csv'
ALPHA = 0.9                        # Default Pareto weighting
PARETO_TOP_WEIGHT = 0.8
PARETO_BOTTOM_WEIGHT = 0.2
MOVING_AVERAGE = True              # Use a 14-day moving average to smooth out profit threshold changes to reflect miner lag.

# -------------------------------
# Efficiency forecasting scenario
# -------------------------------
EFFICIENCY_SCENARIO = "frontier"   # options: "frozen", "frontier"

# -------------------------------
# Dynamic Hashprice Model Parameters
# -------------------------------
GROWTH_FACTOR = 1.0                          # Sensitivity for hashrate growth in dynamic hashprice update
ALPHA_VALUES = [i * 0.1 for i in range(11)]  # Exit fraction values from 0.0 to 1.0 in increments of 0.1

# Logistic function sensitivity for on-/off-grid split
LOGISTIC_SENSITIVITY = 0.5

# -------------------------------
# Water Consumption Factors (liters per MWh)
# -------------------------------
W_AIR = 0.1                      # For dry-air cooling (minimal water use)
W_WATER_MINING = 1.0             # For water-cooling in mining
W_HPC = 2.5                      # For HPC/AI water usage

# -------------------------------
# Growth Rate Limit Parameters
# -------------------------------
BASELINE_GROWTH = 0.00000000000005       # Baseline hashrate growth when price is zero
APPLY_GROWTH_LIMITS = False   # Toggle to apply growth rate limits in hashrate update
MIN_GROWTH = -0.05            # Minimum allowed growth rate (e.g., -5%)
MAX_GROWTH = 0.1              # Maximum allowed growth rate (e.g., 10%)

# -------------------------------
# Debugging Controls
# -------------------------------
LOG_LEVEL = logging.WARNING

# -------------------------------
# Run overrides from the environment
# -------------------------------
# scenarios/run_scenarios.py runs main.py once per scenario and passes each run's
# settings in PRICE_* environment variables, so neither config.py nor main.py is
# hand-edited per run. With none of these variables set, every value above (and the
# hardcoded values in main.py) is used unchanged. A malformed value, or a PRICE_*
# variable that is not listed here, raises ValueError at import.
import os as _os
import re as _re

RUN_OVERRIDE_VARS = (
    # read below
    "PRICE_EFFICIENCY_SCENARIO", "PRICE_FORECAST_MODEL",
    "PRICE_FORECAST_TARGET_DATE", "PRICE_FORECAST_TARGET_PRICE",
    "PRICE_BLOCK_PACE_DATA", "PRICE_PRICE_DATA",
    # read in main.py
    "PRICE_NUM_STEPS", "PRICE_HISTORICAL_CUTOFF", "PRICE_OUTPUT_NAME",
)

_unknown = sorted(k for k in _os.environ if k.startswith("PRICE_") and k not in RUN_OVERRIDE_VARS)
if _unknown:
    raise ValueError(f"unknown PRICE_* environment variable(s) {_unknown}; "
                     f"the recognised ones are {list(RUN_OVERRIDE_VARS)}")


def env_override(name, parse, default):
    """Return parse(value of environment variable ``name``) if it is set, else ``default``."""
    if name not in RUN_OVERRIDE_VARS:
        raise ValueError(f"{name} is not in RUN_OVERRIDE_VARS")
    raw = _os.environ.get(name)
    if raw is None:
        return default
    try:
        return parse(raw.strip())
    except (ValueError, TypeError) as exc:
        raise ValueError(f"environment variable {name}={raw!r} is invalid: {exc}") from None


def parse_choice(*choices):
    def parse(s):
        if s not in choices:
            raise ValueError(f"expected one of {list(choices)}")
        return s
    return parse


def parse_positive_int(s):
    value = int(s)  # accepts "1260000" and "1_260_000"; rejects "1e6" and "1.0"
    if value <= 0:
        raise ValueError("must be a positive integer")
    return value


def parse_price_or_none(s):
    """'none' (any case) -> None; otherwise a positive finite number (int if integral text)."""
    if s.lower() == "none":
        return None
    try:
        value = int(s)
    except ValueError:
        value = float(s)
    if not (value > 0 and value != float("inf")):  # also rejects nan
        raise ValueError("must be 'none' or a positive finite number")
    return value


def parse_naive_datetime(s):
    """ISO date or date-time without a UTC offset, e.g. 2032-01-01 or 2025-11-08T17:58:32."""
    value = datetime.fromisoformat(s)
    if value.tzinfo is not None:
        raise ValueError("must not carry a UTC offset (the model uses UTC-naive datetimes)")
    return value


def parse_output_name(s):
    """Base name of the output CSV, written by main.py to ../data/<name>.csv."""
    if not _re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", s) or s.endswith(".csv"):
        raise ValueError("must be 1-200 characters from [A-Za-z0-9._-], start with a letter "
                         "or digit, and not end in .csv")
    return s


def parse_data_csv(s):
    """An input CSV in ../data/: a bare file name, or ../data/<name>.csv.
    Returns the ../data/<name>.csv form that the defaults above use."""
    name = s[len("../data/"):] if s.startswith("../data/") else s
    if not _re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\.csv", name) or ".." in name:
        raise ValueError("must be a file name <name>.csv or ../data/<name>.csv, with <name> from "
                         "[A-Za-z0-9._-] and no other directory")
    return "../data/" + name


EFFICIENCY_SCENARIO = env_override("PRICE_EFFICIENCY_SCENARIO",
                                   parse_choice("frozen", "frontier"), EFFICIENCY_SCENARIO)
FORECAST_MODEL = env_override("PRICE_FORECAST_MODEL",
                              parse_choice("powerlaw", "fixed", "constant", "linear", "logistic"),
                              FORECAST_MODEL)
FORECAST_TARGET_DATE = env_override("PRICE_FORECAST_TARGET_DATE", parse_naive_datetime,
                                    FORECAST_TARGET_DATE)
FORECAST_TARGET_PRICE = env_override("PRICE_FORECAST_TARGET_PRICE", parse_price_or_none,
                                     FORECAST_TARGET_PRICE)
# Input files. TX_BLOCK_DATA and MACHINE_DATA_FILE have no override.
BLOCK_PACE_DATA = env_override("PRICE_BLOCK_PACE_DATA", parse_data_csv, BLOCK_PACE_DATA)
PRICE_DATA = env_override("PRICE_PRICE_DATA", parse_data_csv, PRICE_DATA)
# main.py reads the remaining three variables after its data loading; validate them
# here too, so a malformed value fails at import rather than minutes into a run.
for _name, _parse in (("PRICE_NUM_STEPS", parse_positive_int),
                      ("PRICE_HISTORICAL_CUTOFF", parse_naive_datetime),
                      ("PRICE_OUTPUT_NAME", parse_output_name)):
    env_override(_name, _parse, None)
