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
C_ELEC = 50                        # Electricity cost in $/MWh for modern era
S = 0.02
T_STAR = pd.Timestamp("2018-01-01") 
C_ELEC_0 = 100
S_0 = 0.07
# S = 0.07 C = 110 RMSLE best for modern era & **S = 0.02 C = 50 for most recent calibration**
# S = 0.08 C = 130 RMSLE best for hobby era & **S = 0.07 C = 100 for most recent calibration**
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
EFFICIENCY_SCENARIO = "frozen"   # options: "frozen", "frontier"

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