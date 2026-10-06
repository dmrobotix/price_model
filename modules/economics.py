# Responsibilities:
# - Calculate expected revenue per hash:
# R_t = (R_block * T_t) / 2^256
# - Convert revenue into a price and energy adjusted metric:
# R_pe_t = 3.6e21 * R_t * P_USD
# - Update hashrate based on economic profitability:
# H_t = H_t-1 * [ 1 + lambda * max(0, (R_pe_t / C_elec) - 1) ]
# - Include additional functions (from dynamic hashprice, logistic function for on-off grid split, etc.)

from config import LOWER_BLOCK_MULTIPLIER, UPPER_BLOCK_MULTIPLIER, DEFAULT_MULTIPLIER, T_STAR
import pandas as pd
import numpy as np
import logging

def calc_expected_revenue(R_block, tx_fees, target):
    """
    Calculate the expected revenue per hash (BTC/hash).

    This function computes the expected revenue for a single hash by using the block subsidy
    and the current network target. The calculation is based on the assumption that each hash
    has an equal probability of being below the current target threshold.

    The formula used is:
        R_t = (R_block * target) / 2^256

    Where:
      - R_block: The block subsidy (in BTC), which is the reward given for mining a block.
      - tx_fees: The transaction fees for the block (in BTC).
      - target: The current network target, a dimensionless value that sets the threshold
                for a valid block hash. A higher target corresponds to lower mining difficulty.
      - 2^256: The total number of possible SHA-256 hash outcomes, used to normalize the reward
              to a per-hash basis.

    This results in R_t having units of BTC/hash, indicating the expected amount of Bitcoin
    earned for each individual hash computed.

    Args:
        R_block (float): Block reward in Bitcoin (BTC).
        tx_fees (float): Block's transaction fees (BTC).
        target (float): Current network target (dimensionless).

    Returns:
        float: Expected revenue per hash in BTC/hash.
    """
    # Multiply block reward by target and divide by the total number of possible hash outcomes.
    return ((R_block + tx_fees) * target) / 2**256

def economic_to_block_time(R_pe: float, base_time: float, C_elec_1: float,
                           S_1: float, C_elec_0: float, S_0: float, ts: pd.Timestamp,
                           modern_era: bool | None = None) -> float:
#                             lower_multiplier: float = 0.5, upper_multiplier: float = 1.5) -> float:
    """
    Map an economic metric (price-energy adjusted revenue, R_pe) to an average block time,
    using a power function that adjusts block time inversely to the ratio R_pe/C_elec.
    
    If R_pe is greater than C_elec (higher profitability), the block time decreases.
    If R_pe is less than C_elec (lower profitability), the block time increases.
    
    Args:
        R_pe (float): Price-energy adjusted revenue.
        base_time (float): Baseline block time (e.g., 600 seconds).
        C_elec (float): Electricity cost.
        s (float): Sensitivity scaling factor.
        
        modern_era (bool | None): True for the modern era (S_1, C_elec_1), False for
            the early era (S_0, C_elec_0). The simulation passes it from the block
            height (modules/boundaries.py). If None, the era is taken from ts < T_STAR,
            which is right only when ts is a real (historical) UTC time.

    Returns:
        float: The adjusted block time in seconds.
    """
    if modern_era is None:
        modern_era = not (ts < T_STAR)
    if not modern_era:
        S = S_0
        C_elec = C_elec_0
    else:
        S = S_1
        C_elec = C_elec_1
        
    # Check if R_pe is zero:
    if R_pe == 0:
        original_multiplier = DEFAULT_MULTIPLIER
    else:
        ratio = R_pe / C_elec
        original_multiplier = ratio ** (-S)
    return base_time * original_multiplier


# def economic_to_block_time(R_pe: float, base_time: float, C_elec: float,
#                            S: float, lower_multiplier: float = 0.5, upper_multiplier: float = 1.5) -> float:
# #                             lower_multiplier: float = 0.5, upper_multiplier: float = 1.5) -> float:
#     """
#     Map an economic metric (price-energy adjusted revenue, R_pe) to an average block time,
#     using an exponential function that adjusts block time inversely to the ratio R_pe/C_elec.
    
#     If R_pe is greater than C_elec (higher profitability), the block time decreases.
#     If R_pe is less than C_elec (lower profitability), the block time increases.
    
#     The output multiplier is clamped between lower_multiplier and upper_multiplier to avoid
#     unrealistic block times.
    
#     Args:
#         R_pe (float): Price-energy adjusted revenue.
#         base_time (float): Baseline block time (e.g., 600 seconds).
#         C_elec (float): Electricity cost.
#         s (float): Sensitivity scaling factor.
#         lower_multiplier (float): Minimum allowed multiplier (default 0.5).
#         upper_multiplier (float): Maximum allowed multiplier (default 1.5).
        
#     Returns:
#         float: The adjusted block time in seconds.
#     """
#     # Check if R_pe is zero:
#     if R_pe == 0:
#         original_multiplier = DEFAULT_MULTIPLIER
#     else:
    
#         # Compute deviation: at equilibrium, R_pe/C_elec == 1 implies no change.
#         x = S * ( (R_pe / C_elec) - 1 )
        
#         # Compute the multiplier using an exponential function.
#         original_multiplier = np.exp(-x)

#         # ratio = R_pe / C_elec
#         # original_multiplier = ratio ** (-S)
        
#         # Clamp the multiplier to a realistic range.
#         # multiplier = np.clip(original_multiplier, lower_multiplier, upper_multiplier)

#     #return base_time * multiplier
#     return base_time * original_multiplier

def calc_price_energy_adjusted_revenue(R_t, P_USD, efficiency):
    """
    Convert revenue per hash to USD per kWh.

    Args:
        R_t (float): Expected revenue per hash (BTC/hash).
        P_USD (float): Bitcoin price in USD ($/BTC).
        efficiency (float): Mining hardware efficiency (J/hash).

    Returns:
        float: Price-energy adjusted revenue in USD/kWh.
    """
    # Multiply by 3.6e9 to convert from $/J to $/MWh (1 kWh = 3.6e6 J)
    return (3.6e9 * R_t * P_USD) / efficiency

def build_shock_profile(max_hist_block: int, initial_block_height: int, num_steps: int, shock_duration: int = 100_000, L: float = 0.10) -> np.ndarray:
    """
    Build a per-block logistic shock fraction profile.

    Args:
        real_paces: DataFrame indexed by block Height with a 'Time' column.
        last_hist_time: Timestamp when historical data ends.
        initial_block_height: Starting block height (e.g., initial_state['block_height'] + 1).
        num_steps: Total number of simulation steps (blocks) to generate.
        shock_duration: Number of blocks over which the shock ramps in.
        L: Peak shock fraction (e.g., 0.10 for 10%).

    Returns:
        A NumPy array of length initial_block_height + num_steps + 1,
        where each entry is the shock fraction for that block.
    """
    # Determine start and end of shock
    s0 = int(max_hist_block + 1)
    s1 = s0 + shock_duration

    # Build logistic ramp from 0 -> L over [s0, s1)
    d = shock_duration
    k   = 10.0 / d
    mid = 0.5*(s0 + s1)
    x   = np.arange(s0, s1)  # length = d

    # raw logistic (0→1)  
    raw = 1.0 / (1.0 + np.exp(-k*(x-mid)))

    # normalize to sum=1 then scale by L  
    raw_sum = raw.sum()
    profile = (raw / raw_sum) * L

    # embed into full array
    max_block = initial_block_height + num_steps + 1
    sf_array  = np.zeros(max_block)
    # Check if s0 is within the valid range
    if s0 >= max_block:
        logging.warning("Shock profile starts after simulation ends. Returning zeros.")
        return sf_array

    # Clamp s1 to max_block to prevent overflow
    s1 = min(s1, max_block)
    profile = profile[:s1 - s0]
    
    sf_array[s0:s1] = profile
    return sf_array

def apply_hashrate_shock(prev_hashrate: float, shock_fraction: float) -> float:
    """
    Apply an exogenous hashrate shock.

    This function reduces the prior network hashrate by a given fraction,
    modeling X % of hashpower dropping off-line instantaneously or over a ramp.

    Formula:
        H_eff = (1 - shock_fraction) * prev_hashrate

    Args:
        prev_hashrate (float):
            The network hashrate before the shock (hashes per second).
        shock_fraction (float):
            Fraction of hashrate lost, between 0.0 (no loss) and 1.0 (total loss).

    Returns:
        float:
            The effective hashrate after applying the shock.

    Raises:
        ValueError:
            If shock_fraction is outside [0.0, 1.0], or prev_hashrate is negative.
    """
    if prev_hashrate < 0:
        raise ValueError(f"prev_hashrate must be non-negative, got {prev_hashrate}")
    if not (0.0 <= shock_fraction <= 1.0):
        raise ValueError(f"shock_fraction must be between 0 and 1, got {shock_fraction}")

    return prev_hashrate * (1.0 - shock_fraction)