# modules/network.py

"""
Network Module

This module contains functions for calculating network-based quantities,
such as block probabilities, expected hashes, stochastic block time sampling,
and performing difficulty retargeting using updated target and hashrate calculations.
"""

import numpy as np
from datetime import datetime
from config import SHA256_MAX, DIFFICULTY_BASE, EXPECTED_BLOCK_TIME, DEFAULT_TARGET, PERIOD_BLOCKS, BLOCKTIME_SHAPE_K 

def calc_probability(target: float) -> float:
    """
    Calculate the probability of a winning hash.

    Formula:
        p = target / 2^256

    Args:
        target (float): The current target value (T_t).

    Returns:
        float: The probability of finding a winning hash.
    """
    return target / SHA256_MAX

def calc_expected_hashes(difficulty: float) -> float:
    """
    Calculate the expected number of hashes needed to find a block.

    Formula:
        Expected Hashes = difficulty * 2^32

    Args:
        difficulty (float): The current difficulty (D_t).

    Returns:
        float: The expected number of hashes needed.
    """
    return difficulty * DIFFICULTY_BASE

# def calc_stochastic_block_time(economic_mean: float) -> float:
#     """
#     Calculate the actual block time stochastically by drawing from an exponential distribution.

#     The economic model (e.g., profitability) provides an average block time (economic_mean). 
#     The realized block time is then sampled from an exponential distribution with this mean.

#     Args:
#         economic_mean (float): The average block time (in seconds) determined via the economic metric.

#     Returns:
#         float: A stochastically sampled block time in seconds.
#     """
#     return np.random.exponential(scale=economic_mean)

def calc_stochastic_block_time(economic_mean: float) -> float:
    """
    Draw a stochastic inter-block time with mean equal to `economic_mean`.
    Uses a Gamma(k, θ) with θ = economic_mean / k so E[T] = economic_mean.
    k=1 reduces exactly to the exponential distribution.
    """
    k = float(BLOCKTIME_SHAPE_K)
    if not np.isfinite(k) or k <= 0.0:
        # Fallback: pure exponential (original behavior)
        return float(np.random.exponential(scale=economic_mean))
    theta = float(economic_mean) / k
    return float(np.random.gamma(shape=k, scale=theta))

def calc_block_time(difficulty: float, hashrate: float) -> float:
    """
    Calculate the expected block time given the current difficulty and hashrate. Used during hashrate
    shock only.
    
    Formula (deterministic):
        T_block = (difficulty * DIFFICULTY_BASE) / hashrate
    """
    if hashrate <= 0:
        raise ValueError("Hashrate must be a positive number")
    return (difficulty * DIFFICULTY_BASE) / hashrate

def update_target_float(prev_target: float, actual_timespan: float, expected_timespan: float) -> float:
    """
    Update the network target based on the ratio of actual to expected timespan,
    clamping the timespan to [expected/4, expected*4] and the result to max target.
    """
    # 1) Bound the timespan like Bitcoin Core does:
    min_ts = expected_timespan / 4.0
    max_ts = expected_timespan * 4.0
    clamped_timespan = max(min(actual_timespan, max_ts), min_ts)
    # print(clamped_timespan)

    # 2) Compute new target from the *clamped* timespan
    adjustment = clamped_timespan / expected_timespan
    new_target = prev_target * adjustment

    # 3) Enforce the maximum target (difficulty floor)
    if new_target > DEFAULT_TARGET:
        new_target = DEFAULT_TARGET

    return new_target


def target_to_compact(target: int) -> int:
    """Encode full target int into Bitcoin 'compact' (nBits) format."""
    if target <= 0:
        return 0

    size = (target.bit_length() + 7) // 8

    if size <= 3:
        mantissa = target << (8 * (3 - size))
    else:
        mantissa = target >> (8 * (size - 3))

    mantissa &= 0xFFFFFF

    # If mantissa's highest bit is set, shift down and increase exponent
    if mantissa & 0x800000:
        mantissa >>= 8
        size += 1

    return (size << 24) | mantissa


def compact_to_target(bits: int) -> int:
    """Decode Bitcoin 'compact' (nBits) into full target int."""
    size = (bits >> 24) & 0xFF
    mantissa = bits & 0xFFFFFF

    if size <= 3:
        return mantissa >> (8 * (3 - size))
    return mantissa << (8 * (size - 3))

def update_target(prev_target: int, first_ts: datetime, last_ts: datetime) -> int:
    """
    Integer-only difficulty retarget, Bitcoin-Core-style.

    prev_target: Python int representing the previous target (like a uint256).
    first_ts, last_ts: datetimes for the start and end of the 2016-block window.
    """
    # 2016 * 600 = 1_209_600 seconds
    expected_timespan = PERIOD_BLOCKS * EXPECTED_BLOCK_TIME  # int

    # Actual elapsed time in *integer* seconds
    actual_timespan = int((last_ts - first_ts).total_seconds())

    # Clamp timespan to [expected/4, expected*4] using only ints
    min_ts = expected_timespan // 4      # integer division
    max_ts = expected_timespan * 4       # still an int

    if actual_timespan < min_ts:
        clamped_timespan = min_ts
    elif actual_timespan > max_ts:
        clamped_timespan = max_ts
    else:
        clamped_timespan = actual_timespan

    # Integer-only retarget:
    # new_target = prev_target * clamped / expected
    new_target = (prev_target * clamped_timespan) // expected_timespan

    # Enforce the maximum target (difficulty 1 floor) and avoid zero/negative.
    if new_target > DEFAULT_TARGET:
        new_target = DEFAULT_TARGET
    if new_target <= 0:
        new_target = 1

    # --- Core-style quantization through nBits ---
    bits = target_to_compact(new_target)
    new_target = compact_to_target(bits)

    # clamp again after quantization
    if new_target > DEFAULT_TARGET:
        new_target = DEFAULT_TARGET
    if new_target <= 0:
        new_target = 1

    return int(new_target)


def update_target_from_seconds(prev_target: int, first_ts_s: int, last_ts_s: int) -> int:
    """
    Integer-only difficulty retarget, Bitcoin-Core-style.

    prev_target: Python int representing the previous target (like a uint256).
    first_ts_s, last_ts_s: seconds for the start and end of the 2016-block window.
    """
    expected_timespan = PERIOD_BLOCKS * EXPECTED_BLOCK_TIME
    actual_timespan = int(last_ts_s - first_ts_s)

    min_ts = expected_timespan // 4
    max_ts = expected_timespan * 4
    clamped = min(max(actual_timespan, min_ts), max_ts)

    new_target = (prev_target * clamped) // expected_timespan
    if new_target > DEFAULT_TARGET:
        new_target = DEFAULT_TARGET
    if new_target <= 0:
        new_target = 1

    bits = target_to_compact(new_target)
    new_target = compact_to_target(bits)

    if new_target > DEFAULT_TARGET:
        new_target = DEFAULT_TARGET
    if new_target <= 0:
        new_target = 1

    return int(new_target)


def update_difficulty(new_target: float, default_target: float) -> float:
    """
    Update the network difficulty based on the new target.

    Since target and difficulty are inversely related, the new difficulty is computed as:
        new_difficulty = default_target / new_target

    Args:
        new_target (float): The updated network target.
        default_target (float): The baseline target when difficulty is 1.

    Returns:
        float: The updated network difficulty.
    """
    computed_difficulty = default_target / new_target
    return max(1.0, computed_difficulty)

def update_hashrate(new_difficulty: float, block_time: float) -> float:
    """ 
    Update the network hashrate using the new difficulty and the realized block time.

    Based on the relationship:
        T_block = (new_difficulty * DIFFICULTY_BASE) / hashrate
    we solve for hashrate:
        hashrate = (new_difficulty * DIFFICULTY_BASE) / T_block

    Args:
        new_difficulty (float): The updated network difficulty.
        block_time (float): The actual block time (in seconds) measured (or sampled) for the period.

    Returns:
        float: The updated network hashrate.
    """
    if block_time <= 0:
        raise ValueError("Block time must be a positive number")
    return (new_difficulty * DIFFICULTY_BASE) / block_time

def calc_target_from_difficulty(difficulty: float, default_target: float) -> float:
    """
    [Deprecated] Calculate the full network target based on the current difficulty.

    In the Bitcoin protocol, as difficulty increases, the target (i.e., the threshold
    a valid hash must be below) decreases. This is computed as:

        target = default_target / difficulty

    Args:
        difficulty (float): The current network difficulty.
        default_target (float): The baseline target corresponding to difficulty 1.

    Returns:
        float: The computed network target.
    """
    return default_target / difficulty

def calc_core_hashrate(difficulties: np.ndarray, block_times: np.ndarray, window: int = None) -> np.ndarray:
    """
    Compute network hashrate exactly as Bitcoin Core does:
      H = workDiff / timeDiff,
    where workDiff = sum(D_i * 2**32) over the window,
          timeDiff = sum(T_block_i)        over the same window.

    If window is None, uses the *entire* array (one value at the very end).
    If window is an integer N, returns a rolling hashrate of length len(difficulties),
    with NaN for the first N-1 entries and then windowed H.
    """
    # per-block *work* in "hash-at-diff-1" units:
    work = difficulties.astype(float) * DIFFICULTY_BASE

    if window is None or window >= len(work):
        # single overall rate
        total_work = np.nansum(work)
        total_time = np.nansum(block_times)
        H = np.full_like(work, fill_value=(total_work/total_time))
        return H

    # rolling sums
    cwork = np.convolve(work, np.ones(window), mode='full')[:len(work)]
    ctime = np.convolve(block_times, np.ones(window), mode='full')[:len(work)]

    # shift so that index i holds the sum over [i-window+1 .. i]
    cwork[window-1:] = cwork[window-1:]
    ctime[window-1:] = ctime[window-1:]
    cwork[:window-1] = np.nan
    ctime[:window-1] = np.nan

    return cwork / ctime
