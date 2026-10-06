# modules/boundaries.py
"""
Period boundaries of the PRICE runs and calibrations.

The rule (authors' decision, 2026-10-05)
----------------------------------------
Every boundary is a UTC instant. All runs and calibrations split their periods at
the same block heights, and those heights are found from the UTC block times.

Block times come from `Block_Time_Seconds`, which is UTC epoch seconds. The `Time`
column of combined_block_data.csv is never used for a boundary: it holds U.S.
Eastern local time (EST or EDT, with daylight saving) for blocks 0 to 888,382 and
UTC from block 888,383 (2025-03-18) on.

For a boundary instant t, the boundary block B is the first block whose
Block_Time_Seconds is later than t:

    B = min{ h : Block_Time_Seconds(h) > t }

Block B is the LAST block of the earlier period. Block B + 1 is the FIRST block of
the later period.

This is the convention of the historical/forecast switch in simulation.py. Block h
is simulated in historical mode when the clock before it is mined, which is the
timestamp of block h - 1, is at or before the cutoff. Every block below B is
stamped at or before t, so block B still copies its real time. Block B is stamped
after t, so block B + 1 is the first block whose time the economic model sets.
A block stamped at or before t that comes after B (timestamps are not monotonic)
does not move the boundary, because the switch happens once.

The three boundaries, in combined_block_data.csv (and in the _latest file, which
has the same Block_Time_Seconds over the blocks the two share):

    instant (UTC)  B        block B-1               block B                 period after B starts at
    2010-07-17     68,607   2010-07-16 23:55:58     2010-07-17 00:11:43     68,608  (economic model, hindcast/calibration)
    2018-01-01     501,961  2017-12-31 23:48:38     2018-01-01 00:08:28     501,962 (modern era: S, C_elec)
    2025-01-01     877,259  2024-12-31 23:56:16     2025-01-01 00:21:15     877,260 (out-of-sample test)

So the periods are, in block heights:
    before the economic model   0 .. 68,607
    early era (S_0, C_elec_0)   68,608 .. 501,961   (and every block before it)
    2018-2024                   501,962 .. 877,259
    out-of-sample test          877,260 ..

Before 2026-10-05 several of these heights were found from the Eastern `Time`
column (68,633, 501,995 and 877,280 are the first blocks stamped at or after
Eastern midnight), or from the model's simulated clock (505,227).
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import T_STAR

# Genesis block time, UTC epoch seconds (2009-01-03 18:15:05 UTC). The code used
# 1230988505 before 2026-10-05, which is the Eastern clock reading 13:15:05 taken as UTC.
GENESIS_TIME_S = 1231006505

# Boundary instants, UTC (naive datetimes in this code base always mean UTC).
ECON_START_UTC = datetime(2010, 7, 17)   # start of the minute price data; cutoff of hindcast and grids
ERA_SPLIT_UTC = T_STAR                   # 2018-01-01: early era -> modern era
OOS_START_UTC = datetime(2025, 1, 1)     # start of the out-of-sample test

BOUNDARY_INSTANTS = {
    "econ_start": ECON_START_UTC,
    "era_split": ERA_SPLIT_UTC,
    "oos_start": OOS_START_UTC,
}

# Boundary block B of each instant in the block data the runs use. These are not
# inputs: check_boundary_blocks() recomputes them from the data and raises if they
# differ, so a changed data file cannot shift a period silently.
EXPECTED_BOUNDARY_BLOCKS = {
    "econ_start": 68_607,
    "era_split": 501_961,
    "oos_start": 877_259,
}


def utc_seconds(instant) -> int:
    """Integer UTC epoch seconds of a datetime, pandas Timestamp or ISO string.

    A naive value is read as UTC (never as the host's local time, which is what
    datetime.timestamp() would do). An aware value is converted to UTC.
    """
    ts = pd.Timestamp(instant)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    return int(ts.tz_convert(timezone.utc).timestamp())


def _block_time_series(block_times_s) -> pd.Series:
    """Block_Time_Seconds as a Series indexed by height 0, 1, ..., N.

    Accepts a Series indexed by height, a DataFrame with a Block_Time_Seconds column
    (indexed by height, as load_block_paces returns it), or a 1-D array whose
    position is the height. Raises if the heights are not exactly 0..N.
    """
    if isinstance(block_times_s, pd.DataFrame):
        s = block_times_s["Block_Time_Seconds"]
    elif isinstance(block_times_s, pd.Series):
        s = block_times_s
    else:
        arr = np.asarray(block_times_s)
        s = pd.Series(arr, index=np.arange(len(arr)))
    heights = s.index.to_numpy()
    if len(heights) == 0 or heights[0] != 0 or not np.all(np.diff(heights) == 1):
        raise ValueError("block times must cover every height from 0 upward, in order")
    if s.isna().any():
        raise ValueError("block times contain missing values")
    return s.astype(np.int64)


def boundary_block(block_times_s, instant) -> int:
    """The boundary block B of a UTC instant: the first block stamped later than it.

    B is the last block of the earlier period; B + 1 starts the later period.
    Raises ValueError if no block in the data is stamped later than the instant.
    """
    s = _block_time_series(block_times_s)
    t = utc_seconds(instant)
    later = np.flatnonzero(s.to_numpy() > t)
    if later.size == 0:
        raise ValueError(f"no block in the data is stamped later than {pd.Timestamp(instant)} UTC")
    return int(s.index[later[0]])


def boundary_blocks(block_times_s) -> dict:
    """Boundary block B of each instant in BOUNDARY_INSTANTS, computed from the data."""
    s = _block_time_series(block_times_s)
    return {name: boundary_block(s, inst) for name, inst in BOUNDARY_INSTANTS.items()}


def check_boundary_blocks(block_times_s) -> dict:
    """Compute the boundary blocks from the data, check them against
    EXPECTED_BOUNDARY_BLOCKS, and return them. Also checks the genesis time."""
    s = _block_time_series(block_times_s)
    if int(s.iloc[0]) != GENESIS_TIME_S:
        raise ValueError(f"block 0 is stamped {int(s.iloc[0])}, expected genesis {GENESIS_TIME_S}")
    got = boundary_blocks(s)
    if got != EXPECTED_BOUNDARY_BLOCKS:
        raise ValueError(f"boundary blocks {got} differ from EXPECTED_BOUNDARY_BLOCKS "
                         f"{EXPECTED_BOUNDARY_BLOCKS}; the block data changed")
    return got


def load_block_times(path: str) -> pd.Series:
    """Block_Time_Seconds of a block CSV (columns Height, Block_Time_Seconds), indexed by height."""
    df = pd.read_csv(path, usecols=["Height", "Block_Time_Seconds"])
    df = df.dropna().astype(np.int64).sort_values("Height")
    return _block_time_series(df.set_index("Height")["Block_Time_Seconds"])
