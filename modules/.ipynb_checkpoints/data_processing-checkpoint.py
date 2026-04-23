# modules/data_processing.py

"""
Data Processing Module

This module contains functions for loading and processing historical blockchain data.
It includes functions for reading target history files, cleaning the data,
and computing block pace (i.e., inter-block times).
"""

import pandas as pd
import numpy as np
from modules.network import calc_core_hashrate, compact_to_target  # To calculate H from D and T
from config import DEFAULT_TARGET, BLOCK_PACE_DATA, TX_BLOCK_DATA  
from decimal import Decimal, InvalidOperation
import re

import re
import numpy as np
import pandas as pd

def parse_bits_int(x):
    if x is None or pd.isna(x):
        return None

    # numeric types
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if np.isfinite(x):
            return int(x)
        return None

    s = str(x).strip().lower()
    if s == "" or s == "nan":
        return None

    # accept optional 0x prefix
    if s.startswith("0x"):
        s = s[2:]

    # Bitcoin nBits is typically rendered as exactly 8 hex chars.
    # If it looks like 8 hex chars (even if digit-only), interpret as hex.
    if re.fullmatch(r"[0-9a-f]{8}", s):
        return int(s, 16)

    # otherwise, if it contains hex letters, treat as hex
    if re.fullmatch(r"[0-9a-f]", s) and any(c in "abcdef" for c in s):
        return int(s, 16)

    # fallback decimal
    return int(s)

def parse_target_int(x):
    """
    Parse Bitcoin 'target' as an exact Python int.

    Handles:
      - decimal strings
      - scientific notation strings (via Decimal)
      - hex strings (0x... or plain hex)
    """
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        return None

    if isinstance(x, (int,)):
        return x

    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None

    # 0x-prefixed hex
    if s.lower().startswith("0x"):
        try:
            return int(s, 16)
        except ValueError:
            return None

    # plain integer decimal
    if re.fullmatch(r"[0-9]", s):
        try:
            return int(s)
        except ValueError:
            return None

    # plain hex without 0x (rare, but possible)
    if re.fullmatch(r"[0-9a-fA-F]", s) and any(c.isalpha() for c in s):
        try:
            return int(s, 16)
        except ValueError:
            return None

    # scientific notation or decimal-like strings
    try:
        return int(Decimal(s))
    except (InvalidOperation, ValueError):
        return None

def get_last_tx_fee_time(file_path: str = TX_BLOCK_DATA) -> pd.Timestamp:
    """
    Return the last (max) timestamp available in the transaction-fee dataset.

    This is intended to mirror how main.py derives `last_hist_time` for price:
        last_hist_time = market_prices["Time"].max()

    The function is robust to common timestamp encodings:
      1) A datetime-like column containing ISO strings or pandas-parseable datetimes.
      2) A numeric epoch column in seconds / milliseconds / nanoseconds (unit inferred).

    Returns:
        pd.Timestamp: UTC-naive timestamp representing the latest available fee timestamp.

    Raises:
        ValueError: If no usable timestamp column is found or all timestamps are NaT/NaN.
    """
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    def _epoch_unit_from_magnitude(x: float) -> str:
        """
        Infer epoch unit from magnitude.
          - seconds:      ~1e9  (e.g., 1730000000)
          - milliseconds: ~1e12 (e.g., 1730000000000)
          - nanoseconds:  ~1e18 (e.g., 1730000000000000000)
        """
        if x >= 1e17:
            return "ns"
        if x >= 1e11:
            return "ms"
        return "s"

    # 1) Prefer datetime-ish columns, but handle numeric-epoch columns too (e.g., "timestamp" = 1231469665000)
    datetime_candidates = [
        "Time", "time",
        "Timestamp", "timestamp",
        "Datetime", "datetime",
        "Date", "date",
    ]
    for col in datetime_candidates:
        if col in df.columns:
            # If the column is numeric (or numeric-like), treat as epoch with inferred unit.
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().any():
                max_val = float(numeric.max())
                unit = _epoch_unit_from_magnitude(max_val)
                ts = pd.to_datetime(numeric, errors="coerce", utc=True, unit=unit).dropna()
                if ts.empty:
                    raise ValueError(
                        f"{file_path}: Column '{col}' looks numeric but could not be converted as epoch ({unit})."
                    )
                return ts.max().tz_convert(None)

            # Otherwise, attempt normal datetime parsing
            ts = pd.to_datetime(df[col], errors="coerce", utc=True).dropna()
            if ts.empty:
                raise ValueError(f"{file_path}: Column '{col}' exists but contains no parseable timestamps.")
            return ts.max().tz_convert(None)

    # 2) Fall back to epoch-seconds columns
    seconds_candidates = [
        "Block_Time_Seconds", "block_time_seconds",
        "timestamp_s", "Timestamp_s",
        "unix_time", "UnixTime",
        "time_s",
    ]
    for col in seconds_candidates:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                raise ValueError(f"{file_path}: Column '{col}' exists but contains no numeric epoch seconds.")
            max_val = float(s.max())
            unit = _epoch_unit_from_magnitude(max_val)
            return pd.to_datetime(int(max_val), unit=unit, utc=True).tz_convert(None)

    raise ValueError(
        f"{file_path}: No recognized timestamp column found. "
        f"Available columns: {list(df.columns)}"
    )

def load_target_history(file_path: str, sep: str="\t", date_format: str="%d %b %Y, %H:%M:%S") -> pd.DataFrame:
    """
    Load historical target data from a file.

    Args:
        file_path (str): Path to the target history file.
        sep (str): Separator used in the file (default is tab).
        date_format (str): Format of the date in the file (default is '%d %b %Y, %H:%M:%S').

    Returns:
        pd.DataFrame: A DataFrame containing the loaded and processed data.
    """
    # Read the file into a DataFrame
    df = pd.read_csv(file_path, sep=sep)
    # Strip any extra whitespace from the column headers
    df.columns = df.columns.str.strip()
    # Convert the "Time" column to datetime using the specified format
    df["Time"] = pd.to_datetime(df["Time"], format=date_format, errors="coerce")
    # Sort the DataFrame by the Time column
    df.sort_values("Time", inplace=True)
    return df

def compute_block_pace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the block pace (inter-block times) from a DataFrame that contains a "Time" column.

    The function calculates the time difference between consecutive blocks
    and adds two new columns:
      - "Block_Pace_Seconds": block time difference in seconds
      - "Block_Pace_Minutes": block time difference in minutes

    Args:
        df (pd.DataFrame): DataFrame containing at least a "Time" column.

    Returns:
        pd.DataFrame: The input DataFrame with added columns for block pace.
                    Rows with missing block pace values (NaN) are dropped.
    """
    # Calculate the difference between consecutive "Time" values
    df["Block_Pace_Seconds"] = df["Time"].diff().dt.total_seconds()
    # Convert the block pace from seconds to minutes
    df["Block_Pace_Minutes"] = df["Block_Pace_Seconds"] / 60.0
    # Drop rows where block pace is NaN (typically the first row)
    df_clean = df.dropna(subset=["Block_Pace_Seconds"])
    return df_clean

def save_block_pace_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the DataFrame with block pace data to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame containing block pace data.
        output_path (str): The file path where the CSV should be saved.
    """
    df.to_csv(output_path, index=False)

def load_block_paces(file_path_1: str, file_path_2: str) -> pd.DataFrame:
    """
    Load inter-block intervals and timestamps, utilizing Block_Time_Seconds as the source of truth.
    """
    # 1) Read CSV 1
    hist_1 = pd.read_csv(
        file_path_1,
        parse_dates=["Time"], # Parse initially, but we will overwrite
        dtype={"Target": "string", "Bits": "string"}
    )
    hist_1.columns = hist_1.columns.str.strip()
    hist_1 = hist_1.dropna(subset=["Height"])
    hist_1["Height"] = hist_1["Height"].astype(int)

    # 2) Process Time based on Block_Time_Seconds (Integer)
    if "Block_Time_Seconds" in hist_1.columns:
        hist_1["Block_Time_Seconds"] = pd.to_numeric(hist_1["Block_Time_Seconds"], errors="coerce")
        hist_1 = hist_1.dropna(subset=["Block_Time_Seconds"])
        hist_1["Block_Time_Seconds"] = hist_1["Block_Time_Seconds"].astype(np.int64)
        
        # Overwrite Time with UTC-derived naive datetime
        hist_1["Time"] = pd.to_datetime(hist_1["Block_Time_Seconds"], unit="s", utc=True).dt.tz_convert(None)
        
        # Sort and recalculate intervals
        hist_1 = hist_1.sort_values("Height")
        hist_1["Inter_Block_Interval_Seconds"] = hist_1["Block_Time_Seconds"].diff().fillna(0).astype(float)
        # Fix Genesis block interval (or just leave as 0)
        
    # 3) Process Bits/Target
    bits_col = next((c for c in ["Bits", "nBits", "bits", "nbits"] if c in hist_1.columns), None)
    if bits_col:
        hist_1["Bits_int"] = hist_1[bits_col].apply(parse_bits_int)
        hist_1 = hist_1.dropna(subset=["Bits_int"])
        hist_1["Target"] = hist_1["Bits_int"].apply(lambda b: compact_to_target(int(b)))
    else:
        hist_1["Target"] = hist_1["Target"].apply(parse_target_int)
        hist_1 = hist_1.dropna(subset=["Target"])

    # 4) Read CSV 2 (Fees)
    hist_2 = pd.read_csv(file_path_2)
    hist_2 = hist_2.dropna(subset=["Height"])
    hist_2["Height"] = hist_2["Height"].astype(int)
    if "total_fees" in hist_2.columns:
        hist_2["total_fees"] = hist_2["total_fees"] / 100_000_000
    hist_2 = hist_2.rename(columns={"total_fees": "Total_Fees_BTC"})
    
    # 5) Merge
    hist = pd.merge(hist_1, hist_2[["Height", "Total_Fees_BTC"]], on="Height", how="left")
    hist["Total_Fees_BTC"] = hist["Total_Fees_BTC"].fillna(0.0)
    
    # 6) Finalize
    df = hist.set_index("Height", drop=False)[
        ["Time", "Inter_Block_Interval_Seconds", "Block_Time_Seconds", "Target", "Height", "Total_Fees_BTC"]
    ].copy()
    
    return df

# def load_block_paces(file_path_1: str, file_path_2: str) -> pd.DataFrame:
#     """
#     Load inter-block intervals and timestamps, merge with second CSV data,
#     and re-index the DataFrame on Height.
    
#     Returns a DataFrame indexed by Height with columns:
#       - 'Time'                          : datetime of each block
#       - 'Inter_Block_Interval_Seconds'  : seconds since previous block
#       - 'Target'
#       - 'Height'
#       - 'Total_Fees_BTC' (from file_path_2, converted from satoshis)
#     """
    
#     # 1) Read CSV 1 (main data), parse 'Time', clean 'Height'
#     #hist_1 = pd.read_csv(file_path_1, parse_dates=["Time"])
#     hist_1 = pd.read_csv(
#     file_path_1,
#     parse_dates=["Time"],
#     dtype={"Target": "string", "Bits": "string"},   # preserve exact text; do NOT parse as float
#     )
#     hist_1.columns = hist_1.columns.str.strip()
#     hist_1 = hist_1.dropna(subset=["Height"])
#     hist_1["Height"] = hist_1["Height"].astype(int)

#     # Canonical UTC time from epoch seconds (prevents DST artifacts)
#     if "Block_Time_Seconds" in hist_1.columns:
#         hist_1["Block_Time_Seconds"] = pd.to_numeric(hist_1["Block_Time_Seconds"], errors="coerce")
#         hist_1 = hist_1.dropna(subset=["Block_Time_Seconds"])
#         hist_1["Block_Time_Seconds"] = hist_1["Block_Time_Seconds"].astype(np.int64)
    
#         # overwrite Time with UTC-derived datetime
#         hist_1["Time"] = pd.to_datetime(hist_1["Block_Time_Seconds"], unit="s", utc=True).dt.tz_convert(None)
    
#         # (optional but recommended) recompute inter-block interval from epoch
#         hist_1 = hist_1.sort_values("Height")
#         hist_1["Inter_Block_Interval_Seconds"] = hist_1["Block_Time_Seconds"].diff().fillna(0).astype(float)

#     # # Parse Target to exact integer
#     # if "Target" in hist_1.columns:
#     #     hist_1["Target"] = hist_1["Target"].apply(parse_target_int)
#     #     hist_1 = hist_1.dropna(subset=["Target"])
#     # else:
#     #     raise ValueError("CSV missing 'Target' column. Consider using 'Bits' and decoding target instead.")

#     # Prefer canonical target decoded from Bits/nBits if present
#     bits_col = None
#     for c in ["Bits", "nBits", "bits", "nbits"]:
#         if c in hist_1.columns:
#             bits_col = c
#             break
    
#     if bits_col is not None:
#         hist_1["Bits_int"] = hist_1[bits_col].apply(parse_bits_int)
#         hist_1 = hist_1.dropna(subset=["Bits_int"])
#         hist_1["Target"] = hist_1["Bits_int"].apply(lambda b: compact_to_target(int(b)))
#     else:
#         # Fallback to Target column parsing ONLY if Bits is unavailable
#         if "Target" not in hist_1.columns:
#             raise ValueError("Historical file missing both Target and Bits/nBits columns.")
#         hist_1["Target"] = hist_1["Target"].apply(parse_target_int)
#         hist_1 = hist_1.dropna(subset=["Target"])
    

    
#     # 2) Read CSV 2 (new data) and process it
#     hist_2 = pd.read_csv(file_path_2)
#     hist_2 = hist_2.dropna(subset=["Height"])
#     hist_2["Height"] = hist_2["Height"].astype(int)
    
#     # Convert 'total_fees' from satoshis to BTC (1 BTC = 100,000,000 sats)
#     if "total_fees" in hist_2.columns:
#         hist_2["total_fees"] = hist_2["total_fees"] / 100_000_000
    
#     # Rename the column to 'Total_Fees_BTC'
#     hist_2 = hist_2.rename(columns={"total_fees": "Total_Fees_BTC"})
    
#     # Select only the merge key and the new column(s) you want to add
#     hist_2_subset = hist_2[["Height", "Total_Fees_BTC"]]
    
#     # 3) Merge the two DataFrames on 'Height'
#     # We use a 'left' merge to keep all records from hist_1
#     hist = pd.merge(hist_1, hist_2_subset, on="Height", how="left")

#     # Fill any missing transaction fees with 0.0
#     hist["Total_Fees_BTC"] = hist["Total_Fees_BTC"].fillna(0.0)
    
#     # 4) Filter (This line was commented out in original, keeping it)
#     #hist = hist[hist["Height"] < 100446]
    
#     # 5) Re-index on Height, then select all required columns
#     # 'Total_Fees_BTC' now refers to the new, converted column from hist_2
#     df = (
#         hist
#         .set_index("Height", drop=False)[["Time", "Inter_Block_Interval_Seconds", "Block_Time_Seconds", "Target", "Height", "Total_Fees_BTC", "Bits" ]]
#         .copy()
#     )
    
#     # 6) Replace any negative intervals (commented out in original)
#     # df["Inter_Block_Interval_Seconds"] = df["Inter_Block_Interval_Seconds"].mask(
#     #     df["Inter_Block_Interval_Seconds"] < 0, 600
#     # )

#     # df["Inter_Block_Interval_Seconds"] = df["Inter_Block_Interval_Seconds"].mask(
#     #     df["Inter_Block_Interval_Seconds"] == 0, 1
#     # )
    
#     return df
def get_max_block_height(file_path: str) -> int:
    """
    Load a block-paces CSV and return the highest block height found in the file.
    
    Steps:
      1) Read the CSV (parsing Time for consistency, though not used here).
      2) Drop any rows where Height is missing.
      3) Convert the Height column to integer.
      4) Return the maximum Height.
    """
    # 1) Read CSV, parse 'Time' column (optional)
    hist = pd.read_csv(file_path, parse_dates=["Time"])
    
    # 2) Drop rows missing Height
    hist = hist.dropna(subset=["Height"])
    
    # 3) Ensure Height is integer
    hist["Height"] = hist["Height"].astype(int)
    
    # 4) Return the maximum block height
    return int(hist["Height"].max())

def calculate_hashrate_rmse(
    simulation_df: pd.DataFrame,
    historical_file: str = BLOCK_PACE_DATA, 
    use_moving_avg: bool = True,
    window: int = 2016
) -> float:
    """
    Calculates the Root Mean Squared Error (RMSE) between the simulated
    and historical hashrate data.
    ... (rest of docstring) ...
    """
    
    # --- 1. Prepare simulation series ---
    simulation_df = simulation_df.copy()
    simulation_df['Timestamp'] = pd.to_datetime(simulation_df['Timestamp'])
    sim_df = simulation_df.set_index('Timestamp')

    if not sim_df.index.is_unique:
        print("Warning: Found duplicate timestamps in simulation data. Averaging.")
        sim_df = sim_df.groupby(sim_df.index).mean(numeric_only=True) 
        
    if use_moving_avg:
        sim_series = sim_df['H'].rolling(window='30D').mean()
    else:
        sim_series = sim_df['H'] 

    sim_th = sim_series * 1e-12
    start_time = sim_th.index.min()
    end_time   = sim_th.index.max()
    sim_th     = sim_th.loc[start_time:end_time]
    sim_th     = sim_th.rename("sim_hashrate")

    
    # --- 2. Load & calculate historical series ---
    hist_df = pd.read_csv(historical_file)
    hist_df.columns = hist_df.columns.str.strip()
    hist_df['Time'] = pd.to_datetime(hist_df['Time'])
    hist_df = hist_df.set_index('Time')
    
    # 1. Define helper to convert target to a numeric float
    def safe_target_to_numeric(target_val):
        try:
            # Convert to float. Using float handles large integers
            # and allows for NaNs.
            val = float(target_val) 
            if val == 0:
                # A target of 0 is invalid and will cause divide-by-zero
                return np.nan 
            return val
        except (ValueError, TypeError):
            # Handles any non-numeric strings, empty values, etc.
            return np.nan

    # 2. Apply this conversion.
    hist_df['Target'] = hist_df['Target'].apply(safe_target_to_numeric)
    
    # --- END OF FIX ---

    # Handle duplicate timestamps
    if not hist_df.index.is_unique:
        print("Warning: Found duplicate timestamps in historical data. Averaging.")
        # Now that 'Target' is float64, it will be correctly averaged.
        hist_df = hist_df.groupby(hist_df.index).mean(numeric_only=True)
    
    hist_df = hist_df.sort_index() 
    
    # This line will now work
    hist_df = hist_df.dropna(subset=['Target', 'Inter_Block_Interval_Seconds'])
    
    # This line will now work
    hist_df['D'] = DEFAULT_TARGET / hist_df['Target']
    
    d_arr = hist_df['D'].to_numpy()
    t_arr = hist_df['Inter_Block_Interval_Seconds'].to_numpy()
    h_arr = calc_core_hashrate(d_arr, t_arr, window=window)
    h_arr_th = h_arr * 1e-12
    
    hist_th = pd.Series(h_arr_th, index=hist_df.index, name="hist_hashrate")
    hist_th = hist_th.loc[start_time:end_time]
    hist_th = hist_th.rename("hist_hashrate") 

    
    # --- 3. Align data and Calculate Error ---
    combined_df = pd.concat([sim_th, hist_th], axis=1)
    resampled_df = combined_df.resample('D').mean()
    resampled_df = resampled_df.interpolate(method='linear')
    resampled_df = resampled_df.dropna()
    print(resampled_df.head())
    print(resampled_df.tail())

    if resampled_df.empty:
        print("Warning: No overlapping data after alignment. Cannot calculate error.")
        return np.nan

    squared_errors = (resampled_df['sim_hashrate'] - resampled_df['hist_hashrate']) ** 2
    mean_squared_error = squared_errors.mean()
    rmse = np.sqrt(mean_squared_error)

    return rmse


def calculate_hashrate_rmse_block_level(
    simulation_df: pd.DataFrame,
    historical_file: str = BLOCK_PACE_DATA,
    window: int = 2016,
) -> float:
    """
    Block-level RMSE between simulated and historical hashrate.

    - Aligns by block height (not by day).
    - Uses Core-style 2016-block window hashrate on both sides.
    """

    # ---------- 1. Simulation side ----------
    sim_df = simulation_df.copy()

    # Ensure block_height exists and is integer
    if 'block_height' not in sim_df.columns:
        raise ValueError("simulation_df must contain 'block_height' column.")

    sim_df = sim_df.dropna(subset=['H', 'block_height'])
    sim_df['block_height'] = sim_df['block_height'].astype(int)
    sim_df = sim_df.sort_values('block_height').set_index('block_height')

    # H is already Core-style hashrate in H/s (from main.py calc_core_hashrate)
    # convert to TH/s
    sim_th = sim_df['H'] * 1e-12
    sim_th.name = "sim_hashrate"

    # ---------- 2. Historical side ----------
    # hist_df = pd.read_csv(
    # historical_file,
    # dtype={"Target": "string"},   # keep exact text
    # )
    hist_df = load_block_paces(historical_file, TX_BLOCK_DATA).copy()
    # then use hist_df["Target"] (already canonical if Bits exists)
    hist_df["D"] = hist_df["Target"].map(lambda t: DEFAULT_TARGET / int(t))
    
    hist_df.columns = hist_df.columns.str.strip()
    
    required_cols = ["Height", "Target", "Inter_Block_Interval_Seconds"]
    missing = [c for c in required_cols if c not in hist_df.columns]
    if missing:
        raise ValueError(f"Historical file missing columns: {missing}")
    
    hist_df = hist_df.dropna(subset=required_cols).copy()
    hist_df["Height"] = hist_df["Height"].astype(int)
    
    # Parse target exactly
    hist_df["Target_int"] = hist_df["Target"].apply(parse_target_int)
    hist_df = hist_df.dropna(subset=["Target_int"])
    
    hist_df["Inter_Block_Interval_Seconds"] = pd.to_numeric(
        hist_df["Inter_Block_Interval_Seconds"],
        errors="coerce",
    )
    hist_df = hist_df.dropna(subset=["Inter_Block_Interval_Seconds"])
    
    # Difficulty from exact integer target
    hist_df["D"] = hist_df["Target_int"].map(lambda t: DEFAULT_TARGET / int(t))
    
    # Compute Core-style hashrate over same window
    d_arr = hist_df["D"].to_numpy()
    t_arr = hist_df["Inter_Block_Interval_Seconds"].to_numpy()
    h_arr = calc_core_hashrate(d_arr, t_arr, window=window)
    
    hist_df["hist_hashrate"] = h_arr * 1e-12  # TH/s
    hist_df = hist_df.set_index("Height", drop=False)
    
    # hist_df = pd.read_csv(historical_file)
    # hist_df.columns = hist_df.columns.str.strip()

    # required_cols = ['Height', 'Target', 'Inter_Block_Interval_Seconds']
    # missing = [c for c in required_cols if c not in hist_df.columns]
    # if missing:
    #     raise ValueError(f"Historical file missing columns: {missing}")

    # hist_df = hist_df.dropna(subset=required_cols)
    # hist_df['Height'] = hist_df['Height'].astype(int)

    # # Safe conversion for Target
    # def safe_target_to_numeric(target_val):
    #     try:
    #         val = float(target_val)
    #         if val == 0:
    #             return np.nan
    #         return val
    #     except (ValueError, TypeError):
    #         return np.nan

    # hist_df['Target'] = hist_df['Target'].apply(safe_target_to_numeric)
    # hist_df = hist_df.dropna(subset=['Target'])

    # # Difficulty from historical target
    # hist_df['D'] = DEFAULT_TARGET / hist_df['Target']

    # # Compute Core-style hashrate over the same window
    # d_arr = hist_df['D'].to_numpy()
    # t_arr = hist_df['Inter_Block_Interval_Seconds'].to_numpy()
    # h_arr = calc_core_hashrate(d_arr, t_arr, window=window)  # H/s

    # hist_df['hist_hashrate'] = h_arr * 1e-12  # TH/s
    # hist_df = hist_df.set_index('Height')
    hist_th = hist_df['hist_hashrate']

    # ---------- 3. Align by block height & compute RMSE ----------
    combined = pd.concat([sim_th, hist_th], axis=1, join='inner')

    # Drop blocks where either side is NaN (first window-1 blocks, etc.)
    combined = combined.dropna(subset=['sim_hashrate', 'hist_hashrate'])

    if combined.empty:
        print("Warning: No overlapping blocks after alignment. Cannot calculate RMSE.")
        return np.nan

    squared_errors = (combined['sim_hashrate'] - combined['hist_hashrate']) ** 2
    mean_squared_error = squared_errors.mean()
    rmse = float(np.sqrt(mean_squared_error))

    relrmse = rmse/combined['hist_hashrate'].mean()
    nrmse = rmse/(combined['hist_hashrate'].max() - combined['hist_hashrate'].min())
    stdrmse = rmse/combined['hist_hashrate'].std()

   #  Optional: print a quick sanity check slice
    print(combined.head())
    print(combined.tail())
    print(f"Block-level RMSE between simulation and historical hashrate: {rmse:.3e} TH/s")
    print(f'Relative RMSE: {relrmse:.5f}')
    print(f'Normalized RMSE: {nrmse:.5f}')
    print(f"STD RMSE: {stdrmse:.5f} and STD: {combined['hist_hashrate'].std():.5f}")

    return rmse

def calculate_hashrate_rmsle_block_level(
    simulation_df: pd.DataFrame,
    historical_file: str = BLOCK_PACE_DATA,
    window: int = 2016,
) -> float:
    """
    Block-level RMSLE between simulated and historical hashrate.
    
    Metric: Root Mean Squared Logarithmic Error
    - Aligns by block height.
    - Uses Core-style 2016-block window hashrate on both sides.
    - Applies log1p transformation to handle scale differences across eras.
    """

    # ---------- 1. Simulation side ----------
    sim_df = simulation_df.copy()

    # Ensure block_height exists and is integer
    if 'block_height' not in sim_df.columns:
        raise ValueError("simulation_df must contain 'block_height' column.")

    sim_df = sim_df.dropna(subset=['H', 'block_height'])
    sim_df['block_height'] = sim_df['block_height'].astype(int)
    sim_df = sim_df.sort_values('block_height').set_index('block_height')

    # H is already Core-style hashrate in H/s (from main.py calc_core_hashrate)
    # convert to TH/s
    sim_th = sim_df['H'] * 1e-12
    sim_th.name = "sim_hashrate"

    # ---------- 2. Historical side ----------
    # Load using the project's standard loader
    hist_df = load_block_paces(historical_file, TX_BLOCK_DATA).copy()
    
    # Clean and Format
    required_cols = ["Height", "Target", "Inter_Block_Interval_Seconds"]
    missing = [c for c in required_cols if c not in hist_df.columns]
    if missing:
        raise ValueError(f"Historical file missing columns: {missing}")
    
    hist_df = hist_df.dropna(subset=required_cols).copy()
    hist_df["Height"] = hist_df["Height"].astype(int)
    
    # Calculate Difficulty from Target
    # Using astype(float) as seen in sensitivity_analysis scripts
    hist_df["Target"] = hist_df["Target"].astype(float)
    hist_df["D"] = DEFAULT_TARGET / hist_df["Target"]
    
    # Compute Core-style hashrate over the same window
    d_arr = hist_df["D"].to_numpy()
    t_arr = hist_df["Inter_Block_Interval_Seconds"].to_numpy()
    h_arr = calc_core_hashrate(d_arr, t_arr, window=window)
    
    hist_df["hist_hashrate"] = h_arr * 1e-12  # TH/s
    hist_df = hist_df.set_index("Height", drop=False)
    hist_th = hist_df['hist_hashrate']

    # ---------- 3. Align by block height & compute RMSLE ----------
    combined = pd.concat([sim_th, hist_th], axis=1, join='inner')

    # Drop blocks where either side is NaN (first window-1 blocks, etc.)
    combined = combined.dropna(subset=['sim_hashrate', 'hist_hashrate'])

    if combined.empty:
        print("Warning: No overlapping blocks after alignment. Cannot calculate RMSLE.")
        return np.nan

    # Clip negative values just in case (hashrate > 0)
    sim_vals = combined['sim_hashrate'].clip(lower=0)
    hist_vals = combined['hist_hashrate'].clip(lower=0)

    # Log transform (log1p adds 1 automatically to handle zeros safely)
    log_sim = np.log1p(sim_vals)
    log_hist = np.log1p(hist_vals)

    # Mean Squared Logarithmic Error
    msle = ((log_sim - log_hist) ** 2).mean()
    rmsle = np.sqrt(msle)

    # Optional: Debug prints
    print(f"Block-level RMSLE calculated over {len(combined)} blocks.")
    print(f"RMSLE: {rmsle:.4f}")

    return rmsle