# modules/visualization.py

"""
Visualization Module

This module provides functions to visualize simulation and hindcasting results.
It uses both matplotlib and Plotly to generate plots such as:
  - Simulation block times over time
  - Network hashrate trends over time
  - Energy consumption over time
  - An overview plot combining several key metrics
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
import plotly.graph_objects as go
import pandas as pd
from modules.network import calc_core_hashrate
from config import DEFAULT_TARGET, C_ELEC, BLOCK_PACE_DATA        

def plot_block_times(simulation_df: pd.DataFrame) -> None:
    """
    Plot block times over simulation time using matplotlib.

    The simulation_df should have at least the following columns:
      - 'Timestamp': The simulated time (or block number if time not available)
      - 'Block_Time_Seconds': Block time for each simulation step in seconds

    Args:
        simulation_df (pd.DataFrame): DataFrame containing simulation results.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(simulation_df['Timestamp'], simulation_df['Block_Time_Seconds'], 
             marker='o', linestyle='-', color='blue')
    plt.xlabel("Time")
    plt.ylabel("Block Time (seconds)")
    plt.title("Simulation Block Times Over Time")
    plt.ticklabel_format(style='plain', axis='y', useOffset=False)
    plt.grid(True)
    plt.show()

def plot_hashrate(
    simulation_df: pd.DataFrame,
    historical_file: str = BLOCK_PACE_DATA, 
    use_moving_avg: bool = True,
    window: int = 2016
) -> None:
    """
    Plot network hashrate over simulation time...
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
        sim_label  = 'Simulated 30D MA Hashrate'
        sim_marker = None 
    else:
        sim_series = sim_df['H']
        sim_label  = 'Simulated Hashrate'
        sim_marker = None 

    sim_th = sim_series * 1e-12
    start_time = sim_th.index.min()
    end_time   = sim_th.index.max()
    sim_th     = sim_th.loc[start_time:end_time]

    
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

    # Handle duplicate timestamps
    if not hist_df.index.is_unique:
        print("Warning: Found duplicate timestamps in historical data. Averaging.")
        # Now that 'Target' is float64, it will be correctly averaged.
        hist_df = hist_df.groupby(hist_df.index).mean(numeric_only=True)
    
    hist_df = hist_df.sort_index() 
    
    # This line will now work
    hist_df = hist_df.dropna(subset=['Target', 'Inter_Block_Interval_Seconds'])
    #hist_df['Target'] = hist_df['Target'].astype(float)
    
    # This line will also work
    hist_df['D'] = DEFAULT_TARGET / hist_df['Target']
    
    d_arr = hist_df['D'].to_numpy()
    t_arr = hist_df['Inter_Block_Interval_Seconds'].to_numpy()
    h_arr = calc_core_hashrate(d_arr, t_arr, window=window)
    h_arr_th = h_arr * 1e-12
    
    hist_th = pd.Series(h_arr_th, index=hist_df.index, name="Historical Hashrate")
    hist_th = hist_th.loc[start_time:end_time]


    # --- 3. Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(
        sim_th.index, sim_th.values,
        marker=sim_marker, linestyle='-',
        color='green', label=sim_label
    )
    plt.plot(
        hist_th.index, hist_th.values,
        marker=None, linestyle='-', 
        color='red', label=f'Historical {window}-Block MA Hashrate'
    )
    plt.xlabel("Time")
    plt.ylabel("Hashrate (TH/s)")
    title = "Network Hashrate: Simulation vs Historical"
    if use_moving_avg:
        title += " (Simulated 30-Day MA)"
    plt.title(title)
    plt.ticklabel_format(style='plain', axis='y', useOffset=False)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../data/hashrate_comparison.png") 
    print("Hashrate comparison plot saved to ../data/hashrate_comparison.png")
    plt.show()

    # # --- 3b. Interactive Plotly plot ---
    # import plotly.graph_objs as go
    # from plotly.offline import plot as plotly_plot  # or use fig.show() in notebooks

    # # Build traces
    # sim_trace = go.Scatter(
    #     x=sim_th.index,
    #     y=sim_th.values,
    #     mode='lines',  # or 'lines+markers' if you want markers
    #     name=sim_label,
    #     line=dict(width=2)
    # )

    # hist_trace = go.Scatter(
    #     x=hist_th.index,
    #     y=hist_th.values,
    #     mode='lines',
    #     name=f'Historical {window}-Block MA Hashrate',
    #     line=dict(width=2)
    # )

    # # Create figure
    # fig = go.Figure(data=[sim_trace, hist_trace])

    # # Title text reused from Matplotlib section
    # plotly_title = "Network Hashrate: Simulation vs Historical"
    # if use_moving_avg:
    #     plotly_title += " (Simulated 30-Day MA)"

    # fig.update_layout(
    #     title=plotly_title,
    #     xaxis_title="Time",
    #     yaxis_title="Hashrate (TH/s)",
    #     hovermode='x unified',
    #     legend=dict(
    #         orientation="h",
    #         yanchor="bottom",
    #         y=1.02,
    #         xanchor="right",
    #         x=1
    #     ),
    #     margin=dict(l=60, r=30, t=80, b=60)
    # )

    # # Optional: disable y-axis scientific notation for large numbers
    # fig.update_yaxes(tickformat=".0f")

    # # Show in notebook / interactive environment
    # fig.show()


def plot_energy_consumption(simulation_df: pd.DataFrame) -> None:
    """
    Plot energy consumption over simulation time using matplotlib.

    The simulation_df should have at least the following columns:
      - 'Timestamp': The simulated time
      - 'E': Energy consumption in Watt-hours for each simulation step

    Args:
        simulation_df (pd.DataFrame): DataFrame containing simulation results.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(simulation_df['Timestamp'], simulation_df['E'] / 1e6, 
             marker='o', linestyle='-', color='red')
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption (MWh)")
    plt.title("Energy Consumption Over Time")
    plt.ticklabel_format(style='plain', axis='y', useOffset=False)
    plt.grid(True)
    plt.show()

def plot_simulation_overview(simulation_df: pd.DataFrame) -> None:
    """
    Create an overview plot using Plotly that shows key simulation metrics on one graph.

    The simulation_df must include columns:
      - 'Timestamp'
      - 'H' (Hashrate)
      - 'Block_Time_Seconds'
      - 'E' (Energy Consumption)

    Args:
        simulation_df (pd.DataFrame): DataFrame containing simulation results.
    """
    fig = go.Figure()

    # Add hashrate trace
    fig.add_trace(go.Scatter(
        x=simulation_df['Timestamp'], 
        y=simulation_df['H'], 
        mode='lines+markers', 
        name='Hashrate (H/s)'
    ))

    # Add block time trace
    fig.add_trace(go.Scatter(
        x=simulation_df['Timestamp'], 
        y=simulation_df['Block_Time_Seconds'], 
        mode='lines+markers', 
        name='Block Time (s)'
    ))

    # Add energy consumption trace
    fig.add_trace(go.Scatter(
        x=simulation_df['Timestamp'], 
        y=simulation_df['E'], 
        mode='lines+markers', 
        name='Energy (Wh)'
    ))

    fig.update_layout(
        title="Simulation Overview",
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="x unified"
    )
    fig.show()

def plot_dynamic_efficiency(simulation_df: pd.DataFrame) -> None:
    """
    Plot dynamic machine efficiency over simulation time using matplotlib.
    
    The simulation_df should have at least:
      - 'Timestamp': the simulation time (datetime)
      - 'efficiency' (or specified column): dynamic efficiency in J/hash
    
    Args:
        simulation_df (pd.DataFrame): DataFrame containing simulation results.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(simulation_df['Timestamp'], simulation_df['efficiency'],  # Convert J/hash to W/TH/s
             marker='o', linestyle='-', color='purple')
    plt.xlabel("Time")
    plt.ylabel("Dynamic Efficiency (W/TH/s)")
    plt.title("Dynamic Machine Efficiency Over Time")
    plt.ticklabel_format(style='plain', axis='y', useOffset=False)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("example_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_block_time_distribution(simulation_df, bins=30):
    """
    Plot a histogram of the block times from the simulation.

    Args:
        simulation_df (pd.DataFrame): DataFrame containing simulation results.
            This must include a column named "Block_Time_Seconds" (or similar)
            that holds the block time (in seconds) for each simulation step.
        bins (int): Number of bins for the histogram (default is 30).

    Returns:
        None. Displays a matplotlib histogram.
    """
    # Ensure the DataFrame contains the block time column.
    if "Block_Time_Seconds" not in simulation_df.columns:
        raise ValueError("The simulation DataFrame must contain 'Block_Time_Seconds' column.")

    block_times = simulation_df["Block_Time_Seconds"]

    plt.figure(figsize=(10, 6))
    plt.hist(block_times, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Block Times")
    plt.xlabel("Block Time (seconds)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def plot_annual_energy(
    simulation_df: pd.DataFrame,
    energy_col: str = "E",
    timestamp_col: str = "Timestamp",
) -> None:
    """
    Plot annual network energy consumption in TWh.

    Parameters
    ----------
    simulation_df : pd.DataFrame
        Must contain:
          • timestamp_col – datetime64 column for binning by year
          • energy_col    – energy per block in Wh
    energy_col : str, optional
        Column holding block-level energy in Wh (default "E").
    timestamp_col : str, optional
        Column to use for calendar years (default "Timestamp").
    """
    if timestamp_col not in simulation_df.columns or energy_col not in simulation_df.columns:
        raise ValueError(
            f"DataFrame must contain '{timestamp_col}' and '{energy_col}' columns."
        )

    df = simulation_df.copy()
    df = df.dropna(subset=[timestamp_col, energy_col])

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["Year"] = df[timestamp_col].dt.year

    annual_TWh = df.groupby("Year")[energy_col].sum() / 1e12
    print(annual_TWh)

    plt.figure(figsize=(8, 4))
    annual_TWh.plot(kind="bar", color="steelblue")
    plt.ylabel("Energy consumption (TWh)")
    plt.title("Annual Bitcoin-network energy use (simulation)")
    plt.tight_layout()
    plt.show()

def plot_efficiency_over_time(
    simulation_df: pd.DataFrame,
    efficiency_col: str = "efficiency",
    units: str = "W/TH",               # "W/TH" (default) or "J/hash"
    rolling_days: int | None = 30      # None → no smoothing
) -> None:
    """
    Plot mining‑hardware efficiency over time.

    Parameters
    ----------
    simulation_df : pd.DataFrame
        Must contain 'Timestamp' (datetime64) and `efficiency_col`.
    efficiency_col : str, optional
        Column holding efficiency values (default "efficiency").
    units : {"W/TH", "J/hash"}, optional
        • "W/TH"   : assumes values already scaled by 1e12
        • "J/hash" : divides by 1e12 on the fly
    rolling_days : int or None, optional
        Trailing‑mean window size (days). Set to None/0 for raw series.
    """
    # Guard‑rails
    if "Timestamp" not in simulation_df.columns or efficiency_col not in simulation_df.columns:
        raise ValueError("DataFrame must contain 'Timestamp' and the specified efficiency column.")

    # Copy & optionally convert units
    df = simulation_df[["Timestamp", efficiency_col]].copy()
    if units.lower() == "j/hash":
        df[efficiency_col] = df[efficiency_col] / 1e12
        ylabel = "Efficiency (J / hash)"
    elif units.lower() == "w/th":
        ylabel = "Efficiency (W / TH·s)"
    else:
        raise ValueError("units must be either 'W/TH' or 'J/hash'.")

    # Optional rolling mean
    if rolling_days and rolling_days > 1:
        df = df.set_index("Timestamp")
        df[efficiency_col] = df[efficiency_col].rolling(
            window=f"{rolling_days}D", min_periods=1
        ).mean()
        df = df.reset_index()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Timestamp"], df[efficiency_col], color="darkmagenta")
    ax.ticklabel_format(axis="y", style="plain")   # no scientific notation
    ax.set_title("Simulated Mining‑Hardware Efficiency Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

def plot_price_usd(simulation_df):
    """
    Plot the simulated Bitcoin price (P_USD) over time.

    Args:
        simulation_df (pd.DataFrame): DataFrame containing at least:
            - 'Timestamp': datetime for each block
            - 'P_USD'    : simulated Bitcoin price in USD
    """
    # Guard‐rails
    if 'Timestamp' not in simulation_df.columns or 'P_USD' not in simulation_df.columns:
        raise ValueError("DataFrame must contain 'Timestamp' and 'P_USD' columns.")

    # Ensure Timestamp is datetime
    sim = simulation_df.copy()
    sim['Timestamp'] = pd.to_datetime(sim['Timestamp'])

    plt.figure(figsize=(10, 6))
    plt.plot(sim['Timestamp'], sim['P_USD'], marker='o', linestyle='-', label='P_USD')
    plt.xlabel("Time")
    plt.ylabel("Bitcoin Price (USD)")
    plt.title("Simulated Bitcoin Price Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cost_per_btc(
    simulation_df: pd.DataFrame,
    C_elec: float = C_ELEC,
    energy_col: str = "E",
    subsidy_col: str = "R_block",
    fee_col: str = "TX_fee",
    rolling_window: str = "30D",
    overlay_price: bool = True,
) -> None:
    """
    Compute and plot the marginal electricity cost per BTC over time:
        C_BTC(t) = C_elec * Energy_per_BTC(t)

    where:
        - C_elec is in $/MWh (from config.C_ELEC by default)
        - Energy_per_BTC is in MWh/BTC (derived from simulation_df)

    Parameters
    ----------
    simulation_df : pd.DataFrame
        Must contain at least:
          - 'Timestamp'      : datetime per block
          - energy_col       : block-level energy in Wh (default 'E')
          - subsidy_col      : block subsidy in BTC (default 'R_block')
          - fee_col          : transaction fees in BTC (default 'TX_fee')
          - 'P_USD'          : BTC price in USD (only needed if overlay_price=True)
    C_elec : float, optional
        Electricity price in $/MWh. Defaults to config.C_ELEC.
    energy_col : str, optional
        Column name holding block-level energy in Wh.
    subsidy_col : str, optional
        Column name holding block subsidy in BTC.
    fee_col : str, optional
        Column name holding transaction fees in BTC.
    rolling_window : str, optional
        Pandas offset for rolling mean (e.g., '30D', '7D'). Set to None to disable smoothing.
    overlay_price : bool, optional
        If True, overlays P_USD on a secondary y-axis for visual comparison.

    Returns
    -------
    None. Shows a matplotlib plot.
    """
    if "Timestamp" not in simulation_df.columns:
        raise ValueError("simulation_df must contain 'Timestamp' column.")
    for col in (energy_col, subsidy_col, fee_col):
        if col not in simulation_df.columns:
            raise ValueError(f"simulation_df must contain '{col}' column.")

    df = simulation_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp")

    # 1) Energy per BTC (MWh/BTC)
    #    - E_block in Wh
    #    - BTC_block = subsidy + fees
    #    - Energy_per_BTC_MWh = E_block / 1e6 / BTC_block
    E_Wh = df[energy_col].astype(float)
    BTC_block = (df[subsidy_col].fillna(0.0) +
                 df[fee_col].fillna(0.0)).astype(float)

    # Avoid division by zero (e.g., weird early blocks)
    BTC_block = BTC_block.replace(0.0, np.nan)

    energy_per_btc_MWh = (E_Wh / 1e6) / BTC_block  # MWh/BTC

    # 2) Cost per BTC in USD
    #    C_elec is $/MWh → multiply by MWh/BTC → $/BTC
    C_btc = C_elec * energy_per_btc_MWh
    df["C_BTC_USD"] = C_btc

    # 3) Optional rolling (Kristoufek-style smoothed marginal cost)
    if rolling_window is not None:
        df = df.set_index("Timestamp")
        df["C_BTC_USD_smoothed"] = df["C_BTC_USD"].rolling(
            window=rolling_window, min_periods=1
        ).mean()
        if overlay_price and "P_USD" in df.columns:
            df["P_USD_smoothed"] = df["P_USD"].rolling(
                window=rolling_window, min_periods=1
            ).mean()
        df = df.reset_index()
    else:
        df["C_BTC_USD_smoothed"] = df["C_BTC_USD"]
        if overlay_price and "P_USD" in df.columns:
            df["P_USD_smoothed"] = df["P_USD"]

    # 4) Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(
        df["Timestamp"],
        df["C_BTC_USD_smoothed"],
        label=f"Electricity cost per BTC (smoothed {rolling_window})",
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Cost per BTC (USD)")
    ax1.ticklabel_format(axis="y", style="plain")
    ax1.grid(True, alpha=0.3)

    # Optional overlay: BTC price
    if overlay_price and "P_USD_smoothed" in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(
            df["Timestamp"],
            df["P_USD_smoothed"],
            linestyle="--",
            label="BTC price (smoothed)",
        )
        ax2.set_ylabel("BTC price (USD)")
        ax2.ticklabel_format(axis="y", style="plain")

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
    else:
        ax1.legend(loc="upper left")

    title = "Simulated marginal electricity cost per BTC"
    if rolling_window is not None:
        title += f" (rolling {rolling_window})"
    ax1.set_title(title)

    fig.tight_layout()
    plt.show()

def plot_margin_index(
    simulation_df: pd.DataFrame,
    C_elec: float = C_ELEC,
    energy_col: str = "E",
    subsidy_col: str = "R_block",
    fee_col: str = "TX_fee",
    price_col: str = "P_USD",
    rolling_window: str = "30D",
) -> None:
    """
    Compute and plot the profit-margin index over time:

        MarginIndex(t) = P_USD(t) / C_BTC(t)

    where:
        - C_BTC(t) = C_elec * Energy_per_BTC(t)
        - C_elec is in $/MWh
        - Energy_per_BTC is in MWh/BTC derived from simulation_df

    Parameters
    ----------
    simulation_df : pd.DataFrame
        Must contain at least:
          - 'Timestamp'           : datetime per block
          - energy_col            : block-level energy in Wh (default 'E')
          - subsidy_col           : block subsidy in BTC (default 'R_block')
          - fee_col               : transaction fees in BTC (default 'TX_fee')
          - price_col             : BTC price in USD (default 'P_USD')
    C_elec : float, optional
        Electricity price in $/MWh. Defaults to config.C_ELEC.
    energy_col : str, optional
        Column name holding block-level energy in Wh.
    subsidy_col : str, optional
        Column name holding block subsidy in BTC.
    fee_col : str, optional
        Column name holding transaction fees in BTC.
    price_col : str, optional
        Column name holding BTC price in USD.
    rolling_window : str, optional
        Pandas offset string for rolling mean (e.g., '30D', '7D').
        Set to None to disable smoothing.

    Returns
    -------
    None. Shows a matplotlib plot.
    """
    required_cols = ["Timestamp", energy_col, subsidy_col, fee_col, price_col]
    missing = [c for c in required_cols if c not in simulation_df.columns]
    if missing:
        raise ValueError(f"simulation_df is missing required columns: {missing}")

    df = simulation_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp")

    # --- 1) Energy per BTC in MWh/BTC ---
    # E_block in Wh
    E_Wh = df[energy_col].astype(float)

    # BTC produced in block (subsidy + fees)
    BTC_block = (
        df[subsidy_col].fillna(0.0) +
        df[fee_col].fillna(0.0)
    ).astype(float)

    # Avoid division by zero
    BTC_block = BTC_block.replace(0.0, np.nan)

    # Convert energy: Wh -> MWh, then divide by BTC
    energy_per_btc_MWh = (E_Wh / 1e6) / BTC_block  # MWh/BTC

    # --- 2) Electricity cost per BTC in USD ---
    C_btc = C_elec * energy_per_btc_MWh  # $/BTC

    df["C_BTC_USD"] = C_btc

    # --- 3) Margin index: Price / Cost ---
    P_usd = df[price_col].astype(float)
    df["MarginIndex"] = P_usd / df["C_BTC_USD"]

    # --- 4) Optional smoothing in time ---
    if rolling_window is not None:
        df = df.set_index("Timestamp")

        # Time-based rolling window (e.g., '30D')
        df["MarginIndex_smoothed"] = df["MarginIndex"].rolling(
            window=rolling_window, min_periods=1
        ).mean()

        df = df.reset_index()
    else:
        df["MarginIndex_smoothed"] = df["MarginIndex"]

    # --- 5) Plot MarginIndex over time ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df["Timestamp"],
        df["MarginIndex_smoothed"],
        label=f"Price / electricity cost per BTC (rolling {rolling_window})"
        if rolling_window is not None else "Price / electricity cost per BTC",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Margin index (dimensionless)")
    ax.set_title("BTC price to simulated electricity cost per BTC")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    plt.show()


def plot_time_varying_power_price_ceiling(
    simulation_df: pd.DataFrame,
    energy_col: str = "E",
    subsidy_col: str = "R_block",
    fee_col: str = "TX_fee",
    price_col: str = "P_USD",
    M_target: float | None = None,
    margin_smooth_window: str | None = "30D",
    overlay_assumed_CELEC: bool = True,
) -> None:
    """
    Compute and plot the time-varying maximum *average* power price implied
    by a target price-to-electricity-cost margin M_target.

    k_max(t) = (1 / M_target) * P_BTC(t) / EnergyPerBTC(t)

    where:
      - EnergyPerBTC(t) comes from simulated block-level energy,
      - P_BTC(t) is the simulated BTC price in USD,
      - M_target is a chosen equilibrium margin (default: long-run median of the
        observed MarginIndex in the simulation).

    Parameters
    ----------
    simulation_df : pd.DataFrame
        Must contain at least:
          - 'Timestamp'
          - energy_col  : block-level energy in Wh (default 'E')
          - subsidy_col : block subsidy in BTC (default 'R_block')
          - fee_col     : transaction fees in BTC (default 'TX_fee')
          - price_col   : BTC price in USD (default 'P_USD')
    M_target : float, optional
        Target margin index (price / elec-cost-per-BTC). If None, use the
        long-run median of the simulated MarginIndex.
    margin_smooth_window : str or None
        Optional time-based rolling window (e.g. '30D') to smooth the
        MarginIndex before computing its median (when M_target is None).
    overlay_assumed_CELEC : bool
        If True, overlay the model's assumed C_ELEC (in $/MWh).

    Returns
    -------
    None. Shows a matplotlib plot.
    """
    required_cols = ["Timestamp", energy_col, subsidy_col, fee_col, price_col]
    missing = [c for c in required_cols if c not in simulation_df.columns]
    if missing:
        raise ValueError(f"simulation_df is missing required columns: {missing}")

    df = simulation_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp")

    # --- 1) Energy per BTC in MWh/BTC ---
    E_Wh = df[energy_col].astype(float)

    BTC_block = (
        df[subsidy_col].fillna(0.0) +
        df[fee_col].fillna(0.0)
    ).astype(float)

    BTC_block = BTC_block.replace(0.0, np.nan)
    energy_per_btc_MWh = (E_Wh / 1e6) / BTC_block  # MWh/BTC
    df["EnergyPerBTC_MWh"] = energy_per_btc_MWh

    # --- 2) Electricity cost per BTC using assumed C_ELEC ---
    df["C_elec_BTC_USD"] = C_ELEC * df["EnergyPerBTC_MWh"]

    # --- 3) Observed margin index ---
    P_usd = df[price_col].astype(float)
    df["MarginIndex"] = P_usd / df["C_elec_BTC_USD"]

    # Optionally smooth the margin before taking its long-run median
    if M_target is None:
        tmp = df.set_index("Timestamp")
        if margin_smooth_window is not None:
            tmp["MarginIndex_smoothed"] = (
                tmp["MarginIndex"]
                .rolling(window=margin_smooth_window, min_periods=1)
                .mean()
            )
            margin_series = tmp["MarginIndex_smoothed"]
        else:
            margin_series = tmp["MarginIndex"]

        M_target = float(np.nanmedian(margin_series.values))

    # --- 4) Time-varying max average power price ---
    # k_max(t) = (1 / M_target) * P_BTC(t) / EnergyPerBTC(t)
    df["k_max_USD_per_MWh"] = (P_usd / df["EnergyPerBTC_MWh"]) / M_target

    # --- 5) Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df["Timestamp"],
        df["k_max_USD_per_MWh"],
        label=f"Implied max average power price (M_target = {M_target:.2f})",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Max average power price ($/MWh)")
    ax.set_title("Time-varying maximum average power price implied by model")
    ax.grid(True, alpha=0.3)

    if overlay_assumed_CELEC:
        ax.axhline(
            C_ELEC,
            linestyle="--",
            label=f"Assumed C_ELEC (model input) = {C_ELEC:.2f} $/MWh",
        )

    ax.legend(loc="best")
    fig.tight_layout()
    plt.show()

def compute_and_plot_annual_energy_from_hashrate(
    hashrate,
    efficiency,
    pue=1.10,
    energy_unit="TWh",
):
    """
    Compute annual energy consumption from hashrate using daily power/energy,
    then print and plot the annual values.

    Parameters
    ----------
    hashrate : pandas.Series
        Time series of network hashrate with a DatetimeIndex.
        Units assumed to be TH/s.
    efficiency : float or pandas.Series
        Mining efficiency in J/TH. Can be:
          - scalar (constant over time), or
          - Series with DatetimeIndex, which will be aligned to daily data.
    pue : float, optional
        Power Usage Effectiveness. Default is 1.10.
    energy_unit : {"MWh", "GWh", "TWh"}, optional
        Unit for reporting and plotting annual energy.

    Returns
    -------
    annual_energy : pandas.Series
        Annual energy consumption in the requested unit, indexed by year-end timestamp.
    """

    if not isinstance(hashrate, pd.Series):
        raise TypeError("hashrate must be a pandas.Series")

    if not isinstance(hashrate.index, pd.DatetimeIndex):
        raise TypeError("hashrate index must be a DatetimeIndex")

    # 1) Resample hashrate to daily mean (TH/s)
    H_daily = hashrate.resample("D").mean()

    # 2) Build a daily efficiency series (J/TH)
    if np.isscalar(efficiency):
        eta_daily = pd.Series(float(efficiency), index=H_daily.index)
    elif isinstance(efficiency, pd.Series):
        if not isinstance(efficiency.index, pd.DatetimeIndex):
            raise TypeError("efficiency Series must have a DatetimeIndex")
        # Align to daily index: resample then ffill to avoid gaps
        eta_daily = (
            efficiency.resample("D").mean()
            .reindex(H_daily.index)
            .ffill()
        )
    else:
        raise TypeError("efficiency must be a float or pandas.Series")

    # 3) Daily power (W): H_daily [TH/s] * eta_daily [J/TH] = J/s = W, times PUE
    power_W_daily = H_daily * eta_daily * pue

    # 4) Daily energy: E_d = P_d * 24 h
    #    Wh: W * h  -> multiply by 24
    daily_energy_Wh = power_W_daily * 24.0

    # 5) Aggregate to annual energy
    annual_energy_Wh = daily_energy_Wh.resample("Y").sum()

    # 6) Convert to requested unit
    factor = {"MWh": 1e6, "GWh": 1e9, "TWh": 1e12}
    if energy_unit not in factor:
        raise ValueError("energy_unit must be one of 'MWh', 'GWh', or 'TWh'")

    annual_energy = annual_energy_Wh / factor[energy_unit]

    # 7) Print year + energy
    print(f"Annual energy consumption ({energy_unit}):")
    for ts, val in annual_energy.items():
        print(f"{ts.year}: {val:.3f} {energy_unit}")

    # 8) Bar plot
    years = annual_energy.index.year.astype(str)
    plt.figure(figsize=(8, 4))
    plt.bar(years, annual_energy.values)
    plt.xlabel("Year")
    plt.ylabel(f"Energy consumption ({energy_unit})")
    plt.title("Annual energy consumption from hashrate (daily power/energy)")
    plt.tight_layout()
    plt.show()

    return annual_energy

