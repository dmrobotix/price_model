# modules/monte_carlo.py

from __future__ import annotations
import os
import math
import copy
from typing import Literal, Dict, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from modules.simulation import run_simulation
from modules.network import calc_core_hashrate
from modules.energy import calc_power_demand, calc_energy_consumption
from modules.economics import calc_expected_revenue, calc_price_energy_adjusted_revenue
from config import PERIOD_BLOCKS

MetricName = Literal["target","hashrate","R_t","hashprice_usd_th_day","ea_hashprice_usd_mwh"]

# ---------- helpers ----------

def _ensure_derived_columns(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required derived columns exist on a per-run simulation_df:
      - Block_Time_Seconds
      - Cumulative_Seconds
      - Timestamp (already present in main.py, but make robust)
      - H (Core-style rolling hashrate with 2016-block window)
      - R_t (expected revenue per hash)
      - hashprice_usd_th_day
      - ea_hashprice_usd_mwh (== R_pe, USD/MWh)
    """
    df = sim_df.copy()

    # Block time & cumulative time
    if "T_block" in df and "Block_Time_Seconds" not in df:
        df["Block_Time_Seconds"] = df["T_block"]
    df["Block_Time_Seconds"] = df.get("Block_Time_Seconds", 0.0)
    df["Block_Time_Seconds"] = df["Block_Time_Seconds"].fillna(0.0)
    if "Cumulative_Seconds" not in df:
        df["Cumulative_Seconds"] = df["Block_Time_Seconds"].cumsum()

    # Timestamp (must be present by caller; keep as datetime)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Core-style hashrate over 2016-block window
    d_arr = df["D"].to_numpy()
    t_arr = df["Block_Time_Seconds"].to_numpy()
    H_arr = calc_core_hashrate(d_arr, t_arr, window=2016)
    df["H"] = H_arr

    # Expected revenue per hash (BTC/hash)
    if "R_t" not in df.columns:
        # Prefer vectorized using columns if present; fallback is safe
        if "R_block" in df.columns and "target" in df.columns:
            df["R_t"] = (df["R_block"] * df["target"]) / (2**256)
        else:
            df["R_t"] = df.apply(lambda r: calc_expected_revenue(r["R_block"], r["target"]), axis=1)

    # Hashprice ($/TH/s/day) = R_t [BTC/hash] * P_USD [$ / BTC] * 86400 s/day * 1e12 hash/TH
    df["hashprice_usd_th_day"] = df["R_t"] * df["P_USD"] * 86400.0 * 1e12

    # Energy-adjusted hashprice (USD/MWh)
    if "R_pe" in df.columns:
        df["ea_hashprice_usd_mwh"] = df["R_pe"]
    else:
        # compute if not present
        if "efficiency" not in df.columns:
            raise ValueError("efficiency column missing; needed to compute energy-adjusted hashprice.")
        df["ea_hashprice_usd_mwh"] = (3.6e9 * df["R_t"] * df["P_USD"]) / df["efficiency"]

    # --- Power (W) via energy module (overwrite stale column) ---
    if "efficiency" not in df.columns:
        raise ValueError("efficiency column missing; needed for power/energy.")
    
    # Core-style H was computed just above
    df["P"] = calc_power_demand(df["H"].astype(float), df["efficiency"].astype(float))
    
    # --- Per-block energy (Wh) via energy module (overwrite stale column) ---
    hours = df["Block_Time_Seconds"].astype(float) / 3600.0
    df["E"] = calc_energy_consumption(df["P"].astype(float), hours)


    # # Power (W) via energy module
    # if "P" not in df.columns:
    #     if "efficiency" not in df.columns:
    #         raise ValueError("efficiency column missing; needed for power/energy.")
    #     try:
    #         # If the functions accept array-like, this will vectorize automatically
    #         df["P"] = calc_power_demand(df["H"], df["efficiency"])
    #     except Exception:
    #         # Fallback to explicit vectorization if the function expects scalars
    #         df["P"] = pd.Series(
    #             np.vectorize(calc_power_demand)(
    #                 df["H"].to_numpy(dtype=float),
    #                 df["efficiency"].to_numpy(dtype=float),
    #             ),
    #             index=df.index,
    #         )

    # # Per-block energy (Wh) via energy module
    # if "E" not in df.columns:
    #     hours = df["Block_Time_Seconds"] / 3600.0
    #     try:
    #         df["E"] = calc_energy_consumption(df["P"], hours)
    #     except Exception:
    #         df["E"] = pd.Series(
    #             np.vectorize(calc_energy_consumption)(
    #                 df["P"].to_numpy(dtype=float),
    #                 hours.to_numpy(dtype=float),
    #             ),
    #             index=df.index,
    #         )

    return df


def _resample_by_time(df: pd.DataFrame, freq: Literal["D","W","M"], metrics: List[MetricName]) -> pd.DataFrame:
    """
    Resample to daily/weekly/monthly using the mean within the bin.
    Index: Timestamp (period end).
    """
    g = df.set_index("Timestamp").resample(freq)
    agg = g[metrics].mean()
    agg.index.name = "Timestamp"
    return agg


def _window_2016_blocks(df: pd.DataFrame, cutoff_ts: pd.Timestamp, metrics: List[MetricName]) -> pd.DataFrame:
    """
    Aggregate by Core retarget periods (2016 blocks) strictly after cutoff.
    We compute a period index k = floor(block_height/2016) and take means over each period.
    """
    out = df.copy()
    # Align to periods by integer division of block_height
    if "block_height" not in out.columns:
        # If absent, create a 0..N-1 index as a safe fallback
        out["block_height"] = np.arange(len(out), dtype=int)

    out["period_index"] = (out["block_height"] // PERIOD_BLOCKS).astype(int)
    out = out[out["Timestamp"] > cutoff_ts]

    # Per-period mean for metrics
    agg = out.groupby("period_index", as_index=True)[metrics].mean()
    agg.index.name = "PeriodIndex"
    return agg


def _common_time_index(resampled_runs: List[pd.DataFrame]) -> pd.DatetimeIndex:
    """
    Compute the intersection of all datetime indices (to avoid excessive NaNs).
    """
    idx = None
    for df in resampled_runs:
        cur = df.index
        idx = cur if idx is None else idx.intersection(cur)
    return idx if idx is not None else pd.DatetimeIndex([])


def _stack_and_percentiles(aligned_frames: List[pd.DataFrame], metrics: List[MetricName]) -> pd.DataFrame:
    """
    Given a list of frames on the same index, compute P10/P50/P90 per timestamp for each metric.
    Output columns: {metric}_p10, {metric}_p50, {metric}_p90
    """
    if not aligned_frames:
        return pd.DataFrame()

    # 3D array: (n_times, n_metrics, n_runs)
    index = aligned_frames[0].index
    m = len(metrics)
    n = len(aligned_frames)
    data = np.stack([df[metrics].to_numpy() for df in aligned_frames], axis=2)  # shape (T, m, N)

    p10 = np.nanpercentile(data, 10, axis=2)
    p50 = np.nanpercentile(data, 50, axis=2)
    p90 = np.nanpercentile(data, 90, axis=2)

    out = pd.DataFrame(index=index)
    for j, name in enumerate(metrics):
        out[f"{name}_p10"] = p10[:, j]
        out[f"{name}_p50"] = p50[:, j]
        out[f"{name}_p90"] = p90[:, j]
    return out


def _stack_and_percentiles_by_period(aligned_frames: List[pd.DataFrame], metrics: List[MetricName]) -> pd.DataFrame:
    """
    For 2016-block period tables (indexed by PeriodIndex).
    """
    if not aligned_frames:
        return pd.DataFrame()

    # Align by overlapping integer index 0..K
    common_idx = None
    for df in aligned_frames:
        idx = df.index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    if common_idx is None or len(common_idx) == 0:
        return pd.DataFrame()

    # Align all frames to common_idx
    aligned = [df.loc[common_idx, metrics] for df in aligned_frames]
    # 3D array (K, m, N)
    K = len(common_idx)
    m = len(metrics)
    N = len(aligned)
    data = np.stack([df.to_numpy() for df in aligned], axis=2)

    p10 = np.nanpercentile(data, 10, axis=2)
    p50 = np.nanpercentile(data, 50, axis=2)
    p90 = np.nanpercentile(data, 90, axis=2)

    out = pd.DataFrame(index=common_idx)
    out.index.name = "PeriodIndex"
    for j, name in enumerate(metrics):
        out[f"{name}_p10"] = p10[:, j]
        out[f"{name}_p50"] = p50[:, j]
        out[f"{name}_p90"] = p90[:, j]
    return out


def _annual_energy_mwh(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annual energy in MWh for a single run.
    Returns a two-column frame: Year, E_annual_MWh
    """
    tmp = df.copy()
    tmp["Year"] = tmp["Timestamp"].dt.year
    # E is Wh/block; convert to MWh
    out = tmp.groupby("Year", as_index=False)["E"].sum()
    out["E_annual_MWh"] = out["E"] / 1e6
    return out[["Year","E_annual_MWh"]]

def _annual_energy_twh(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annual energy in TWh for a single run.
    Returns a two-column frame: Year, E_annual_TWh
    """
    tmp = df.copy()
    tmp["Year"] = tmp["Timestamp"].dt.year
    out = tmp.groupby("Year", as_index=False)["E"].sum()  # E is Wh
    out["E_annual_TWh"] = out["E"] / 1e12                 # Wh → TWh
    return out[["Year","E_annual_TWh"]]


def _ci95(mean: float, std: float, n: int) -> float:
    return 1.96 * (std / math.sqrt(n))


# ---------- public API ----------

def run_mc_and_summarize(
    initial_state: dict,
    params: dict,
    num_steps: int,
    N: int = 200,
    seed_base: int = 12345,
    window: Literal["D","W","M","2016b"] = "M",
    scenario_name: str = "baseline",
    out_dir: str = "../data/mc",
    save_plots: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Run N Monte Carlo simulations, build P10–P90 fan bands for requested metrics,
    and compute annual energy stats with 95% CI on the mean.

    Returns a dict with keys:
      - 'bands_time'  (if window in D/W/M): percentiles over time
      - 'bands_2016b' (if window == 2016b): percentiles by period index
      - 'annual_energy_stats': per-year mean, std, N, CI, (and flags for forecast years)

    Also writes CSVs to {out_dir}/{scenario_name}/.
    """
    os.makedirs(f"{out_dir}/{scenario_name}", exist_ok=True)

    metrics: List[MetricName] = [
        "target",
        "hashrate",
        "R_t",
        "hashprice_usd_th_day",
        "ea_hashprice_usd_mwh",
    ]

    # Collect per-run processed frames + annual energy
    resampled_runs: List[pd.DataFrame] = []
    period_runs: List[pd.DataFrame] = []
    annual_list: List[pd.DataFrame] = []

    cutoff_ts = pd.to_datetime(params["historical_cutoff"])

    for run in range(N):
        # fresh state & seeded RNG
        state0 = copy.deepcopy(initial_state)
        np.random.seed(seed_base + run)

        history = run_simulation(state0, params, num_steps, initial_state.get("block_height", 0))
        df = pd.DataFrame(history).copy()

        # Timestamps & block_time might already be in the history; ensure derived cols
        # (Main pipeline usually fills Timestamp in main.py; we rely on simulation history having it.)
        if "Timestamp" not in df.columns:
            # fallback: reconstruct from cumulative seconds + initial timestamp
            if "T_block" not in df.columns:
                df["T_block"] = 0.0
            df["T_block"] = df["T_block"].fillna(0.0)
            df["Cumulative_Seconds"] = df["T_block"].cumsum()
            start_date = initial_state["sim_timestamp"]
            df["Timestamp"] = pd.to_datetime(start_date) + pd.to_timedelta(df["Cumulative_Seconds"], unit="s")

        df = _ensure_derived_columns(df)

        # Slice strictly after historical cutoff for fan charts
        df_fc = df[df["Timestamp"] > cutoff_ts].copy()

        # Map internal 'hashrate' metric name onto df column 'H'
        df_fc = df_fc.assign(hashrate=df_fc["H"])

        # Time resampling (D/W/M) or 2016-block periods
        if window in ("D","W","M"):
            r = _resample_by_time(df_fc, window, metrics)
            resampled_runs.append(r)
        elif window == "2016b":
            p = _window_2016_blocks(df, cutoff_ts, metrics)
            period_runs.append(p)
        else:
            raise ValueError("window must be one of 'D','W','M','2016b'.")

        # Annual energy (whole series; we’ll flag forecast years later)
        annual_list.append(_annual_energy_twh(df))

        # Save raw per-run (optional for audit), lightly compressed
        df.to_csv(f"{out_dir}/{scenario_name}/simulation_run_{run:03d}.csv", index=False)

    # -------- Fan-band aggregation & CSVs --------
    outputs: Dict[str, pd.DataFrame] = {}

    if window in ("D","W","M"):
        # Intersect time indexes
        common_idx = _common_time_index(resampled_runs)
        aligned = [r.loc[common_idx] for r in resampled_runs]
        bands = _stack_and_percentiles(aligned, metrics)
        # Save
        bands.to_csv(f"{out_dir}/{scenario_name}/bands_{window}.csv")
        outputs["bands_time"] = bands
    else:
        # 2016-block periods
        bands_p = _stack_and_percentiles_by_period(period_runs, metrics)
        bands_p.to_csv(f"{out_dir}/{scenario_name}/bands_2016b.csv")
        outputs["bands_2016b"] = bands_p

    # -------- Annual energy stats (mean & 95% CI) --------
    # Outer join across runs, then compute stats per year
    annual_all = None
    for i, a in enumerate(annual_list):
        a = a.set_index("Year").rename(columns={"E_annual_TWh": f"E_run_{i:03d}"})
        annual_all = a if annual_all is None else annual_all.join(a, how="outer")
    annual_all = annual_all.sort_index()

    # Row-wise stats
    vals = annual_all.to_numpy()
    mean = np.nanmean(vals, axis=1)
    std  = np.nanstd(vals, axis=1, ddof=1)
    n    = np.sum(~np.isnan(vals), axis=1)
    ci95 = np.array([_ci95(m, s, int(nn)) if int(nn) > 1 else np.nan for m, s, nn in zip(mean, std, n)])

    energy_stats = pd.DataFrame({
        "Year": annual_all.index,
        "E_mean_TWh": mean,
        "E_std_TWh": std,
        "N": n,
        "E_CI95_TWh": ci95,
        "E_CI95_low_TWh": mean - ci95,
        "E_CI95_high_TWh": mean + ci95,
    }).set_index("Year")

    # Mark forecast-only years (strictly > cutoff year, but also ensure dates actually extend past cutoff)
    cutoff_year = pd.to_datetime(params["historical_cutoff"]).year
    energy_stats["is_forecast_year"] = energy_stats.index.astype(int) > cutoff_year

    energy_stats.to_csv(f"{out_dir}/{scenario_name}/annual_energy_stats.csv")
    outputs["annual_energy_stats"] = energy_stats

    # -------- Optional: quick plots (PNG) --------
    if save_plots:
        import matplotlib.pyplot as plt

        def _plot_fan(df_bands: pd.DataFrame, y_base: str, title: str, x_is_period=False, fname="plot.png"):
            x = df_bands.index.values
            p10 = df_bands[f"{y_base}_p10"].values
            p50 = df_bands[f"{y_base}_p50"].values
            p90 = df_bands[f"{y_base}_p90"].values
            plt.figure(figsize=(10,5))
            plt.fill_between(x, p10, p90, alpha=0.25, label="P10–P90")
            plt.plot(x, p50, lw=1.8, label="Median")
            plt.title(title)
            plt.xlabel("PeriodIndex" if x_is_period else "Time")
            plt.ylabel(y_base)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{out_dir}/{scenario_name}/{fname}", dpi=200)
            plt.close()

        if window in ("D","W","M"):
            bands = outputs["bands_time"]
            _plot_fan(bands, "target", "Target fan (P10–P90)", False, f"target_fan_{window}.png")
            _plot_fan(bands, "hashrate", "Hashrate fan (P10–P90)", False, f"hashrate_fan_{window}.png")
            _plot_fan(bands, "R_t", "Expected revenue per hash fan", False, f"Rt_fan_{window}.png")
            _plot_fan(bands, "hashprice_usd_th_day", "Hashprice ($/TH/s/day) fan", False, f"hashprice_fan_{window}.png")
            _plot_fan(bands, "ea_hashprice_usd_mwh", "Energy‑adjusted hashprice (USD/MWh) fan", False, f"ea_hashprice_fan_{window}.png")
        else:
            bands_p = outputs["bands_2016b"]
            _plot_fan(bands_p, "target", "Target fan by 2016‑block periods", True, "target_fan_2016b.png")
            _plot_fan(bands_p, "hashrate", "Hashrate fan by 2016‑block periods", True, "hashrate_fan_2016b.png")
            _plot_fan(bands_p, "R_t", "Expected revenue per hash fan (2016b)", True, "Rt_fan_2016b.png")
            _plot_fan(bands_p, "hashprice_usd_th_day", "Hashprice fan (2016b)", True, "hashprice_fan_2016b.png")
            _plot_fan(bands_p, "ea_hashprice_usd_mwh", "Energy‑adjusted hashprice fan (2016b)", True, "ea_hashprice_fan_2016b.png")

        # Annual energy bars with CI (forecast years highlighted)
        es = outputs["annual_energy_stats"].reset_index()
        plt.figure(figsize=(10,4))
        colors = np.where(es["is_forecast_year"], "tab:blue", "tab:gray")
        plt.bar(es["Year"].astype(int), es["E_mean_TWh"], yerr=es["E_CI95_TWh"], capsize=3, color=colors)
        plt.title("Annualized energy (TWh) — mean ± 95% CI")
        plt.xlabel("Year")
        plt.ylabel("Energy (TWh)")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{scenario_name}/annual_energy_mean_ci.png", dpi=200)
        plt.close()

        # years  = es["Year"].astype(int).to_numpy()
        # means  = es["E_mean_TWh"].to_numpy()
        # yerr   = np.where(es["is_forecast_year"], es["E_CI95_TWh"].to_numpy(), 0.0)
        # colors = np.where(es["is_forecast_year"], "tab:blue", "tab:gray")
        
        # fig, ax = plt.subplots(figsize=(12,4))
        # ax.bar(years, means, yerr=yerr, capsize=3, color=colors)
        
        # #force whole-number years on the x-axis
        # ax.xaxis.set_major_locator(mticker.FixedLocator(years))
        # ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        # ax.set_xlim(years.min() - 0.5, years.max() + 0.5)
        
        # ax.set_title("Annualized energy (TWh) — mean ± 95% CI")
        # ax.set_xlabel("Year")
        # ax.set_ylabel("Energy (TWh)")
        # fig.tight_layout()
        # fig.savefig(f"{out_dir}/{scenario_name}/annual_energy_mean_ci.png", dpi=200)
        # plt.close(fig)

    return outputs
