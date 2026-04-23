# validate_hashrate.py
"""
Validate miner-model hashrate forecasts against realized data.

Inputs (set in CONFIG below):
- REALIZED_HASHRATE_CSV: path to observed network hashrate CSV
- SIM_RUNS_DIR_GLOB: glob for per-run CSVs (e.g., ../data/mc/baseline/simulation_run_*.csv)
- BANDS_CSV: optional precomputed bands (e.g., ../data/mc/baseline/bands_M.csv)
- VALIDATION_START / END: time window for validation (inclusive)
- WINDOW: 'M' (monthly), 'W' (weekly), 'D' (daily)
- HISTORICAL_CUTOFF: if provided, used only to filter bands/runs (> cutoff)

Outputs in OUT_DIR:
- validation_hashrate_metrics.csv
- validation_hashrate_timeseries.csv
- validation_hashrate.png
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime

# ------------------ CONFIG ------------------
REALIZED_HASHRATE_CSV = "../data/hash-rate_03052025_30days.csv"
SIM_RUNS_DIR_GLOB     = "../data/mc/test_baseline/simulation_run_*.csv"   # used if BANDS_CSV is None
BANDS_CSV             = "../data/mc/test_baseline/bands_M.csv"            # set to None to recompute from runs
OUT_DIR               = "../data/validation"

# Validation window (hindcast period you want to score)
VALIDATION_START = "2016-01-01"
VALIDATION_END   = "2025-03-31"

# Windowing choice: 'M' (monthly), 'W' (weekly), 'D' (daily)
WINDOW = "M"

# Optional: fan bands should start strictly after this cutoff (matches your plotting convention)
HISTORICAL_CUTOFF = None  # e.g., "2015-12-31"

# Realized hashrate column parsing:
# If your file includes units in the column name (e.g., 'Hashrate (EH/s)'), the script
# tries to auto-detect the unit. Otherwise set REALIZED_COL and REALIZED_UNIT explicitly.
REALIZED_DATE_COL = None       # autodetects among ['Date','Time','Timestamp'] if None
REALIZED_COL      = None       # first non-date numeric column if None
REALIZED_UNIT     = "TH/s"     # one of {"H/s","TH/s","PH/s","EH/s"}; if None, try to parse from header

# --------------------------------------------

# --- segment scoring ---
EVENT_WINDOWS = [
    ("2016-01-01", "2021-04-30", "pre_ban"),
    ("2021-05-01", "2021-09-30", "china_ban"),
    ("2021-10-01", "2024-12-31", "post_ban"),
]

def evaluate_by_segments(ts_df: pd.DataFrame):
    rows = []
    for start, end, label in EVENT_WINDOWS:
        m = (ts_df.index >= pd.to_datetime(start)) & (ts_df.index <= pd.to_datetime(end))
        g = ts_df.loc[m]
        if g.empty:
            continue
        inside = (g["H_realized_Hs"] >= g["H_p10_Hs"]) & (g["H_realized_Hs"] <= g["H_p90_Hs"])
        rows.append({
            "segment": label,
            "start": start,
            "end": end,
            "n_points": int(len(g)),
            "coverage_p10_p90": float(np.mean(inside)),
            "mae_vs_median": float(np.mean(np.abs(g["H_realized_Hs"] - g["H_p50_Hs"]))),
            "rmse_vs_median": float(np.sqrt(np.mean((g["H_realized_Hs"] - g["H_p50_Hs"])**2))),
            "median_bandwidth": float(np.median(g["H_p90_Hs"] - g["H_p10_Hs"])),
        })
    return pd.DataFrame(rows)

os.makedirs(OUT_DIR, exist_ok=True)

UNIT_FACTORS = {"H/s": 1.0, "TH/s": 1e12, "PH/s": 1e15, "EH/s": 1e18}

def _guess_date_col(df: pd.DataFrame):
    for c in ["Timestamp","Time","Date","date","time","timestamp"]:
        if c in df.columns:
            return c
    # fallback: first datetime-like column
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            pass
    raise ValueError("Could not find a date/time column in realized CSV.")

def _guess_value_col(df: pd.DataFrame, date_col: str):
    num_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        # try non-strict: first non-date column
        others = [c for c in df.columns if c != date_col]
        if not others:
            raise ValueError("No value column found in realized CSV.")
        return others[0]
    return num_cols[0]

def _infer_unit_from_name(colname: str):
    s = colname.lower()
    if "eh" in s: return "EH/s"
    if "ph" in s: return "PH/s"
    if "th" in s: return "TH/s"
    if "h/s" in s or "hash/s" in s or "hashrate" in s: return "H/s"
    # some files use "EH/s (7-day SMA)" etc.
    for u in UNIT_FACTORS:
        if u.lower().replace("/","") in s.replace("/",""):
            return u
    return None

def load_realized_series(path: str) -> pd.Series:
    df = pd.read_csv(path)
    dcol = REALIZED_DATE_COL or _guess_date_col(df)
    vcol = REALIZED_COL or _guess_value_col(df, dcol)

    ser = df[[dcol, vcol]].copy()
    ser[dcol] = pd.to_datetime(ser[dcol])
    ser = ser.dropna().sort_values(dcol)
    unit = REALIZED_UNIT or _infer_unit_from_name(vcol) or "EH/s"  # default to EH/s (common)
    factor = UNIT_FACTORS[unit]
    y = pd.Series(ser[vcol].astype(float).to_numpy() * factor, index=ser[dcol].to_numpy(), name="H_realized_Hs")
    # Resample to requested window using mean (consistent with your bands)
    if WINDOW in ("D","W","M"):
        y = y.resample(WINDOW).mean()
    else:
        raise ValueError("WINDOW must be one of 'D','W','M'.")
    return y

def load_bands_csv(path: str) -> pd.DataFrame:
    bands = pd.read_csv(path, parse_dates=[0])
    # Expect a datetime index or a column named Timestamp
    if "Timestamp" in bands.columns:
        bands = bands.set_index("Timestamp")
    else:
        bands = bands.set_index(bands.columns[0])
        bands.index.name = "Timestamp"
    # Optional cutoff filter
    if HISTORICAL_CUTOFF:
        bands = bands[bands.index > pd.to_datetime(HISTORICAL_CUTOFF)]
    # Require hashrate columns
    required = {"hashrate_p10","hashrate_p50","hashrate_p90"}
    missing = required - set(bands.columns)
    if missing:
        raise ValueError(f"bands CSV missing columns: {missing}")
    # Already windowed by your MC code; still, align to WINDOW if different
    if WINDOW in ("D","W","M"):
        bands = bands.resample(WINDOW).mean()
    return bands

def _calc_core_hashrate_from_D_Tblock(d: np.ndarray, tblock: np.ndarray, window: int = 2016) -> np.ndarray:
    """Replicates Core-style estimator: H = sum(D_i * 2**32) / sum(T_i) over a rolling window."""
    work = d.astype(float) * (2.0**32)
    if window is None or window <= 1 or window > len(work):
        total_work = np.nansum(work)
        total_time = np.nansum(tblock.astype(float))
        return np.full_like(work, fill_value=(total_work/total_time), dtype=float)
    # cumulative rolling sums via convolution
    kernel = np.ones(window, dtype=float)
    cwork = np.convolve(work, kernel, mode="full")[:len(work)]
    ctime = np.convolve(tblock.astype(float), kernel, mode="full")[:len(work)]
    H = cwork / ctime
    H[:window-1] = np.nan  # not enough history
    return H

def build_bands_from_runs(run_glob: str) -> pd.DataFrame:
    """Compute monthly/weekly/daily P10/P50/P90 for hashrate from per-run CSVs."""
    paths = sorted(glob.glob(run_glob))
    if not paths:
        raise FileNotFoundError(f"No simulation runs found for glob: {run_glob}")

    resampled = []
    for p in paths:
        df = pd.read_csv(p, parse_dates=["Timestamp"])
        if HISTORICAL_CUTOFF:
            df = df[df["Timestamp"] > pd.to_datetime(HISTORICAL_CUTOFF)]

        # prefer 'H' or 'hashrate'; fallback to compute from D + T_block
        if "H" in df.columns:
            Hs = pd.Series(df["H"].astype(float).to_numpy(), index=df["Timestamp"].to_numpy(), name="H")
        elif "hashrate" in df.columns:
            Hs = pd.Series(df["hashrate"].astype(float).to_numpy(), index=df["Timestamp"].to_numpy(), name="H")
        elif {"D","T_block"}.issubset(df.columns):
            H_np = _calc_core_hashrate_from_D_Tblock(df["D"].to_numpy(), df["T_block"].to_numpy(), window=2016)
            Hs = pd.Series(H_np, index=df["Timestamp"].to_numpy(), name="H")
        else:
            raise ValueError(f"{os.path.basename(p)} missing hashrate; expected 'H' or ('D' & 'T_block').")

        # resample to chosen window (mean)
        Hs = Hs.resample(WINDOW).mean()
        resampled.append(Hs)

    # Align to common index (intersection to avoid NaNs)
    common_idx = resampled[0].index
    for s in resampled[1:]:
        common_idx = common_idx.intersection(s.index)
    aligned = [s.loc[common_idx] for s in resampled]
    data = np.stack([s.to_numpy() for s in aligned], axis=1)  # shape (T, N)

    p10 = np.nanpercentile(data, 10, axis=1)
    p50 = np.nanpercentile(data, 50, axis=1)
    p90 = np.nanpercentile(data, 90, axis=1)

    bands = pd.DataFrame({
        "hashrate_p10": p10,
        "hashrate_p50": p50,
        "hashrate_p90": p90,
    }, index=common_idx)
    bands.index.name = "Timestamp"
    return bands

def evaluate_validation(realized: pd.Series, bands: pd.DataFrame, start: str, end: str):
    """Compute coverage, MAE, RMSE, and sharpness on the selected window."""
    start_ts = pd.to_datetime(start) if start else bands.index.min()
    end_ts   = pd.to_datetime(end)   if end   else bands.index.max()

    # align both to the intersection and slice to window
    idx = realized.index.intersection(bands.index)
    realized = realized.loc[idx]
    bands    = bands.loc[idx]
    mask = (realized.index >= start_ts) & (realized.index <= end_ts)
    r = realized.loc[mask]
    b = bands.loc[mask]

    # metrics
    inside = (r.values >= b["hashrate_p10"].values) & (r.values <= b["hashrate_p90"].values)
    coverage = float(np.mean(inside)) if len(inside) else np.nan
    mae = float(np.mean(np.abs(r.values - b["hashrate_p50"].values))) if len(r) else np.nan
    rmse = float(np.sqrt(np.mean((r.values - b["hashrate_p50"].values)**2))) if len(r) else np.nan
    sharpness = float(np.median((b["hashrate_p90"] - b["hashrate_p10"]).values)) if len(b) else np.nan

    metrics = {
        "window": WINDOW,
        "n_points": int(len(r)),
        "coverage_p10_p90": coverage,   # should be ≈ 0.80 if bands are well calibrated
        "mae_vs_median": mae,
        "rmse_vs_median": rmse,
        "median_bandwidth": sharpness,
    }
    # row-wise table for export
    ts = pd.DataFrame({
        "H_realized_Hs": r.values,
        "H_p10_Hs": b["hashrate_p10"].values,
        "H_p50_Hs": b["hashrate_p50"].values,
        "H_p90_Hs": b["hashrate_p90"].values,
    }, index=r.index)
    ts.index.name = "Timestamp"
    return metrics, ts

def plot_validation(ts: pd.DataFrame, out_path: str, title="Hashrate validation ({} window)"):
    title = title.format(WINDOW)
    years = pd.Index(ts.index).year.values
    x = ts.index

    fig, ax = plt.subplots(figsize=(12,4))
    ax.fill_between(x, ts["H_p10_Hs"], ts["H_p90_Hs"], alpha=0.25, label="P10–P90")
    ax.plot(x, ts["H_p50_Hs"], lw=1.8, label="Median (P50)")
    ax.plot(x, ts["H_realized_Hs"], lw=1.5, label="Realized")

    # In plot_validation(...), after drawing the lines, shade the ban window:
    for start, end, label in EVENT_WINDOWS:
        if label == "china_ban":
            ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color="gray", alpha=0.15, label="China ban")
            break
    # avoid duplicate legend entry
    handles, labels = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            h2.append(h); l2.append(l); seen.add(l)
    ax.legend(h2, l2)

    # nicer x-axis ticks for monthly/weekly
    if WINDOW == "M":
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
    ax.set_title(title)
    ax.set_ylabel("Hashrate (H/s)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


realized = load_realized_series(REALIZED_HASHRATE_CSV)

if BANDS_CSV and os.path.exists(BANDS_CSV):
    bands = load_bands_csv(BANDS_CSV)
else:
    bands = build_bands_from_runs(SIM_RUNS_DIR_GLOB)

# Evaluate
metrics, ts = evaluate_validation(realized, bands, VALIDATION_START, VALIDATION_END)

# After you compute `metrics, ts = evaluate_validation(...)` add:
by_seg = evaluate_by_segments(ts)
by_seg.to_csv(os.path.join(OUT_DIR, "validation_hashrate_by_segment.csv"), index=False)
print(by_seg)

# Save artifacts
ts.to_csv(os.path.join(OUT_DIR, "validation_hashrate_timeseries.csv"))
pd.DataFrame([metrics]).to_csv(os.path.join(OUT_DIR, "validation_hashrate_metrics.csv"), index=False)

# Plot
plot_validation(ts, os.path.join(OUT_DIR, "validation_hashrate.png"))

# Print short summary
print("Validation window:", VALIDATION_START, "→", VALIDATION_END)
print("Points:", metrics["n_points"])
print("Coverage P10–P90:", round(metrics["coverage_p10_p90"], 3))
print("MAE vs median (H/s):", f"{metrics['mae_vs_median']:.3e}")
print("RMSE vs median (H/s):", f"{metrics['rmse_vs_median']:.3e}")
print("Median band width (H/s):", f"{metrics['median_bandwidth']:.3e}")

# ==== Residual-based interval calibration (drop-in for validate_hashrate.py) ====
# Requires: ts (DataFrame from evaluate_validation) already built.
# Columns in ts: H_realized_Hs, H_p10_Hs, H_p50_Hs, H_p90_Hs with DatetimeIndex.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

CALIB_FIT_START = "2016-01-01"   # fit on a stable regime (pre-ban)
CALIB_FIT_END   = "2021-04-30"
CALIB_APPLY_START = "2021-10-01" # evaluate on post-ban (hold-out)
CALIB_APPLY_END   = "2024-12-31"

def _segment(df, start, end):
    m = (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))
    return df.loc[m].copy()

def _metrics(r, b10, b50, b90):
    inside = (r >= b10) & (r <= b90)
    return {
        "n_points": int(len(r)),
        "coverage_p10_p90": float(np.mean(inside)) if len(r) else np.nan,
        "mae_vs_median": float(np.mean(np.abs(r - b50))) if len(r) else np.nan,
        "rmse_vs_median": float(np.sqrt(np.mean((r - b50)**2))) if len(r) else np.nan,
        "median_bandwidth": float(np.median(b90 - b10)) if len(r) else np.nan,
    }

def _calibrate_additive(ts_fit):
    # residuals in H/s
    e = (ts_fit["H_realized_Hs"] - ts_fit["H_p50_Hs"]).to_numpy()
    q10, q90 = np.nanpercentile(e, [10, 90])
    return {"add_q10": q10, "add_q90": q90}

def _apply_additive(ts_all, add_params):
    q10, q90 = add_params["add_q10"], add_params["add_q90"]
    b50 = ts_all["H_p50_Hs"].to_numpy()
    b10 = b50 + q10
    b90 = b50 + q90
    return b10, b50, b90

def _calibrate_multiplicative(ts_fit, eps=1e-12):
    # ratios; guard against division by ~0 early in history
    ratio = (ts_fit["H_realized_Hs"] / (ts_fit["H_p50_Hs"].abs() + eps)).to_numpy()
    q10, q90 = np.nanpercentile(ratio, [10, 90])
    return {"mul_q10": q10, "mul_q90": q90}

def _apply_multiplicative(ts_all, mul_params):
    q10, q90 = mul_params["mul_q10"], mul_params["mul_q90"]
    b50 = ts_all["H_p50_Hs"].to_numpy()
    b10 = b50 * q10
    b90 = b50 * q90
    return b10, b50, b90

def _plot_compare(x, r, b10_a, b50_a, b90_a, b10_m, b50_m, b90_m, out_png, window_label):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.fill_between(x, b10_a, b90_a, alpha=0.18, label="Additive P10–P90")
    ax.plot(x, b50_a, lw=1.5, label="Additive P50")
    ax.fill_between(x, b10_m, b90_m, alpha=0.18, label="Multiplicative P10–P90")
    ax.plot(x, b50_m, lw=1.5, label="Multiplicative P50")
    ax.plot(x, r, lw=1.8, label="Realized")
    ax.set_title(f"Hashrate validation — recalibrated bands ({window_label})")
    ax.set_ylabel("Hashrate (H/s)")
    ax.grid(True, alpha=0.3)
    # keep reasonable tick count
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

# --- Fit on pre-ban, evaluate on post-ban ---
ts_fit   = _segment(ts, CALIB_FIT_START, CALIB_FIT_END)
ts_apply = _segment(ts, CALIB_APPLY_START, CALIB_APPLY_END)

# Fit
add_params = _calibrate_additive(ts_fit)
mul_params = _calibrate_multiplicative(ts_fit)

# Apply
b10_a, b50_a, b90_a = _apply_additive(ts_apply, add_params)
b10_m, b50_m, b90_m = _apply_multiplicative(ts_apply, mul_params)
r = ts_apply["H_realized_Hs"].to_numpy()
x = ts_apply.index.to_numpy()

# Metrics
metrics_add = _metrics(r, b10_a, b50_a, b90_a)
metrics_mul = _metrics(r, b10_m, b50_m, b90_m)

# Save metrics & series
out_dir = OUT_DIR
os.makedirs(out_dir, exist_ok=True)
pd.DataFrame([{
    "calib_fit": f"{CALIB_FIT_START}→{CALIB_FIT_END}",
    "eval_apply": f"{CALIB_APPLY_START}→{CALIB_APPLY_END}",
    "scheme": "additive", **metrics_add
}]).to_csv(os.path.join(out_dir, "recalib_hashrate_metrics_additive.csv"), index=False)
pd.DataFrame([{
    "calib_fit": f"{CALIB_FIT_START}→{CALIB_FIT_END}",
    "eval_apply": f"{CALIB_APPLY_START}→{CALIB_APPLY_END}",
    "scheme": "multiplicative", **metrics_mul
}]).to_csv(os.path.join(out_dir, "recalib_hashrate_metrics_multiplicative.csv"), index=False)

ts_out = pd.DataFrame({
    "H_realized_Hs": r,
    "H_p10_add": b10_a, "H_p50_add": b50_a, "H_p90_add": b90_a,
    "H_p10_mul": b10_m, "H_p50_mul": b50_m, "H_p90_mul": b90_m,
}, index=ts_apply.index)
ts_out.index.name = "Timestamp"
ts_out.to_csv(os.path.join(out_dir, "recalib_hashrate_timeseries.csv"))

# Plot
_plot_compare(x, r, b10_a, b50_a, b90_a, b10_m, b50_m, b90_m,
              os.path.join(out_dir, "recalib_hashrate_compare.png"),
              window_label="post-ban")
print("Recalibration done. See files in", out_dir)
print("Additive metrics:", metrics_add)
print("Multiplicative metrics:", metrics_mul)
