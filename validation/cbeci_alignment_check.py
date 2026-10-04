"""Alignment check: 14-day threshold window lag under Coin Metrics revenue (daily offline).

Variants for the threshold on day d: A = mean of revenue days d-14..d-1 (the model), B = d-13..d,
C = d-15..d-2. Reads the outputs of cbeci_revenue_check.py (run-model, download, analyze) and
writes data/results/cbeci_revenue_check/alignment/. Uses one evaluation per calendar day, so its
WAPE differs from the per-block replay of cbeci_revenue_check.py by about 0.001 pp.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "validation"))
import cbeci_revenue_check as crc
import modules.efficiency_cbeci as ec
from cbeci_replication import CBECI_TWH, WINDOWS

OUT = os.path.join(crc.OUT_DIR, "alignment")
if os.path.exists(OUT):
    raise FileExistsError(OUT)
D = crc.OUT_DIR
rt = dict(float_precision="round_trip")
daily = pd.read_csv(os.path.join(D, "model_daily_revenue_appends.csv"), parse_dates=["revenue_day"], **rt)
comp = pd.read_csv(os.path.join(D, "daily_revenue_comparison.csv"), parse_dates=["date_utc"], **rt)
cm = pd.read_csv(os.path.join(D, "coinmetrics_daily.csv"), parse_dates=["date_utc"], index_col="date_utc", **rt)
prev = pd.read_csv(os.path.join(D, "daily_psi_by_revenue_source.csv"), parse_dates=["date_utc"],
                   index_col="date_utc", **rt)["daily_cm"]
# comp rows are in the same order as daily rows (row 0 = genesis, no revenue day)
assert len(comp) == len(daily) and comp["date_utc"].iloc[1:].equals(daily["revenue_day"].iloc[1:].rename("date_utc"))
have = comp[["revenue_usd_cm", "H_THs_cm"]].notna().all(axis=1) & (comp["H_THs_cm"] > 0)
theta = (comp["R_per_TH_cm"] / crc.P_PER_J).where(have, comp["theta_ours"])
thetas = [float(x) for x in theta.to_numpy()]
ma = crc._ma_by_count(thetas)
machines = ec._load_machines(os.path.join(ROOT, "data", "cbeci_machines_090325.csv"))
# revenue_day per entry; genesis entry counted from the start
rday = daily["revenue_day"].fillna(pd.Timestamp("1900-01-01")).dt.normalize().to_numpy()

def psi_lag(L):
    """psi on day d from the 14 entries with revenue_day <= d - L."""
    sel = crc.FleetSelector(machines, ec._months_between, ec._age_weight)
    days = pd.date_range(daily["append_day"].iloc[0] if "append_day" in daily else "2009-01-03",
                         "2023-12-31", freq="D")
    k = np.searchsorted(rday, (days - pd.Timedelta(days=L)).to_numpy(), side="right")
    vals = [sel.evaluate(d.date(), ma[int(kk)]) * 1e12 if kk > 0 else np.nan for d, kk in zip(days, k)]
    return pd.Series(vals, index=days)

daily["append_day"] = pd.to_datetime(pd.read_csv(os.path.join(D, "model_daily_revenue_appends.csv"))["append_day"])
H = cm["HashRate"].dropna()
conv = {"A_lag1_ours": 1, "B_sameday": 0, "C_lag2": 2}
psi = {k: psi_lag(L) for k, L in conv.items()}
rng = slice("2011-01-01", "2023-12-31")
chk = (psi["A_lag1_ours"].loc[rng] / prev.loc[rng] - 1).abs().max()
print("check A vs earlier daily_cm psi, max rel diff 2011-2023:", chk)
ann = pd.DataFrame({"CBECI_TWh": pd.Series(CBECI_TWH)}); ann.index.name = "year"
for k, p in psi.items():
    ann[f"TWh_{k}"] = crc._annual_twh(H, p).reindex(ann.index)
    ann[f"PctDiff_{k}"] = 100 * (ann[f"TWh_{k}"] - ann["CBECI_TWh"]) / ann["CBECI_TWh"]
rows = []
for k in psi:
    for lo, hi in WINDOWS:
        s = ann.loc[lo:hi]
        rows.append({"convention": k, "window": f"{lo}-{hi}", "MAPE_pct": s[f"PctDiff_{k}"].abs().mean(),
                     "WAPE_pct": (s[f"TWh_{k}"] - s["CBECI_TWh"]).abs().sum() / s["CBECI_TWh"].sum() * 100})
err = pd.DataFrame(rows)
os.makedirs(OUT)
ann.to_csv(os.path.join(OUT, "annual_energy_by_alignment.csv"), float_format="%.6f")
err.to_csv(os.path.join(OUT, "errors_by_alignment.csv"), index=False, float_format="%.6f")
pd.DataFrame(psi).rename_axis("date_utc").to_csv(os.path.join(OUT, "daily_psi_by_alignment.csv"), float_format="%.10g")
pd.DataFrame([{"check": "A reproduces daily_cm psi, max rel diff 2011-2023", "value": chk}]).to_csv(
    os.path.join(OUT, "checks.csv"), index=False)
print(ann[[c for c in ann if c.startswith("PctDiff")]].to_string(float_format=lambda x: f"{x:+.3f}"))
print(err.pivot_table(index="convention", columns="window", values=["WAPE_pct", "MAPE_pct"]).to_string(float_format=lambda x: f"{x:.3f}"))
