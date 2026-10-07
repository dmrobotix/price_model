"""
CBECI revenue check: do the daily revenue inputs explain the remaining CBECI gap?

Question
--------
The CBECI replication (validation/cbeci_replication.py) reproduces CBECI's annual
electricity estimates to a WAPE of 1.26 % (2011-2023, one-day hashrate). This script
tests whether the remaining per-year differences come from the daily revenue that
enters the profitability threshold.

The model (modules/efficiency_cbeci.compute_dynamic_efficiency, fed by
modules/simulation.simulate_step) computes, once per UTC calendar day d,

    R_d = (sum of the day's block subsidies x P_BTC + fees_BTC x P_BTC) / (H_d x 86,400)   [$/TH]
    theta_d = R_d / P_elec                                                   [J/TH]

where P_BTC is the minute price at the first block stamped on the next day, each
block is credited with its own protocol subsidy (since 2026-10-06; before that every
block of the day was priced at the subsidy of the block after the first block of the
next day, which was wrong on a day containing a halving, and on the day before a
halving when the halving block is the first block after the next day's opening block), fees_BTC are the day's block fees, and
H_d is the Core-style hashrate over the day's blocks (sum of work / sum of block
intervals). theta_d is appended to a 14-entry moving average at the first block of
the next day. CBECI (https://ccaf.io/cbnsi/cbeci/methodology, Eqs. 1-3) uses daily
issuance + fees in USD and the daily hashrate from Coin Metrics.

Stages (run in this order; each refuses to overwrite its own outputs)
--------------------------------------------------------------------
  run-model : the validation model path (cbeci_replication.main), instrumented to
              record every daily moving-average append (R_d, theta_d and its inputs)
              and every block's efficiency and 14-day threshold. About 40-50 min.
  download  : Coin Metrics community API daily BTC metrics, raw JSON saved.
  analyze   : revenue comparison, offline psi reconstruction (validated against the
              model run), and annual energy / errors under each revenue source.

Usage
-----
    cd <repo root>/validation
    PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg <python> cbeci_revenue_check.py run-model
    <python> cbeci_revenue_check.py download
    <python> cbeci_revenue_check.py analyze

Outputs go to <repo root>/data/results/cbeci_revenue_check/.
No model file is modified: the instrumentation replaces names inside the imported
modules for this process only.
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
VALIDATION_DIR = os.path.join(ROOT, "validation")
if VALIDATION_DIR not in sys.path:
    sys.path.insert(0, VALIDATION_DIR)

OUT_DIR = os.path.join(ROOT, "data", "results", "cbeci_revenue_check")
MODEL_RUN_DIR = os.path.join(OUT_DIR, "model_run")
V3_DIR = os.path.join(ROOT, "data", "results", "cbeci_replication_v3")


def _refuse_existing(paths):
    existing = [p for p in paths if os.path.exists(p)]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing output files: {existing}")


# ======================================================================
# Stage 1: instrumented model run
# ======================================================================
def run_model():
    import modules.efficiency_cbeci as ec
    import modules.simulation as sim
    import cbeci_replication as cr
    from modules.network import calc_core_hashrate

    daily_csv = os.path.join(OUT_DIR, "model_daily_revenue_appends.csv")
    block_csv = os.path.join(OUT_DIR, "model_per_block.csv.gz")
    _refuse_existing([daily_csv, block_csv, MODEL_RUN_DIR])
    os.makedirs(OUT_DIR, exist_ok=True)

    daily_rows = []
    block_tdate = []      # target_date (ns since epoch) passed to compute_dynamic_efficiency
    block_eff = []        # returned efficiency (J/hash)
    block_theta = []      # 14-day threshold moving average used for this block (J/TH)
    block_ma_len = []     # number of entries in the moving average
    orig_cde = sim.compute_dynamic_efficiency

    def recording_cde(**kw):
        prev_day = ec._last_rev_day
        out = orig_cde(**kw)
        if ec._last_rev_day != prev_day:
            dd = np.asarray(kw["daily_difficulties"], dtype=float)
            bt = np.asarray(kw["daily_block_times"], dtype=float)
            H = float(calc_core_hashrate(dd, bt, window=None)[0])
            subs = kw.get("daily_block_subsidies")
            # Subsidy revenue basis of the append: the sum of the blocks' own subsidies
            # (the model's computation), or R_block x n for a model without the argument.
            subsidy_btc = float(np.sum(subs)) if subs is not None else float(kw["R_block"]) * len(dd)
            daily_rows.append({
                "revenue_day": prev_day,                # day whose blocks were aggregated
                "append_day": ec._last_rev_day,         # day on which the entry was appended
                "target_date": pd.Timestamp(kw["target_date"]),
                "num_blocks": len(dd),
                "R_block": float(kw["R_block"]),        # subsidy of the firing block (informational)
                "subsidy_btc": subsidy_btc,             # sum of the day's block subsidies
                "P_BTC": float(kw["P_BTC"]),
                "fees_btc": float(kw["TX_FEE_btc"]),
                "sum_difficulty": float(np.nansum(dd)),
                "sum_block_times_s": float(np.nansum(bt)),
                "H_d_Hs": H,
                "revenue_per_TH": float(ec._daily_revenue_ma[-1]),
                "theta_J_per_TH": float(ec._daily_threshold_ma[-1]),
            })
        block_tdate.append(pd.Timestamp(kw["target_date"]).value)
        block_eff.append(float(out[0]))
        block_theta.append(ec._threshold_14d_ma())
        block_ma_len.append(len(ec._daily_threshold_ma))
        return out

    sim.compute_dynamic_efficiency = recording_cde

    captured = {}
    orig_run = cr.run_simulation

    def capturing_run(*a, **k):
        hist = orig_run(*a, **k)
        captured["height"] = np.array([h["block_height"] for h in hist], dtype=np.int64)
        captured["ts"] = np.array([pd.Timestamp(h["sim_timestamp"]).value for h in hist], dtype=np.int64)
        captured["eff"] = np.array([h["efficiency"] for h in hist], dtype=float)
        return hist

    cr.run_simulation = capturing_run

    t0 = time.time()
    sys.argv = ["cbeci_replication.py", "--out-dir", MODEL_RUN_DIR]
    cr.main()
    print(f"model run took {(time.time() - t0) / 60:.1f} min")

    # history[0] is the initial state (no call); history[i], i >= 1, is block i,
    # whose efficiency came from the i-th call.
    n_calls = len(block_eff)
    if n_calls != len(captured["eff"]) - 1:
        raise RuntimeError(f"{n_calls} efficiency calls vs {len(captured['eff']) - 1} blocks")
    if not np.array_equal(np.asarray(block_eff), captured["eff"][1:]):
        raise RuntimeError("recorded efficiencies differ from history")

    blocks = pd.DataFrame({
        "block_height": captured["height"][1:],
        "sim_timestamp": pd.to_datetime(captured["ts"][1:]),
        "target_date": pd.to_datetime(np.asarray(block_tdate, dtype=np.int64)),
        "efficiency_J_per_hash": captured["eff"][1:],
        "theta_ma_J_per_TH": block_theta,
        "ma_len": block_ma_len,
    })
    _refuse_existing([daily_csv, block_csv])
    pd.DataFrame(daily_rows).to_csv(daily_csv, index=False)
    blocks.to_csv(block_csv, index=False, float_format="%.17g")
    print(f"wrote {daily_csv} ({len(daily_rows)} rows) and {block_csv} ({len(blocks)} rows)")


# ======================================================================
# Stage 2: Coin Metrics community API download
# ======================================================================
CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
CM_START, CM_END = "2010-07-01", "2023-12-31"
# Requested by the task. FeeTotUSD and RevUSD are tried one at a time so that a
# metric missing from the community tier is recorded rather than failing the batch.
CM_WANTED = ["IssTotUSD", "FeeTotUSD", "RevUSD", "HashRate", "PriceUSD"]
# Native-unit metrics used to rebuild what the community tier withholds:
# FeeTotUSD is rebuilt as FeeTotNtv x PriceUSD (checked on issuance below).
CM_EXTRA = ["IssTotNtv", "FeeTotNtv", "BlkCnt"]
CM_RAW_DIR = os.path.join(OUT_DIR, "coinmetrics_raw")


def _cm_get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def download():
    avail_csv = os.path.join(OUT_DIR, "coinmetrics_availability.csv")
    daily_csv = os.path.join(OUT_DIR, "coinmetrics_daily.csv")
    _refuse_existing([CM_RAW_DIR, avail_csv, daily_csv])
    os.makedirs(CM_RAW_DIR)

    avail, frames = [], []
    for metric in CM_WANTED + CM_EXTRA:
        q = {"assets": "btc", "metrics": metric, "frequency": "1d",
             "start_time": CM_START, "end_time": CM_END, "page_size": 10000,
             "paging_from": "start"}
        url = CM_URL + "?" + urllib.parse.urlencode(q)
        rows, page = [], 0
        while url:
            status, body = _cm_get(url)
            with open(os.path.join(CM_RAW_DIR, f"{metric}_page{page}.json"), "x") as f:
                json.dump({"url": url, "http_status": status, "body": body}, f)
            if "error" in body:
                avail.append({"metric": metric, "available": False, "http_status": status,
                              "message": body["error"].get("message", "")})
                break
            rows.extend(body.get("data", []))
            url = body.get("next_page_url")
            page += 1
        else:
            avail.append({"metric": metric, "available": True, "http_status": 200,
                          "message": f"{len(rows)} rows"})
            if rows:
                s = pd.DataFrame(rows)
                s["date_utc"] = pd.to_datetime(s["time"]).dt.tz_convert(None).dt.normalize()
                frames.append(s.set_index("date_utc")[metric].astype(float))
        time.sleep(1.0)  # community tier rate limit

    pd.DataFrame(avail).to_csv(avail_csv, index=False)
    cm = pd.concat(frames, axis=1).sort_index()
    cm.to_csv(daily_csv, float_format="%.17g")
    print(pd.DataFrame(avail).to_string())
    print(cm.describe().T.to_string())


# ======================================================================
# Stage 3: analysis
# ======================================================================
C_ELEC = 50.0                       # $/MWh, CBECI Assumption 1, every year
P_PER_J = float(C_ELEC) / 3.6e9     # exactly as compute_dynamic_efficiency computes it
PUE_CBECI = 1.10
MA_DAYS = 14
ANALYSIS_FIRST_YEAR, ANALYSIS_LAST_YEAR = 2011, 2023


class FleetSelector:
    """
    Per-evaluation fleet efficiency, replicating compute_dynamic_efficiency for
    dates before the historical cutoff (the frontier machine is never included
    there): deployment window, profitable set, the previous-non-empty-set rule,
    age weights and the weighted mean. Returns J/hash, as the model does.
    The deployment mask and age weights depend only on the calendar date, because
    _months_between reads year, month and day only; they are cached per date.
    """

    def __init__(self, machines, months_between, age_weight):
        self.eta_TH = machines["eta_TH_J_per_TH"].to_numpy(copy=False)
        self.eta_hash_max = float(machines["eta_hash_J_per_hash"].max())
        self.dep_ts = machines["deployment_ts"].to_numpy()
        self._mb, self._aw = months_between, age_weight
        self._date_cache = {}
        self.prev_mask = None

    def _date_terms(self, day):
        if day not in self._date_cache:
            now_ts = pd.Timestamp(day)
            M = np.array([self._mb(pd.Timestamp(d), now_ts) for d in self.dep_ts], dtype=float)
            deployed = (M >= 0) & (M < 60)
            w_all = np.array([self._aw(m) for m in M], dtype=float)
            self._date_cache[day] = (deployed, w_all)
        return self._date_cache[day]

    def evaluate(self, day, theta_J_per_TH):
        deployed, w_all = self._date_terms(day)
        eta_TH = self.eta_TH
        mask = (eta_TH <= theta_J_per_TH) & deployed
        if not mask.any():
            if self.prev_mask is not None and self.prev_mask.any():
                mask = self.prev_mask.copy()
            else:
                return self.eta_hash_max
        else:
            self.prev_mask = mask.copy()
        weights = np.zeros(len(eta_TH), dtype=float)
        idx = np.where(mask)[0]
        weights[idx] = w_all[idx]
        valid = weights > 0
        if not np.any(valid):
            sub = eta_TH[mask]
            eta_avg = float(np.mean(sub)) if len(sub) > 0 else float(np.min(eta_TH))
        else:
            wsum = float(np.sum(weights[valid]))
            eta_avg = float(np.sum(eta_TH[valid] * weights[valid]) / wsum)
        return eta_avg / 1e12


def _ma_by_count(thetas):
    """ma[k] = 14-entry moving average after k appends, summed in deque order."""
    out = [0.0]
    for k in range(1, len(thetas) + 1):
        window = thetas[max(0, k - MA_DAYS):k]
        out.append(float(sum(window) / len(window)))
    return out


def _annual_twh(H_daily, psi_daily):
    """Annual TWh as in cbeci_replication.annual_twh (psi forward-filled)."""
    full_idx = pd.date_range(H_daily.index.min(), H_daily.index.max(), freq="D")
    H_full = H_daily.reindex(full_idx)
    psi_d = psi_daily.reindex(full_idx).ffill()
    e_twh = H_full * psi_d * PUE_CBECI * 24.0 / 1e12
    return e_twh.groupby(e_twh.index.year).sum(min_count=1)


def analyze():
    import modules.efficiency_cbeci as ec
    from config import USE_FRACTION
    from cbeci_replication import CBECI_TWH, WINDOWS

    if USE_FRACTION:
        raise RuntimeError("config.USE_FRACTION is True; the threshold here assumes it is False")

    out = {
        "daily": os.path.join(OUT_DIR, "daily_revenue_comparison.csv"),
        "yearly": os.path.join(OUT_DIR, "yearly_revenue_comparison.csv"),
        "topdays": os.path.join(OUT_DIR, "largest_deviation_days.csv"),
        "psi": os.path.join(OUT_DIR, "daily_psi_by_revenue_source.csv"),
        "annual": os.path.join(OUT_DIR, "annual_energy_by_revenue_source.csv"),
        "errors": os.path.join(OUT_DIR, "errors_by_revenue_source.csv"),
        "checks": os.path.join(OUT_DIR, "reconstruction_checks.csv"),
        "changedays": os.path.join(OUT_DIR, "psi_change_days.csv"),
        "summary": os.path.join(OUT_DIR, "summary.txt"),
    }
    _refuse_existing(out.values())
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    checks = []

    def check(name, value, note=""):
        checks.append({"check": name, "value": value, "note": note})
        say(f"  [check] {name}: {value}  {note}")

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    daily = pd.read_csv(os.path.join(OUT_DIR, "model_daily_revenue_appends.csv"),
                        parse_dates=["revenue_day", "append_day", "target_date"],
                        float_precision="round_trip")
    blocks = pd.read_csv(os.path.join(OUT_DIR, "model_per_block.csv.gz"),
                         parse_dates=["sim_timestamp", "target_date"],
                         float_precision="round_trip")
    cm = pd.read_csv(os.path.join(OUT_DIR, "coinmetrics_daily.csv"),
                     parse_dates=["date_utc"], index_col="date_utc", float_precision="round_trip")
    v3_psi = pd.read_csv(os.path.join(V3_DIR, "daily_efficiency_J_per_TH.csv"),
                         parse_dates=["date_utc"], index_col="date_utc", float_precision="round_trip")["psi_J_per_TH"]
    run_psi = pd.read_csv(os.path.join(MODEL_RUN_DIR, "daily_efficiency_J_per_TH.csv"),
                          parse_dates=["date_utc"], index_col="date_utc", float_precision="round_trip")["psi_J_per_TH"]
    v3_annual = pd.read_csv(os.path.join(V3_DIR, "annual_comparison.csv"), index_col="year",
                            float_precision="round_trip")

    say("CBECI revenue check")
    say("=" * 70)
    say("Data availability (Coin Metrics community API, daily, btc, "
        f"{CM_START} to {CM_END}):")
    av = pd.read_csv(os.path.join(OUT_DIR, "coinmetrics_availability.csv"))
    for _, r in av.iterrows():
        say(f"  {r['metric']:<10s} available={r['available']}  {r['message']}")
    say("  FeeTotUSD is rebuilt as FeeTotNtv x PriceUSD (IssTotUSD = IssTotNtv x PriceUSD is checked")
    say("  below). Coin Metrics daily rows are UTC days labelled by their start date; PriceUSD is")
    say("  the CM reference rate at the end of that UTC day; BTC HashRate is")
    say("  (BlkCnt/144) x DiffMean x (2^32/1e12)/600 TH/s, i.e. the day's work / 86,400 s.")
    say("")
    say("Section 0. Reconstruction checks")
    # The instrumented run is the validation run.
    both = pd.concat([v3_psi.rename("v3"), run_psi.rename("run")], axis=1)
    same = ((both["v3"] == both["run"]) | (both["v3"].isna() & both["run"].isna())).all()
    check("instrumented run daily psi identical to cbeci_replication_v3", bool(same),
          f"max |diff| = {np.nanmax(np.abs(both['v3'] - both['run'])):.3g} J/TH")

    # Recorded R_d and theta follow the model formula.
    H_TH = daily["H_d_Hs"] / 1e12
    if "subsidy_btc" not in daily.columns:
        # Output of a run made before 2026-10-06: every block was priced at R_block.
        daily["subsidy_btc"] = daily["R_block"] * daily["num_blocks"]
    R_formula = (daily["subsidy_btc"] * daily["P_BTC"]
                 + daily["fees_btc"] * daily["P_BTC"]) / (H_TH * 86400.0)
    ok = H_TH > 0
    check("recorded revenue_per_TH vs formula, max rel diff",
          float(np.nanmax(np.abs(daily.loc[ok, "revenue_per_TH"] / R_formula[ok] - 1))))
    check("recorded theta vs revenue_per_TH / P_per_J, max rel diff",
          float(np.nanmax(np.abs(daily.loc[ok, "theta_J_per_TH"]
                                 / (daily.loc[ok, "revenue_per_TH"] / P_PER_J) - 1))))
    # The first append (genesis block) has no previous day; every later one does.
    rd = daily["revenue_day"].iloc[1:]
    check("revenue_day missing only on the first append, strictly increasing after",
          bool(pd.isna(daily["revenue_day"].iloc[0]) and rd.notna().all()
               and rd.is_monotonic_increasing and rd.is_unique))

    # ------------------------------------------------------------------
    # Section 1. Revenue comparison per revenue day
    # ------------------------------------------------------------------
    d = daily.copy()
    d["date_utc"] = d["revenue_day"].dt.normalize()
    d = d.set_index("date_utc")
    comp = pd.DataFrame(index=d.index)
    comp["n_blocks_ours"] = d["num_blocks"]
    comp["subsidy_btc_ours"] = d["subsidy_btc"]
    comp["fees_btc_ours"] = d["fees_btc"]
    comp["price_ours"] = d["P_BTC"]
    comp["subsidy_usd_ours"] = comp["subsidy_btc_ours"] * d["P_BTC"]
    comp["fees_usd_ours"] = d["fees_btc"] * d["P_BTC"]
    comp["revenue_usd_ours"] = comp["subsidy_usd_ours"] + comp["fees_usd_ours"]
    comp["H_THs_ours"] = d["H_d_Hs"] / 1e12
    comp["R_per_TH_ours"] = d["revenue_per_TH"]
    comp["theta_ours"] = d["theta_J_per_TH"]

    cmj = cm.reindex(comp.index)
    comp["n_blocks_cm"] = cmj["BlkCnt"]
    comp["subsidy_btc_cm"] = cmj["IssTotNtv"]
    comp["fees_btc_cm"] = cmj["FeeTotNtv"]
    comp["price_cm"] = cmj["PriceUSD"]
    comp["subsidy_usd_cm"] = cmj["IssTotUSD"]
    comp["fees_usd_cm"] = cmj["FeeTotNtv"] * cmj["PriceUSD"]      # FeeTotUSD rebuilt
    comp["revenue_usd_cm"] = comp["subsidy_usd_cm"] + comp["fees_usd_cm"]
    comp["H_THs_cm"] = cmj["HashRate"]
    comp["R_per_TH_cm"] = comp["revenue_usd_cm"] / (comp["H_THs_cm"] * 86400.0)
    comp["theta_cm"] = comp["R_per_TH_cm"] / P_PER_J

    iss_chk = (cm["IssTotUSD"] / (cm["IssTotNtv"] * cm["PriceUSD"]) - 1).abs()
    check("Coin Metrics IssTotUSD vs IssTotNtv x PriceUSD, max rel diff", float(iss_chk.max()),
          "supports rebuilding FeeTotUSD as FeeTotNtv x PriceUSD")

    pairs = {
        "n_blocks": ("n_blocks_ours", "n_blocks_cm"),
        "subsidy_btc": ("subsidy_btc_ours", "subsidy_btc_cm"),
        "fees_btc": ("fees_btc_ours", "fees_btc_cm"),
        "price": ("price_ours", "price_cm"),
        "subsidy_usd": ("subsidy_usd_ours", "subsidy_usd_cm"),
        "fees_usd": ("fees_usd_ours", "fees_usd_cm"),
        "revenue_usd": ("revenue_usd_ours", "revenue_usd_cm"),
        "hashrate": ("H_THs_ours", "H_THs_cm"),
        "R_per_TH": ("R_per_TH_ours", "R_per_TH_cm"),
    }
    for k, (a, b) in pairs.items():
        comp[f"reldiff_{k}"] = comp[a] / comp[b] - 1.0
    # Log decomposition of the revenue-per-TH ratio:
    # ln(R_o/R_c) = ln(BTC_o/BTC_c) + ln(P_o/P_c) - ln(H_o/H_c)
    btc_o = comp["subsidy_btc_ours"] + comp["fees_btc_ours"]
    btc_c = comp["subsidy_btc_cm"] + comp["fees_btc_cm"]
    comp["ln_ratio_R"] = np.log(comp["R_per_TH_ours"] / comp["R_per_TH_cm"])
    comp["ln_ratio_btc"] = np.log(btc_o / btc_c)
    comp["ln_ratio_price"] = np.log(comp["price_ours"] / comp["price_cm"])
    comp["ln_ratio_hashrate"] = np.log(comp["H_THs_ours"] / comp["H_THs_cm"])
    comp["year"] = comp.index.year

    have_cm = comp[["revenue_usd_cm", "H_THs_cm"]].notna().all(axis=1) & (comp["H_THs_cm"] > 0)
    win = have_cm & comp["year"].between(ANALYSIS_FIRST_YEAR, ANALYSIS_LAST_YEAR)
    check("revenue days with Coin Metrics revenue and hashrate", int(have_cm.sum()),
          f"of {len(comp)} model revenue days; first = {comp.index[have_cm].min().date()}")
    missing_cm = comp.index[win.eq(False) & comp["year"].between(ANALYSIS_FIRST_YEAR, ANALYSIS_LAST_YEAR)]
    cal = pd.date_range(f"{ANALYSIS_FIRST_YEAR}-01-01", f"{ANALYSIS_LAST_YEAR}-12-31", freq="D")
    check("2011-2023 calendar days with no model revenue entry",
          int(len(cal.difference(comp.index))), f"{list(cal.difference(comp.index).date)[:10]}")
    check("2011-2023 model revenue days without Coin Metrics data", int(len(missing_cm)))

    say("")
    say("Section 1. Daily revenue inputs, ours / Coin Metrics - 1 (2011-2023 days)")
    w = comp.loc[win]
    say("  median relative difference over all days:")
    for k in pairs:
        say(f"    {k:<12s} {100 * w[f'reldiff_{k}'].median():+9.4f} %   "
            f"(mean {100 * w[f'reldiff_{k}'].mean():+8.4f} %, "
            f"5th/95th pct {100 * w[f'reldiff_{k}'].quantile(.05):+.3f} / "
            f"{100 * w[f'reldiff_{k}'].quantile(.95):+.3f} %)")

    yrows = []
    for y, g in w.groupby("year"):
        r = {"year": y, "days": len(g)}
        for k in pairs:
            r[f"mean_reldiff_{k}_pct"] = 100 * g[f"reldiff_{k}"].mean()
            r[f"median_reldiff_{k}_pct"] = 100 * g[f"reldiff_{k}"].median()
        for k in ("subsidy_usd", "fees_usd", "revenue_usd"):
            a, b = pairs[k]
            r[f"sum_ratio_{k}_pct"] = 100 * (g[a].sum() / g[b].sum() - 1)
        r["mean_ln_ratio_R_pct"] = 100 * g["ln_ratio_R"].mean()
        r["mean_ln_ratio_btc_pct"] = 100 * g["ln_ratio_btc"].mean()
        r["mean_ln_ratio_price_pct"] = 100 * g["ln_ratio_price"].mean()
        r["mean_ln_ratio_hashrate_pct"] = 100 * g["ln_ratio_hashrate"].mean()
        r["sd_ln_ratio_R_pct"] = 100 * g["ln_ratio_R"].std()
        yrows.append(r)
    yearly = pd.DataFrame(yrows).set_index("year")
    say("")
    say("  per-year mean relative difference (%), ours vs Coin Metrics:")
    cols = ["mean_reldiff_price_pct", "mean_reldiff_subsidy_usd_pct", "mean_reldiff_fees_usd_pct",
            "mean_reldiff_revenue_usd_pct", "mean_reldiff_hashrate_pct", "mean_reldiff_R_per_TH_pct",
            "median_reldiff_R_per_TH_pct", "sd_ln_ratio_R_pct"]
    say(yearly[cols].rename(columns=lambda c: c.replace("_pct", "").replace("mean_reldiff_", "mean_")
                            .replace("median_reldiff_", "median_")).to_string(float_format=lambda x: f"{x:+.3f}"))
    say("")
    say("  per-year mean of log-ratio components (%, ln R ratio = btc + price - hashrate):")
    say(yearly[["mean_ln_ratio_R_pct", "mean_ln_ratio_btc_pct", "mean_ln_ratio_price_pct",
                "mean_ln_ratio_hashrate_pct"]].to_string(float_format=lambda x: f"{x:+.3f}"))

    top = w.reindex(w["ln_ratio_R"].abs().sort_values(ascending=False).index).head(30)
    top_cols = ["n_blocks_ours", "n_blocks_cm", "price_ours", "price_cm", "revenue_usd_ours",
                "revenue_usd_cm", "H_THs_ours", "H_THs_cm", "R_per_TH_ours", "R_per_TH_cm",
                "reldiff_R_per_TH", "ln_ratio_btc", "ln_ratio_price", "ln_ratio_hashrate"]
    say("")
    say("  15 largest |ln(R_ours/R_cm)| days (all 30 in largest_deviation_days.csv):")
    say(top[["n_blocks_ours", "n_blocks_cm", "reldiff_price", "reldiff_revenue_usd",
             "reldiff_hashrate", "reldiff_R_per_TH"]].head(15)
        .to_string(float_format=lambda x: f"{x:+.4f}"))

    # ------------------------------------------------------------------
    # Section 2. psi replays
    # ------------------------------------------------------------------
    machines = ec._load_machines(os.path.join(ROOT, "data", "cbeci_machines_090325.csv"))

    # Theta series per revenue day under each revenue source. Where Coin Metrics
    # has no value (before 2010-07-18), the model's own theta is kept.
    def theta_series(R):
        th = (R / P_PER_J).where(have_cm, comp["theta_ours"])
        return [float(x) for x in th.to_numpy()]

    btc_ours = btc_o
    R_variants = {
        "ours": comp["R_per_TH_ours"],
        "cm": comp["R_per_TH_cm"],
        # one input swapped at a time, the rest ours
        "cm_hashrate_only": comp["revenue_usd_ours"] / (comp["H_THs_cm"] * 86400.0),
        "cm_price_only": btc_ours * comp["price_cm"] / (comp["H_THs_ours"] * 86400.0),
        "cm_btc_only": btc_c * comp["price_ours"] / (comp["H_THs_ours"] * 86400.0),
    }
    thetas = {"ours": [float(x) for x in comp["theta_ours"].to_numpy()]}
    for k, R in R_variants.items():
        if k != "ours":
            thetas[k] = theta_series(R)

    # --- 2a. per-block replay (exact model path, revenue swapped) ---
    td_day = blocks["target_date"].dt.normalize().to_numpy()
    td_int = td_day.astype("datetime64[D]").astype(np.int64)
    runmax = np.maximum.accumulate(td_int)
    inc = np.concatenate([[True], runmax[1:] > runmax[:-1]])
    kcount = np.cumsum(inc)
    say("")
    say("Section 2. Fleet efficiency psi")
    check("appends inferred from block dates == recorded appends",
          bool(kcount[-1] == len(daily)
               and np.array_equal(td_day[inc], daily["append_day"].dt.normalize().to_numpy())))
    ma_ours = _ma_by_count(thetas["ours"])
    rec_theta = blocks["theta_ma_J_per_TH"].to_numpy()
    check("per-block 14-day theta rebuilt == recorded (exact)",
          bool(np.array_equal(np.asarray(ma_ours)[kcount], rec_theta)),
          f"max abs diff {np.max(np.abs(np.asarray(ma_ours)[kcount] - rec_theta)):.3g} J/TH")

    day_keys = [pd.Timestamp(x).date() for x in td_day]

    def replay_blocks(theta_list):
        ma = _ma_by_count(theta_list)
        sel = FleetSelector(machines, ec._months_between, ec._age_weight)
        eff = np.empty(len(kcount))
        last_key, last_val = None, None
        for i in range(len(kcount)):
            key = (day_keys[i], int(kcount[i]))
            if key != last_key:
                last_val = sel.evaluate(key[0], ma[key[1]])
                last_key = key
            eff[i] = last_val
        return eff

    def daily_psi_from_blocks(eff):
        s = pd.Series(eff * 1e12, index=blocks["sim_timestamp"].to_numpy())
        p = s.resample("D").mean()
        p.index = p.index.normalize()
        return p

    t0 = time.time()
    eff_ours = replay_blocks(thetas["ours"])
    rec_eff = blocks["efficiency_J_per_hash"].to_numpy()
    n_exact = int(np.sum(eff_ours == rec_eff))
    check("per-block replay with our revenue: blocks identical to the model",
          f"{n_exact} of {len(rec_eff)}",
          f"max rel diff {np.max(np.abs(eff_ours / rec_eff - 1)):.3g}; {time.time() - t0:.0f} s")
    psi = {"model_v3": v3_psi, "replay_ours": daily_psi_from_blocks(eff_ours)}
    for k in ("cm", "cm_hashrate_only", "cm_price_only", "cm_btc_only"):
        psi[f"replay_{k}"] = daily_psi_from_blocks(replay_blocks(thetas[k]))

    # --- 2b. daily offline reconstruction (one evaluation per calendar day) ---
    append_days = daily["append_day"].dt.normalize().to_numpy()

    def replay_daily(theta_list):
        ma = _ma_by_count(theta_list)
        sel = FleetSelector(machines, ec._months_between, ec._age_weight)
        days = pd.date_range(pd.Timestamp(append_days[0]), "2023-12-31", freq="D")
        k_of_day = np.searchsorted(append_days, days.to_numpy(), side="right")
        vals = [sel.evaluate(day.date(), ma[int(k)]) * 1e12 for day, k in zip(days, k_of_day)]
        return pd.Series(vals, index=days)

    psi["daily_ours"] = replay_daily(thetas["ours"])
    psi["daily_cm"] = replay_daily(thetas["cm"])

    rng = slice(f"{ANALYSIS_FIRST_YEAR}-01-01", f"{ANALYSIS_LAST_YEAR}-12-31")
    a = psi["replay_ours"].loc[rng]
    b = v3_psi.loc[rng]
    check("per-block replay daily psi vs v3, max rel diff (2011-2023)",
          float(np.nanmax(np.abs(a / b - 1))))
    rel = (psi["daily_ours"].loc[rng] / b - 1)
    check("daily offline psi vs v3 (2011-2023): days with |rel diff| > 1e-9",
          f"{int((rel.abs() > 1e-9).sum())} of {int(rel.notna().sum())}",
          f"median |rel| {rel.abs().median():.3g}, max |rel| {rel.abs().max():.4g} "
          f"on {rel.abs().idxmax().date()}")

    # ------------------------------------------------------------------
    # Section 3. Annual energy and errors
    # ------------------------------------------------------------------
    H_cm = cm["HashRate"].dropna()
    cbeci = pd.Series(CBECI_TWH, name="CBECI_TWh")
    ann = pd.DataFrame({"CBECI_TWh": cbeci})
    ann.index.name = "year"
    ann["TWh_v3_published_H1day"] = v3_annual["TWh_H_1day"]
    for name, p in psi.items():
        ann[f"TWh_{name}"] = _annual_twh(H_cm, p).reindex(ann.index)
    names = ["TWh_v3_published_H1day"] + [f"TWh_{n}" for n in psi]
    for n in names:
        ann[n.replace("TWh_", "PctDiff_")] = 100 * (ann[n] - ann["CBECI_TWh"]) / ann["CBECI_TWh"]
    ann["daily_approx_effect_pct"] = 100 * (ann["TWh_daily_ours"] / ann["TWh_model_v3"] - 1)
    ann["cm_revenue_effect_pct_blockreplay"] = 100 * (ann["TWh_replay_cm"] / ann["TWh_replay_ours"] - 1)
    ann["cm_revenue_effect_pct_daily"] = 100 * (ann["TWh_daily_cm"] / ann["TWh_daily_ours"] - 1)
    # CBECI publishes annual TWh to 0.01 TWh; half a unit of that rounding, in percent.
    ann["cbeci_rounding_halfwidth_pct"] = 100 * 0.005 / ann["CBECI_TWh"]

    erows = []
    for n in names:
        pc = n.replace("TWh_", "PctDiff_")
        for lo, hi in WINDOWS:
            sub = ann.loc[lo:hi]
            mape = sub[pc].abs().mean()
            wape = (sub[n] - sub["CBECI_TWh"]).abs().sum() / sub["CBECI_TWh"].sum() * 100.0
            erows.append({"series": n.replace("TWh_", ""), "window": f"{lo}-{hi}",
                          "MAPE_pct": mape, "WAPE_pct": wape})
    err = pd.DataFrame(erows)

    say("")
    say("Section 3. Annual energy, H = Coin Metrics HashRate, PUE 1.10, $50/MWh")
    say("  percent difference from CBECI by year:")
    show = ["PctDiff_v3_published_H1day", "PctDiff_model_v3", "PctDiff_replay_ours",
            "PctDiff_replay_cm", "PctDiff_daily_ours", "PctDiff_daily_cm"]
    say(ann[show].rename(columns=lambda c: c.replace("PctDiff_", ""))
        .to_string(float_format=lambda x: f"{x:+.3f}"))
    say("")
    say("  effect of each swap on annual energy (% change vs our revenue, block replay):")
    eff_tab = pd.DataFrame({
        k: 100 * (ann[f"TWh_replay_{k}"] / ann["TWh_replay_ours"] - 1)
        for k in ("cm", "cm_hashrate_only", "cm_price_only", "cm_btc_only")})
    eff_tab["daily_approx_effect"] = ann["daily_approx_effect_pct"]
    say(eff_tab.to_string(float_format=lambda x: f"{x:+.3f}"))
    say("")
    say("  WAPE / MAPE (%):")
    say(err.pivot_table(index="series", columns="window", values=["WAPE_pct", "MAPE_pct"],
                        sort=False).to_string(float_format=lambda x: f"{x:.3f}"))

    # Days on which psi differs between our revenue and Coin Metrics revenue,
    # with each day's contribution to the annual-energy change.
    dpsi = (psi["replay_cm"] - psi["replay_ours"]).loc[rng]
    chg = dpsi[dpsi.abs() > 1e-12 * psi["replay_ours"].loc[rng]]
    Hc = H_cm.reindex(chg.index)
    cd = pd.DataFrame({
        "psi_ours_J_per_TH": psi["replay_ours"].reindex(chg.index),
        "psi_cm_J_per_TH": psi["replay_cm"].reindex(chg.index),
        "H_cm_THs": Hc,
        "dTWh": Hc * chg * PUE_CBECI * 24.0 / 1e12,
    })
    cd["dTWh_pct_of_CBECI_year"] = 100 * cd["dTWh"] / cbeci.reindex(cd.index.year).to_numpy()
    say("")
    say("Section 4. Days on which psi differs (CM revenue vs ours, block replay), 2013:")
    say(cd.loc["2013"].to_string(float_format=lambda x: f"{x:.4g}"))
    say("  sum of the changed days' contribution by year (% of CBECI annual TWh):")
    say(cd.groupby(cd.index.year)["dTWh_pct_of_CBECI_year"].sum()
        .to_string(float_format=lambda x: f"{x:+.3f}"))
    m13 = comp.loc[comp.index.notna() & (comp["year"] == 2013)]
    say("  2013 price, ours / Coin Metrics - 1, by month (%):")
    say((100 * m13["reldiff_price"]).groupby(m13.index.month).agg(["mean", "median", "min", "max"])
        .to_string(float_format=lambda x: f"{x:+.2f}"))
    say("  CBECI rounding half-width (0.005 TWh) as % of each year:")
    say(ann["cbeci_rounding_halfwidth_pct"].to_string(float_format=lambda x: f"{x:.2f}"))

    # psi ratio per year
    pr = (psi["replay_cm"] / psi["replay_ours"]).loc[rng]
    say("")
    say("  mean daily psi ratio, CM revenue / our revenue (block replay), by year (%):")
    say((100 * (pr.groupby(pr.index.year).mean() - 1)).to_string(float_format=lambda x: f"{x:+.3f}"))
    ndiff = (pr.sub(1).abs() > 1e-12).groupby(pr.index.year).sum()
    say("  days per year on which psi changes at all:")
    say(ndiff.to_string())

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    _refuse_existing(out.values())
    comp.to_csv(out["daily"], float_format="%.10g")
    yearly.to_csv(out["yearly"], float_format="%.6f")
    top[top_cols].to_csv(out["topdays"], float_format="%.10g")
    pd.DataFrame(psi).rename_axis("date_utc").to_csv(out["psi"], float_format="%.10g")
    ann.to_csv(out["annual"], float_format="%.6f")
    err.to_csv(out["errors"], index=False, float_format="%.6f")
    pd.DataFrame(checks).to_csv(out["checks"], index=False)
    cd.rename_axis("date_utc").to_csv(out["changedays"], float_format="%.10g")
    with open(out["summary"], "x") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nOutputs written to {OUT_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("stage", choices=["run-model", "download", "analyze"])
    args = ap.parse_args()
    if args.stage == "run-model":
        run_model()
    elif args.stage == "download":
        download()
    else:
        analyze()


if __name__ == "__main__":
    main()
