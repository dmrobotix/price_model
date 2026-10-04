"""
CBECI machine-list check: does our machine table explain the remaining CBECI gap?

Question
--------
The CBECI replication with Coin Metrics revenue and hashrate
(validation/cbeci_revenue_check.py, series "replay_cm") differs from CBECI's
published annual electricity estimates by +3.4 % in 2013, -2.5 % / -2.9 % in
2015-16 and +2.4 % / +2.2 % in 2018-19. This script tests whether the machine
table explains those residuals. Our table is data/cbeci_machines_090325.csv.
CBECI publishes its list at http://sha256.cbeci.org.

Stages (run in this order; each refuses to overwrite its own outputs)
--------------------------------------------------------------------
  fetch   : request http://sha256.cbeci.org (records the redirect), then the
            Google Sheet it points to, exported as .xlsx (all tabs, with cached
            formula values) and as .csv (the linked tab). Also queries the Wayback
            Machine CDX index for archived copies of the sheet. Raw responses are
            saved under raw/.
  analyze : parse CBECI's list, match it to ours by name, compare the shared
            fields, build alternative machine tables, and replay the daily fleet
            efficiency with Coin Metrics revenue and hashrate for each table.

Machine-table variants replayed (all loaded by modules.efficiency_cbeci._load_machines,
so the release-date handling and the CBECI manufacturer filter are the model's own):
  ours            : data/cbeci_machines_090325.csv, unchanged. Must reproduce the
                    revenue check's replay_cm / daily_cm series.
  ours_cbeci_eff  : ours, with Efficiency (J/Gh) replaced by CBECI's value for every
                    matched machine (CBECI's value = the sheet's cached value of its efficiency
                    formula cell, at full precision; it equals Power / Hashing power / 1000
                    to within 4.4e-10).
  cbeci_list      : CBECI's current list, every row, full-precision efficiency.
  cbeci_list_v1_1 : additional diagnostic. CBECI's list restricted to rows whose
                    "Included in version" is 1.0.x or 1.1.x, i.e. without the hardware
                    added in versions 1.2.0 and later.

Revenue and hashrate inputs are the revenue check's saved outputs
(data/results/cbeci_revenue_check/): the instrumented model run's per-block target
dates and daily appends, and the Coin Metrics daily metrics. The fleet selection
is cbeci_revenue_check.FleetSelector, imported unchanged.

Usage
-----
    cd <repo root>/validation
    PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg <python> cbeci_machine_list_check.py fetch
    PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg <python> cbeci_machine_list_check.py analyze

Outputs go to <repo root>/data/results/cbeci_machine_list_check/.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
VALIDATION_DIR = os.path.join(ROOT, "validation")
if VALIDATION_DIR not in sys.path:
    sys.path.insert(0, VALIDATION_DIR)

OUT_DIR = os.path.join(ROOT, "data", "results", "cbeci_machine_list_check")
RAW_DIR = os.path.join(OUT_DIR, "raw")
TABLE_DIR = os.path.join(OUT_DIR, "machine_tables")
REV_DIR = os.path.join(ROOT, "data", "results", "cbeci_revenue_check")
OUR_TABLE = os.path.join(ROOT, "data", "cbeci_machines_090325.csv")

CBECI_URL = "http://sha256.cbeci.org"
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
LIST_TAB = "list"   # the SHA-256 tab; the other tabs are ETH, ETC, DOGELTC

# Thresholds for reporting a matched machine as different (set by the task).
EFF_REL_TOL = 0.01      # 1 %
DATE_TOL_DAYS = 15

OUR_COLUMNS = ["Miner_name", "Type", "Date of release", "UNIX_date_of_release",
               "Hashing Power (TH/s)", "Power (W)", "Efficiency (J/Gh)", "Weight in kg"]


def _refuse_existing(paths):
    existing = [p for p in paths if os.path.exists(p)]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing output files: {existing}")


def _sha256(b):
    return hashlib.sha256(b).hexdigest()


# ======================================================================
# Stage 1: fetch
# ======================================================================
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _get(url, timeout=60, follow=True):
    """GET url; returns (status, headers dict, body bytes, final url, error text)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    opener = (urllib.request.build_opener() if follow
              else urllib.request.build_opener(_NoRedirect))
    try:
        with opener.open(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read(), r.geturl(), ""
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read(), url, ""
    except Exception as e:  # network failure: recorded, not raised
        return None, {}, b"", url, f"{type(e).__name__}: {e}"


def fetch():
    _refuse_existing([OUT_DIR])
    os.makedirs(RAW_DIR)
    log = []

    def save(name, url, status, headers, body, final_url, err):
        path = os.path.join(RAW_DIR, name)
        with open(path, "xb") as f:
            f.write(body)
        with open(path + ".headers.json", "x") as f:
            json.dump({"url": url, "final_url": final_url, "http_status": status,
                       "headers": headers, "error": err}, f, indent=1)
        rec = {"file": name, "url": url, "final_url": final_url, "http_status": status,
               "content_type": headers.get("Content-Type", headers.get("content-type", "")),
               "bytes": len(body), "sha256": _sha256(body), "error": err,
               "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        log.append(rec)
        print(f"  {name}: status={status} bytes={len(body)} {err}")
        return rec

    # 1. The CBECI address, without following the redirect.
    st, hd, body, fu, err = _get(CBECI_URL, follow=False)
    save("sha256.cbeci.org_response.html", CBECI_URL, st, hd, body, fu, err)
    location = hd.get("Location") or hd.get("location")
    if not location:
        raise RuntimeError(f"{CBECI_URL} did not redirect (status {st}, error {err!r}); "
                           "see raw/ for the response")
    m = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", location)
    if not m:
        raise RuntimeError(f"redirect target is not a Google Sheet: {location}")
    sheet_id = m.group(1)
    gid = (re.search(r"gid=(\d+)", location) or [None, None])[1]
    print(f"  redirect -> {location}")

    # 2. The page it redirects to (Google Sheets app HTML), saved as received.
    st, hd, body, fu, err = _get(location)
    save("redirect_target.html", location, st, hd, body, fu, err)

    # 3. The sheet's exports.
    base = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export"
    st, hd, body, fu, err = _get(base + "?format=xlsx")
    rec = save("sheet.xlsx", base + "?format=xlsx", st, hd, body, fu, err)
    if st != 200 or not body.startswith(b"PK"):
        raise RuntimeError(f"xlsx export failed: {rec}")
    if gid:
        u = base + f"?format=csv&gid={gid}"
        st, hd, body, fu, err = _get(u)
        save(f"sheet_gid{gid}.csv", u, st, hd, body, fu, err)

    # 4. Wayback Machine CDX index for the sheet (non-fatal).
    for key, target in (("sheet", f"docs.google.com/spreadsheets/d/{sheet_id}*"),
                        ("cbeci", "sha256.cbeci.org*")):
        u = ("https://web.archive.org/cdx/search/cdx?" + urllib.parse.urlencode(
            {"url": target, "output": "txt", "fl": "timestamp,original,statuscode,mimetype"}))
        st, hd, body, fu, err = _get(u, timeout=90)
        save(f"wayback_cdx_{key}.txt", u, st, hd, body, fu, err)

    with open(os.path.join(RAW_DIR, "fetch_log.json"), "x") as f:
        json.dump({"sheet_id": sheet_id, "gid": gid, "redirect": location,
                   "responses": log}, f, indent=1)
    print(f"raw responses written to {RAW_DIR}")


# ======================================================================
# Stage 2: analysis helpers
# ======================================================================
MAKERS = {"bitmain", "microbt", "canaan", "ebang", "innosilicon", "bitfury", "bitfily",
          "pantech", "halong", "asicminer", "gmo", "holic", "strongu", "aladdin", "aisen",
          "intel", "nvidia", "amd", "xilinx"}
PRODUCT_LINES = {"antminer", "whatsminer", "avalonminer"}


def norm_name(name):
    """
    Normalised name used for the second matching tier:
      lower case; surrounding whitespace removed;
      "Hyd." / "Hyd" -> "hydro";  "Imm." / "Imm" -> "immersion";
      hashrate tags "(14Th)", "16Th/s", "52T" -> "14th", "16th", "52th";
      parentheses removed, whitespace collapsed;
      a leading manufacturer token (MAKERS) dropped;
      the product-line words antminer / whatsminer / avalonminer dropped.
    """
    s = str(name).strip().lower()
    s = re.sub(r"\bhyd\.?(?=\s|$)", "hydro", s)
    s = re.sub(r"\bimm\.?(?=\s|$)", "immersion", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*th(?:/s)?\b", r"\1th", s)
    s = re.sub(r"(\d+(?:\.\d+)?)t\b", r"\1th", s)
    s = re.sub(r"[()]", " ", s)
    tok = s.split()
    if tok and tok[0] in MAKERS:
        tok = tok[1:]
    tok = [t for t in tok if t not in PRODUCT_LINES]
    return " ".join(tok)


def _num(x):
    """Number from a cell that may be a string with thousands separators."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    t = str(x).strip().replace(",", "")
    if t == "":
        return np.nan
    return float(t)


def parse_cbeci_xlsx(path):
    """CBECI's 'list' tab: one row per machine, with cached formula values."""
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[LIST_TAB]
    rows = list(ws.iter_rows(values_only=True))
    meta = {"title": rows[0][0], "last_updated_text": rows[1][0], "note": rows[2][0]}
    hdr_i = next(i for i, r in enumerate(rows) if r and r[0] == "Miner_name")
    hdr = [("" if h is None else str(h)) for h in rows[hdr_i]]

    def col(prefix):
        hits = [i for i, h in enumerate(hdr) if re.sub(r"\s+", " ", h).startswith(prefix)]
        if len(hits) != 1:
            raise ValueError(f"header {prefix!r}: {len(hits)} matches in {hdr}")
        return hits[0]

    c = {"name": col("Miner_name"), "type": col("Type"), "date": col("Date of release"),
         "unix": col("UNIX_date_of_release"), "hash": col("Hashing power"),
         "power": col("Power (W)"), "eff": col("Efficiency (J/Gh)"), "weight": col("Weight in kg"),
         "eff_alt": col("Efficiency: suggested"), "version": col("Included in"),
         "comment": col("Additional comments")}
    meta["header"] = hdr
    recs = []
    for i, r in enumerate(rows[hdr_i + 1:], start=hdr_i + 2):
        if r is None or r[c["name"]] is None or str(r[c["name"]]).strip() == "":
            if r is not None and any(v is not None for v in r):
                raise ValueError(f"xlsx row {i}: values without a Miner_name: {r}")
            continue
        d = r[c["date"]]
        if isinstance(d, datetime):
            rel = pd.Timestamp(d)
        else:
            rel = pd.to_datetime(str(d).strip(), format="%d/%m/%Y")
        recs.append({
            "xlsx_row": i,
            "Miner_name": str(r[c["name"]]),
            "Type": r[c["type"]],
            "release_date": rel,
            "UNIX_date_of_release": _num(r[c["unix"]]),
            "hashrate_THs": _num(r[c["hash"]]),
            "power_W": _num(r[c["power"]]),
            "eff_J_per_Gh": _num(r[c["eff"]]),
            "weight_kg": _num(r[c["weight"]]),
            "eff_suggested_alternative": r[c["eff_alt"]],
            "included_in_version": None if r[c["version"]] is None else str(r[c["version"]]).strip(),
            "comment": r[c["comment"]],
        })
    return pd.DataFrame(recs), meta


def parse_ours(path):
    """Our table, with the release date parsed as _load_machines parses it."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    unix = raw["UNIX_date_of_release"].map(_num)
    dtxt = raw["Date of release"].str.strip()
    dmy = pd.to_datetime(dtxt.mask(dtxt == ""), format="%d/%m/%Y", errors="raise")
    rel = pd.to_datetime(unix, unit="s").fillna(dmy)
    return pd.DataFrame({
        "row": np.arange(len(raw)),
        "Miner_name": raw["Miner_name"],
        "Type": raw["Type"],
        "release_date": rel,
        "UNIX_date_of_release": unix,
        "hashrate_THs": raw["Hashing Power (TH/s)"].map(_num),
        "power_W": raw["Power (W)"].map(_num),
        "eff_J_per_Gh": raw["Efficiency (J/Gh)"].map(_num),
        "weight_kg": raw["Weight in kg"].map(_num),
    }), raw


def passes_filter(name, release):
    """CBECI manufacturer filter as modules.efficiency_cbeci applies it."""
    import modules.efficiency_cbeci as ec
    maker = str(name).strip().split()[0] if str(name).strip() else ""
    return not (release >= ec.CBECI_MANUFACTURER_FILTER_FROM and maker not in ec.CBECI_MANUFACTURERS)


def match_tables(ours, cb):
    """
    Two-tier one-to-one name match.
      Tier 1 'exact'      : identical Miner_name after removing surrounding whitespace.
      Tier 2 'normalised' : identical norm_name(), among rows not matched in tier 1.
    A name or normalised key that occurs more than once within a list, or that has
    more than one candidate on the other side, is reported as ambiguous and left
    unmatched.
    """
    o = ours.assign(k1=ours["Miner_name"].str.strip(), k2=ours["Miner_name"].map(norm_name))
    c = cb.assign(k1=cb["Miner_name"].str.strip(), k2=cb["Miner_name"].map(norm_name))
    ambiguous = []
    for side, df in (("ours", o), ("cbeci", c)):
        for k in ("k1", "k2"):
            dup = df[df[k].duplicated(keep=False)]
            for key, g in dup.groupby(k):
                ambiguous.append({"side": side, "key_type": k, "key": key,
                                  "names": " | ".join(g["Miner_name"])})
    pairs, used_o, used_c = [], set(), set()
    for tier, k in (("exact", "k1"), ("normalised", "k2")):
        oc = o[~o.index.isin(used_o)]
        cc = c[~c.index.isin(used_c)]
        for key, og in oc.groupby(k):
            cg = cc[cc[k] == key]
            if len(og) == 1 and len(cg) == 1:
                pairs.append((og.index[0], cg.index[0], tier))
                used_o.add(og.index[0])
                used_c.add(cg.index[0])
            elif len(cg) > 0:
                ambiguous.append({"side": "both", "key_type": k, "key": key,
                                  "names": " | ".join(list(og["Miner_name"]) + ["<->"]
                                                      + list(cg["Miner_name"]))})
    return pairs, sorted(set(o.index) - used_o), sorted(set(c.index) - used_c), \
        pd.DataFrame(ambiguous).drop_duplicates() if ambiguous else pd.DataFrame(
            columns=["side", "key_type", "key", "names"])


def write_machine_csv(df, path):
    """Write a table in our column layout (Date of release as DD/MM/YYYY)."""
    _refuse_existing([path])
    df[OUR_COLUMNS].to_csv(path, index=False)


# ======================================================================
# Stage 2: analysis
# ======================================================================
def analyze():
    import modules.efficiency_cbeci as ec
    import cbeci_revenue_check as rc
    from config import USE_FRACTION
    from cbeci_replication import CBECI_TWH, WINDOWS

    if USE_FRACTION:
        raise RuntimeError("config.USE_FRACTION is True; the threshold here assumes it is False")

    out = {
        "cbeci_parsed": os.path.join(OUT_DIR, "cbeci_list_parsed.csv"),
        "matching": os.path.join(OUT_DIR, "matching_table.csv"),
        "ambiguous": os.path.join(OUT_DIR, "ambiguous_matches.csv"),
        "differences": os.path.join(OUT_DIR, "differences.csv"),
        "model_sets": os.path.join(OUT_DIR, "model_machine_sets.csv"),
        "psi": os.path.join(OUT_DIR, "daily_psi_by_machine_table.csv"),
        "annual": os.path.join(OUT_DIR, "annual_energy_by_machine_table.csv"),
        "errors": os.path.join(OUT_DIR, "errors_by_machine_table.csv"),
        "checks": os.path.join(OUT_DIR, "checks.csv"),
        "summary": os.path.join(OUT_DIR, "summary.txt"),
    }
    tables = {
        "ours_cbeci_eff": os.path.join(TABLE_DIR, "ours_with_cbeci_efficiency.csv"),
        "cbeci_list": os.path.join(TABLE_DIR, "cbeci_list.csv"),
        "cbeci_list_v1_1": os.path.join(TABLE_DIR, "cbeci_list_v1_1_and_earlier.csv"),
    }
    _refuse_existing(list(out.values()) + list(tables.values()))
    lines, checks = [], []

    def say(s=""):
        print(s)
        lines.append(s)

    def check(name, value, note=""):
        checks.append({"check": name, "value": value, "note": note})
        say(f"  [check] {name}: {value}  {note}")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 60)

    # ------------------------------------------------------------------
    # Section 0. Source
    # ------------------------------------------------------------------
    with open(os.path.join(RAW_DIR, "fetch_log.json")) as f:
        flog = json.load(f)
    cb, meta = parse_cbeci_xlsx(os.path.join(RAW_DIR, "sheet.xlsx"))
    ours, our_raw = parse_ours(OUR_TABLE)

    say("CBECI machine-list check")
    say("=" * 70)
    say("Section 0. Source of CBECI's list")
    say(f"  {CBECI_URL} answers with an HTTP redirect to a Google Sheet:")
    say(f"    {flog['redirect']}")
    for r in flog["responses"]:
        say(f"    {r['file']:<34s} status={r['http_status']} bytes={r['bytes']:>7d} "
            f"fetched {r['fetched_utc']} sha256 {r['sha256'][:16]}... {r['error']}")
    say(f"  Sheet title: {meta['title']!r}; {meta['last_updated_text']!r}")
    say(f"  Tab {LIST_TAB!r} header: {meta['header'][:11]}")
    say(f"  Rows parsed: {len(cb)} machines (CBECI); {len(ours)} machines (ours).")
    say("  Fields CBECI provides per machine: name, type (CPU/GPU/FPGA/ASIC for the Taylor (2016)")
    say("  rows, blank afterwards), release date (always the 1st of a month except one row), UNIX")
    say("  release date (formula of the date; blank on later rows), hashing power (TH/s), power (W),")
    say("  efficiency (J/Gh; a formula = power / hashing power / 1000), weight (kg), a column")
    say("  'Efficiency: suggested alternative(s)', the version in which the machine was included,")
    say("  and a free-text comment. There is no manufacturer column (the maker is the first word")
    say("  of the name) and no separate 'adjusted' efficiency.")
    n_alt = int(cb["eff_suggested_alternative"].notna().sum())
    say(f"  Rows with a value in 'Efficiency: suggested alternative(s)': {n_alt}")
    say("  Versions present (machines per version):")
    say("    " + ", ".join(f"{v}: {n}" for v, n in
                           cb["included_in_version"].fillna("<blank>").value_counts().sort_index().items()))
    say(f"  Comments present on {int(cb['comment'].notna().sum())} rows; distinct texts:")
    for t in cb["comment"].dropna().unique():
        say(f"    - {t}")

    # Internal consistency of CBECI's sheet.
    eff_formula = cb["power_W"] / cb["hashrate_THs"] / 1000.0
    check("CBECI cached efficiency == power / hashrate / 1000, max rel diff",
          float(np.nanmax(np.abs(cb["eff_J_per_Gh"] / eff_formula - 1))))
    unix_dt = pd.to_datetime(cb["UNIX_date_of_release"], unit="s")
    have_u = unix_dt.notna()
    check("CBECI UNIX date == release date where present",
          f"{int((unix_dt[have_u] == cb.loc[have_u, 'release_date']).sum())} of {int(have_u.sum())}")
    not_first = cb[cb["release_date"].dt.day != 1]
    check("CBECI release dates not on the 1st of a month", len(not_first),
          "; ".join(f"{n} {d.date()}" for n, d in zip(not_first["Miner_name"], not_first["release_date"])))
    gcsv = [r["file"] for r in flog["responses"] if r["file"].startswith("sheet_gid")]
    if gcsv:
        lines_csv = pd.read_csv(os.path.join(RAW_DIR, gcsv[0]), skiprows=3, dtype=str,
                                keep_default_na=False)
        lines_csv = lines_csv[lines_csv["Miner_name"].str.strip() != ""]
        same = (len(lines_csv) == len(cb)
                and (lines_csv["Miner_name"].to_numpy() == cb["Miner_name"].to_numpy()).all())
        check("CSV export (displayed values) has the same rows and names as the xlsx", bool(same),
              f"{len(lines_csv)} CSV rows vs {len(cb)} xlsx rows")

    # Our date parse == _load_machines (on the rows the loader keeps).
    lm_ours = ec._load_machines(OUR_TABLE)
    chk = ours.set_index("Miner_name")["release_date"]
    check("our release-date parse == _load_machines (rows kept by the loader)",
          bool(all(chk.loc[n] == t for n, t in zip(lm_ours["miner_name"], lm_ours["release_ts"]))),
          f"{len(lm_ours)} rows kept of {len(ours)}")

    # ------------------------------------------------------------------
    # Section 1. Matching
    # ------------------------------------------------------------------
    pairs, only_o, only_c, amb = match_tables(ours, cb)
    say("")
    say("Section 1. Name matching")
    say("  Rule: tier 1 = identical name after removing surrounding whitespace; tier 2 = identical")
    say("  normalised name (lower case; 'Hyd.'/'Hyd' -> 'hydro'; 'Imm.' -> 'immersion'; hashrate")
    say("  tags '(14Th)', '16Th/s', '52T' -> '14th', '16th', '52th'; parentheses removed; leading")
    say("  manufacturer word and the words Antminer/Whatsminer/AvalonMiner dropped). One-to-one only.")
    tiers = pd.Series([t for _, _, t in pairs]).value_counts()
    say(f"  matched: {len(pairs)} ({', '.join(f'{k} {v}' for k, v in tiers.items())}); "
        f"only ours: {len(only_o)}; only CBECI: {len(only_c)}; ambiguous keys: {len(amb)}")
    if len(amb):
        say(amb.to_string(index=False))

    rows = []
    for io, ic, tier in pairs:
        o, c = ours.loc[io], cb.loc[ic]
        rows.append({"status": f"matched_{tier}", "name_ours": o["Miner_name"], "name_cbeci": c["Miner_name"],
                     "ours_row": int(o["row"]), "cbeci_xlsx_row": int(c["xlsx_row"]),
                     "release_ours": o["release_date"], "release_cbeci": c["release_date"],
                     "eff_J_per_TH_ours": 1000 * o["eff_J_per_Gh"], "eff_J_per_TH_cbeci": 1000 * c["eff_J_per_Gh"],
                     "hashrate_ours": o["hashrate_THs"], "hashrate_cbeci": c["hashrate_THs"],
                     "power_ours": o["power_W"], "power_cbeci": c["power_W"],
                     "weight_ours": o["weight_kg"], "weight_cbeci": c["weight_kg"],
                     "type_ours": o["Type"] or None, "type_cbeci": c["Type"],
                     "cbeci_version": c["included_in_version"]})
    for io in only_o:
        o = ours.loc[io]
        rows.append({"status": "only_ours", "name_ours": o["Miner_name"], "ours_row": int(o["row"]),
                     "release_ours": o["release_date"], "eff_J_per_TH_ours": 1000 * o["eff_J_per_Gh"],
                     "hashrate_ours": o["hashrate_THs"], "power_ours": o["power_W"],
                     "weight_ours": o["weight_kg"], "type_ours": o["Type"] or None})
    for ic in only_c:
        c = cb.loc[ic]
        rows.append({"status": "only_cbeci", "name_cbeci": c["Miner_name"], "cbeci_xlsx_row": int(c["xlsx_row"]),
                     "release_cbeci": c["release_date"], "eff_J_per_TH_cbeci": 1000 * c["eff_J_per_Gh"],
                     "hashrate_cbeci": c["hashrate_THs"], "power_cbeci": c["power_W"],
                     "weight_cbeci": c["weight_kg"], "type_cbeci": c["Type"],
                     "cbeci_version": c["included_in_version"]})
    mt = pd.DataFrame(rows)
    mt["release_date_diff_days"] = (mt["release_ours"] - mt["release_cbeci"]).dt.days
    mt["eff_rel_diff"] = mt["eff_J_per_TH_ours"] / mt["eff_J_per_TH_cbeci"] - 1
    # Is our efficiency CBECI's value rounded to 4 decimals of J/Gh (the CSV display)?
    mt["eff_ours_eq_cbeci_rounded_4dp"] = np.isclose(
        mt["eff_J_per_TH_ours"] / 1000, np.round(mt["eff_J_per_TH_cbeci"] / 1000, 4), rtol=0, atol=1e-12)
    mt["hashrate_rel_diff"] = mt["hashrate_ours"] / mt["hashrate_cbeci"] - 1
    mt["power_rel_diff"] = mt["power_ours"] / mt["power_cbeci"] - 1
    rel = mt["release_ours"].fillna(mt["release_cbeci"])
    mt["release_year"] = rel.dt.year
    name = mt["name_ours"].fillna(mt["name_cbeci"])
    mt["in_model_set"] = [passes_filter(n, r) for n, r in zip(name, rel)]
    # Calendar years in which the machine can be in the fleet: deployment (release +
    # 2 months) up to 60 months after deployment.
    dep = rel.map(ec._deployment_date)
    mt["fleet_window_first_year"] = dep.dt.year
    mt["fleet_window_last_year"] = (dep + pd.DateOffset(months=60) - pd.Timedelta(days=1)).dt.year
    mt["flag_eff_gt_1pct"] = mt["eff_rel_diff"].abs() > EFF_REL_TOL
    mt["flag_date_gt_15d"] = mt["release_date_diff_days"].abs() > DATE_TOL_DAYS
    mt = mt.sort_values(["release_year", "status", "name_ours"], na_position="last").reset_index(drop=True)

    m = mt[mt["status"].str.startswith("matched")]
    say("")
    say("  Matched machines, field comparison:")
    say(f"    release date identical: {int((m['release_date_diff_days'] == 0).sum())} of {len(m)}; "
        f"|diff| > {DATE_TOL_DAYS} days: {int(m['flag_date_gt_15d'].sum())}")
    say(f"    efficiency |rel diff| > 1 %: {int(m['flag_eff_gt_1pct'].sum())}; "
        f"max |rel diff| {m['eff_rel_diff'].abs().max():.3e} "
        f"({m.loc[m['eff_rel_diff'].abs().idxmax(), 'name_ours']})")
    say(f"    our efficiency == CBECI's rounded to 4 decimals of J/Gh: "
        f"{int(m['eff_ours_eq_cbeci_rounded_4dp'].sum())} of {len(m)}")
    say(f"    hashrate identical: {int((m['hashrate_rel_diff'].abs() < 1e-12).sum())}; "
        f"power identical: {int((m['power_rel_diff'].abs() < 1e-12).sum())} of {len(m)}")
    for yrs in ((2013,), (2015, 2016), (2018, 2019)):
        mm = m[m["release_year"].isin(yrs)]
        say(f"    released in {'/'.join(map(str, yrs))}: {len(mm)} matched, max |eff rel diff| "
            f"{mm['eff_rel_diff'].abs().max():.3e}, max |date diff| {mm['release_date_diff_days'].abs().max()} d")
    say("  Largest efficiency differences among matched machines (top 10):")
    say(m.reindex(m["eff_rel_diff"].abs().sort_values(ascending=False).index).head(10)[
        ["name_ours", "release_year", "eff_J_per_TH_ours", "eff_J_per_TH_cbeci", "eff_rel_diff",
         "in_model_set"]].to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    say("")
    show = ["name_ours", "name_cbeci", "release_ours", "release_cbeci", "eff_J_per_TH_ours",
            "eff_J_per_TH_cbeci", "cbeci_version", "in_model_set"]
    say("  Machines only in CBECI's list:")
    oc = mt[mt["status"] == "only_cbeci"]
    say(oc[["name_cbeci", "release_cbeci", "eff_J_per_TH_cbeci", "cbeci_version", "in_model_set",
            "fleet_window_first_year"]].to_string(index=False, float_format=lambda x: f"{x:.4g}")
        if len(oc) else "    none")
    say("  Machines only in ours:")
    oo = mt[mt["status"] == "only_ours"]
    say(oo[["name_ours", "release_ours", "eff_J_per_TH_ours", "in_model_set"]]
        .to_string(index=False, float_format=lambda x: f"{x:.4g}") if len(oo) else "    none")
    if any(mt["status"] == "matched_normalised"):
        say("  Pairs matched only after normalisation:")
        say(mt[mt["status"] == "matched_normalised"][show].to_string(index=False))
    diffs = mt[(mt["status"] != "matched_exact") | mt["flag_eff_gt_1pct"] | mt["flag_date_gt_15d"]]
    first_only_c = oc["release_cbeci"].min() if len(oc) else None
    say(f"  Earliest release among machines only in CBECI's list: "
        f"{first_only_c.date() if first_only_c is not None else 'n/a'}")

    # ------------------------------------------------------------------
    # Section 2. Alternative machine tables
    # ------------------------------------------------------------------
    os.makedirs(TABLE_DIR, exist_ok=True)

    def cb_layout(df):
        return pd.DataFrame({
            "Miner_name": df["Miner_name"].to_numpy(),
            "Type": df["Type"].fillna("").to_numpy(),
            "Date of release": df["release_date"].dt.strftime("%d/%m/%Y").to_numpy(),
            "UNIX_date_of_release": df["UNIX_date_of_release"].map(
                lambda x: "" if pd.isna(x) else str(int(round(x)))).to_numpy(),
            "Hashing Power (TH/s)": df["hashrate_THs"].map(repr).to_numpy(),
            "Power (W)": df["power_W"].map(repr).to_numpy(),
            "Efficiency (J/Gh)": df["eff_J_per_Gh"].map(repr).to_numpy(),
            "Weight in kg": df["weight_kg"].map(lambda x: "" if pd.isna(x) else repr(x)).to_numpy(),
        })

    write_machine_csv(cb_layout(cb), tables["cbeci_list"])
    v11 = cb["included_in_version"].fillna("").str.match(r"^1\.[01]\.")
    write_machine_csv(cb_layout(cb[v11]), tables["cbeci_list_v1_1"])
    alt = our_raw.copy()
    for io, ic, _ in pairs:
        alt.loc[ours.loc[io, "row"], "Efficiency (J/Gh)"] = repr(float(cb.loc[ic, "eff_J_per_Gh"]))
    write_machine_csv(alt, tables["ours_cbeci_eff"])

    variant_paths = {"ours": OUR_TABLE, **tables}
    machines = {k: ec._load_machines(p) for k, p in variant_paths.items()}
    say("")
    say("Section 2. Machine tables replayed (after _load_machines: date handling + manufacturer filter)")
    for k, df in machines.items():
        say(f"  {k:<16s} {len(df):>4d} machines kept  ({os.path.relpath(variant_paths[k], ROOT)})")
    a, b = machines["ours"], machines["ours_cbeci_eff"]
    same_other = (len(a) == len(b) and (a["miner_name"].to_numpy() == b["miner_name"].to_numpy()).all()
                  and (a["deployment_ts"].to_numpy() == b["deployment_ts"].to_numpy()).all())
    check("ours_cbeci_eff differs from ours only in efficiency (names, order, deployment equal)",
          bool(same_other))
    ms_rows = []
    for k, df in machines.items():
        for _, r in df.iterrows():
            ms_rows.append({"variant": k, "miner_name": r["miner_name"], "release": r["release_ts"].date(),
                            "deployment": r["deployment_ts"].date(), "eta_J_per_TH": r["eta_TH_J_per_TH"]})
    model_sets = pd.DataFrame(ms_rows)
    # Set comparison of the model-relevant machines by the window in which they can be
    # in the fleet before 2024.
    so, sc = set(machines["ours"]["miner_name"].str.strip()), set(machines["cbeci_list"]["miner_name"].str.strip())
    say(f"  model set, ours vs cbeci_list: {len(so & sc)} shared names, "
        f"{len(sc - so)} only in cbeci_list, {len(so - sc)} only in ours")
    late = machines["cbeci_list"][machines["cbeci_list"]["miner_name"].str.strip().isin(sc - so)]
    if len(late):
        say(f"    cbeci_list-only machines in the model set deploy from "
            f"{late['deployment_ts'].min().date()} to {late['deployment_ts'].max().date()}")
    dropped = cb[~v11]
    say(f"  cbeci_list_v1_1 drops {len(dropped)} rows added in version 1.2.0 or later; their release dates "
        f"run {dropped['release_date'].min().date()} to {dropped['release_date'].max().date()};")
    say("    those released before 2020-10: " + (", ".join(
        f"{n.strip()} ({d.date()}, v{v})" for n, d, v in
        zip(dropped["Miner_name"], dropped["release_date"], dropped["included_in_version"])
        if d < pd.Timestamp("2020-10-01")) or "none"))

    # ------------------------------------------------------------------
    # Section 3. Replays with Coin Metrics revenue and hashrate
    # ------------------------------------------------------------------
    daily = pd.read_csv(os.path.join(REV_DIR, "model_daily_revenue_appends.csv"),
                        parse_dates=["revenue_day", "append_day", "target_date"],
                        float_precision="round_trip")
    blocks = pd.read_csv(os.path.join(REV_DIR, "model_per_block.csv.gz"),
                         usecols=["sim_timestamp", "target_date", "efficiency_J_per_hash"],
                         parse_dates=["sim_timestamp", "target_date"], float_precision="round_trip")
    cm = pd.read_csv(os.path.join(REV_DIR, "coinmetrics_daily.csv"), parse_dates=["date_utc"],
                     index_col="date_utc", float_precision="round_trip")
    rc_psi = pd.read_csv(os.path.join(REV_DIR, "daily_psi_by_revenue_source.csv"), parse_dates=["date_utc"],
                         index_col="date_utc", float_precision="round_trip")
    rc_ann = pd.read_csv(os.path.join(REV_DIR, "annual_energy_by_revenue_source.csv"), index_col="year",
                         float_precision="round_trip")

    # Coin Metrics theta per revenue day, built exactly as cbeci_revenue_check.analyze builds it.
    d = daily.copy()
    d["date_utc"] = d["revenue_day"].dt.normalize()
    d = d.set_index("date_utc")
    cmj = cm.reindex(d.index)
    revenue_usd_cm = cmj["IssTotUSD"] + cmj["FeeTotNtv"] * cmj["PriceUSD"]
    H_THs_cm = cmj["HashRate"]
    R_cm = revenue_usd_cm / (H_THs_cm * 86400.0)
    have_cm = pd.concat([revenue_usd_cm, H_THs_cm], axis=1).notna().all(axis=1) & (H_THs_cm > 0)
    theta_cm = [float(x) for x in (R_cm / rc.P_PER_J).where(have_cm, d["theta_J_per_TH"]).to_numpy()]
    ma_cm = rc._ma_by_count(theta_cm)

    # Per-block replay keys, as in cbeci_revenue_check.analyze.
    td_day = blocks["target_date"].dt.normalize().to_numpy()
    td_int = td_day.astype("datetime64[D]").astype(np.int64)
    runmax = np.maximum.accumulate(td_int)
    inc = np.concatenate([[True], runmax[1:] > runmax[:-1]])
    kcount = np.cumsum(inc)
    check("appends inferred from block dates == recorded appends",
          bool(kcount[-1] == len(daily)
               and np.array_equal(td_day[inc], daily["append_day"].dt.normalize().to_numpy())))
    day_keys = [pd.Timestamp(x).date() for x in td_day]
    block_ts = blocks["sim_timestamp"].to_numpy()
    append_days = daily["append_day"].dt.normalize().to_numpy()

    def replay_blocks(mdf, ma):
        sel = rc.FleetSelector(mdf, ec._months_between, ec._age_weight)
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
        s = pd.Series(eff * 1e12, index=block_ts)
        p = s.resample("D").mean()
        p.index = p.index.normalize()
        return p

    def replay_daily(mdf, ma):
        sel = rc.FleetSelector(mdf, ec._months_between, ec._age_weight)
        days = pd.date_range(pd.Timestamp(append_days[0]), "2023-12-31", freq="D")
        k_of_day = np.searchsorted(append_days, days.to_numpy(), side="right")
        vals = [sel.evaluate(day.date(), ma[int(k)]) * 1e12 for day, k in zip(days, k_of_day)]
        return pd.Series(vals, index=days)

    # Sanity: with our own revenue and our table, the block replay is the model run.
    ma_ours = rc._ma_by_count([float(x) for x in daily["theta_J_per_TH"].to_numpy()])
    eff_chk = replay_blocks(machines["ours"], ma_ours)
    rec = blocks["efficiency_J_per_hash"].to_numpy()
    check("block replay, our revenue + our table == model run, blocks identical",
          f"{int(np.sum(eff_chk == rec))} of {len(rec)}")
    del eff_chk

    psi = {}
    t0 = time.time()
    for k, mdf in machines.items():
        psi[f"block_{k}"] = daily_psi_from_blocks(replay_blocks(mdf, ma_cm))
        psi[f"daily_{k}"] = replay_daily(mdf, ma_cm)
        print(f"  replayed {k} ({time.time() - t0:.0f} s)")

    rng = slice("2011-01-01", "2023-12-31")
    for ours_key, rc_key in (("block_ours", "replay_cm"), ("daily_ours", "daily_cm")):
        a_, b_ = psi[ours_key].loc[rng], rc_psi[rc_key].loc[rng]
        check(f"{ours_key} daily psi vs revenue check {rc_key} (CSV at 10 sig. fig.), max rel diff",
              float(np.nanmax(np.abs(a_ / b_ - 1))))

    H_cm = cm["HashRate"].dropna()
    cbeci = pd.Series(CBECI_TWH, name="CBECI_TWh")
    ann = pd.DataFrame({"CBECI_TWh": cbeci})
    ann.index.name = "year"
    for k, p in psi.items():
        ann[f"TWh_{k}"] = rc._annual_twh(H_cm, p).reindex(ann.index)
    for k in psi:
        ann[f"PctDiff_{k}"] = 100 * (ann[f"TWh_{k}"] - ann["CBECI_TWh"]) / ann["CBECI_TWh"]
    for ours_key, rc_key in (("block_ours", "replay_cm"), ("daily_ours", "daily_cm")):
        check(f"annual TWh {ours_key} vs revenue check TWh_{rc_key} (CSV at 6 dp), max abs diff",
              float((ann[f"TWh_{ours_key}"] - rc_ann[f"TWh_{rc_key}"]).abs().max()))
    for k in machines:
        if k != "ours":
            ann[f"effect_vs_ours_pct_block_{k}"] = 100 * (ann[f"TWh_block_{k}"] / ann["TWh_block_ours"] - 1)
    ann["cbeci_rounding_halfwidth_pct"] = 100 * 0.005 / ann["CBECI_TWh"]

    erows = []
    for k in psi:
        for lo, hi in WINDOWS:
            sub = ann.loc[lo:hi]
            erows.append({"series": k, "window": f"{lo}-{hi}",
                          "MAPE_pct": sub[f"PctDiff_{k}"].abs().mean(),
                          "WAPE_pct": (sub[f"TWh_{k}"] - sub["CBECI_TWh"]).abs().sum()
                          / sub["CBECI_TWh"].sum() * 100.0})
    err = pd.DataFrame(erows)

    say("")
    say("Section 3. Annual energy, H = Coin Metrics HashRate, Coin Metrics revenue, PUE 1.10, $50/MWh")
    say("  'block_*' = per-block replay (the revenue check's replay_cm path);")
    say("  'daily_*' = one fleet evaluation per calendar day (the revenue check's daily_cm path).")
    say("  percent difference from CBECI by year:")
    say(ann[[f"PctDiff_{k}" for k in psi]].rename(columns=lambda c: c.replace("PctDiff_", ""))
        .to_string(float_format=lambda x: f"{x:+.3f}"))
    say("")
    say("  effect of each table on annual energy (% change vs ours, per-block replay):")
    say(ann[[c for c in ann.columns if c.startswith("effect_vs_ours")]]
        .rename(columns=lambda c: c.replace("effect_vs_ours_pct_block_", ""))
        .to_string(float_format=lambda x: f"{x:+.4f}"))
    say("")
    say("  WAPE / MAPE (%):")
    say(err.pivot_table(index="series", columns="window", values=["WAPE_pct", "MAPE_pct"], sort=False)
        .to_string(float_format=lambda x: f"{x:.3f}"))

    # Days on which psi differs from ours, per variant and year.
    say("")
    say("  days per year on which daily psi differs from ours (per-block replay, |rel| > 1e-12):")
    nd = {}
    for k in machines:
        if k == "ours":
            continue
        r = (psi[f"block_{k}"] / psi["block_ours"] - 1).loc[rng]
        nd[k] = (r.abs() > 1e-12).groupby(r.index.year).sum()
        nd[f"{k}_max_abs_rel"] = r.abs().groupby(r.index.year).max()
    say(pd.DataFrame(nd).to_string(float_format=lambda x: f"{x:.2e}"))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    _refuse_existing(out.values())
    cb.to_csv(out["cbeci_parsed"], index=False)
    mt.to_csv(out["matching"], index=False, float_format="%.10g")
    amb.to_csv(out["ambiguous"], index=False)
    diffs.to_csv(out["differences"], index=False, float_format="%.10g")
    model_sets.to_csv(out["model_sets"], index=False, float_format="%.10g")
    pd.DataFrame(psi).rename_axis("date_utc").to_csv(out["psi"], float_format="%.10g")
    ann.to_csv(out["annual"], float_format="%.6f")
    err.to_csv(out["errors"], index=False, float_format="%.6f")
    pd.DataFrame(checks).to_csv(out["checks"], index=False)
    with open(out["summary"], "x") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nOutputs written to {OUT_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("stage", choices=["fetch", "analyze"])
    args = ap.parse_args()
    if args.stage == "fetch":
        fetch()
    else:
        analyze()


if __name__ == "__main__":
    main()
