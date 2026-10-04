#!/usr/bin/env python
"""Compare a finished run's CSV with a saved original, row by row.

Both files are read in lockstep in chunks, every value as text. A value counts
as identical only when its text is identical (both empty counts as identical).
For numeric columns whose text differs, the report gives the maximum absolute
and relative difference, where relative = |run - original| / |original| over
rows with a non-zero original; rows where the original is 0 and the run is not
are counted separately. Timestamp differences are reported in seconds.

Before the column comparison, the two files are hashed: byte-identical files
are reported as such.

Exit status: 0 if every compared column is identical in every row and the row
counts match; 1 otherwise; 2 on bad input.
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

KEY_COLUMNS = ["block_height", "Timestamp", "H_sim", "efficiency", "P_USD", "E_sim", "R_pe"]
TIME_COLUMNS = {"Timestamp", "Hist_Timestamp", "last_retarget_ts"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


class ColumnStats:
    def __init__(self, name):
        self.name = name
        self.rows = 0
        self.text_diff = 0          # rows whose text differs
        self.missing_mismatch = 0   # one side empty, the other not
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.zero_orig_nonzero_run = 0
        self.first_diff_block = None
        self.first_diff_values = None
        self.numeric = True

    def update(self, run, orig, blocks):
        self.rows += len(run)
        r_na, o_na = run.isna().to_numpy(), orig.isna().to_numpy()
        same = (run.fillna("") == orig.fillna("")).to_numpy()
        diff = ~same
        if not diff.any():
            return
        self.text_diff += int(diff.sum())
        self.missing_mismatch += int((r_na != o_na).sum())
        if self.first_diff_block is None:
            i = int(np.argmax(diff))
            self.first_diff_block = blocks[i]
            self.first_diff_values = (run.iloc[i], orig.iloc[i])
        both = diff & ~r_na & ~o_na
        if not both.any():
            return
        if self.name in TIME_COLUMNS:
            a = pd.to_datetime(run[both], errors="coerce")
            b = pd.to_datetime(orig[both], errors="coerce")
            d = (a - b).dt.total_seconds().abs().to_numpy()
            if np.isfinite(d).any():
                self.max_abs = max(self.max_abs, float(np.nanmax(d)))
            return
        a = pd.to_numeric(run[both], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(orig[both], errors="coerce").to_numpy(dtype=float)
        if np.isnan(a).any() or np.isnan(b).any():
            self.numeric = False  # e.g. a boolean or text column; text counts still apply
        ok = ~np.isnan(a) & ~np.isnan(b)
        if not ok.any():
            return
        a, b = a[ok], b[ok]
        ad = np.abs(a - b)
        self.max_abs = max(self.max_abs, float(ad.max()))
        nz = b != 0
        if nz.any():
            self.max_rel = max(self.max_rel, float((ad[nz] / np.abs(b[nz])).max()))
        self.zero_orig_nonzero_run += int(((b == 0) & (a != 0)).sum())

    def line(self):
        if self.text_diff == 0:
            return f"  {self.name:<22} identical in all {self.rows} rows"
        unit = " s" if self.name in TIME_COLUMNS else ""
        s = (f"  {self.name:<22} DIFFERS in {self.text_diff} of {self.rows} rows; "
             f"max abs diff {self.max_abs:.6g}{unit}")
        if self.name not in TIME_COLUMNS:
            s += f", max rel diff {self.max_rel:.6g}"
            if self.zero_orig_nonzero_run:
                s += f", {self.zero_orig_nonzero_run} rows original=0 run!=0"
        if self.missing_mismatch:
            s += f", {self.missing_mismatch} rows empty on one side only"
        s += (f"; first at block {self.first_diff_block} "
              f"(run {self.first_diff_values[0]!r}, original {self.first_diff_values[1]!r})")
        return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_csv")
    ap.add_argument("original_csv")
    ap.add_argument("--columns", default=",".join(KEY_COLUMNS),
                    help="comma-separated columns to compare (default: %(default)s)")
    ap.add_argument("--all-columns", action="store_true", help="compare every column the files share")
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--no-hash", action="store_true", help="skip the whole-file hash comparison")
    ap.add_argument("--head", type=int, default=0,
                    help="compare only the first N data rows of each file (partial check)")
    args = ap.parse_args()

    for p in (args.run_csv, args.original_csv):
        if not os.path.isfile(p):
            print(f"ERROR: {p} not found", file=sys.stderr)
            return 2
    print(f"run:      {args.run_csv} ({os.path.getsize(args.run_csv)} bytes)")
    print(f"original: {args.original_csv} ({os.path.getsize(args.original_csv)} bytes)")
    if args.head < 0:
        print("ERROR: --head must be >= 0", file=sys.stderr)
        return 2
    if args.head:
        args.no_hash = True  # a whole-file hash says nothing about a prefix
        print(f"PARTIAL CHECK: only the first {args.head} rows of each file are compared")
    if not args.no_hash:
        h_run, h_orig = sha256(args.run_csv), sha256(args.original_csv)
        print(f"sha256 run {h_run}\nsha256 orig {h_orig}")
        print("files are BYTE-IDENTICAL" if h_run == h_orig else "files are not byte-identical")

    hdr_run = list(pd.read_csv(args.run_csv, nrows=0).columns)
    hdr_orig = list(pd.read_csv(args.original_csv, nrows=0).columns)
    if hdr_run != hdr_orig:
        print(f"header differs: only in run {sorted(set(hdr_run) - set(hdr_orig))}, "
              f"only in original {sorted(set(hdr_orig) - set(hdr_run))}, "
              f"same set, different order: {set(hdr_run) == set(hdr_orig)}")
    else:
        print(f"headers identical ({len(hdr_run)} columns)")
    cols = ([c for c in hdr_orig if c in hdr_run] if args.all_columns
            else [c.strip() for c in args.columns.split(",") if c.strip()])
    missing = [c for c in cols if c not in hdr_run or c not in hdr_orig]
    if missing:
        print(f"ERROR: column(s) {missing} not in both files", file=sys.stderr)
        return 2
    if "block_height" not in cols:
        cols = ["block_height"] + cols

    stats = {c: ColumnStats(c) for c in cols}
    n_run = n_orig = 0
    misaligned = None
    nrows = args.head or None
    it_run = pd.read_csv(args.run_csv, usecols=cols, dtype=str, keep_default_na=False,
                         na_values=[""], chunksize=args.chunksize, nrows=nrows)
    it_orig = pd.read_csv(args.original_csv, usecols=cols, dtype=str, keep_default_na=False,
                          na_values=[""], chunksize=args.chunksize, nrows=nrows)
    pending_run = pending_orig = None
    while True:
        # Refill each side independently; chunks can end at different rows only if one
        # file is shorter, so the comparison runs over the common prefix.
        if pending_run is None:
            pending_run = next(it_run, None)
        if pending_orig is None:
            pending_orig = next(it_orig, None)
        if pending_run is None or pending_orig is None:
            break
        k = min(len(pending_run), len(pending_orig))
        a, b = pending_run.iloc[:k], pending_orig.iloc[:k]
        blocks = b["block_height"].to_numpy()
        if misaligned is None and not (a["block_height"].to_numpy() == blocks).all():
            i = int(np.argmax(a["block_height"].to_numpy() != blocks))
            misaligned = (n_orig + i, a["block_height"].iloc[i], blocks[i])
        for c in cols:
            stats[c].update(a[c].reset_index(drop=True), b[c].reset_index(drop=True), blocks)
        n_run += k
        n_orig += k
        pending_run = pending_run.iloc[k:] if k < len(pending_run) else None
        pending_orig = pending_orig.iloc[k:] if k < len(pending_orig) else None
    # Count any rows left on the longer side.
    for rest, it, side in ((pending_run, it_run, "run"), (pending_orig, it_orig, "orig")):
        extra = (len(rest) if rest is not None else 0) + sum(len(ch) for ch in it)
        if side == "run":
            n_run += extra
        else:
            n_orig += extra

    print(f"rows: run {n_run}, original {n_orig}" + ("" if n_run == n_orig else "  ROW COUNTS DIFFER"))
    if misaligned:
        print(f"ROWS MISALIGNED: first at row {misaligned[0]}: run block {misaligned[1]}, "
              f"original block {misaligned[2]}")
    for c in cols:
        print(stats[c].line())
    identical = n_run == n_orig and misaligned is None and all(s.text_diff == 0 for s in stats.values())
    scope = f" (first {args.head} rows only)" if args.head else ""
    print("RESULT: " + ("REPRODUCED: every compared column identical in every row" + scope if identical
                        else "NOT REPRODUCED" + scope))
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
