#!/usr/bin/env python3
"""Figure 3 of the Paper 3 manuscript: the two-panel calibration heatmap.

Panel (a) shows the RMSLE of the pre-2018 grid over the early-era elasticity S_0 and
electricity cost C_elec_0. Panel (b) shows the RMSLE of the 2018-2024 grid over S and
C_elec. The minimum of each grid is marked with a cross and labelled in a box beside the
marker with its parameters and RMSLE. The panel letters sit above the top-left corner of
each axes.

The plotting code is based on `plot_sensitivity_two_panel` from
data/results/scenario_results.ipynb (cell index 10): the same rcParams, viridis_r
pcolormesh, white contours, minimum marker, LaTeX-style axis labels, width and
constrained layout. These things differ from the notebook:
  * the font is Liberation Serif (metric-compatible with the manuscript's Times New
    Roman); the notebook sets no font, so matplotlib used DejaVu Sans. Mathtext uses the
    STIX fonts, which match a Times-style text face;
  * the height follows the aspect ratio of the drawing the figure replaces (see below),
    and the figure is saved without bbox_inches="tight", which would change that ratio;
  * the legend of the notebook is replaced by a boxed annotation beside the minimum
    marker (parameters and RMSLE), so that no second cross appears in the data area;
  * the panel letters sit above the top-left corner of each axes instead of inside it;
  * the annotation font is 10 pt at widths of 10 in or more and 7.5 pt below that, so
    that the box fits inside the axes at print size;
  * the inputs are the CSVs of the calibration grid driver (calibration/run_grid.py).

Aspect ratio: the wp:extent of the picture inside the text box whose caption starts
"Figure 3." in the manuscript. The notebook width of 16.0 in is kept and the height is
16.0 * cy / cx. Pass --size-in WIDTH HEIGHT to set the size directly (this skips the
docx).

The CSVs may carry extra columns; only S_0/C_elec_0 (pre-2018) or S/C_elec (modern) and
RMSLE are read. The script refuses to overwrite an existing output file.

Usage:
    python calibration/plot_grid_figure3.py
    python calibration/plot_grid_figure3.py --out some/other/figure.png
"""

import argparse
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The grid whose optimum config.py holds (the paper's calibration). Until 2026-10-06 the
# default was data/results/grid_2026-10-01, the grid run before the clock and UTC fixes.
GRID_DIR = "data/results/grid_2026-10-rerun"
DEFAULT_PRE = os.path.join(GRID_DIR, "grid_pre2018.csv")
DEFAULT_MODERN = os.path.join(GRID_DIR, "grid_2018_2024.csv")
DEFAULT_OUT = os.path.join(GRID_DIR, "figure3_calibration_grid.png")
DEFAULT_DOCX = ("/home/jynurso/gitea/bitcoin-mining-research/Thesis_Papers/Paper_3/"
                "network_model_paper_3_short_version_restructured-tracked.docx")
CAPTION_START = re.compile(r"\s*Figure\s*3\.")
NOTEBOOK_WIDTH_IN = 16.0
DPI = 300
EMU_PER_INCH = 914400
FONT = "Liberation Serif"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def extent_from_docx(path):
    """Return (cx_emu, cy_emu, media_part) of the picture in the Figure 3 text box."""
    z = zipfile.ZipFile(path)
    doc = ET.fromstring(z.read("word/document.xml"))
    rels = ET.fromstring(z.read("word/_rels/document.xml.rels"))
    target = {r.get("Id"): r.get("Target") for r in rels}
    w = "{%s}" % NS["w"]
    # The caption and the picture share one paragraph inside the text box (the box is
    # stored twice, in mc:Choice and mc:Fallback; both copies carry the same extent).
    for para in doc.iter(w + "p"):
        text = "".join(t.text or "" for t in para.iter(w + "t"))
        if not CAPTION_START.match(text):
            continue
        inline = para.find(".//wp:inline", NS)
        if inline is None:
            continue
        ext = inline.find("wp:extent", NS)
        blip = inline.find(".//a:blip", NS)
        rid = blip.get("{%s}embed" % NS["r"])
        return int(ext.get("cx")), int(ext.get("cy")), "word/" + target[rid]
    sys.exit("no picture found beside a caption starting 'Figure 3.' in %s" % path)


def ensure_font():
    """Make Liberation Serif available to matplotlib.

    matplotlib reads installed fonts from a cache that can predate the installation, so
    the font may be missing from font_manager.fontManager although it is on disk. In that
    case every LiberationSerif-*.ttf found on the system is added for this run only.
    """
    def have():
        return FONT in {f.name for f in font_manager.fontManager.ttflist}
    if not have():
        for path in font_manager.findSystemFonts(fontext="ttf"):
            if os.path.basename(path).startswith("LiberationSerif-"):
                font_manager.fontManager.addfont(path)
    if not have():
        sys.exit("font %s not found" % FONT)


def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "font.family": "serif",
        "font.serif": [FONT],
        "mathtext.fontset": "stix",
    })


def pivot_grid(df, row_col, col_col):
    pv = (df.pivot(index=row_col, columns=col_col, values="RMSLE")
          .sort_index(ascending=True).sort_index(axis=1, ascending=True))
    x = pv.columns.values.astype(float)
    y = pv.index.values.astype(float)
    xx, yy = np.meshgrid(x, y)
    return xx, yy, pv.values.astype(float)


def draw_panel(fig, ax, df, s_col, c_col, xlabel, ylabel, letter, s_symbol, label_fs):
    """One heatmap panel; returns (best_S, best_C, best_rmsle).

    s_symbol is the mathtext symbol of the elasticity in the minimum label; label_fs is
    the font size of that label in points.
    """
    xx, yy, zz = pivot_grid(df, c_col, s_col)
    hm = ax.pcolormesh(xx, yy, zz, cmap="viridis_r", shading="auto")
    ax.contour(xx, yy, zz, colors="white", alpha=0.30, linewidths=0.8)

    idx = df["RMSLE"].astype(float).idxmin()
    best_s = float(df.loc[idx, s_col])
    best_c = float(df.loc[idx, c_col])
    best_rmsle = float(df.loc[idx, "RMSLE"])
    ax.plot(best_s, best_c, marker="x", markersize=10, markeredgewidth=2.5,
            linestyle="None", color="#111111", zorder=3)

    # Label the minimum beside the marker. The label goes to the left of the marker when
    # the marker lies in the right half of the axes, and to the right otherwise; it goes
    # below the marker in the upper half and above it in the lower half. So the label
    # stays inside the axes and does not cover the marker.
    fx = (best_s - xx.min()) / (xx.max() - xx.min())
    fy = (best_c - yy.min()) / (yy.max() - yy.min())
    dx = -16 if fx > 0.5 else 16
    dy = -16 if fy > 0.5 else 16
    ax.annotate(
        rf"Minimum: {s_symbol} = {best_s:.2f}, \${best_c:.0f}/MWh" "\n"
        rf"RMSLE = {best_rmsle:.3f}",
        xy=(best_s, best_c), xytext=(dx, dy), textcoords="offset points",
        ha="right" if dx < 0 else "left", va="top" if dy < 0 else "bottom",
        multialignment="left",
        fontsize=label_fs, color="#111111", zorder=4,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                  edgecolor="#555555", linewidth=0.6))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", axis="both")
    ax.set_axisbelow(True)
    ax.text(0.0, 1.02, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=13, fontweight="bold")
    cb = fig.colorbar(hm, ax=ax, pad=0.02)
    cb.set_label("RMSLE (Hashrate)")
    return best_s, best_c, best_rmsle


def read_grid(path, need):
    df = pd.read_csv(path)
    if not set(need) <= set(df.columns):
        sys.exit("%s lacks columns %s" % (path, sorted(set(need) - set(df.columns))))
    if df["RMSLE"].isna().any():
        sys.exit("%s has missing RMSLE values" % path)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pre2018", default=DEFAULT_PRE, help="pre-2018 grid CSV (S_0, C_elec_0, RMSLE)")
    ap.add_argument("--modern", default=DEFAULT_MODERN, help="2018-2024 grid CSV (S, C_elec, RMSLE)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output PNG (refuses to overwrite)")
    ap.add_argument("--docx", default=DEFAULT_DOCX,
                    help="manuscript whose Figure 3 drawing fixes the aspect ratio")
    ap.add_argument("--size-in", nargs=2, type=float, metavar=("W", "H"),
                    help="figure size in inches; skips reading the docx")
    args = ap.parse_args()

    def rooted(p):
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    out = rooted(args.out)
    if os.path.exists(out):
        sys.exit("refusing to overwrite %s" % out)
    df_pre = read_grid(rooted(args.pre2018), ["S_0", "C_elec_0", "RMSLE"])
    df_mod = read_grid(rooted(args.modern), ["S", "C_elec", "RMSLE"])

    if args.size_in:
        width, height = args.size_in
        print("figure size %.4f x %.4f in (--size-in)" % (width, height))
    else:
        cx, cy, media = extent_from_docx(args.docx)
        width, height = NOTEBOOK_WIDTH_IN, NOTEBOOK_WIDTH_IN * cy / cx
        print("manuscript extent %d x %d EMU (%.4f x %.4f in), media part %s"
              % (cx, cy, cx / EMU_PER_INCH, cy / EMU_PER_INCH, media))
        print("figure size %.4f x %.4f in (width of the notebook, aspect of the extent)"
              % (width, height))

    ensure_font()
    print("font", FONT)
    set_pub_style()

    # At the notebook width the label is 10 pt. At print width (about 7 in) the axis and
    # tick fonts stay as above, and a 10 pt label is wider than the left part of panel
    # (a), so it is set smaller there.
    label_fs = 10 if width >= 10 else 7.5
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(width, height),
                             constrained_layout=True)
    b0 = draw_panel(fig, axes[0], df_pre, "S_0", "C_elec_0", r"$S_0$ (Elasticity)",
                    r"$C_{\mathrm{elec},0}$ (\$/MWh)", "(a)", "$S_0$", label_fs)
    b1 = draw_panel(fig, axes[1], df_mod, "S", "C_elec", r"$S$ (Elasticity)",
                    r"$C_{\mathrm{elec}}$ (\$/MWh)", "(b)", "$S$", label_fs)
    print("panel (a) minimum: S_0=%g C_elec_0=%g RMSLE=%.6f" % b0)
    print("panel (b) minimum: S=%g C_elec=%g RMSLE=%.6f" % b1)

    fig.savefig(out, dpi=DPI)
    print("wrote", out)


if __name__ == "__main__":
    main()
