#!/usr/bin/env python3
"""Figure 2 of the Paper 3 manuscript: annual electricity consumption from CBECI's
best-guess estimates and from this repository's implementation of the CBECI methodology.

Reads <in-dir>/annual_comparison.csv, written by validation/cbeci_replication.py, and
writes <in-dir>/figure2_cbeci_replication.png at 300 dpi, or the path given by --out. It
refuses to overwrite an existing figure. The font is Liberation Serif, with DejaVu Serif
as the fallback; the script prints which one it used.

Two stacked panels share the year axis:
  (a) annual consumption in TWh on a logarithmic axis, CBECI against the implementation
      run with one-day hashrate and $50/MWh (columns CBECI_TWh and TWh_H_1day);
  (b) the percent difference of the implementation from CBECI (column PctDiff_H_1day).

The figure size in inches is read from the manuscript, so that the new image has the
aspect ratio of the drawing it replaces and the drawing's extent need not change. The
size is the wp:extent of the picture inside the Figure 2 caption's text box, found through
the bookmark Ref_Figure1_number_only. Pass --size-in WIDTH HEIGHT to bypass the docx.

Usage:
    python validation/plot_cbeci_replication.py
    python validation/plot_cbeci_replication.py --in-dir data/results/cbeci_replication_v3
    python validation/plot_cbeci_replication.py --out data/results/cbeci_replication_v3/figure2_cbeci_replication_liberation.png
"""

import argparse
import os
import sys
import zipfile
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IN_DIR = "data/results/cbeci_replication_v3"
DEFAULT_DOCX = ("/home/jynurso/gitea/bitcoin-mining-research/Thesis_Papers/Paper_3/"
                "network_model_paper_3_short_version_restructured-tracked.docx")
OUT_NAME = "figure2_cbeci_replication.png"
DPI = 300
EMU_PER_INCH = 914400

# One series colour per source. Both also differ in line style and marker shape, so the
# two series stay distinguishable in grayscale.
CBECI_COLOR = "#eb6834"
IMPL_COLOR = "#2a78d6"
TEXT_COLOR = "#222222"
GRID_COLOR = "#dcdcdc"
ZERO_COLOR = "#555555"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}
BOOKMARK = "Ref_Figure1_number_only"


def figure_size_from_docx(path):
    """Return (width_in, height_in) of the Figure 2 picture, plus its media part name."""
    z = zipfile.ZipFile(path)
    doc = ET.fromstring(z.read("word/document.xml"))
    rels = ET.fromstring(z.read("word/_rels/document.xml.rels"))
    target = {r.get("Id"): r.get("Target") for r in rels}
    w = "{%s}" % NS["w"]
    parent = {c: p for p in doc.iter() for c in p}
    marks = [b for b in doc.iter(w + "bookmarkStart") if b.get(w + "name") == BOOKMARK]
    if not marks:
        sys.exit("bookmark %s not found in %s" % (BOOKMARK, path))
    # The first bookmark of that name sits in the Figure 2 caption, in the paragraph
    # that also holds the picture (mc:Choice copy).
    para = parent[marks[0]]
    inline = para.find(".//wp:inline", NS)
    if inline is None:
        sys.exit("no inline picture beside the Figure 2 caption bookmark")
    ext = inline.find("wp:extent", NS)
    blip = inline.find(".//a:blip", NS)
    rid = blip.get("{%s}embed" % NS["r"])
    return (int(ext.get("cx")) / EMU_PER_INCH, int(ext.get("cy")) / EMU_PER_INCH,
            "word/" + target[rid])


def pick_font():
    """Liberation Serif (metric-compatible with Times New Roman), else DejaVu Serif.

    matplotlib reads installed fonts from a cache that can predate the installation, so
    Liberation Serif may be on disk but missing from font_manager.fontManager. In that
    case every LiberationSerif-*.ttf found on the system is added for this run only.
    """
    def names():
        return {f.name for f in font_manager.fontManager.ttflist}
    if "Liberation Serif" not in names():
        for path in font_manager.findSystemFonts(fontext="ttf"):
            if os.path.basename(path).startswith("LiberationSerif-"):
                font_manager.fontManager.addfont(path)
    for name in ("Liberation Serif", "DejaVu Serif"):
        if name in names():
            return name
    return "serif"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in-dir", default=DEFAULT_IN_DIR,
                    help="directory holding annual_comparison.csv (relative to the repo root)")
    ap.add_argument("--out",
                    help="output PNG (relative to the repo root); default <in-dir>/" + OUT_NAME)
    ap.add_argument("--docx", default=DEFAULT_DOCX,
                    help="manuscript whose Figure 2 drawing fixes the figure size")
    ap.add_argument("--size-in", nargs=2, type=float, metavar=("W", "H"),
                    help="figure size in inches; skips reading the docx")
    args = ap.parse_args()

    in_dir = args.in_dir if os.path.isabs(args.in_dir) else os.path.join(ROOT, args.in_dir)
    if args.out:
        out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    else:
        out = os.path.join(in_dir, OUT_NAME)
    if os.path.exists(out):
        sys.exit("refusing to overwrite %s" % out)
    df = pd.read_csv(os.path.join(in_dir, "annual_comparison.csv"))
    need = {"year", "CBECI_TWh", "TWh_H_1day", "PctDiff_H_1day"}
    if not need <= set(df.columns):
        sys.exit("annual_comparison.csv lacks columns %s" % sorted(need - set(df.columns)))
    df = df.sort_values("year")
    if list(df["year"]) != list(range(2011, 2024)):
        sys.exit("expected exactly the years 2011-2023, got %s" % list(df["year"]))

    if args.size_in:
        width, height = args.size_in
        media = "(--size-in)"
    else:
        width, height, media = figure_size_from_docx(args.docx)
    font = pick_font()
    print("figure size %.4f x %.4f in (media part %s), font %s" % (width, height, media, font))

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [font],
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "axes.edgecolor": ZERO_COLOR,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
    })

    fig = plt.figure(figsize=(width, height), dpi=DPI)
    # Margins in inches, converted to figure fractions, so they hold at any figure size.
    left, right, top, bottom, gap = 0.80, 0.12, 0.26, 0.42, 0.28
    avail = height - top - bottom - gap
    h_a, h_b = avail * 0.62, avail * 0.38
    x0, w_ax = left / width, (width - left - right) / width
    ax_b = fig.add_axes([x0, bottom / height, w_ax, h_b / height])
    ax_a = fig.add_axes([x0, (bottom + h_b + gap) / height, w_ax, h_a / height], sharex=ax_b)

    years = df["year"].to_numpy()

    # ---- panel (a)
    ax_a.plot(years, df["CBECI_TWh"], color=CBECI_COLOR, linestyle="--", linewidth=1.5,
              marker="s", markersize=4.0, label="CBECI best guess", zorder=3)
    ax_a.plot(years, df["TWh_H_1day"], color=IMPL_COLOR, linestyle="-", linewidth=1.5,
              marker="o", markersize=3.6,
              label="Our implementation", zorder=4)
    ax_a.set_yscale("log")
    ax_a.set_ylim(0.07, 220)
    ax_a.set_yticks([0.1, 1, 10, 100])
    ax_a.set_yticklabels(["0.1", "1", "10", "100"])
    ax_a.minorticks_off()
    ax_a.set_ylabel("Annual electricity\nconsumption (TWh)")
    ax_a.legend(loc="upper left", frameon=False, handlelength=2.6, borderaxespad=0.2)
    ax_a.tick_params(axis="x", labelbottom=False)

    # ---- panel (b)
    pct = df["PctDiff_H_1day"].to_numpy()
    ax_b.bar(years, pct, width=0.6, color=IMPL_COLOR, zorder=3)
    ax_b.axhline(0, color=ZERO_COLOR, linewidth=0.6, zorder=4)
    ax_b.set_ylim(-5, 12)
    ax_b.set_yticks([-5, 0, 5, 10])
    ax_b.set_ylabel("Difference from\nCBECI (%)")
    peak = int(abs(pct).argmax())
    ax_b.annotate("%+.1f%%" % pct[peak], (years[peak], pct[peak]), xytext=(0, 2),
                  textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax_b.set_xticks(years)
    ax_b.set_xlim(years.min() - 0.7, years.max() + 0.7)
    ax_b.set_xlabel("Year")

    for ax, letter in ((ax_a, "(a)"), (ax_b, "(b)")):
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6)
        ax.xaxis.grid(False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.text(-left * 0.97 / (width - left - right), 1.0, letter, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9, fontweight="bold")
        ax.tick_params(length=2.5, width=0.6)
    fig.align_ylabels([ax_a, ax_b])

    fig.savefig(out, dpi=DPI)
    print("wrote", out)


if __name__ == "__main__":
    main()
