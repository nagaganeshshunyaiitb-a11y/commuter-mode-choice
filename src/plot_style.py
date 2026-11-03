"""Shared matplotlib style so that every figure in the report looks the same.

Colours: one fixed hue per mode (never re-assigned between charts), a single
blue ramp for magnitudes and a blue-grey-red diverging map for signed values.
The six-colour order was checked for colour-blind separation.
"""
import logging

import matplotlib as mpl

# Files only, no windows. The default Tk backend on Windows also throws
# "main thread is not in main loop" when joblib worker threads collect figures.
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap

from . import config as C

# DejaVu (the fallback font on machines without Segoe UI) has no semibold
# face; matplotlib substitutes bold and warns about it on every title.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODE_COLORS = dict(zip(C.MODES, SERIES[:6]))
GROUP_COLORS = {"Sustainable": SERIES[0], "Intermediate (auto)": SERIES[1], "Private motorised": SERIES[2]}

BLUES = LinearSegmentedColormap.from_list(
    "proj_blues", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("proj_div", ["#2a78d6", "#f0efec", "#e34948"])


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
        "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.labelsize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "text.color": INK, "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "savefig.dpi": 200, "savefig.bbox": "tight", "figure.dpi": 100,
    })


def plain_log(ax, axis="x"):
    """Log scale with tick labels written as 0.1, 1, 10 rather than powers of ten."""
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    target = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    target.set_major_formatter(fmt)


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    return path


def mode_legend(ax, modes=None, ncol=6, below_inches=0.62):
    """Legend centred under the x-axis label, a fixed physical distance below the axes."""
    modes = modes or C.MODES
    handles = [mpl.patches.Patch(color=MODE_COLORS[m], label=C.MODE_LABELS[m]) for m in modes]
    ax_h = ax.get_position().height * ax.figure.get_figheight()
    ax.legend(handles=handles, ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, -below_inches / ax_h),
              handlelength=1.0, columnspacing=1.2)
