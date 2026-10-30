"""One consistent, colour-blind-checked look for every figure in the report."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots in fixed order (validated palette: blue, orange, aqua, yellow)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
UP, DOWN = "#e34948", "#2a78d6"   # pushes risk up / down


def apply():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
        "ytick.color": INK_2, "text.color": INK, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False,
    })
