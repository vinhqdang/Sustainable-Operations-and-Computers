"""Graphical abstract (13 x 5 cm, 300 dpi) from results/multigrid_summary.csv."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from common import ROOT

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "font.size": 7.5, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
T = pd.read_csv(ROOT / "results" / "multigrid_summary.csv")
T = T[T.rho == 0.5]
order = ["GB", "EU", "US", "BR", "GLOBAL"]
names = ["Great\nBritain", "Europe", "United\nStates", "Brazil", "Four\ncontinents"]
base = ["Greedy-Spatial", "Forecast-Reserve", "MPC", "MPC-Protect", "MPC+conformal"]
mpc = [T[(T.fleet == f) & (T.policy == "MPC")].gap.iloc[0] for f in order]
best = [T[(T.fleet == f) & T.policy.isin(base)].gap.min() for f in order]
car = [T[(T.fleet == f) & (T.policy == "CARMA")].gap.iloc[0] for f in order]

fig = plt.figure(figsize=(13 / 2.54, 5 / 2.54), dpi=300)
ax0 = fig.add_axes([0.0, 0.0, 0.40, 1.0]); ax0.axis("off"); ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
def box(x, y, w, h, txt, fc, ec="0.2", dashed=False, fs=7):
    ax0.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=0.8,
                                 ls="--" if dashed else "-"))
    ax0.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)
ax0.text(0.02, 0.93, "Anticipate demand,\nnot only carbon", fontsize=9.5, fontweight="bold", va="top")
box(0.03, 0.47, 0.42, 0.17, "Real jobs", "#FFFFFF")
box(0.03, 0.26, 0.42, 0.17, "Ghost jobs\n(expected arrivals)", "#FFFFFF", dashed=True)
box(0.60, 0.26, 0.37, 0.38, "Min-cost\nflow plan\nevery hour", "#F6D5BE", ec="#D55E00")
for y in (0.555, 0.345):
    ax0.add_patch(FancyArrowPatch((0.46, y), (0.59, 0.45), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="0.2"))
ax0.text(0.03, 0.15, "Reserves scarce clean capacity\nfor work that has not yet arrived", fontsize=6.8, va="top")

ax = fig.add_axes([0.52, 0.22, 0.47, 0.50])
x = np.arange(5); w = 0.27
ax.bar(x - w, mpc, w, color="#0072B2", label="Myopic MPC")
ax.bar(x, best, w, color="#009E73", hatch="////", edgecolor="white", linewidth=0.3, label="Best non-anticipating baseline")
ax.bar(x + w, car, w, color="#D55E00", label="CARMA")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.3)
ax.set_ylabel("Gap to oracle (%)", fontsize=7); ax.tick_params(axis="y", labelsize=6.5)
fig.text(0.50, 0.93, "Smaller gap to the clairvoyant optimum on five grids", fontsize=7.6, fontweight="bold", va="center")
ax.legend(frameon=False, fontsize=6.2, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, columnspacing=1.0, handlelength=1.2, borderaxespad=0.1)
out = ROOT / "submission"
out.mkdir(exist_ok=True)
fig.savefig(out / "Graphical_abstract.pdf"); fig.savefig(out / "Graphical_abstract.png"); fig.savefig(out / "Graphical_abstract.tiff")
from PIL import Image
print(Image.open(out / "Graphical_abstract.png").size)
