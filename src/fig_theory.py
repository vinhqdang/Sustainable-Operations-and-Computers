"""Figure 7 (competitive-ratio landscape) and Figure 9 (carbon attribution by site)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 300})


def rstar(theta):
    if theta <= 1.0001:
        return 1.0
    p = np.linspace(1, theta, 20001)
    return (1 + (p - 1) * (theta - p) / (p ** 2 - p - 1 + theta)).max()


th = np.logspace(0, 3, 200)
fig, ax = plt.subplots(figsize=(3.6, 2.8))
ax.loglog(th, th, color="#d95f02", lw=1.6, label="CARMA, robustness bound $\\theta$ (Thm 2)")
ax.loglog(th, (1 + th) / 2, color="#1f78b4", lw=1.4, ls="--", label="Myopic MPC, worst case $\\geq(1+\\theta)/2$ (Thm 1)")
ax.loglog(th, [rstar(t) for t in th], color="0.2", lw=1.6, label="Any online algorithm, $R^*(\\theta)$ (Thm 1)")
ax.loglog(th, np.ones_like(th), color="#d95f02", lw=1.2, ls=":", label="CARMA, exact predictions (Thm 2)")
ax.fill_between(th, [rstar(t) for t in th], (1 + th) / 2, color="0.9", zorder=0)
ax.set_xlabel("Carbon-intensity ratio $\\theta = U/L$"); ax.set_ylabel("Competitive ratio")
ax.legend(frameon=False, fontsize=6.3, loc="upper left")
fig.savefig(FIG / "fig7_ratios.pdf"); plt.close(fig)

site = pd.read_csv(RES / "fairness_sites.csv")
order = ["South Scotland", "North West England", "West Midlands", "London"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
for ax, game, title in zip(axs, ["hindsight", "online"], ["(a) Hindsight optimum", "(b) CARMA (realised)"]):
    g = site[site.game == game]
    x = np.arange(4); wdt = 0.2
    for k, (m, c) in enumerate(zip(["Location-based", "Work-proportional", "Dual (Theorem 3)"], ["0.65", "#a6cee3", "#d95f02"])):
        v = g[g.method == m].set_index("site").loc[order].attributed
        ax.bar(x + (k - 1) * wdt, v.values, wdt, color=c, label=m)
    sa = g[g.method == "Dual (Theorem 3)"].set_index("site").loc[order].standalone
    ax.scatter(x, sa.values, marker="_", s=500, color="k", lw=1.6, label="Stand-alone value", zorder=3)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(["South\nScotland", "North West\nEngland", "West\nMidlands", "London"], fontsize=7.5)
    ax.set_title(title, fontsize=9, loc="left")
axs[0].set_ylabel("Mean weekly attribution (tCO$_2$)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.16))
fig.savefig(FIG / "fig9_fairness.pdf"); plt.close(fig)
print("ok")

r = pd.read_csv(RES / "robustness.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
for pol, c, mk in [("MPC", "#1f78b4", "s"), ("CARMA", "#d95f02", "o")]:
    g = r[(r.kind == "sigma") & (r.policy == pol)].groupby("val").gap
    axs[0].errorbar(g.mean().index, g.mean().values, yerr=1.96 * g.std().values / np.sqrt(g.count().values),
                    fmt=mk + "-", color=c, ms=4, capsize=2, lw=1.4, label=pol)
axs[0].axvline(31.2, color="0.6", lw=0.8, ls=":")
axs[0].text(35, 11, "MAE of the\nGBM forecaster", fontsize=7, color="0.4")
axs[0].set_xlabel("Forecast noise $\\sigma$ (gCO$_2$/kWh)"); axs[0].set_ylabel("Gap to oracle (%)")
axs[0].legend(frameon=False, fontsize=7.5); axs[0].set_title("(a) Carbon-forecast error", fontsize=9, loc="left")
g = r[r.kind == "scale"].groupby("val").gap
axs[1].errorbar(g.mean().index, g.mean().values, yerr=1.96 * g.std().values / np.sqrt(g.count().values),
                fmt="o-", color="#d95f02", ms=4, capsize=2, lw=1.4, label="CARMA, profile scaled by $g$")
sh = r[r.kind == "shift"].gap
axs[1].errorbar([1.0], [sh.mean()], yerr=[1.96 * sh.std() / np.sqrt(len(sh))], fmt="D", color="#6a3d9a", ms=5,
                capsize=2, label="CARMA, profile shifted 12 h")
axs[1].axhline(g.mean().loc[0.0], color="#1f78b4", ls="--", lw=1, label="No anticipation ($g=0$, = MPC)")
axs[1].set_xlabel("Demand-profile scale $g$ (1 = unbiased)"); axs[1].set_ylabel("Gap to oracle (%)")
axs[1].legend(frameon=False, fontsize=6.8, loc="upper center"); axs[1].set_title("(b) Demand-prediction error", fontsize=9, loc="left")
axs[1].set_ylim(0, 38)
fig.savefig(FIG / "fig8_robust.pdf"); plt.close(fig)
print("robust fig ok")
