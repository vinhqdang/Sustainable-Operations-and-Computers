"""Figure 7 (competitive-ratio landscape) and Figure 9 (carbon attribution by site)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 300,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def rstar(theta):
    if theta <= 1.0001:
        return 1.0
    p = np.linspace(1, theta, 20001)
    return (1 + (p - 1) * (theta - p) / (p ** 2 - p - 1 + theta)).max()


th = np.logspace(0, 3, 200)
fig, ax = plt.subplots(figsize=(3.6, 3.3))
ax.loglog(th, th, color="#D55E00", lw=1.6, label="Every on-time policy under (A1), CARMA included: $\\theta$ (Prop. 3)")
ax.loglog(th, (1 + th) / 2, color="#0072B2", lw=1.4, ls="--", label="Myopic MPC, worst case $\\geq(1+\\theta)/2$ (Thm 1)")
ax.loglog(th, [rstar(t) for t in th], color="0.2", lw=1.6, label="Any online algorithm without demand\ninformation, $R^*(\\theta)$ (Thm 1)")
ax.loglog(th, np.ones_like(th), color="#D55E00", lw=1.2, ls=":",
          label="CARMA with exact demand and carbon\npredictions, under (A1)-(A2) (Thm 2)")
ax.fill_between(th, [rstar(t) for t in th], (1 + th) / 2, color="0.9", zorder=0)
ax.set_xlabel("Unit-emission-cost ratio $\\theta = U/L$"); ax.set_ylabel("Competitive ratio")
ax.legend(frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
fig.savefig(FIG / "fig7_ratios.pdf"); plt.close(fig)

site = pd.read_csv(RES / "fairness_sites.csv")
order = ["South Scotland", "North West England", "West Midlands", "London"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
for ax, game, title in zip(axs, ["hindsight", "online"], ["(a) Hindsight optimum", "(b) CARMA (realized)"]):
    g = site[site.game == game]
    x = np.arange(4); wdt = 0.18
    meths = ["Origin-based", "Host-based", "Shapley", "Dual (Theorem 3)"]
    for k, (m, c, h) in enumerate(zip(meths, ["0.6", "#56B4E9", "#CC79A7", "#D55E00"], ["", "////", "xxxx", ""])):
        v = g[g.method == m].set_index("site").loc[order].attributed
        lab = "Dual (Theorem 3); online: adjusted dual" if m.startswith("Dual") else m
        ax.bar(x + (k - 1.5) * wdt, v.values, wdt, color=c, label=lab, hatch=h, edgecolor="white", linewidth=0.3)
    sa = g[g.method == "Dual (Theorem 3)"].set_index("site").loc[order].standalone
    ax.scatter(x, sa.values, marker="_", s=500, color="k", lw=1.6, label="Stand-alone value", zorder=3)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(["South\nScotland", "North West\nEngland", "West\nMidlands", "London"], fontsize=7.5)
    ax.set_title(title, fontsize=9, loc="left")
axs[0].set_ylabel("Mean attribution per episode (tCO$_2$)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.16))
fig.savefig(FIG / "fig9_fairness.pdf"); plt.close(fig)
print("ok")

r = pd.read_csv(RES / "robustness.csv")
mae = json.load(open(RES / "robustness_mae.json"))


def ci(v, B=5000, seed=0):
    """Percentile bootstrap CI of the mean over the (4-week-spaced) weeks."""
    v = np.asarray(v, float); rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def eb(ax, g, **kw):
    mu = g.mean(); lo, hi = [], []
    for k in mu.index:
        c = ci(g.get_group(k).values); lo.append(mu[k] - c[0]); hi.append(c[1] - mu[k])
    ax.errorbar(mu.index, mu.values, yerr=[lo, hi], **kw)
    return mu


fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
for pol, c, mk in [("MPC", "#0072B2", "s"), ("CARMA", "#D55E00", "o")]:
    for kind, ls, lab in [("sigma", "-", "i.i.d. noise"), ("corr", "--", "day-level correlated")]:
        g = r[(r.kind == kind) & (r.policy == pol)]
        if kind == "corr":
            g = pd.concat([r[(r.kind == "sigma") & (r.policy == pol) & (r.val == 0)], g])
        eb(axs[0], g.groupby("val").gap, fmt=mk + ls, color=c, ms=4, capsize=2, lw=1.4, label=f"{pol}, {lab}",
           mfc="white" if kind == "corr" else c)
# the GBM forecaster placed at the i.i.d. noise level with the same realized MAE
sig = np.array([0, 25, 50, 100, 200]); m_iid = np.r_[0, [mae[f"iid_{s_}"] for s_ in sig[1:]]]
s_eq = float(np.interp(mae["GBM"], m_iid, sig))
axs[0].axvline(s_eq, color="0.6", lw=0.8, ls=":")
axs[0].text(s_eq + 4, 1.0, f"GBM forecaster (MAE {mae['GBM']:.1f})", fontsize=6.8, color="0.4")
axs[0].set_xlabel("Forecast noise $\\sigma$ (gCO$_2$/kWh)"); axs[0].set_ylabel("Gap to oracle (%)")
axs[0].set_ylim(bottom=0); axs[0].set_title("(a) Carbon-forecast error", fontsize=9, loc="left")
axs[0].legend(frameon=False, fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
mu = eb(axs[1], r[r.kind == "scale"].groupby("val").gap, fmt="o-", color="#D55E00", ms=4, capsize=2, lw=1.4,
        label="CARMA, profile scaled by $g$")
sh = r[r.kind == "shift"].gap; c_ = ci(sh.values)
axs[1].errorbar([1.0], [sh.mean()], yerr=[[sh.mean() - c_[0]], [c_[1] - sh.mean()]], fmt="D", color="#CC79A7", ms=5,
                capsize=2, label="CARMA, profile shifted 12 h")
axs[1].axhline(mu.loc[0.0], color="#0072B2", ls="--", lw=1, label="No anticipation ($g=0$, = MPC)")
axs[1].set_xlabel("Demand-profile scale $g$ (1 = unbiased)"); axs[1].set_ylabel("Gap to oracle (%)")
axs[1].set_ylim(bottom=0); axs[1].set_title("(b) Demand-prediction error", fontsize=9, loc="left")
axs[1].legend(frameon=False, fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
fig.savefig(FIG / "fig8_robust.pdf"); plt.close(fig)
json.dump({"gbm_equivalent_sigma": s_eq}, open(RES / "robustness_gbm_sigma.json", "w"))
print("robust fig ok")
