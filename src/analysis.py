"""Tables, statistics and figures for the scheduling experiments (revision 1).

Statistics: each (week, load, regime) cell has 3 independent workload traces; tests
use week-level means (52 paired weeks). The estimand is the mean paired difference of
the gap to the oracle between a policy and CARMA. Confidence intervals and two-sided
p-values come from a moving-block bootstrap over weeks (block length 4; 5000 draws
for intervals, 100000 for p-values, so that Holm-adjusted p-values can fall below 0.001),
which respects the serial correlation of consecutive weeks; the p-value is the share
of bootstrap means of the centred series at least as far from zero as the observed
mean. p-values are Holm-adjusted jointly over all comparisons of the main experiment
(both regimes and all loads), and separately for the trace and sensitivity sets.
Block lengths 2 and 8 are reported as a sensitivity check."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from common import *

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 300})
ORDER = ["ASAP-Local", "Greedy-Spatial", "Forecast-Reserve", "MPC", "MPC-Protect", "MPC+conformal",
         "MPC-Perfect", "Scenario-MPC", "CARMA", "CARMA+conformal", "CARMA-Perfect", "Oracle"]
PAL = {"ASAP-Local": "0.75", "Greedy-Spatial": "0.55", "Forecast-Reserve": "#a6cee3", "MPC": "#1f78b4",
       "MPC-Protect": "#33a02c", "MPC+conformal": "#6a3d9a", "MPC-Perfect": "#b2df8a", "CARMA": "#d95f02",
       "CARMA+conformal": "#fdbf6f", "CARMA-Perfect": "#e31a1c", "Oracle": "0.1",
       "CARMA (in-sample profile)": "#fb9a99", "Scenario-MPC": "#8c510a"}
KEY = ["experiment", "migrate", "rho", "t0", "rep"]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj


def _boot_idx(n, B, block, seed):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block)); starts = rng.integers(0, n - block + 1, (B, nb))
    return (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]


def block_boot(x, B=5000, block=4, seed=0):
    """95% CI of the mean of a weekly series by moving-block bootstrap."""
    x = np.asarray(x, float)
    m = x[_boot_idx(len(x), B, block, seed)].mean(1)
    return np.percentile(m, [2.5, 97.5])


def block_boot_p(x, B=100000, block=4, seed=1):
    """Two-sided moving-block bootstrap p-value for H0: mean = 0."""
    x = np.asarray(x, float)
    if np.allclose(x, 0):
        return 1.0
    xc = x - x.mean()
    m = xc[_boot_idx(len(x), B, block, seed)].mean(1)
    return float((1 + (np.abs(m) >= abs(x.mean())).sum()) / (B + 1))


def add_rel(d):
    base = d[d.label == "ASAP-Local"][KEY + ["emis_t"]].rename(columns={"emis_t": "base"})
    orc = d[d.label == "Oracle"][KEY + ["emis_t", "emis_adj_t"]].rename(columns={"emis_t": "orc", "emis_adj_t": "orc_adj"})
    d = d.merge(base, on=KEY).merge(orc, on=KEY)
    d["red"] = 100 * (1 - d.emis_t / d.base)
    d["gap"] = 100 * (d.emis_t / d.orc - 1)
    d["gap_adj"] = 100 * (d.emis_adj_t / d.orc_adj - 1)
    return d


def weekly(g, label, col="emis_t"):
    return g[g.label == label].groupby("t0")[col].mean()


def summarise(d, group_cols, order):
    rows, tests = [], []
    for key, g in d.groupby(group_cols, sort=False):
        carma = weekly(g, "CARMA")
        for l in order:
            x = g[g.label == l]
            if len(x) == 0:
                continue
            r = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
            r.update(policy=l, emis=x.emis_t.mean(), red=x.red.mean(), gap=x.gap.mean(), gap_adj=x.gap_adj.mean(),
                     late=100 * x.late_frac.mean(), mig=100 * x.mig_frac.mean(), ms=x.ms_per_decision.mean(),
                     n=len(x))
            if l not in ("CARMA", "Oracle"):
                wl = weekly(g, l).loc[carma.index]
                gd = (g[g.label == l].groupby("t0").gap.mean() - g[g.label == "CARMA"].groupby("t0").gap.mean()).loc[carma.index]
                r["p_raw"] = block_boot_p(gd.values)
                r["wins"] = int((carma.values < wl.values - 1e-9).sum()); r["weeks"] = len(carma)
                lo, hi = block_boot(gd.values)
                r.update(dgap=gd.mean(), dgap_lo=lo, dgap_hi=hi)
                for bl in (2, 8):
                    lo_, hi_ = block_boot(gd.values, block=bl)
                    r.update({f"dgap_lo_b{bl}": lo_, f"dgap_hi_b{bl}": hi_, f"p_raw_b{bl}": block_boot_p(gd.values, block=bl)})
            rows.append(r)
    t = pd.DataFrame(rows)
    m = t.p_raw.notna()
    t.loc[m, "p"] = holm(t.loc[m, "p_raw"].values)
    for bl in (2, 8):
        if f"p_raw_b{bl}" in t:
            t.loc[m, f"p_b{bl}"] = holm(t.loc[m, f"p_raw_b{bl}"].values)
    return t


def fmt_p(p):
    if pd.isna(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def latex_main(tab):
    for regime, fn in [(True, "table_main_st.tex"), (False, "table_main_t.tex")]:
        lines = []
        for rho in [0.3, 0.5, 0.7]:
            g = tab[(tab.migrate == regime) & (tab.rho == rho)]
            if not regime:
                g = g[g.policy != "Greedy-Spatial"]
            best = g[~g.policy.isin(["Oracle", "MPC-Perfect", "CARMA-Perfect"])].gap.min()
            for i, r in enumerate(g.itertuples()):
                lab = f"$\\rho={rho}$" if i == 0 else ""
                gap = f"{r.gap:.1f}"
                if abs(r.gap - best) < 1e-9:
                    gap = r"\textbf{" + gap + "}"
                mig = f" & {r.mig:.1f}" if regime else ""
                ms = "--" if pd.isna(r.ms) else (f"{r.ms:.0f}" if r.ms >= 10 else f"{r.ms:.1f}")
                lines.append(f"{lab} & {r.policy} & {r.emis:.2f} & {r.red:.1f} & {gap} & {r.gap_adj:.1f} & {r.late:.2f}{mig} & {ms} & {fmt_p(r.p)}\\\\")
            lines.append(r"\addlinespace")
        open(RES / fn, "w").write("\n".join(lines[:-1]))


def decomposition(tab):
    out = []
    for (mig, rho), g in tab.groupby(["migrate", "rho"], sort=False):
        v = g.set_index("policy").gap
        out.append(dict(migrate=mig, rho=rho, mpc=v["MPC"], protect=v["MPC-Protect"], carma=v["CARMA"],
                        forecast_value_myopic=v["MPC"] - v["MPC-Perfect"],
                        anticipation_value=v["MPC"] - v["CARMA"],
                        forecast_value_anticip=v["CARMA"] - v["CARMA-Perfect"], carma_perfect=v["CARMA-Perfect"]))
    return pd.DataFrame(out)


def fig_gap(d):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    pols = ["Greedy-Spatial", "Forecast-Reserve", "MPC", "MPC-Protect", "MPC+conformal", "MPC-Perfect",
            "Scenario-MPC", "CARMA", "CARMA+conformal", "CARMA-Perfect"]
    for ax, mig in zip(axs, [True, False]):
        g = d[d.migrate == mig]
        pl = [p for p in pols if (g.label == p).any() and (mig or p != "Greedy-Spatial")]
        w = 0.8 / len(pl)
        for k, p in enumerate(pl):
            wk = g[g.label == p].groupby(["rho", "t0"]).gap.mean()
            mu, lo, hi = [], [], []
            for rho in [0.3, 0.5, 0.7]:
                v = wk.loc[rho].sort_index().values
                c = block_boot(v); mu.append(v.mean()); lo.append(v.mean() - c[0]); hi.append(c[1] - v.mean())
            x = np.arange(3) + (k - (len(pl) - 1) / 2) * w
            ax.bar(x, mu, w, yerr=[lo, hi], color=PAL[p], label=p,
                   error_kw=dict(lw=0.6, capsize=1.2), edgecolor="none")
        ax.set_xticks(range(3)); ax.set_xticklabels([f"$\\rho$ = {r}" for r in [0.3, 0.5, 0.7]])
        ax.set_ylabel("Gap to clairvoyant oracle (%)")
        ax.set_title("(a) Spatio-temporal (migration allowed)" if mig else "(b) Temporal-only (no migration)",
                     fontsize=9, loc="left")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(FIG / "fig4_gap.pdf"); plt.close(fig)


def fig_weekly(d):
    g = d[(d.migrate == True) & (d.rho == 0.5)]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.7, 1]})
    for p in ["Greedy-Spatial", "MPC", "MPC-Protect", "CARMA", "Oracle"]:
        x = g[g.label == p].groupby("week").red.mean()
        ax[0].plot(pd.to_datetime(x.index), x.values, color=PAL[p],
                   lw=1.3 if p != "Oracle" else 1.0, ls="--" if p == "Oracle" else "-", label=p)
    ax[0].set_ylabel("Reduction vs ASAP-Local (%)")
    ax[0].legend(frameon=False, fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.33))
    ax[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax[0].set_title("(a) Weekly emission reduction, 2024 ($\\rho$ = 0.5)", fontsize=9, loc="left")
    a = g[g.label == "MPC"].groupby("t0").gap.mean(); b = g[g.label == "CARMA"].groupby("t0").gap.mean().loc[a.index]
    ax[1].scatter(a, b, s=10, color=PAL["CARMA"])
    lim = [0, max(a.max(), b.max()) * 1.05]
    ax[1].plot(lim, lim, color="0.6", lw=0.8)
    ax[1].set_xlabel("MPC gap to oracle (%)"); ax[1].set_ylabel("CARMA gap to oracle (%)")
    ax[1].set_title("(b) Paired weekly gaps (mean of 3 traces)", fontsize=9, loc="left")
    fig.savefig(FIG / "fig5_weekly.pdf"); plt.close(fig)


def fig_tuning():
    d = pd.concat([pd.read_csv(RES / "tuning_stage1.csv"), pd.read_csv(RES / "tuning_stage2.csv")])
    d = d[d.regime == "st"]
    orc = d[d.policy == "Oracle"][["t0", "rho", "emis_t"]].rename(columns={"emis_t": "orc"})
    d = d.merge(orc, on=["t0", "rho"]); d["gap"] = 100 * (d.emis_t / d.orc - 1)
    cfg = json.load(open(RES / "carma_config.json"))["st"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    s1 = d[(d.policy == "CARMA") & (d.cube == "pred")].groupby(["kappa", "look"]).gap.mean().unstack()
    mpc = d[(d.policy == "MPC") & (d.cube == "pred")].gap.mean()
    for L, c in zip(s1.columns, ["#fdae6b", "#e6550d", "#a63603"]):
        ax[0].plot(s1.index, s1[L], "o-", color=c, ms=3.5, lw=1.3, label=f"look-ahead $\\ell$ = {int(L)} h")
    ax[0].axhline(mpc, color=PAL["MPC"], ls="--", lw=1, label="MPC ($\\kappa$ = 0)")
    ax[0].set_xlabel("Anticipation weight $\\kappa$"); ax[0].set_ylabel("Validation gap to oracle (%)")
    ax[0].legend(frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.62))
    ax[0].set_title("(a) Demand anticipation", fontsize=9, loc="left")
    for pol, c, sub in [("MPC", PAL["MPC"], d.policy == "MPC"),
                        ("CARMA", PAL["CARMA"], (d.policy == "CARMA") & (d.kappa == cfg["kappa"]) & (d.look == cfg["look"]))]:
        g = d[sub]
        lv = g[g.cube.str.startswith("q")].groupby("cube").gap.mean()
        xs = [int(k[1:]) / 100 for k in lv.index]
        o = np.argsort(xs)
        ax[1].plot(np.array(xs)[o], lv.values[o], "o-", color=c, ms=3.5, lw=1.3, label=f"{pol}, conformal $U^\\beta$")
        ax[1].axhline(g[g.cube == "pred"].gap.mean(), color=c, ls=":", lw=1.2, label=f"{pol}, point forecast")
    ax[1].set_xlabel("Conformal level $\\beta$"); ax[1].set_ylabel("Validation gap to oracle (%)")
    ax[1].legend(frameon=False, fontsize=6.5); ax[1].set_title("(b) Carbon cost view", fontsize=9, loc="left")
    fig.savefig(FIG / "fig6_tuning.pdf"); plt.close(fig)


def latex_sens(st):
    order = ["urgent=0.1", "urgent=0.5", "eta=0.005", "eta=0.06", "load=0.4, profile=0.5", "load=0.6, profile=0.5",
             "cap=150/400/400/600", "cap=600/400/400/600", "cap=300/400/400/1200", "always-on (idle 0.2 kWh)",
             "marginal accounting"]
    names = {"urgent=0.1": "Urgent share $\\upsilon=0.1$", "urgent=0.5": "Urgent share $\\upsilon=0.5$",
             "eta=0.005": "Network energy $\\eta=0.005$ kWh/GB", "eta=0.06": "Network energy $\\eta=0.06$ kWh/GB",
             "load=0.4, profile=0.5": "Load $\\rho=0.4$, profile learned at 0.5",
             "load=0.6, profile=0.5": "Load $\\rho=0.6$, profile learned at 0.5",
             "cap=150/400/400/600": "Clean-site capacity halved (150)",
             "cap=600/400/400/600": "Clean-site capacity doubled (600)",
             "cap=300/400/400/1200": "London capacity doubled (1200)",
             "always-on (idle 0.2 kWh)": "Always-on servers (idle 0.2 kWh)$^{a}$",
             "marginal accounting": "Marginal-emission accounting$^{b}$"}
    lines = []
    for e in order:
        g = st[st.experiment == e].set_index("policy")
        if len(g) == 0:
            continue
        lines.append(f"{names[e]} & {g.red['Oracle']:.1f} & {g.gap['Greedy-Spatial']:.1f} & {g.gap['Forecast-Reserve']:.1f} & "
                     f"{g.gap['MPC']:.1f} & {g.gap['MPC-Protect']:.1f} & \\textbf{{{g.gap['CARMA']:.1f}}} & "
                     f"{int(g.wins['MPC'])}/{int(g.weeks['MPC'])} & {g.late['MPC']:.2f} / {g.late['CARMA']:.2f}\\\\")
    open(RES / "table_sens.tex", "w").write("\n".join(lines))


def latex_trace(tt):
    lines = []
    for r in tt.itertuples():
        ms = "--" if pd.isna(r.ms) else (f"{r.ms:.0f}" if r.ms >= 10 else f"{r.ms:.1f}")
        lines.append(f"{r.policy} & {r.emis:.2f} & {r.red:.1f} & {r.gap:.1f} & {r.late:.2f} & {r.mig:.1f} & {ms} & {fmt_p(r.p)}\\\\")
    open(RES / "table_trace.tex", "w").write("\n".join(lines))


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    d = add_rel(pd.read_csv(RES / "test_main.csv"))
    tab = summarise(d, ["migrate", "rho"], ORDER)
    tab.to_csv(RES / "summary_main.csv", index=False)
    latex_main(tab)
    dec = decomposition(tab); dec.to_csv(RES / "decomposition.csv", index=False)
    print(tab.drop(columns=["p_raw"]).round(3).to_string()); print(dec.round(2).to_string())
    fig_gap(d); fig_weekly(d); fig_tuning()
    for which, fn in [("sens", latex_sens), ("trace", latex_trace)]:
        try:
            s = add_rel(pd.read_csv(RES / f"test_{which}.csv"))
        except FileNotFoundError:
            continue
        grp = ["experiment"]
        order = ORDER + ["CARMA (in-sample profile)"]
        t = summarise(s, grp, order)
        t.to_csv(RES / f"summary_{which}.csv", index=False)
        fn(t)
        print(t.drop(columns=["p_raw"]).round(3).to_string())
