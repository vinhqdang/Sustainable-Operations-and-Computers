"""Tables, statistics and figures for the scheduling experiments."""
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 300})
ORDER = ["ASAP-Local", "Greedy-Spatial", "Forecast-Reserve", "MPC", "MPC+conformal", "MPC-Perfect",
         "CARMA", "CARMA+conformal", "CARMA-Perfect", "Oracle"]
PAL = {"ASAP-Local": "0.75", "Greedy-Spatial": "0.55", "Forecast-Reserve": "#a6cee3",
       "MPC": "#1f78b4", "MPC+conformal": "#6a3d9a", "MPC-Perfect": "#b2df8a",
       "CARMA": "#d95f02", "CARMA+conformal": "#fdbf6f", "CARMA-Perfect": "#e31a1c", "Oracle": "0.1"}


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj


def add_rel(d):
    key = ["experiment", "migrate", "rho", "t0"]
    base = d[d.label == "ASAP-Local"][key + ["emis_t"]].rename(columns={"emis_t": "base"})
    orc = d[d.label == "Oracle"][key + ["emis_t"]].rename(columns={"emis_t": "orc"})
    d = d.merge(base, on=key).merge(orc, on=key)
    d["red"] = 100 * (1 - d.emis_t / d.base)
    d["gap"] = 100 * (d.emis_t / d.orc - 1)
    return d


def main_table(d):
    rows = []
    for (mig, rho), g in d.groupby(["migrate", "rho"], sort=False):
        carma = g[g.label == "CARMA"].set_index("t0")
        others = [l for l in ORDER if l in set(g.label) and l not in ("CARMA", "Oracle")]
        ps = []
        for l in others:
            x = g[g.label == l].set_index("t0").loc[carma.index]
            ps.append(wilcoxon(carma.emis_t, x.emis_t).pvalue)
        padj = dict(zip(others, holm(ps)))
        for l in ORDER:
            x = g[g.label == l]
            if len(x) == 0:
                continue
            rows.append(dict(regime="Spatio-temporal" if mig else "Temporal-only", rho=rho, policy=l,
                             emis=x.emis_t.mean(), red=x.red.mean(), red_sd=x.red.std(), gap=x.gap.mean(),
                             late=100 * x.late_frac.mean(), mig=100 * x.mig_frac.mean(),
                             ms=x.ms_per_decision.mean(), p_vs_carma=padj.get(l, np.nan),
                             carma_better_weeks=int((carma.emis_t.values < g[g.label == l].set_index("t0").loc[carma.index].emis_t.values - 1e-9).sum()) if l in padj else np.nan))
    return pd.DataFrame(rows)


def fmt_p(p):
    if np.isnan(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def latex_main(tab):
    for regime, fn in [("Spatio-temporal", "table_main_st.tex"), ("Temporal-only", "table_main_t.tex")]:
        lines = []
        for rho in [0.3, 0.5, 0.7]:
            g = tab[(tab.regime == regime) & (tab.rho == rho)]
            if regime == "Temporal-only":
                g = g[g.policy != "Greedy-Spatial"]  # identical to ASAP-Local without migration
            best = g[~g.policy.isin(["Oracle", "MPC-Perfect", "CARMA-Perfect"])].gap.min()
            for i, r in enumerate(g.itertuples()):
                lab = f"$\\rho={rho}$" if i == 0 else ""
                gap = f"{r.gap:.1f}"
                if abs(r.gap - best) < 1e-9:
                    gap = r"\textbf{" + gap + "}"
                mig = f" & {r.mig:.1f}" if regime == "Spatio-temporal" else ""
                lines.append(f"{lab} & {r.policy} & {r.emis:.2f} & {r.red:.1f} & {gap} & {r.late:.2f}{mig} & {fmt_p(r.p_vs_carma)}\\\\")
            lines.append(r"\addlinespace")
        open(RES / fn, "w").write("\n".join(lines[:-1]))


def decomposition(tab):
    out = []
    for (regime, rho), g in tab.groupby(["regime", "rho"], sort=False):
        v = g.set_index("policy").gap
        out.append(dict(regime=regime, rho=rho, mpc=v["MPC"], forecast_value=v["MPC"] - v["MPC-Perfect"],
                        anticipation_value=v["MPC"] - v["CARMA"], carma=v["CARMA"],
                        carma_perfect=v["CARMA-Perfect"]))
    return pd.DataFrame(out)


def fig_gap(d):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=False)
    pols = ["Greedy-Spatial", "Forecast-Reserve", "MPC", "MPC+conformal", "MPC-Perfect", "CARMA",
            "CARMA+conformal", "CARMA-Perfect"]
    for ax, mig in zip(axs, [True, False]):
        g = d[d.migrate == mig]
        w = 0.1
        for k, p in enumerate(pols):
            m = g[g.label == p].groupby("rho").gap
            mu, se = m.mean(), m.std() / np.sqrt(m.count())
            x = np.arange(3) + (k - (len(pols) - 1) / 2) * w
            ax.bar(x, mu.values, w, yerr=1.96 * se.values, color=PAL[p], label=p,
                   error_kw=dict(lw=0.6, capsize=1.2), edgecolor="none")
        ax.set_xticks(range(3)); ax.set_xticklabels([f"$\\rho$ = {r}" for r in [0.3, 0.5, 0.7]])
        ax.set_ylabel("Gap to clairvoyant oracle (%)")
        ax.set_title("(a) Spatio-temporal (migration allowed)" if mig else "(b) Temporal-only (no migration)",
                     fontsize=9, loc="left")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.13))
    fig.savefig(FIG / "fig4_gap.pdf"); plt.close(fig)


def fig_weekly(d):
    g = d[(d.migrate == True) & (d.rho == 0.5)]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.7, 1]})
    for p in ["Greedy-Spatial", "MPC", "CARMA", "Oracle"]:
        x = g[g.label == p].sort_values("t0")
        ax[0].plot(pd.to_datetime(x.week), x.red, color=PAL[p], lw=1.3 if p != "Oracle" else 1.0,
                   ls="--" if p == "Oracle" else "-", label=p)
    ax[0].set_ylabel("Reduction vs ASAP-Local (%)")
    ax[0].legend(frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.33))
    import matplotlib.dates as mdates
    ax[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax[0].set_title("(a) Weekly emission reduction, 2024 ($\\rho$ = 0.5)", fontsize=9, loc="left")
    a = g[g.label == "MPC"].set_index("t0").gap; b = g[g.label == "CARMA"].set_index("t0").gap.loc[a.index]
    ax[1].scatter(a, b, s=10, color=PAL["CARMA"])
    lim = [0, max(a.max(), b.max()) * 1.05]
    ax[1].plot(lim, lim, color="0.6", lw=0.8)
    ax[1].set_xlabel("MPC gap to oracle (%)"); ax[1].set_ylabel("CARMA gap to oracle (%)")
    ax[1].set_title("(b) Paired weekly gaps", fontsize=9, loc="left")
    fig.savefig(FIG / "fig5_weekly.pdf"); plt.close(fig)


def fig_tuning():
    d = pd.concat([pd.read_csv(RES / "tuning_stage1.csv"), pd.read_csv(RES / "tuning_stage2.csv")])
    orc = d[d.policy == "Oracle"][["t0", "rho", "emis_t"]].rename(columns={"emis_t": "orc"})
    d = d.merge(orc, on=["t0", "rho"]); d["gap"] = 100 * (d.emis_t / d.orc - 1)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    s1 = d[(d.policy == "CARMA") & (d.cube == "pred")].groupby(["kappa", "look"]).gap.mean().unstack()
    mpc = d[(d.policy == "MPC") & (d.cube == "pred")].gap.mean()
    for L, c in zip(s1.columns, ["#fdae6b", "#e6550d", "#a63603"]):
        ax[0].plot(s1.index, s1[L], "o-", color=c, ms=3.5, lw=1.3, label=f"look-ahead $\\ell$ = {int(L)} h")
    ax[0].axhline(mpc, color=PAL["MPC"], ls="--", lw=1, label="MPC ($\\kappa$ = 0)")
    ax[0].set_xlabel("Anticipation weight $\\kappa$"); ax[0].set_ylabel("Validation gap to oracle (%)")
    ax[0].set_ylim(5, 22)
    ax[0].legend(frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.62)); ax[0].set_title("(a) Demand anticipation", fontsize=9, loc="left")
    cfg = json.load(open(RES / "carma_config.json"))
    for pol, c, sub in [("MPC", PAL["MPC"], d.policy == "MPC"),
                        ("CARMA", PAL["CARMA"], (d.policy == "CARMA") & (d.kappa == cfg["kappa"]) & (d.look == cfg["look"]))]:
        g = d[sub]
        lv = g[g.cube.str.startswith("q")].groupby("cube").gap.mean()
        xs = [int(k[1:]) / 100 for k in lv.index]
        ax[1].plot(xs, lv.values, "o-", color=c, ms=3.5, lw=1.3, label=f"{pol}, conformal $U^\\beta$")
        ax[1].axhline(g[g.cube == "pred"].gap.mean(), color=c, ls=":", lw=1.2, label=f"{pol}, point forecast")
    ax[1].set_xlabel("Conformal level $\\beta$"); ax[1].set_ylabel("Validation gap to oracle (%)")
    ax[1].set_ylim(5, None)
    ax[1].legend(frameon=False, fontsize=6.5); ax[1].set_title("(b) Carbon cost view", fontsize=9, loc="left")
    fig.savefig(FIG / "fig6_tuning.pdf"); plt.close(fig)


def sens_table(s):
    rows = []
    for e, g in s.groupby("experiment", sort=False):
        c = g[g.label == "CARMA"].set_index("t0")
        for l in ["ASAP-Local", "Greedy-Spatial", "Forecast-Reserve", "MPC", "CARMA", "Oracle"]:
            x = g[g.label == l]
            p = wilcoxon(c.emis_t, x.set_index("t0").loc[c.index].emis_t).pvalue if l not in ("CARMA", "Oracle") else np.nan
            rows.append(dict(setting=e, policy=l, red=x.red.mean(), gap=x.gap.mean(),
                             late=100 * x.late_frac.mean(), mig=100 * x.mig_frac.mean(), p=p))
    return pd.DataFrame(rows)


def latex_sens(st, s):
    names = {"urgent=0.1": "Urgent share $\\upsilon=0.1$", "urgent=0.5": "Urgent share $\\upsilon=0.5$",
             "eta=0.005": "Network energy $\\eta=0.005$ kWh/GB", "eta=0.06": "Network energy $\\eta=0.06$ kWh/GB",
             "load=0.4, profile=0.5": "Load $\\rho=0.4$, profile learned at 0.5",
             "load=0.6, profile=0.5": "Load $\\rho=0.6$, profile learned at 0.5"}
    order = ["urgent=0.1", "urgent=0.5", "eta=0.005", "eta=0.06", "load=0.4, profile=0.5", "load=0.6, profile=0.5"]
    lines = []
    for e in order:
        g = st[st.setting == e].set_index("policy")
        x = s[s.experiment == e]
        c = x[x.label == "CARMA"].set_index("t0").emis_t
        m = x[x.label == "MPC"].set_index("t0").emis_t.loc[c.index]
        wins = int((c < m - 1e-9).sum())
        lines.append(f"{names[e]} & {g.red['Oracle']:.1f} & {g.gap['Greedy-Spatial']:.1f} & {g.gap['Forecast-Reserve']:.1f} & "
                     f"{g.gap['MPC']:.1f} & \\textbf{{{g.gap['CARMA']:.1f}}} & {wins}/{len(c)} & {g.late['MPC']:.2f} / {g.late['CARMA']:.2f}\\\\")
    open(RES / "table_sens.tex", "w").write("\n".join(lines))
    print("max p (CARMA vs others):", st.p.max())


if __name__ == "__main__":
    d = add_rel(pd.read_csv(RES / "test_main.csv"))
    tab = main_table(d)
    tab.to_csv(RES / "summary_main.csv", index=False)
    latex_main(tab)
    dec = decomposition(tab); dec.to_csv(RES / "decomposition.csv", index=False)
    pd.set_option("display.width", 200)
    print(tab.round(3).to_string())
    print(dec.round(2).to_string())
    fig_gap(d); fig_weekly(d); fig_tuning()
    try:
        s = add_rel(pd.read_csv(RES / "test_sens.csv"))
        st = sens_table(s); st.to_csv(RES / "summary_sens.csv", index=False)
        latex_sens(st, s)
        print(st.round(3).to_string())
    except FileNotFoundError:
        pass
