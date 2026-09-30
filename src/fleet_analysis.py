"""Per-fleet statistics (FLEET=xx python fleet_analysis.py) and cross-fleet summary
(python fleet_analysis.py all). Same estimands as analysis.py: gap to the oracle,
paired weekly differences with CARMA, moving-block bootstrap."""
import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
import analysis as A
from fleets import FLEETS

ORDER = ["ASAP-Local", "Greedy-Spatial", "Forecast-Reserve", "MPC", "MPC-Protect", "MPC+conformal", "MPC-Perfect",
         "CARMA", "CARMA-Perfect", "Oracle"]


def one():
    d = A.add_rel(pd.read_csv(RES / "fleet_test.csv"))
    main = d[(d.experiment == "main") & (d.label != "Scenario-MPC")]
    tab = A.summarise(main, ["rho"], ORDER)
    tab.to_csv(RES / "fleet_summary_main.csv", index=False)
    rev = d[d.experiment == "reversed"]
    rtab = A.summarise(rev, ["experiment"], ORDER)
    rtab.to_csv(RES / "fleet_summary_reversed.csv", index=False)
    # Scenario-MPC against CARMA on the common weeks (one trace, rho = 0.5)
    sc = d[(d.experiment == "main") & (d.rho == 0.5) & (d.rep == 0)]
    a = sc[sc.label == "Scenario-MPC"].set_index("t0").gap
    b = sc[sc.label == "CARMA"].set_index("t0").gap.loc[a.index]
    diff = (a - b).sort_index()
    lo, hi = A.block_boot(diff.values, block=2)
    out = dict(weeks=len(diff), scen_gap=float(a.mean()), carma_gap_same_weeks=float(b.mean()),
               scen_minus_carma=float(diff.mean()), lo=float(lo), hi=float(hi),
               scen_better_weeks=int((diff < -1e-9).sum()),
               scen_ms=float(sc[sc.label == "Scenario-MPC"].ms_per_decision.mean()),
               carma_ms=float(sc[sc.label == "CARMA"].ms_per_decision.mean()))
    json.dump(out, open(RES / "fleet_scenario.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(tab[["rho", "policy", "emis", "red", "gap", "late", "ms", "dgap", "dgap_lo", "dgap_hi", "p", "wins"]].round(3).to_string())
    print(rtab[["policy", "red", "gap", "dgap", "dgap_lo", "dgap_hi", "p", "wins"]].round(3).to_string())
    print(out)


def allf():
    rows, scen = [], []
    order = ["GB", "EU", "US", "BR", "GLOBAL"]
    for f in order:
        base = ROOT / "results" if f == "GB" else ROOT / "results" / "fleets" / f.lower()
        if f == "GB":
            t = pd.read_csv(base / "summary_main.csv"); t = t[t.migrate == True].copy()
        else:
            t = pd.read_csv(base / "fleet_summary_main.csv")
        t["fleet"] = f
        rows.append(t)
        if f != "GB":
            scen.append(dict(fleet=f, **json.load(open(base / "fleet_scenario.json"))))
        else:
            d = A.add_rel(pd.read_csv(ROOT / "results" / "test_main.csv"))
            sc = d[(d.migrate == True) & (d.rho == 0.5) & (d.rep == 0)]
            a = sc[sc.label == "Scenario-MPC"].set_index("t0").gap.sort_index(); b = sc[sc.label == "CARMA"].set_index("t0").gap.loc[a.index]
            keep = a.index[::2]; diff = (a - b).loc[keep].sort_index(); lo, hi = A.block_boot(diff.values, block=2)
            scen.append(dict(fleet="GB", weeks=len(diff), scen_gap=float(a.loc[keep].mean()), carma_gap_same_weeks=float(b.loc[keep].mean()),
                             scen_minus_carma=float(diff.mean()), lo=float(lo), hi=float(hi), scen_better_weeks=int((diff < -1e-9).sum()),
                             scen_ms=float(sc[sc.label == "Scenario-MPC"].ms_per_decision.mean()), carma_ms=float(sc[sc.label == "CARMA"].ms_per_decision.mean())))
    T = pd.concat(rows); T.to_csv(ROOT / "results" / "multigrid_summary.csv", index=False)
    S = pd.DataFrame(scen); S.to_csv(ROOT / "results" / "multigrid_scenario.csv", index=False)
    lines = []
    for f in order:
        g = T[(T.fleet == f) & (T.rho == 0.5)].set_index("policy")
        base_pols = ["Greedy-Spatial", "Forecast-Reserve", "MPC", "MPC-Protect", "MPC+conformal"]
        bestb = g.loc[base_pols, "gap"].idxmin()
        sc = S[S.fleet == f].iloc[0]
        nm = "Great Britain" if f == "GB" else f
        wins = int(g.wins["MPC"]); wk = int(g.weeks["MPC"])
        lines.append(f"{nm} & {g.red['Oracle']:.1f} & {g.gap['MPC']:.1f} & {g.gap['MPC-Protect']:.1f} & {g.gap['MPC+conformal']:.1f} & "
                     f"\\textbf{{{g.gap['CARMA']:.1f}}} & {sc.scen_gap:.1f} / {sc.carma_gap_same_weeks:.1f} & "
                     f"{g.gap[bestb] - g.gap['CARMA']:.1f} & {wins}/{wk} & {g.gap['CARMA-Perfect']:.1f}\\\\")
    open(ROOT / "results" / "table_multigrid.tex", "w").write("\n".join(lines))
    # all loads
    lines = []
    for f in order:
        for rho in [0.3, 0.5, 0.7]:
            g = T[(T.fleet == f) & (T.rho == rho)].set_index("policy")
            nm = ("Great Britain" if f == "GB" else f) if rho == 0.3 else ""
            bestb = g.loc[["Greedy-Spatial", "Forecast-Reserve", "MPC", "MPC-Protect", "MPC+conformal"], "gap"].idxmin()
            lines.append(f"{nm} & {rho} & {g.red['Oracle']:.1f} & {g.gap['MPC']:.1f} & {g.gap['MPC-Protect']:.1f} & {g.gap['MPC+conformal']:.1f} & "
                         f"\\textbf{{{g.gap['CARMA']:.1f}}} & {bestb} & {g.dgap[bestb]:.1f} ({g.dgap_lo[bestb]:.1f}--{g.dgap_hi[bestb]:.1f}) & {g.late['CARMA']:.2f}\\\\")
        lines.append(r"\addlinespace")
    open(ROOT / "results" / "table_multigrid_loads.tex", "w").write("\n".join(lines[:-1]))
    # figure
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight", "savefig.dpi": 300})
    pols = ["MPC", "MPC-Protect", "MPC+conformal", "CARMA", "CARMA-Perfect"]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    w = 0.16
    for k, p in enumerate(pols):
        mu, err = [], []
        for f in order:
            g = T[(T.fleet == f) & (T.rho == 0.5) & (T.policy == p)].iloc[0]
            mu.append(g.gap)
        ax.bar(np.arange(len(order)) + (k - 2) * w, mu, w, color=A.PAL[p], label=p, edgecolor="none")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(["Great\nBritain", "Europe", "United\nStates", "Brazil", "Four\ncontinents"])
    ax.set_ylabel("Gap to clairvoyant oracle (%)")
    ax.legend(frameon=False, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.savefig(FIG / "fig10_multigrid.pdf"); plt.close(fig)
    pd.set_option("display.width", 250)
    print(T[T.rho == 0.5].pivot_table(index="fleet", columns="policy", values="gap").round(1).to_string()); print(S.round(2).to_string())


if __name__ == "__main__":
    allf() if len(sys.argv) > 1 and sys.argv[1] == "all" else one()
