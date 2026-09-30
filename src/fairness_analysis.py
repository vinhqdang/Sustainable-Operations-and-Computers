"""Coalitional stability of carbon attributions (Table 6, Section 5.8).

Games: (i) hindsight LP value c(C) (Theorem 3; includes the lateness penalty if a
coalition cannot serve its own jobs on time), (ii) hindsight emissions only, and
(iii) the realised emissions of CARMA operated by each coalition (online game).
Attributions: origin-based (each site charged its jobs' emissions wherever they ran),
host-based (each site charged the emissions of the work it executes, the analogue of
GHG Protocol location-based Scope 2), work-proportional, proportional to stand-alone
value, Shapley value of the respective game, and the dual attribution (Eq. 10;
for the online game the excess over the hindsight optimum is split by work)."""
import itertools
import numpy as np
import pandas as pd
from math import factorial
from common import *

S_ = len(SITE_IDS)
FULL = "".join(map(str, range(S_)))
d = pd.read_csv(RES / "fairness_raw.csv", dtype={"coal": str})
lp = d[d.kind == "lp"].set_index(["week", "coal"])
ca = d[d.kind == "carma"].set_index(["week", "coal"])
weeks = sorted(d.week.unique())
coals = [c for c in lp.loc[weeks[0]].index if c != FULL]
TOL = 1e-6


def shapley(v):
    phi = np.zeros(S_)
    for o in range(S_):
        others = [p for p in range(S_) if p != o]
        for r in range(S_):
            for C in itertools.combinations(others, r):
                wgt = factorial(r) * factorial(S_ - r - 1) / factorial(S_)
                key_with = "".join(map(str, sorted(C + (o,))))
                key_wo = "".join(map(str, sorted(C)))
                phi[o] += wgt * (v[key_with] - (v[key_wo] if key_wo else 0.0))
    return phi


rows, ir = [], []
for w in weeks:
    g = lp.loc[(w, FULL)]
    work = np.array([g[f"work{o}"] for o in range(S_)]); share = work / work.sum()
    v_lp = {c: lp.loc[(w, c)].value for c in coals + [FULL]}
    v_em = {c: lp.loc[(w, c)].emis for c in coals + [FULL]}
    v_on = {c: ca.loc[(w, c)].value for c in coals + [FULL]}
    single = lambda v: np.array([v[str(o)] for o in range(S_)])
    dual = np.array([g[f"phi{o}"] for o in range(S_)])
    games = {
        "hindsight": (v_lp, {"Origin-based": np.array([g[f"phys{o}"] for o in range(S_)]),
                             "Host-based": np.array([g[f"host{o}"] for o in range(S_)]),
                             "Work-proportional": v_lp[FULL] * share,
                             "Stand-alone proportional": v_lp[FULL] * single(v_lp) / single(v_lp).sum(),
                             "Shapley": shapley(v_lp), "Dual (Theorem 3)": dual}),
        "emissions-only": (v_em, {"Shapley": shapley(v_em), "Dual (Theorem 3)": dual * v_em[FULL] / v_lp[FULL]}),
        "online": (v_on, {"Origin-based": np.array([ca.loc[(w, FULL)][f"phys{o}"] for o in range(S_)]),
                          "Host-based": np.array([ca.loc[(w, FULL)][f"host{o}"] for o in range(S_)]),
                          "Work-proportional": v_on[FULL] * share,
                          "Stand-alone proportional": v_on[FULL] * single(v_on) / single(v_on).sum(),
                          "Shapley": shapley(v_on),
                          "Dual (Theorem 3)": dual + (v_on[FULL] - v_lp[FULL]) * share}),
    }
    for game, (v, attrs) in games.items():
        for name, phi in attrs.items():
            for c in coals:
                members = [int(ch) for ch in c]
                excess = phi[members].sum() - v[c]
                rows.append(dict(week=w, game=game, method=name, coal=c, size=len(members), excess=excess,
                                 viol=excess > TOL * max(1, abs(v[c]))))
            for o in range(S_):
                ir.append(dict(week=w, game=game, method=name, site=SITES[SITE_IDS[o]], attributed=phi[o],
                               standalone=v[str(o)], negative=phi[o] < -1e-9))
r = pd.DataFrame(rows); r.to_csv(RES / "fairness_checks.csv", index=False)
irr = pd.DataFrame(ir)
summ = r.groupby(["game", "method"]).agg(pair_viol=("viol", "mean"), max_excess=("excess", "max")).reset_index()
wk = r.groupby(["game", "method", "week"]).viol.any().groupby(["game", "method"]).mean().rename("week_viol").reset_index()
sing = r[r["size"] == 1].groupby(["game", "method"]).viol.mean().rename("ir_viol").reset_index()
neg = irr.groupby(["game", "method"]).negative.mean().rename("neg_share").reset_index()
summ = summ.merge(wk, on=["game", "method"]).merge(sing, on=["game", "method"]).merge(neg, on=["game", "method"])
summ.to_csv(RES / "fairness_summary.csv", index=False)
site = irr.groupby(["game", "method", "site"])[["attributed", "standalone"]].mean().reset_index()
site.to_csv(RES / "fairness_sites.csv", index=False)
pool = pd.DataFrame([dict(week=w, grand=lp.loc[(w, FULL)].value, grand_emis=lp.loc[(w, FULL)].emis,
                          sum_single=sum(lp.loc[(w, str(o))].value for o in range(S_)),
                          sum_single_emis=sum(lp.loc[(w, str(o))].emis for o in range(S_)),
                          max_single_slack=max(lp.loc[(w, c)].slack for c in coals),
                          carma_grand=ca.loc[(w, FULL)].value,
                          carma_single=sum(ca.loc[(w, str(o))].value for o in range(S_)),
                          carma_late_max=ca.loc[w].late.max()) for w in weeks])
pool.to_csv(RES / "fairness_pooling.csv", index=False)
# dual non-uniqueness
rg = lp.reset_index()
rg = rg[(rg.coal == FULL) & rg["philo0"].notna()] if "philo0" in rg else pd.DataFrame()
if len(rg):
    width = np.array([[r[f"phihi{o}"] - r[f"philo{o}"] for o in range(S_)] for _, r in rg.iterrows()])
    json_out = dict(weeks=len(rg), max_width_t=float(width.max()), mean_width_t=float(width.mean()),
                    max_rel_width=float((width / np.abs(rg[[f"phi{o}" for o in range(S_)]].to_numpy()).clip(1e-9)).max()))
    pd.Series(json_out).to_json(RES / "fairness_dual_range.json")
    print(json_out)
lines = []
for game, lab in [("hindsight", "(a) Hindsight game, penalised cost $c(C)$ (Theorem~\\ref{thm:core})"),
                  ("emissions-only", "(b) Hindsight game, emissions only (dual rescaled; not covered by Theorem~\\ref{thm:core})"),
                  ("online", "(c) Realised emissions of CARMA (online game)")]:
    ms = [m for m in ["Origin-based", "Host-based", "Work-proportional", "Stand-alone proportional", "Shapley",
                      "Dual (Theorem 3)"] if ((summ.game == game) & (summ.method == m)).any()]
    lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{lab}}}}}\\\\")
    for m in ms:
        x = summ[(summ.game == game) & (summ.method == m)].iloc[0]
        mx = f"{x.max_excess:.2f}" if x.max_excess > 1e-9 else "--"
        name = {"Dual (Theorem 3)": "Dual" if game == "hindsight" else
                ("Rescaled dual" if game == "emissions-only" else "Adjusted dual (heuristic)")}.get(m, m)
        lines.append(f"\\quad {name} & {100 * x.pair_viol:.1f} & {100 * x.week_viol:.1f} & "
                     f"{100 * x.ir_viol:.1f} & {100 * x.neg_share:.1f} & {mx}\\\\")
    lines.append("\\addlinespace")
open(RES / "table_fair.tex", "w").write("\n".join(lines[:-1]))
pd.set_option("display.width", 200)
print(summ.round(4).to_string()); print(site.round(3).to_string()); print(pool.mean().round(3))
