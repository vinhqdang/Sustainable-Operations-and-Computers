"""Core-stability of carbon attributions (Table 6, Section 5.6)."""
import numpy as np
import pandas as pd
from common import *

S_ = len(SITE_IDS)
FULL = "".join(map(str, range(S_)))
d = pd.read_csv(RES / "fairness_raw.csv", dtype={"coal": str})
lp = d[d.kind == "lp"].set_index(["week", "coal"])
ca = d[d.kind == "carma"].set_index(["week", "coal"])
weeks = sorted(d.week.unique())
coals = [c for c in lp.loc[weeks[0]].index if c != FULL]
TOL = 1e-6
rows, ir = [], []
for w in weeks:
    g = lp.loc[(w, FULL)]
    work = np.array([g[f"work{o}"] for o in range(S_)])
    share = work / work.sum()
    cN = g.value
    EN = ca.loc[(w, FULL)].value
    hind = {"Dual (Theorem 3)": np.array([g[f"phi{o}"] for o in range(S_)]),
            "Location-based": np.array([g[f"phys{o}"] for o in range(S_)]),
            "Work-proportional": cN * share}
    onl = {"Dual (Theorem 3)": hind["Dual (Theorem 3)"] + (EN - cN) * share,
           "Location-based": np.array([ca.loc[(w, FULL)][f"phys{o}"] for o in range(S_)]),
           "Work-proportional": EN * share}
    for game, attrs, value in [("hindsight", hind, lambda c: lp.loc[(w, c)].value),
                               ("online", onl, lambda c: ca.loc[(w, c)].value)]:
        for name, phi in attrs.items():
            for c in coals:
                members = [int(ch) for ch in c]
                excess = phi[members].sum() - value(c)
                rows.append(dict(week=w, game=game, method=name, coal=c, size=len(members),
                                 excess=excess, viol=excess > TOL * max(1, abs(value(c)))))
            for o in range(S_):
                ir.append(dict(week=w, game=game, method=name, site=SITES[SITE_IDS[o]],
                               attributed=phi[o], standalone=value(str(o))))
r = pd.DataFrame(rows); r.to_csv(RES / "fairness_checks.csv", index=False)
irr = pd.DataFrame(ir)
summ = r.groupby(["game", "method"]).agg(pair_viol=("viol", "mean"), max_excess=("excess", "max")).reset_index()
wk = r.groupby(["game", "method", "week"]).viol.any().groupby(["game", "method"]).mean().rename("week_viol").reset_index()
summ = summ.merge(wk, on=["game", "method"])
sing = r[r["size"] == 1].groupby(["game", "method"]).viol.mean().rename("ir_viol").reset_index()
summ = summ.merge(sing, on=["game", "method"])
summ.to_csv(RES / "fairness_summary.csv", index=False)
site = irr.groupby(["game", "method", "site"])[["attributed", "standalone"]].mean().reset_index()
site.to_csv(RES / "fairness_sites.csv", index=False)
pool = pd.DataFrame([dict(week=w, grand=lp.loc[(w, FULL)].value,
                          sum_single=sum(lp.loc[(w, str(o))].value for o in range(S_)),
                          carma_grand=ca.loc[(w, FULL)].value,
                          carma_single=sum(ca.loc[(w, str(o))].value for o in range(S_)),
                          carma_late_max=ca.loc[w].late.max()) for w in weeks])
pool.to_csv(RES / "fairness_pooling.csv", index=False)
pd.set_option("display.width", 200)
print(summ.round(4).to_string()); print(site.round(3).to_string()); print(pool.mean().round(3))

lines = []
for game, lab in [("hindsight", "Hindsight optimum"), ("online", "CARMA (realised)")]:
    for i, m in enumerate(["Location-based", "Work-proportional", "Dual (Theorem 3)"]):
        x = summ[(summ.game == game) & (summ.method == m)].iloc[0]
        mx = f"{x.max_excess:.2f}" if x.max_excess > 0 else "--"
        lines.append(f"{lab if i == 0 else ''} & {m} & {100 * x.pair_viol:.1f} & {100 * x.week_viol:.1f} & {100 * x.ir_viol:.1f} & {mx}\\\\")
    lines.append("\\addlinespace")
open(RES / "table_fair.tex", "w").write("\n".join(lines[:-1]))
