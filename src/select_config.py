"""Select hyperparameters per regime from validation (lowest mean oracle gap; for
CARMA, look-aheads within 0.25 points are resolved towards the shorter one)."""
import sys, json
import pandas as pd
from common import RES


def gaps(stage_files):
    d = pd.concat([pd.read_csv(RES / f) for f in stage_files])
    orc = pd.read_csv(RES / "tuning_stage1.csv").query("policy == 'Oracle'")[["t0", "rho", "regime", "emis_t"]]
    d = d[d.policy != "Oracle"].merge(orc.rename(columns={"emis_t": "orc"}), on=["t0", "rho", "regime"])
    d["gap"] = 100 * (d.emis_t / d.orc - 1)
    return d


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "1":
        d = gaps(["tuning_stage1.csv"])
        out = {}
        for reg, g in d[d.policy == "CARMA"].groupby("regime"):
            m = g.groupby(["kappa", "look"]).gap.mean()
            cand = m[m <= m.min() + 0.25].reset_index().sort_values(["look", "kappa"]).iloc[0]
            out[reg] = dict(kappa=float(cand.kappa), look=int(cand.look), val_gap=float(m.min()))
        json.dump(out, open(RES / "tuning_stage1_best.json", "w"), indent=1); print(out)
    else:
        best = json.load(open(RES / "tuning_stage1_best.json"))
        d1 = gaps(["tuning_stage1.csv"]); d2 = gaps(["tuning_stage2.csv"]); dp = gaps(["tuning_stagep.csv"])
        ds = gaps(["tuning_stages.csv"])
        cfg = {}
        for reg in ["st", "t"]:
            b = best[reg]
            c1 = d1[(d1.regime == reg) & (d1.policy == "CARMA") & (d1.kappa == b["kappa"]) & (d1.look == b["look"])]
            views = pd.concat([c1, d2[(d2.regime == reg) & (d2.policy == "CARMA")]]).groupby("cube").gap.mean()
            mv = pd.concat([d1[(d1.regime == reg) & (d1.policy == "MPC")],
                            d2[(d2.regime == reg) & (d2.policy == "MPC")]]).groupby("cube").gap.mean()
            pr = dp[dp.regime == reg].groupby(["r", "tight"]).gap.mean()
            rb = pr.idxmin()
            cfg[reg] = dict(kappa=b["kappa"], look=b["look"], cube=views.idxmin(),
                            best_conformal_cube=views.drop("pred").idxmin(), mpc_best_cube=mv.drop("pred").idxmin(),
                            protect_r=float(rb[0]), protect_tight=int(rb[1]),
                            val_gap_cost_view=views.round(3).to_dict(), val_gap_mpc=mv.round(3).to_dict(),
                            val_gap_protect={f"{k[0]}|{k[1]}": v for k, v in pr.round(3).items()})
            sc = ds[ds.regime == reg].groupby(["K", "look"]).gap.mean()
            if len(sc):
                kb = sc.idxmin()
                cfg[reg].update(scen_K=int(kb[0]), scen_look=int(kb[1]),
                                val_gap_scen={f"{int(k[0])}|{int(k[1])}": v for k, v in sc.round(3).items()})
        json.dump(cfg, open(RES / "carma_config.json", "w"), indent=1)
        print({r: {k: v for k, v in c.items() if not k.startswith("val")} for r, c in cfg.items()})
