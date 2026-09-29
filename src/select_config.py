"""Pick CARMA hyperparameters from the validation results (lowest mean oracle gap;
look-aheads within 0.25 percentage points are resolved towards the cheaper one)."""
import json
import pandas as pd
from common import RES

d = pd.concat([pd.read_csv(RES / "tuning_stage1.csv"), pd.read_csv(RES / "tuning_stage2.csv")])
orc = d[d.policy == "Oracle"][["t0", "rho", "emis_t"]].rename(columns={"emis_t": "orc"})
d = d.merge(orc, on=["t0", "rho"]); d["gap"] = d.emis_t / d.orc - 1
s1 = d[(d.policy == "CARMA") & (d.cube == "pred")].groupby(["kappa", "look"]).gap.mean()
best = s1.min()
cand = s1[s1 <= best + 0.0025].reset_index().sort_values(["look", "kappa"])
kappa, look = float(cand.iloc[0].kappa), int(cand.iloc[0].look)
s2 = d[(d.policy == "CARMA") & (d.kappa == kappa) & (d.look == look)].groupby("cube").gap.mean()
s3 = d[d.policy == "MPC"].groupby("cube").gap.mean()
cfg = dict(kappa=kappa, look=look, cube=s2.idxmin(), best_conformal_cube=s2.drop("pred").idxmin(),
           mpc_best_cube=s3.idxmin(), val_gap_stage1=s1.round(4).reset_index().to_dict("records"),
           val_gap_cost_view=s2.round(4).to_dict(), val_gap_mpc=s3.round(4).to_dict())
json.dump(cfg, open(RES / "carma_config.json", "w"), indent=1)
print({k: cfg[k] for k in ["kappa", "look", "cube", "best_conformal_cube", "mpc_best_cube"]})
