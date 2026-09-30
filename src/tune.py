"""Tuning on the 2023-Q4 validation weeks, separately for each regime.

stage 1: CARMA anticipation weight kappa x look-ahead (point forecasts), plus MPC and oracle
stage 2: carbon cost view (point vs conformal level beta) for MPC and for CARMA at the
         stage-1 winner of the same regime
stage p: MPC-Protect protection share r x urgency horizon
stage s: Scenario-MPC number of scenarios K x look-ahead (spatio-temporal regime)"""
import sys, json, itertools, multiprocessing as mp
import pandas as pd
from experiment import *

REGIMES = {"st": True, "t": False}

if __name__ == "__main__":
    stage = sys.argv[1]
    specs = []
    for reg, mig in REGIMES.items():
        for rho in [0.3, 0.5, 0.7]:
            for i, t0 in enumerate(VAL):
                base = dict(t0=t0, rho=rho, seed=500 + i, migrate=mig, regime=reg)
                if stage == "1":
                    specs.append(dict(base, policy="MPC", cube="pred"))
                    specs.append(dict(base, policy="Oracle", cube="truth"))
                    for k, L in itertools.product([0.5, 0.75, 1.0, 1.25], [6, 12, 18]):
                        specs.append(dict(base, policy="CARMA", cube="pred", kappa=k, look=L))
                elif stage == "2":
                    best = json.load(open(RES / "tuning_stage1_best.json"))[reg]
                    for c in [f"q{int(l * 100)}" for l in LEVELS]:
                        specs.append(dict(base, policy="CARMA", cube=c, kappa=best["kappa"], look=best["look"]))
                        specs.append(dict(base, policy="MPC", cube=c))
                elif stage == "s":
                    if reg == "st":
                        for K, L in itertools.product([4, 8], [6, 12]):
                            specs.append(dict(base, policy="Scenario-MPC", cube="pred", K=K, look=L))
                elif stage == "p":
                    for r, tt in itertools.product([0.1, 0.2, 0.3, 0.5], [1, 2, 4]):
                        specs.append(dict(base, policy="MPC-Protect", cube="pred", r=r, tight=tt))
    specs.sort(key=lambda d: d["policy"] != "CARMA")
    with mp.Pool(4) as pool:
        rows = pool.map(run_one, specs, chunksize=1)
    pd.DataFrame(rows).to_csv(RES / f"tuning_stage{stage}.csv", index=False)
    print(len(rows))
