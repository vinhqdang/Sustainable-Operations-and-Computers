"""Stage-wise tuning of CARMA on the 2023-Q4 validation weeks."""
import sys, itertools, multiprocessing as mp
import pandas as pd
from experiment import *

if __name__ == "__main__":
    stage = sys.argv[1]
    specs = []
    for rho in [0.3, 0.5, 0.7]:
        for i, t0 in enumerate(VAL):
            base = dict(t0=t0, rho=rho, seed=500 + i, migrate=True)
            if stage == "1":
                specs.append(dict(base, policy="MPC", cube="pred"))
                specs.append(dict(base, policy="Oracle", cube="truth"))
                for k, L in itertools.product([0.5, 0.75, 1.0, 1.25], [6, 12, 18]):
                    specs.append(dict(base, policy="CARMA", cube="pred", kappa=k, look=L))
            else:
                k, L = float(sys.argv[2]), int(sys.argv[3])
                for c in [f"q{int(l*100)}" for l in LEVELS]:
                    specs.append(dict(base, policy="CARMA", cube=c, kappa=k, look=L))
                    specs.append(dict(base, policy="MPC", cube=c))
    with mp.Pool(4) as pool:
        rows = pool.map(run_one, specs, chunksize=1)
    pd.DataFrame(rows).to_csv(RES / f"tuning_stage{stage}.csv", index=False)
