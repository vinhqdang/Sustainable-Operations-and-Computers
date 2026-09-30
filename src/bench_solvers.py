"""Wall-clock comparison of the OR-Tools min-cost-flow solver (oracle(), two flow solves) and a general LP solver
(HiGHS via SciPy) on the weekly clairvoyant problems (rho = 0.5, migration)."""
import time, json
import numpy as np
from experiment import *

rows = []
for i, t0 in enumerate(TEST[::13]):
    for rho in [0.3, 0.5, 0.7]:
        jobs = make_jobs(t0, rho, 10_000 + i)
        jobs["ev"] = np.ones(jobs["n"], bool)
        s = time.perf_counter(); o = oracle(jobs, t0, TRUTH); tf = time.perf_counter() - s
        s = time.perf_counter(); l = hindsight_lp(jobs, t0, TRUTH); tl = time.perf_counter() - s
        rows.append(dict(week=i, rho=rho, jobs=int(jobs["n"]), flow_s=tf, lp_s=tl,
                         rel_diff=abs(o["emis_t"] - l["value"]) / l["value"]))
        print(rows[-1], flush=True)
json.dump(rows, open(RES / "bench_solvers.json", "w"), indent=1)
