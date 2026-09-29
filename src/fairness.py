"""Cooperative carbon attribution among sites (Theorem 3 and Section 4.5).

For each 2024 test week (rho = 0.5, migration allowed) the four sites form a
pooling game: coalition S pools the capacity of its sites and serves the jobs
submitted at them. We compute the hindsight value c(S) for all 15 coalitions,
the dual (Owen) attribution, and the realised emissions of CARMA run by the grand
coalition and by every coalition on its own."""
import itertools, json, multiprocessing as mp
import numpy as np
import pandas as pd
from experiment import *
import sim

CFG = json.load(open(RES / "carma_config.json"))
RHO = 0.5
COALS = [c for r in range(1, S + 1) for c in itertools.combinations(range(S), r)]


def week_jobs(i, t0):
    jobs = make_jobs(t0, RHO, 10_000 + i)
    jobs["ev"] = np.ones(jobs["n"], bool)  # the game covers every job of the trace
    return jobs


def restrict(jobs, coal):
    keep = np.isin(jobs["o"], coal)
    return {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in jobs.items()} | {"n": int(keep.sum())}


def task(args):
    kind, i, t0, coal = args
    jobs = week_jobs(i, t0)
    out = dict(kind=kind, week=i, t0=t0, coal="".join(map(str, coal)))
    if kind == "lp":
        r = hindsight_lp(jobs, t0, TRUTH, coal)
        out.update(value=r["value"], **{f"phi{o}": r["phi"][o] for o in range(S)},
                   **{f"phys{o}": r["phys"][o] for o in range(S)},
                   **{f"work{o}": float(jobs["w"][jobs["o"] == o].sum()) for o in range(S)})
    else:  # CARMA operated by coalition `coal` alone
        mask = np.isin(np.arange(S), coal).astype(float)
        base_cap = sim.CAP.copy()
        sim.CAP = base_cap * mask
        try:
            W, M = profile_for(RHO)
            prof = (W * mask[None, :, None], M * mask[None, :, None])
            pol = AnticipatoryMPC(prof, CFG["kappa"], CFG["look"])
            r = simulate(pol, restrict(jobs, list(coal)), t0, TRUTH, CUBES[CFG["cube"]], True)
        finally:
            sim.CAP = base_cap
        out.update(value=r["emis_t"], late=r["late_frac"], **{f"phys{o}": r["emis_by_origin"][o] for o in range(S)})
    return out


if __name__ == "__main__":
    specs = [(k, i, t0, c) for i, t0 in enumerate(TEST) for c in COALS for k in ("lp", "carma")]
    specs.sort(key=lambda s: (s[0] != "carma", -len(s[3])))
    with mp.Pool(4) as pool:
        rows = pool.map(task, specs, chunksize=1)
    pd.DataFrame(rows).to_csv(RES / "fairness_raw.csv", index=False)
    print(len(rows))
