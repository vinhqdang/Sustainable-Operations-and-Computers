"""Graceful degradation of CARMA under controlled prediction errors (Section 4.6).

(a) Carbon-forecast error: the forecast view is the realised intensity plus i.i.d.
    Gaussian noise of standard deviation sigma (clipped at zero), for every issue
    time and horizon h >= 1.
(b) Demand-prediction error: the learned demand profile is scaled by g (g = 1 is the
    unbiased profile; g = 0 removes anticipation) or shifted by 12 hours, which
    predicts arrivals at the wrong time of day.
Week subset: every fourth test week of 2024; rho = 0.5; migration allowed."""
import json, multiprocessing as mp
import numpy as np
import pandas as pd
from experiment import *

CFG = json.load(open(RES / "carma_config.json"))
WEEKS = TEST[::4]
_NOISE = {}


def noisy_cube(sigma):
    if sigma not in _NOISE:
        rng = np.random.default_rng(int(sigma) + 7)
        c = CUBES["truth"].copy()
        c[:, :, 1:] = np.maximum(c[:, :, 1:] + rng.normal(0, sigma, c[:, :, 1:].shape), 0) if sigma > 0 else c[:, :, 1:]
        _NOISE[sigma] = c
    return _NOISE[sigma]


def task(spec):
    kind, val, pol_name, i, t0 = spec
    jobs = make_jobs(t0, 0.5, 30_000 + i)
    W, M = profile_for(0.5)
    cube = CUBES[CFG["cube"]]
    if kind == "sigma":
        cube = noisy_cube(val)
    elif kind == "scale":
        W, M = W * val, M * val
    elif kind == "shift":
        W, M = np.roll(W, int(val), axis=0), np.roll(M, int(val), axis=0)
    if pol_name == "MPC":
        r = simulate(MPC(), jobs, t0, TRUTH, cube, True)
    else:
        r = simulate(AnticipatoryMPC((W, M), CFG["kappa"], CFG["look"]), jobs, t0, TRUTH, cube, True)
    o = oracle(jobs, t0, TRUTH, True)
    return dict(kind=kind, val=val, policy=pol_name, week=i, emis=r["emis_t"], oracle=o["emis_t"],
                late=r["late_frac"], gap=100 * (r["emis_t"] / o["emis_t"] - 1))


if __name__ == "__main__":
    specs = []
    for i, t0 in enumerate(WEEKS):
        for s_ in [0, 25, 50, 100, 200]:
            specs += [("sigma", s_, "CARMA", i, t0), ("sigma", s_, "MPC", i, t0)]
        for g in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
            specs.append(("scale", g, "CARMA", i, t0))
        specs.append(("shift", 12, "CARMA", i, t0))
    specs.sort(key=lambda s: s[2] != "CARMA")
    with mp.Pool(4) as pool:
        rows = pool.map(task, specs, chunksize=1)
    d = pd.DataFrame(rows); d.to_csv(RES / "robustness.csv", index=False)
    print(d.groupby(["kind", "val", "policy"])[["gap", "late"]].mean().round(3))
