"""Graceful degradation (incl. persistent, day-level correlated errors) of CARMA under controlled prediction errors (Section 4.6).

(a) Carbon-forecast error: the forecast view is the realized intensity plus i.i.d.
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


def noisy_cube(sigma, corr=False):
    """i.i.d. noise per (issue time, site, horizon), or, with corr=True, a persistent
    error shared by every forecast of the same target day and site (a systematically
    wrong day, e.g. a missed wind front)."""
    key = (sigma, corr)
    if key not in _NOISE:
        rng = np.random.default_rng(int(sigma) + 7 + 1000 * corr)
        c = CUBES["truth"].copy()
        if sigma > 0:
            T_, S_, H1 = c.shape
            if corr:
                Z = rng.normal(0, sigma, (T_ // 24 + 3, S_))
                tgt = (np.arange(T_)[:, None] + np.arange(1, H1)[None, :]) // 24   # [T, H]
                noise = np.transpose(Z[tgt], (0, 2, 1))                             # [T, S, H]
            else:
                noise = rng.normal(0, sigma, c[:, :, 1:].shape)
            c[:, :, 1:] = np.maximum(c[:, :, 1:] + noise, 0)
        _NOISE[key] = c
    return _NOISE[key]


def task(spec):
    kind, val, pol_name, i, t0 = spec
    jobs = make_jobs(t0, 0.5, 30_000 + i)
    W, M = profile_for(0.5)
    cube = CUBES[CFG["st"]["cube"]]
    if kind == "sigma":
        cube = noisy_cube(val)
    elif kind == "corr":
        cube = noisy_cube(val, corr=True)
    elif kind == "scale":
        W, M = W * val, M * val
    elif kind == "shift":
        W, M = np.roll(W, int(val), axis=0), np.roll(M, int(val), axis=0)
    if pol_name == "MPC":
        r = simulate(MPC(), jobs, t0, TRUTH, cube, True)
    else:
        r = simulate(AnticipatoryMPC((W, M), CFG["st"]["kappa"], CFG["st"]["look"]), jobs, t0, TRUTH, cube, True)
    o = oracle(jobs, t0, TRUTH, True)
    return dict(kind=kind, val=val, policy=pol_name, week=i, emis=r["emis_t"], oracle=o["emis_t"],
                late=r["late_frac"], gap=100 * (r["emis_t"] / o["emis_t"] - 1))


if __name__ == "__main__":
    specs = []
    for i, t0 in enumerate(WEEKS):
        for s_ in [0, 25, 50, 100, 200]:
            specs += [("sigma", s_, "CARMA", i, t0), ("sigma", s_, "MPC", i, t0)]
            if s_ > 0:
                specs += [("corr", s_, "CARMA", i, t0), ("corr", s_, "MPC", i, t0)]
        for g in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
            specs.append(("scale", g, "CARMA", i, t0))
        specs.append(("shift", 12, "CARMA", i, t0))
    specs.sort(key=lambda s: s[2] != "CARMA")
    with mp.Pool(4) as pool:
        rows = pool.map(task, specs, chunksize=1)
    d = pd.DataFrame(rows); d.to_csv(RES / "robustness.csv", index=False)
    # realized MAE of the injected noise (after clipping at zero) and of the GBM
    # forecaster over the same issue times, so that the two can be put on one scale
    ts = np.concatenate([np.arange(t0, t0 + 168) for t0 in WEEKS])
    tr = CUBES["truth"][ts, :, 1:]
    mae = {"GBM": float(np.nanmean(np.abs(CUBES["pred"][ts, :, 1:] - tr)))}
    for s_ in [25, 50, 100, 200]:
        for corr in (False, True):
            mae[f"{'corr' if corr else 'iid'}_{s_}"] = float(np.nanmean(np.abs(noisy_cube(s_, corr)[ts, :, 1:] - tr)))
    json.dump(mae, open(RES / "robustness_mae.json", "w"), indent=1)
    print(d.groupby(["kind", "val", "policy"])[["gap", "late"]].mean().round(3))
