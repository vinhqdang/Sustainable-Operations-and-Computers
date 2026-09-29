"""Shared experiment harness: forecast cubes, workloads and policy factory."""
import numpy as np
import pandas as pd
from common import *
from conformal import load_cubes, upper_cube
from sim import *

IDX, TRUTH, CUBES, VOL = load_cubes()
T = len(IDX)
# clairvoyant cube: the realised intensities for every horizon
_hi = np.minimum(np.arange(T)[:, None] + np.arange(H + 1)[None, :], T - 1)
CUBES["truth"] = np.transpose(TRUTH[_hi], (0, 2, 1))
LEVELS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for lv in LEVELS:
    CUBES[f"q{int(lv * 100)}"], _ = upper_cube(IDX, TRUTH, CUBES["pred"], VOL, lv)


def week_starts(first_monday, n):
    t = int(np.searchsorted(IDX, pd.Timestamp(first_monday)))
    return [t + 168 * i for i in range(n)]


HIST = week_starts("2023-07-03", 13)   # history used to learn the demand profile
VAL = week_starts("2023-10-02", 13)    # policy tuning
TEST = week_starts("2024-01-01", 52)   # out-of-sample evaluation


def profile_for(rho, urgent=0.3):
    return demand_profile([make_jobs(t, rho, 1000 + i, urgent=urgent) for i, t in enumerate(HIST)], HIST)


def run_one(spec):
    """spec: dict(policy, cube, t0, rho, seed, migrate, kappa, look, urgent, eta, profile_rho)"""
    import sim
    sim.E_NET = spec.get("eta", 0.02)
    jobs = make_jobs(spec["t0"], spec["rho"], spec["seed"], urgent=spec.get("urgent", 0.3))
    mig = spec.get("migrate", True)
    p = spec["policy"]
    if p == "Oracle":
        r = oracle(jobs, spec["t0"], TRUTH, mig)
    else:
        if p == "ASAP-Local":
            pol = ASAPLocal()
        elif p == "Greedy-Spatial":
            pol = ASAPGreen()
        elif p == "Forecast-Reserve":
            pol = Reservation()
        elif p == "MPC":
            pol = MPC()
        elif p == "CARMA":
            prof = profile_for(spec.get("profile_rho", spec["rho"]), spec.get("urgent", 0.3))
            pol = AnticipatoryMPC(prof, spec["kappa"], spec["look"])
        else:
            raise ValueError(p)
        r = simulate(pol, jobs, spec["t0"], TRUTH, CUBES[spec["cube"]], mig)
    out = {k: v for k, v in spec.items()}
    out.update({k: float(v) for k, v in r.items() if not isinstance(v, list)})
    out["week"] = str(IDX[spec["t0"]].date())
    return out
