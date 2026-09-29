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
LEVELS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
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


def marginal_factor():
    """Stylised marginal emission factor (gCO2/kWh) for GB (Section 5.7): gas CCGT is
    the marginal plant (394 g/kWh) except in hours with evident surplus low-carbon
    generation. A Scottish site is treated as marginally zero-carbon when Scotland
    is nearly carbon-free while the rest of GB is not (a proxy for export-constrained,
    curtailment-prone hours); every site is marginally zero-carbon when GB-wide
    intensity is below 50 g/kWh."""
    wide = load_hourly()["ci"]
    gb, scot = wide[18].to_numpy(), wide[16].to_numpy()
    mef = np.full((len(wide), S), 394.0)
    constrained = (scot < 25) & (gb > 100)
    mef[constrained, 0] = 0.0          # site 0 = South Scotland
    mef[gb < 50, :] = 0.0
    return mef


MEF = marginal_factor()


def run_one(spec):
    """spec keys: policy, cube, t0, rho, seed, migrate, kappa, look, urgent, eta,
    profile_rho, workload ('synthetic'|'trace'), r, tight, cap (list), eit, idle, acct."""
    import sim
    sim.E_NET = spec.get("eta", 0.02)
    sim.E_IT = spec.get("eit", 0.40)
    base_cap = np.array([300.0, 400.0, 400.0, 600.0])
    sim.CAP = np.array(spec["cap"], float) if spec.get("cap") is not None else base_cap
    urgent = spec.get("urgent", 0.3)
    if spec.get("workload", "synthetic") == "trace":
        from trace import make_trace_jobs, trace_profile
        jobs = make_trace_jobs(spec["t0"], spec["rho"], spec["seed"], urgent)
        prof = trace_profile(spec["rho"], spec["seed"], urgent,
                             days=tuple(spec.get("profile_days", (0, 4))))
    else:
        jobs = make_jobs(spec["t0"], spec["rho"], spec["seed"], urgent=urgent)
        prof = None
    mig = spec.get("migrate", True)
    acct = MEF if spec.get("acct") == "marginal" else None
    p = spec["policy"]
    if p == "Oracle":
        r = oracle(jobs, spec["t0"], TRUTH, mig, acct=acct)
    else:
        if p == "ASAP-Local":
            pol = ASAPLocal()
        elif p == "Greedy-Spatial":
            pol = ASAPGreen()
        elif p == "Forecast-Reserve":
            pol = Reservation()
        elif p == "MPC":
            pol = MPC()
        elif p == "MPC-Protect":
            pol = MPCProtect(spec["r"], spec["tight"])
        elif p == "CARMA":
            if prof is None:
                prof = profile_for(spec.get("profile_rho", spec["rho"]), urgent)
            pol = AnticipatoryMPC(prof, spec["kappa"], spec["look"])
        else:
            raise ValueError(p)
        r = simulate(pol, jobs, spec["t0"], TRUTH, CUBES[spec["cube"]], mig, acct=acct)
    if spec.get("idle"):
        # always-on servers: idle energy of every batch server, identical for all policies
        tt = np.arange(spec["t0"], spec["t0"] + 168)
        I = (TRUTH if acct is None else acct)[tt]
        r["idle_t"] = float((spec.get("idle_kwh", 0.2) * PUE[None, :] * sim.CAP[None, :] * I).sum()) / 1e6
    out = {k: v for k, v in spec.items() if not isinstance(v, (list, tuple))}
    out.update({k: float(v) for k, v in r.items() if not isinstance(v, list)})
    out["week"] = str(IDX[spec["t0"]].date())
    return out
