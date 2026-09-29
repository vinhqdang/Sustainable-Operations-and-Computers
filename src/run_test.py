"""Out-of-sample evaluation on the 52 weeks of 2024 (main results and sensitivity)."""
import sys, json, multiprocessing as mp
import pandas as pd
from experiment import *

CFG = json.load(open(RES / "carma_config.json"))
KAPPA, LOOK, CUBE = CFG["kappa"], CFG["look"], CFG["cube"]
QC, QM = CFG["best_conformal_cube"], CFG["mpc_best_cube"]


def main_specs():
    specs = []
    for mig in [True, False]:
        for rho in [0.3, 0.5, 0.7]:
            for i, t0 in enumerate(TEST):
                b = dict(t0=t0, rho=rho, seed=10_000 + i, migrate=mig, experiment="main")
                specs += [dict(b, policy="ASAP-Local", cube="pred"),
                          dict(b, policy="Greedy-Spatial", cube="pred"),
                          dict(b, policy="Forecast-Reserve", cube="pred"),
                          dict(b, policy="MPC", cube="pred", label="MPC"),
                          dict(b, policy="MPC", cube="truth", label="MPC-Perfect"),
                          dict(b, policy="MPC", cube=QM, label="MPC+conformal"),
                          dict(b, policy="CARMA", cube=QC, kappa=KAPPA, look=LOOK, label="CARMA+conformal"),
                          dict(b, policy="CARMA", cube=CUBE, kappa=KAPPA, look=LOOK, label="CARMA"),
                          dict(b, policy="CARMA", cube="truth", kappa=KAPPA, look=LOOK, label="CARMA-Perfect"),
                          dict(b, policy="Oracle", cube="truth")]
    return specs


def sens_specs():
    specs = []
    weeks = TEST[::2]  # 26 alternating weeks
    settings = [("urgent", 0.1), ("urgent", 0.5), ("eta", 0.005), ("eta", 0.06),
                ("profile_rho", 0.6), ("profile_rho", 0.4)]
    for key, val in settings:
        for i, t0 in enumerate(weeks):
            b = dict(t0=t0, rho=0.5, seed=20_000 + i, migrate=True, experiment=f"{key}={val}")
            b[key] = val
            if key == "profile_rho":  # actual load deviates from the learned profile (0.5)
                b["rho"], b["profile_rho"] = val, 0.5
                b["experiment"] = f"load={val}, profile=0.5"
            specs += [dict(b, policy="ASAP-Local", cube="pred"),
                      dict(b, policy="Greedy-Spatial", cube="pred"),
                      dict(b, policy="Forecast-Reserve", cube="pred"),
                      dict(b, policy="MPC", cube="pred", label="MPC"),
                      dict(b, policy="CARMA", cube=CUBE, kappa=KAPPA, look=LOOK, label="CARMA"),
                      dict(b, policy="Oracle", cube="truth")]
    return specs


if __name__ == "__main__":
    which = sys.argv[1]
    specs = main_specs() if which == "main" else sens_specs()
    for sp_ in specs:
        sp_.setdefault("label", sp_["policy"])
    # expensive jobs first for better load balance
    specs.sort(key=lambda d: d["policy"] != "CARMA")
    with mp.Pool(4) as pool:
        rows = pool.map(run_one, specs, chunksize=1)
    pd.DataFrame(rows).to_csv(RES / f"test_{which}.csv", index=False)
    print(len(rows), "runs")
