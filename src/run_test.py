"""Out-of-sample evaluation in 2024.

main:  synthetic workloads, 52 weeks x 3 seeds x 3 loads x 2 regimes
trace: Alibaba 2018 trace workload (days 5-8) against 52 carbon windows x 3 seeds
sens:  one-factor sensitivity (26 alternating weeks x 2 seeds), rho = 0.5, migration"""
import sys, json, multiprocessing as mp
import pandas as pd
from experiment import *

CFG = json.load(open(RES / "carma_config.json"))
REG = {True: "st", False: "t"}


def policies(b, mig, full=True):
    c = CFG[REG[mig]]
    ps = [dict(b, policy="ASAP-Local", cube="pred"),
          dict(b, policy="Greedy-Spatial", cube="pred"),
          dict(b, policy="Forecast-Reserve", cube="pred"),
          dict(b, policy="MPC", cube="pred", label="MPC"),
          dict(b, policy="MPC-Protect", cube="pred", r=c["protect_r"], tight=c["protect_tight"], label="MPC-Protect"),
          dict(b, policy="CARMA", cube=c["cube"], kappa=c["kappa"], look=c["look"], label="CARMA"),
          dict(b, policy="Oracle", cube="truth")]
    if full:
        ps += [dict(b, policy="MPC", cube="truth", label="MPC-Perfect"),
               dict(b, policy="MPC", cube=c["mpc_best_cube"], label="MPC+conformal"),
               dict(b, policy="CARMA", cube=c["best_conformal_cube"], kappa=c["kappa"], look=c["look"], label="CARMA+conformal"),
               dict(b, policy="CARMA", cube="truth", kappa=c["kappa"], look=c["look"], label="CARMA-Perfect")]
    return ps


def main_specs():
    specs = []
    for mig in [True, False]:
        for rho in [0.3, 0.5, 0.7]:
            for i, t0 in enumerate(TEST):
                for k in range(3):
                    b = dict(t0=t0, rho=rho, seed=10_000 + 100 * k + i, rep=k, migrate=mig, experiment="main")
                    specs += policies(b, mig)
    return specs


def trace_specs():
    specs = []
    for i, t0 in enumerate(TEST):
        for k in range(3):
            b = dict(t0=t0, rho=0.5, seed=40_000 + 100 * k + i, rep=k, migrate=True, workload="trace",
                     experiment="trace")
            specs += policies(b, True, full=False)
            c = CFG["st"]
            specs.append(dict(b, policy="CARMA", cube=c["cube"], kappa=c["kappa"], look=c["look"],
                              profile_days=(4, 8), label="CARMA (in-sample profile)"))
            specs.append(dict(b, policy="CARMA", cube="truth", kappa=c["kappa"], look=c["look"], label="CARMA-Perfect"))
            specs.append(dict(b, policy="MPC", cube="truth", label="MPC-Perfect"))
    return specs


def sens_specs():
    specs = []
    weeks = TEST[::2]
    settings = [("urgent", 0.1), ("urgent", 0.5), ("eta", 0.005), ("eta", 0.06),
                ("load", 0.6), ("load", 0.4), ("cap", (150, 400, 400, 600)), ("cap", (600, 400, 400, 600)),
                ("cap", (300, 400, 400, 1200)), ("idle", 1), ("acct", "marginal")]
    for key, val in settings:
        for i, t0 in enumerate(weeks):
            for k in range(2):
                b = dict(t0=t0, rho=0.5, seed=20_000 + 100 * k + i, rep=k, migrate=True)
                if key == "load":
                    b.update(rho=val, profile_rho=0.5, experiment=f"load={val}, profile=0.5")
                elif key == "cap":
                    b.update(cap=list(val), experiment="cap=" + "/".join(map(str, val)))
                elif key == "idle":
                    b.update(eit=0.2, idle=True, experiment="always-on (idle 0.2 kWh)")
                elif key == "acct":
                    b.update(acct="marginal", experiment="marginal accounting")
                else:
                    b.update({key: val, "experiment": f"{key}={val}"})
                specs += policies(b, True, full=False)
    return specs


def spec_id(sp_):
    return "|".join(f"{k}={sp_[k]}" for k in sorted(sp_))


if __name__ == "__main__":
    import os, csv
    which = sys.argv[1]
    specs = {"main": main_specs, "trace": trace_specs, "sens": sens_specs}[which]()
    for sp_ in specs:
        sp_.setdefault("label", sp_["policy"])
        sp_["sid"] = spec_id(sp_)
    part = RES / f"partial_{which}.jsonl"
    done = set()
    if part.exists():
        for line in open(part):
            done.add(json.loads(line)["sid"])
    todo = [sp_ for sp_ in specs if sp_["sid"] not in done]
    todo.sort(key=lambda d: d["policy"] != "CARMA")
    print(len(specs), "specs;", len(done), "done;", len(todo), "to run", flush=True)
    with mp.Pool(4) as pool, open(part, "a") as fh:
        for k, row in enumerate(pool.imap_unordered(run_one, todo, chunksize=2)):
            fh.write(json.dumps(row, default=float) + "\n"); fh.flush()
            if k % 500 == 0:
                print(k, flush=True)
    rows = [json.loads(l) for l in open(part)]
    ids = {sp_["sid"] for sp_ in specs}
    pd.DataFrame([r for r in rows if r["sid"] in ids]).to_csv(RES / f"test_{which}.csv", index=False)
    print(len(rows), "runs")
