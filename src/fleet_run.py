"""Multi-grid evaluation: the same experiment for one fleet (environment variable FLEET).

    FLEET=US python fleet_run.py tune | select | test | analyze

tune:    13 validation weeks (2023-Q4), rho = 0.5, one trace per week. Every tunable
         method gets a small grid on the validation data of the fleet:
         CARMA kappa in {0.75, 1, 1.25} (look-ahead 12 h, point forecasts),
         MPC-Protect r in {0.2, 0.3, 0.5} (tau = 1 h),
         MPC+conformal level in {q60, q80, q90}.
test:    52 weeks of 2024 x 2 traces x rho in {0.3, 0.5, 0.7}, migration allowed;
         Scenario-MPC on 26 alternating weeks, one trace, rho = 0.5;
         reversed layout (cleanest site largest) on 26 alternating weeks, rho = 0.5.
Results: results/fleets/<fleet>/ ."""
import sys, json, itertools, hashlib, multiprocessing as mp
import numpy as np
import pandas as pd
from experiment import *

FLEET_CFG = RES / "fleet_config.json"


def tune_specs():
    specs = []
    for i, t0 in enumerate(VAL):
        b = dict(t0=t0, rho=0.5, seed=500 + i, migrate=True)
        specs.append(dict(b, policy="MPC", cube="pred", label="MPC"))
        specs.append(dict(b, policy="Oracle", cube="truth", label="Oracle"))
        for lv in ["q60", "q80", "q90"]:
            specs.append(dict(b, policy="MPC", cube=lv, label=f"MPC|{lv}"))
        for k in [0.75, 1.0, 1.25]:
            specs.append(dict(b, policy="CARMA", cube="pred", kappa=k, look=12, label=f"CARMA|{k}"))
        for r in [0.2, 0.3, 0.5]:
            specs.append(dict(b, policy="MPC-Protect", cube="pred", r=r, tight=1, label=f"Protect|{r}"))
    return specs


def select(rows):
    d = pd.DataFrame(rows)
    orc = d[d.label == "Oracle"].set_index("t0").emis_t
    d["gap"] = 100 * (d.emis_t / d.t0.map(orc) - 1)
    m = d[d.label != "Oracle"].groupby("label").gap.mean()
    best = lambda pre: max(m[m.index.str.startswith(pre)].index, key=lambda k: -m[k])
    cfg = dict(kappa=float(best("CARMA|").split("|")[1]), look=12,
               protect_r=float(best("Protect|").split("|")[1]), protect_tight=1,
               mpc_cube=best("MPC|").split("|")[1], val_gap=m.round(3).to_dict())
    return cfg


def test_specs(cfg):
    base = lambda t0, rho, k, i: dict(t0=t0, rho=rho, seed=10_000 + 100 * k + i, rep=k, migrate=True)
    def pols(b, extra=()):
        ps = [dict(b, policy="ASAP-Local", cube="pred"), dict(b, policy="Greedy-Spatial", cube="pred"),
              dict(b, policy="Forecast-Reserve", cube="pred"), dict(b, policy="MPC", cube="pred", label="MPC"),
              dict(b, policy="MPC-Protect", cube="pred", r=cfg["protect_r"], tight=cfg["protect_tight"], label="MPC-Protect"),
              dict(b, policy="MPC", cube=cfg["mpc_cube"], label="MPC+conformal"),
              dict(b, policy="MPC", cube="truth", label="MPC-Perfect"),
              dict(b, policy="CARMA", cube="pred", kappa=cfg["kappa"], look=cfg["look"], label="CARMA"),
              dict(b, policy="CARMA", cube="truth", kappa=cfg["kappa"], look=cfg["look"], label="CARMA-Perfect"),
              dict(b, policy="Oracle", cube="truth")]
        return ps
    specs = []
    for rho in [0.3, 0.5, 0.7]:
        for i, t0 in enumerate(TEST):
            for k in range(2):
                specs += [dict(s, experiment="main") for s in pols(base(t0, rho, k, i))]
    for i, t0 in enumerate(TEST):                       # Scenario-MPC on alternating weeks
        if i % 2 == 0:
            specs.append(dict(base(t0, 0.5, 0, i), policy="Scenario-MPC", cube="pred", K=8, look=12, label="Scenario-MPC",
                              experiment="main"))
    for i, t0 in enumerate(TEST):                       # reversed capacity layout
        if i % 2 == 0:
            for s in pols(base(t0, 0.5, 0, i)):
                if s["policy"] in ("ASAP-Local", "MPC", "MPC-Protect", "CARMA", "Oracle") or s.get("label") == "MPC+conformal":
                    specs.append(dict(s, layout="reversed", experiment="reversed"))
    return specs


def sid(s):
    return "|".join(f"{k}={s[k]}" for k in sorted(s))


def provenance():
    h = hashlib.sha1()
    from common import ROOT, DATA
    for f in sorted((ROOT / "src").glob("*.py")) + [FORECAST_PATH, HOURLY_CACHE, FLEET_CFG]:
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()[:10]


def run(specs, tag):
    for s in specs:
        s.setdefault("label", s["policy"]); s["sid"] = sid(s)
    part = RES / f"partial_{tag}_{provenance()}.jsonl"
    done = set()
    if part.exists():
        done = {json.loads(l)["sid"] for l in open(part)}
    todo = [s for s in specs if s["sid"] not in done]
    todo.sort(key=lambda d: (d["policy"] != "Scenario-MPC", d["policy"] != "CARMA"))
    print(len(specs), "specs;", len(done), "done;", len(todo), "to run", flush=True)
    with mp.Pool(4) as pool, open(part, "a") as fh:
        for k, row in enumerate(pool.imap_unordered(run_one, todo, chunksize=2)):
            fh.write(json.dumps(row, default=float) + "\n"); fh.flush()
            if k % 200 == 0:
                print(k, flush=True)
    ids = {s["sid"] for s in specs}
    rows = [json.loads(l) for l in open(part)]
    return [r for r in rows if r["sid"] in ids]


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "tune":
        rows = run(tune_specs(), "tune")
        pd.DataFrame(rows).to_csv(RES / "fleet_tuning.csv", index=False)
        cfg = select(rows)
        json.dump(cfg, open(FLEET_CFG, "w"), indent=1); print(cfg)
    elif step == "test":
        cfg = json.load(open(FLEET_CFG))
        rows = run(test_specs(cfg), "test")
        pd.DataFrame(rows).to_csv(RES / "fleet_test.csv", index=False)
        print(len(rows), "runs")
