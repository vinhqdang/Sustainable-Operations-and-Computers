"""Forecast accuracy and calibration on the 2024 test year (Tables 2-3, Fig. 2)."""
import json
import numpy as np
import pandas as pd
from common import *
from conformal import load_cubes, upper_cube, valid_pairs

idx, truth, cubes, vol = load_cubes()
T, S = truth.shape
te = np.where((idx >= VAL_END) & (idx < TEST_END))[0]
te = te[te + H < T]
VALID = valid_pairs(T)[te]          # [n, H]: exclude gap-filled issue or target hours
rows = []
for name, key in [("Persistence", "persist"), ("Seasonal naive (24 h)", "snaive"), ("GBM (proposed)", "pred")]:
    for h in range(1, H + 1):
        e = (cubes[key][te, :, h] - truth[te + h])[VALID[:, h - 1]]
        for s in range(S):
            rows.append(dict(model=name, h=h, site=SITES[SITE_IDS[s]], mae=np.abs(e[:, s]).mean(),
                             rmse=np.sqrt((e[:, s] ** 2).mean())))
acc = pd.DataFrame(rows)
acc.to_csv(RES / "forecast_accuracy_by_h.csv", index=False)

# calibration of one-sided upper bounds: static split conformal vs ACI
cal = []
for lv in [0.5, 0.6, 0.7, 0.8, 0.9]:
    for gamma, lab in [(0.0, "Split conformal"), (0.005, "Adaptive (ACI)")]:
        U, _ = upper_cube(idx, truth, cubes["pred"], vol, lv, gamma=gamma)
        hit = truth[te[:, None] + np.arange(1, H + 1)[None, :]]  # [n, H, S]
        u = np.transpose(U[te, :, 1:], (0, 2, 1))
        ok = np.broadcast_to(VALID[:, :, None], hit.shape)
        c = np.where(ok, hit <= u, np.nan)
        cov = np.nanmean(c, axis=0)  # [H, S]
        # monthly coverage spread (worst month)
        months = idx[te].month
        mcov = [np.nanmean(c[months == m]) for m in range(1, 13)]
        cal.append(dict(level=lv, method=lab, coverage=cov.mean(), cov_h1=cov[0].mean(),
                        cov_h24=cov[-1].mean(), worst_month=min(mcov), best_month=max(mcov),
                        mean_excess=float(np.nanmean(np.where(ok, u - np.transpose(cubes["pred"][te, :, 1:], (0, 2, 1)), np.nan)))))
calib = pd.DataFrame(cal)
calib.to_csv(RES / "forecast_calibration.csv", index=False)

summ = acc.groupby(["model"]).agg(mae=("mae", "mean"), rmse=("rmse", "mean")).round(2)
bys = acc.groupby(["model", "site"]).mae.mean().unstack().round(1)
byh = acc.groupby(["model", "h"]).mae.mean().unstack()[[1, 3, 6, 12, 24]].round(1)
print(summ); print(bys); print(byh); print(calib.round(3))
raw = pd.read_csv(DATA / "raw" / "gb_regional_ci_2022_2024.csv.gz", usecols=["time"])
hrs = pd.to_datetime(raw.time).dt.tz_localize(None).dt.floor("h")
full = pd.date_range("2022-01-01", "2024-12-31 23:00", freq="h")
json.dump({"hours": len(full), "hours_without_data": int((~full.isin(hrs)).sum()),
           "filled_hours": [str(x) for x in load_hourly().index[filled_mask()]]},
          open(RES / "data_info.json", "w"))
