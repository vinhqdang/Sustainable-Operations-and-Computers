"""Additional forecast diagnostics: pinball loss of the conformal quantiles,
Diebold-Mariano tests (GBM vs persistence, absolute-error loss, HAC variance),
and the offset between the conformal cost views and the point forecast."""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from common import *
from conformal import load_cubes, upper_cube, valid_pairs

idx, truth, cubes, vol = load_cubes()
T, S = truth.shape


def sel(a, b):
    t = np.where((idx >= a) & (idx < b))[0]
    return t[t + H < T]


te, va = sel(VAL_END, TEST_END), sel(CAL_END, VAL_END)
VP = valid_pairs(T)
Y = lambda ts: np.transpose(truth[ts[:, None] + np.arange(1, H + 1)[None, :]], (0, 2, 1))  # [n,S,H]
rows = []
for lv in [0.5, 0.6, 0.7, 0.8, 0.9]:
    for gamma, lab in [(0.0, "Split conformal"), (0.005, "Adaptive (ACI)")]:
        U, _ = upper_cube(idx, truth, cubes["pred"], vol, lv, gamma=gamma)
        y, q = Y(te), U[te, :, 1:]
        e = np.where(VP[te][:, None, :], y - q, np.nan)
        pin = np.nanmean(np.maximum(lv * e, (lv - 1) * e))
        rows.append(dict(level=lv, method=lab, pinball=pin))
pin = pd.DataFrame(rows)


def dm(e1, e2, h):
    """Diebold-Mariano statistic for absolute-error loss differential, HAC lag h-1."""
    d = np.abs(e1) - np.abs(e2)
    d = d - 0 if d.ndim == 1 else d
    n = len(d); dbar = d.mean(); dc = d - dbar
    v = np.dot(dc, dc) / n
    for k in range(1, h):
        v += 2 * (1 - k / h) * np.dot(dc[k:], dc[:-k]) / n
    stat = dbar / np.sqrt(v / n)
    return stat, 2 * (1 - norm.cdf(abs(stat)))


dmr = []
for h in [1, 3, 6, 12, 18, 24]:
    for s in range(S):
        tv = te[VP[te, h - 1]]
        y = truth[tv + h, s]
        st, p = dm(cubes["pred"][tv, s, h] - y, cubes["persist"][tv, s, h] - y, h)
        st2, p2 = dm(cubes["pred"][tv, s, h] - y, cubes["snaive"][tv, s, h] - y, h)
        dmr.append(dict(h=h, site=SITES[SITE_IDS[s]], dm_persist=st, p_persist=p, dm_snaive=st2, p_snaive=p2))
dmr = pd.DataFrame(dmr)

off = []
for lv in [0.6, 0.8]:
    U, _ = upper_cube(idx, truth, cubes["pred"], vol, lv)
    for per, ts in [("validation", va), ("test", te)]:
        dlt = U[ts, :, 1:] - cubes["pred"][ts, :, 1:]
        for s in range(S):
            off.append(dict(level=lv, period=per, site=SITES[SITE_IDS[s]], mean_offset=dlt[:, s].mean(),
                            offset_h1=dlt[:, s, 0].mean(), offset_h24=dlt[:, s, -1].mean(),
                            sd_offset=dlt[:, s].std()))
off = pd.DataFrame(off)
pin.to_csv(RES / "forecast_pinball.csv", index=False)
dmr.to_csv(RES / "forecast_dm.csv", index=False)
off.to_csv(RES / "costview_offsets.csv", index=False)
pd.set_option("display.width", 200)
print(pin.round(3).to_string()); print(dmr.round(3).to_string()); print(off.round(2).to_string())
