"""Probabilistic multi-horizon carbon-intensity forecaster.

One direct gradient-boosted model per horizon (pooled over sites) predicts y[t+h]-y[t] for every site and
horizon h=1..H from information available at issue time t. Horizon-wise split
conformal calibration on held-out data (normalised by recent volatility) turns
the point model into upper quantile forecasts at any level.
"""
import sys, time, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from common import *

LAGS = [1, 2, 3, 6, 12, 23]


def build_features(wide):
    ci = wide["ci"]
    gb = ci[18]
    T = len(ci)
    tt = ci.index
    rows = []
    for si, r in enumerate(SITE_IDS):
        y = ci[r].to_numpy()
        ys = pd.Series(y)
        vol = ys.diff().abs().rolling(24, min_periods=1).mean().to_numpy()
        m24 = ys.rolling(24, min_periods=1).mean().to_numpy()
        lagv = {L: np.r_[np.full(L, np.nan), y[:-L]] for L in LAGS}
        wind = wide["wind"][r].to_numpy(); gas = wide["gas"][r].to_numpy()
        solar = wide["solar"][r].to_numpy()
        for h in range(1, H + 1):
            tgt = np.r_[y[h:], np.full(h, np.nan)]
            same_hour_yday = np.r_[np.full(24 - h, np.nan), y[:T - (24 - h)]] if h < 24 else y
            wk = 168 - h
            same_hour_lastweek = np.r_[np.full(wk, np.nan), y[:T - wk]]
            th = (tt + pd.Timedelta(hours=h))
            f = pd.DataFrame({
                "t": np.arange(T), "site": si, "h": h,
                "y0": y, "vol": vol, "m24": m24,
                "yday": same_hour_yday - y, "lweek": same_hour_lastweek - y,
                "gb0": gb.to_numpy(), "wind": wind, "gas": gas, "solar": solar,
                "hod": th.hour, "dow": th.dayofweek, "doy_s": np.sin(2 * np.pi * th.dayofyear / 365.25),
                "doy_c": np.cos(2 * np.pi * th.dayofyear / 365.25), "hod0": tt.hour,
                "target": tgt - y,
            })
            for L in LAGS:
                f[f"d{L}"] = y - lagv[L]
            rows.append(f)
    F = pd.concat(rows, ignore_index=True)
    F["time"] = tt[F["t"].to_numpy()]
    filled = filled_mask()
    tgt_i = np.minimum(F["t"].to_numpy() + F["h"].to_numpy(), T - 1)
    F["filled_tgt"] = filled[tgt_i]
    F["filled_in"] = filled[F["t"].to_numpy()]
    return F


def fit_and_forecast():
    wide = load_hourly()
    F = build_features(wide)
    feats = [c for c in F.columns if c not in ("t", "time", "target", "filled_tgt", "filled_in")]
    ok = F.notna().all(axis=1) & ~F.filled_tgt & ~F.filled_in   # never train on gap-filled values
    tr = F[ok & (F.time < pd.Timestamp(TRAIN_END) - pd.Timedelta(hours=H))]
    t0 = time.time()
    models = {}
    for h in range(1, H + 1):
        a = tr[tr.h == h]
        models[h] = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=50, categorical_features=[feats.index("site")],
            random_state=0).fit(a[feats], a["target"])
    fit_s = time.time() - t0
    Fp = F[F[feats].notna().all(axis=1)].copy()
    Fp["pred"] = Fp["y0"]
    for h, m in models.items():
        sel = Fp.h == h
        Fp.loc[sel, "pred"] = Fp.loc[sel, "y0"] + m.predict(Fp.loc[sel, feats])
    Fp["pred"] = Fp["pred"].clip(lower=0)
    Fp["truth"] = Fp["y0"] + Fp["target"]
    # benchmark forecasts
    Fp["persist"] = Fp["y0"]
    Fp["snaive"] = Fp["y0"] + Fp["yday"]
    out = Fp[["t", "time", "site", "h", "y0", "vol", "truth", "pred", "persist", "snaive", "filled_tgt"]]
    out.to_pickle(DATA / "forecasts.pkl")
    json.dump({"fit_seconds": fit_s, "n_train": len(tr), "features": feats},
              open(RES / "forecast_model_info.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    RES.mkdir(exist_ok=True)
    out = fit_and_forecast()
    te = out[(out.time >= CAL_END) & out.truth.notna() & ~out.filled_tgt]
    for m in ["pred", "persist", "snaive"]:
        print(m, (te[m] - te.truth).abs().mean().round(2))
