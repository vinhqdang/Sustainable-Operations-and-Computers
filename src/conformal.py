"""Horizon-wise normalised split-conformal upper quantiles with adaptive
conformal inference (ACI) updates; produces forecast cubes C[t, s, h]."""
import numpy as np
import pandas as pd
from common import *

EPS = 5.0  # gCO2/kWh floor on the volatility scale


def load_cubes():
    """Return truth[T,S], pred[T,S,H+1], persist, snaive, vol[T,S], and time index."""
    f = pd.read_pickle(DATA / "forecasts.pkl")
    wide = load_hourly()
    idx = wide.index
    T, S = len(idx), len(SITE_IDS)
    truth = wide["ci"][SITE_IDS].to_numpy()
    cubes = {}
    for col in ["pred", "persist", "snaive"]:
        c = np.full((T, S, H + 1), np.nan)
        c[:, :, 0] = truth
        c[f.t.to_numpy(), f.site.to_numpy(), f.h.to_numpy()] = f[col].to_numpy()
        cubes[col] = c
    vol = np.full((T, S), np.nan)
    g = f[f.h == 1]
    vol[g.t.to_numpy(), g.site.to_numpy()] = g.vol.to_numpy()
    return idx, truth, cubes, vol


def scores(idx, truth, pred, vol, a, b):
    """Normalised residual scores (truth - pred)/scale over issue times in [a,b)."""
    T = len(idx)
    ts = np.where((idx >= a) & (idx < b))[0]
    out = np.full((len(ts), pred.shape[1], H), np.nan)
    for h in range(1, H + 1):
        ok = ts + h < T
        tt = ts[ok]
        out[ok, :, h - 1] = (truth[tt + h] - pred[tt, :, h]) / (vol[tt] + EPS)
    return out


def upper_cube(idx, truth, pred, vol, level, gamma=0.005, cal=(TRAIN_END, CAL_END), start=CAL_END):
    """Upper forecast U[t,s,h] = pred + q * scale, with q the ACI-adapted conformal
    quantile of calibration scores at target coverage `level`. ACI runs from `start`
    onwards, updating each (s,h) miscoverage level once the outcome is observed."""
    sc = scores(idx, truth, pred, vol, *cal)
    T, S = truth.shape
    U = pred.copy()
    t_start = int(np.searchsorted(idx, pd.Timestamp(start)))
    alpha_star = 1.0 - level
    alpha = np.full((S, H), alpha_star)
    grid = np.linspace(0.0, 1.0, 1001)
    qtab = np.stack([np.nanquantile(sc[:, :, h], grid, axis=0) for h in range(H)], axis=-1)  # [G,S,H]
    hist = []  # for coverage audit
    for t in range(t_start, T):
        lev = np.clip(1.0 - alpha, 0.0, 1.0)
        gi = np.rint(lev * 1000).astype(int)
        q = qtab[gi, np.arange(S)[:, None], np.arange(H)[None, :]]  # [S,H]
        U[t, :, 1:] = pred[t, :, 1:] + q * (vol[t][:, None] + EPS)
        # outcomes that become known at time t: issue time t-h, horizon h
        for h in range(1, H + 1):
            ti = t - h
            if ti < t_start:
                continue
            err = (truth[t] > U[ti, :, h]).astype(float)
            alpha[:, h - 1] += gamma * (alpha_star - err)
            hist.append(err)
    U[:, :, 1:] = np.maximum(U[:, :, 1:], 0)
    cov = 1 - np.mean(hist) if hist else np.nan
    return U, cov
