"""Horizon-wise normalized split-conformal upper quantiles with adaptive
conformal inference (ACI) updates; produces forecast cubes C[t, s, h]."""
import numpy as np
import pandas as pd
from common import *

EPS = 5.0  # gCO2/kWh floor on the volatility scale


def load_cubes():
    """Return truth[T,S], pred[T,S,H+1], persist, snaive, vol[T,S], and time index."""
    f = pd.read_pickle(FORECAST_PATH)
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




def valid_pairs(T):
    """valid[t, h-1]: issue time t and target t+h are both observed (not gap-filled)."""
    f = filled_mask()
    tt = np.minimum(np.arange(T)[:, None] + np.arange(1, H + 1)[None, :], T - 1)
    return ~f[:, None] & ~f[tt] & (np.arange(T)[:, None] + np.arange(1, H + 1)[None, :] < T)


def scores(idx, truth, pred, vol, a, b):
    """Normalized residual scores (truth - pred)/scale for issue times t with both t and
    the target t+h inside [a, b), so that no calibration outcome lies beyond b."""
    T = len(idx)
    ts = np.where((idx >= a) & (idx < b))[0]
    t_b = int(np.searchsorted(idx, pd.Timestamp(b)))
    ok_pair = valid_pairs(T)
    out = np.full((len(ts), pred.shape[1], H), np.nan)
    for h in range(1, H + 1):
        ok = (ts + h < t_b) & ok_pair[ts, h - 1]
        tt = ts[ok]
        out[ok, :, h - 1] = (truth[tt + h] - pred[tt, :, h]) / (vol[tt] + EPS)
    return out


def conformal_table(sc, grid):
    """Split-conformal quantiles: for level g, the ceil((n+1) g)-th smallest of the n
    calibration scores (infinite when that rank exceeds n). Returns [G, S, H]."""
    G = len(grid); _, S_, H_ = sc.shape
    q = np.full((G, S_, H_), np.inf)
    for s in range(S_):
        for h in range(H_):
            v = np.sort(sc[:, s, h][~np.isnan(sc[:, s, h])]); n = len(v)
            k = np.ceil((n + 1) * grid).astype(int)          # 1-based rank
            ok = (k >= 1) & (k <= n)
            q[ok, s, h] = v[k[ok] - 1]
            q[k < 1, s, h] = -np.inf
    return q


def upper_cube(idx, truth, pred, vol, level, gamma=0.005, cal=(TRAIN_END, CAL_END), start=CAL_END):
    """Upper forecast U[t,s,h] = pred + q * scale, with q the conformal quantile of the
    calibration scores at an adaptively updated level (ACI, Gibbs and Candes 2021; used here as an empirical recalibration).
    At each hour t, the outcomes observed at t (issue time t-h, horizon h) are used to
    update the level of every (site, horizon) before the forecasts issued at t are
    formed; feedback for horizon h therefore arrives with a delay of h hours. When the
    level is at least 1 the bound is set to the largest intensity of the site observed
    in the training period (a finite worst case in place of ACI's infinite interval); when it is at most 0 the bound is zero. Gap-filled outcomes give no feedback. Bounds are floored at zero."""
    sc = scores(idx, truth, pred, vol, *cal)
    T, S = truth.shape
    U = pred.copy()
    t_start = int(np.searchsorted(idx, pd.Timestamp(start)))
    alpha_star = 1.0 - level
    alpha = np.full((S, H), alpha_star)
    grid = np.arange(1001) / 1000.0
    qtab = conformal_table(sc, grid)                      # [G,S,H]
    ok_pair = valid_pairs(T)
    u_max = np.nanmax(truth[idx < pd.Timestamp(TRAIN_END)], axis=0)[:, None]   # [S,1]
    hist = []  # for coverage audit
    for t in range(t_start, T):
        # outcomes that become known at time t: issue time t-h, horizon h
        for h in range(1, H + 1):
            ti = t - h
            if ti < t_start or not ok_pair[ti, h - 1].all():
                continue
            err = (truth[t] > U[ti, :, h]).astype(float)
            alpha[:, h - 1] += gamma * (alpha_star - err)
            hist.append(err)
        lev = 1.0 - alpha
        gi = np.clip(np.rint(lev * 1000), 0, 1000).astype(int)
        q = qtab[gi, np.arange(S)[:, None], np.arange(H)[None, :]]  # [S,H]
        q = np.where(lev >= 1.0, np.inf, np.where(lev <= 0.0, -np.inf, q))
        u = pred[t, :, 1:] + q * (vol[t][:, None] + EPS)
        u = np.where(np.isposinf(u), u_max, u)
        U[t, :, 1:] = np.maximum(np.where(np.isneginf(u), 0.0, u), 0.0)
    U[:t_start, :, 1:] = np.maximum(U[:t_start, :, 1:], 0)
    cov = 1 - np.mean(hist) if hist else np.nan
    return U, cov
