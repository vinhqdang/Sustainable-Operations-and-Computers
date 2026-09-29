"""Trace-driven workloads from the Alibaba 2018 cluster trace (batch_task.csv).

Arrival times, work and parallelism come from the trace; deadlines are not recorded
in the trace and are drawn synthetically. Jobs are aggregated into batches by
(arrival hour, origin site, window class); within a batch, work and widths add up,
which leaves the fluid scheduling problem unchanged up to width pooling.
Days 1-4 of the trace (hours 0-95) are used only to learn CARMA's demand profile;
days 5-8 (hours 96-191) are the evaluation workload, preceded by 24 warm-up hours."""
import numpy as np
import pandas as pd
from common import *
from sim import CAP, ORIGIN_P, S, WIN_BINS, capacity

CORES_PER_SERVER = 96.0
_JOBS = None


def trace_jobs():
    global _JOBS
    if _JOBS is None:
        g = pd.read_csv(DATA / "traces" / "alibaba2018_jobs.csv.gz")
        g["h"] = (g.a // 3600).astype(int)
        g = g[(g.h >= 0) & (g.h < 192) & (g.work > 0)]
        _JOBS = g[["h", "work", "width", "dur"]].reset_index(drop=True)
    return _JOBS


def _assign(g, seed, urgent):
    rng = np.random.default_rng(seed)
    n = len(g)
    o = rng.choice(S, size=n, p=ORIGIN_P)
    r = np.maximum(1, np.ceil(g.dur.to_numpy())).astype(int)
    slack = rng.integers(1, 24, n)
    win = np.where(rng.random(n) < urgent, r, np.minimum(r + slack, H))
    b = np.searchsorted(WIN_BINS, win, side="right") - 1          # conservative bin
    return o, WIN_BINS[np.maximum(b, 0)]


def _batches(g, o, win, scale):
    df = pd.DataFrame(dict(h=g.h.to_numpy(), o=o, win=win,
                           w=g.work.to_numpy() / CORES_PER_SERVER * scale,
                           m=g.width.to_numpy() / CORES_PER_SERVER * scale))
    b = df.groupby(["h", "o", "win"], as_index=False)[["w", "m"]].sum()
    b["m"] = np.maximum(b.m, b.w / b.win)   # window-feasible
    return b


def scale_for(rho, seed=0, urgent=0.3):
    g = trace_jobs(); ev = g[g.h >= 96]
    per_hour = ev.work.sum() / CORES_PER_SERVER / 96.0
    avg_cap = capacity(np.arange(24)).sum(axis=1).mean()
    return rho * avg_cap / per_hour


def make_trace_jobs(t0, rho, seed, urgent=0.3):
    """Evaluation workload (trace hours 72-191) aligned so that trace hour 96 = t0."""
    g = trace_jobs(); sc = scale_for(rho)
    o, win = _assign(g, seed, urgent)
    keep = (g.h >= 72).to_numpy()
    b = _batches(g[keep], o[keep], win[keep], sc)
    a = t0 + (b.h.to_numpy() - 96)
    return dict(a=a, o=b.o.to_numpy(), m=b.m.to_numpy(), w=b.w.to_numpy(), d=a + b.win.to_numpy(),
                n=len(b), ev=b.h.to_numpy() >= 96)


def trace_profile(rho, seed, urgent=0.3, days=(0, 4)):
    """Hour-of-day demand profile learned from trace days [days[0], days[1]) and tiled
    to 168 hours; returns (W[168,S,B], M[168,S,B]) as expected by AnticipatoryMPC."""
    g = trace_jobs(); sc = scale_for(rho)
    o, win = _assign(g, seed + 99, urgent)   # independent origin/deadline draws
    keep = ((g.h >= 24 * days[0]) & (g.h < 24 * days[1])).to_numpy()
    b = _batches(g[keep], o[keep], win[keep], sc)
    nd = days[1] - days[0]
    B = len(WIN_BINS)
    W = np.zeros((24, S, B)); M = np.zeros((24, S, B))
    bi = np.searchsorted(WIN_BINS, b.win.to_numpy())
    np.add.at(W, (b.h.to_numpy() % 24, b.o.to_numpy(), bi), b.w.to_numpy())
    np.add.at(M, (b.h.to_numpy() % 24, b.o.to_numpy(), bi), b.m.to_numpy())
    return np.tile(W / nd, (7, 1, 1)), np.tile(M / nd, (7, 1, 1))
