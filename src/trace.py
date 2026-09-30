"""Trace-driven workloads from the Alibaba 2018 cluster trace (batch_task.csv).

Arrival times, work and parallelism come from the trace; deadlines are not recorded
in the trace and are drawn synthetically. Jobs are aggregated into batches by
(arrival hour, origin site, window class); within a batch, work and widths add up.
Pooling widths relaxes the per-job width limits (a batch may run faster than its
slowest member), so the batched workload is a trace-derived fluid approximation.
With fine=True, batches are additionally split by run-length class (ceil of the job
duration in hours), so that only jobs with similar work-to-width ratios are pooled.
The processed job table is built from the raw trace by prepare_trace.py.
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


def _assign(g, seed, urgent, with_r=False):
    rng = np.random.default_rng(seed)
    n = len(g)
    o = rng.choice(S, size=n, p=ORIGIN_P)
    r = np.maximum(1, np.ceil(g.dur.to_numpy())).astype(int)
    slack = rng.integers(1, 24, n)
    win = np.where(rng.random(n) < urgent, r, np.minimum(r + slack, H))
    b = np.searchsorted(WIN_BINS, win, side="right") - 1          # conservative bin
    if with_r:
        return o, WIN_BINS[np.maximum(b, 0)], np.minimum(r, H)
    return o, WIN_BINS[np.maximum(b, 0)]


def _batches(g, o, win, scale, rcls=None):
    df = pd.DataFrame(dict(h=g.h.to_numpy(), o=o, win=win,
                           w=g.work.to_numpy() / CORES_PER_SERVER * scale,
                           m=g.width.to_numpy() / CORES_PER_SERVER * scale,
                           r=0 if rcls is None else rcls))
    b = df.groupby(["h", "o", "win", "r"], as_index=False)[["w", "m"]].sum()
    b["m"] = np.maximum(b.m, b.w / b.win)   # window-feasible
    return b


def scale_for(rho, seed=0, urgent=0.3):
    g = trace_jobs(); ev = g[g.h >= 96]
    per_hour = ev.work.sum() / CORES_PER_SERVER / 96.0
    avg_cap = capacity(np.arange(24)).sum(axis=1).mean()
    return rho * avg_cap / per_hour


def make_trace_jobs(t0, rho, seed, urgent=0.3, fine=False):
    """Evaluation workload (trace hours 72-191) aligned so that trace hour 96 = t0."""
    g = trace_jobs(); sc = scale_for(rho)
    o, win, r = _assign(g, seed, urgent, with_r=True)
    keep = (g.h >= 72).to_numpy()
    b = _batches(g[keep], o[keep], win[keep], sc, r[keep] if fine else None)
    a = t0 + (b.h.to_numpy() - 96)
    return dict(a=a, o=b.o.to_numpy(), m=b.m.to_numpy(), w=b.w.to_numpy(), d=a + b.win.to_numpy(),
                n=len(b), ev=b.h.to_numpy() >= 96)


def trace_history(rho, seed, urgent=0.3, days=(0, 4), fine=False):
    """Batched jobs of trace days [days[0], days[1]) with independent origin/deadline
    draws, one job set per day (aligned so that each day starts at t0=0)."""
    g = trace_jobs(); sc = scale_for(rho)
    o, win, r = _assign(g, seed + 99, urgent, with_r=True)
    out = []
    for d in range(*days):
        keep = ((g.h >= 24 * d) & (g.h < 24 * (d + 1))).to_numpy()
        b = _batches(g[keep], o[keep], win[keep], sc, r[keep] if fine else None)
        a = b.h.to_numpy() - 24 * d
        out.append(dict(a=a, o=b.o.to_numpy(), m=b.m.to_numpy(), w=b.w.to_numpy(),
                        d=a + b.win.to_numpy(), n=len(b), ev=np.ones(len(b), bool)))
    return out


def trace_profile(rho, seed, urgent=0.3, days=(0, 4), fine=False):
    """Hour-of-day demand profile learned from trace days [days[0], days[1]) and tiled
    to 168 hours; returns (W[168,S,B], M[168,S,B]) as expected by AnticipatoryMPC."""
    g = trace_jobs(); sc = scale_for(rho)
    o, win, r = _assign(g, seed + 99, urgent, with_r=True)   # independent origin/deadline draws
    keep = ((g.h >= 24 * days[0]) & (g.h < 24 * days[1])).to_numpy()
    b = _batches(g[keep], o[keep], win[keep], sc, r[keep] if fine else None)
    nd = days[1] - days[0]
    B = len(WIN_BINS)
    W = np.zeros((24, S, B)); M = np.zeros((24, S, B))
    bi = np.searchsorted(WIN_BINS, b.win.to_numpy())
    np.add.at(W, (b.h.to_numpy() % 24, b.o.to_numpy(), bi), b.w.to_numpy())
    np.add.at(M, (b.h.to_numpy() % 24, b.o.to_numpy(), bi), b.m.to_numpy())
    return np.tile(W / nd, (7, 1, 1)), np.tile(M / nd, (7, 1, 1))
