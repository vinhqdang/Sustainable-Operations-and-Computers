"""Spatio-temporal carbon-aware scheduling of deferrable bag-of-tasks jobs.

Time is slotted in hours. Job j arrives at a_j at origin site o_j with work w_j
(server-hours), maximum width m_j (servers) and deadline d_j (exclusive hour).
Executing one server-hour at site s for a job from origin o emits
    E_IT * PUE_s * CI_s(t) + 1[s != o] * D * E_NET * (CI_o(t) + CI_s(t)) / 2   [gCO2].
"""
import time
import numpy as np
from common import *

E_IT = 0.40      # kWh per server-hour (IT energy)
DATA_GB = 2.0    # GB moved per migrated server-hour of work
E_NET = 0.02     # kWh per GB transferred
CAP = np.array([300.0, 400.0, 400.0, 600.0])      # batch servers per site
ORIGIN_P = np.array([0.10, 0.20, 0.25, 0.45])       # share of jobs submitted per site
S = len(SITE_IDS)
DONE_TOL = 0.05  # server-hours treated as complete
BIG = 1.0e4      # penalty (g) per server-hour left unfinished at a deadline


def capacity(hours):
    """Servers available for batch work: capacity net of diurnal interactive load."""
    hod = (hours % 24)
    bg = 0.20 + 0.15 * np.sin(2 * np.pi * (hod - 8) / 24)
    return CAP[None, :] * (1 - bg[:, None])


def unit_cost(ci, migrate=True):
    """Emission per server-hour, shape [..., S_origin, S_exec] from ci [..., S]."""
    comp = E_IT * PUE * ci                                  # [..., S]
    net = DATA_GB * E_NET * 0.5 * (ci[..., :, None] + ci[..., None, :])
    c = comp[..., None, :] + net * (1 - np.eye(S))
    if not migrate:
        c = c + (1 - np.eye(S)) * 1e7
    return c


def make_jobs(t0, rho, seed, hours=168, urgent=0.3, pre=24, post=24):
    """Synthetic arrivals over [t0-pre, t0+hours+post). Jobs arriving inside the
    evaluation week [t0, t0+hours) are flagged `ev`; the others provide warm-up and
    trailing load so that the evaluated week sees steady-state congestion."""
    rng = np.random.default_rng(seed)
    hh = np.arange(t0 - pre, t0 + hours + post)
    idx_week = (hh // 24) % 7  # day index relative to data origin (2022-01-01 = Saturday)
    wd = ((idx_week + 5) % 7) < 5  # Monday..Friday
    diurnal = 1 + 0.5 * np.sin(2 * np.pi * ((hh % 24) - 8) / 24)
    shape = diurnal * np.where(wd, 1.0, 0.7)
    m_mean = (64 - 4) / np.log(16.0)
    r_mean = 3.5
    avg_cap = capacity(np.arange(24)).sum(axis=1).mean()
    lam = rho * avg_cap / (m_mean * r_mean) * shape / shape.mean()
    n = rng.poisson(lam)
    a = np.repeat(hh, n)
    N = len(a)
    o = rng.choice(S, size=N, p=ORIGIN_P)
    m = np.exp(rng.uniform(np.log(4), np.log(64), N))
    r = rng.integers(1, 7, N).astype(float)
    f = np.where(rng.random(N) < urgent, 1.0, rng.uniform(1.5, 4.0, N))
    win = np.minimum(np.ceil(r * f), H).astype(int)
    return dict(a=a, o=o, m=m, w=m * r, d=a + win, n=N, ev=(a >= t0) & (a < t0 + hours))



# --------------------------------------------------------------------------
# Min-cost-flow core. Every allocation problem in this study is a transportation
# problem: source -> job (work) -> job-hour (width) -> hour-site (capacity) -> sink,
# plus a job -> sink slack arc priced at the lateness penalty. Solved exactly by
# network simplex on an integer grid of 0.01 server-hours and 0.01 gCO2.
# --------------------------------------------------------------------------
from ortools.graph.python import min_cost_flow

QW, QC = 100.0, 100.0


def flow_solve(rem, width, orig, rel, win, cost, cap, pen):
    """rel/win: release offset and window length (hours) of each job within the
    planning grid cost[L, o, s], cap[L, s]. Returns (jj, hh, ss, x) triples."""
    J = len(rem); L = cost.shape[0]
    jj = np.repeat(np.arange(J), win)
    hh = np.concatenate([rel[j] + np.arange(win[j]) for j in range(J)]) if J else np.zeros(0, int)
    n_jh = len(jj)
    # node ids
    src, snk = 0, 1
    job_n = 2 + np.arange(J)
    jh_n = 2 + J + np.arange(n_jh)
    hs_n = 2 + J + n_jh + np.arange(L * S)
    tails, heads, caps, costs = [], [], [], []
    r_int = np.rint(rem * QW).astype(np.int64)
    w_int = np.ceil(width * QW - 1e-6).astype(np.int64)
    tails += [np.full(J, src), job_n]
    heads += [job_n, np.full(J, snk)]
    caps += [r_int, r_int]
    costs += [np.zeros(J, np.int64), np.rint(pen * QC).astype(np.int64)]
    tails.append(job_n[jj]); heads.append(jh_n); caps.append(w_int[jj]); costs.append(np.zeros(n_jh, np.int64))
    jh_rep = np.repeat(np.arange(n_jh), S)
    s_rep = np.tile(np.arange(S), n_jh)
    h_rep = hh[jh_rep]
    c_arc = cost[h_rep, orig[jj[jh_rep]], s_rep]
    tails.append(jh_n[jh_rep]); heads.append(hs_n[h_rep * S + s_rep])
    caps.append(w_int[jj[jh_rep]]); costs.append(np.rint(np.minimum(c_arc, 1e9) * QC).astype(np.int64))
    first_x = sum(len(t) for t in tails)
    tails.append(hs_n); heads.append(np.full(L * S, snk))
    caps.append(np.floor(cap.ravel() * QW).astype(np.int64)); costs.append(np.zeros(L * S, np.int64))
    mcf = min_cost_flow.SimpleMinCostFlow()
    T_ = np.concatenate(tails); Hd = np.concatenate(heads)
    mcf.add_arcs_with_capacity_and_unit_cost(T_, Hd, np.concatenate(caps), np.concatenate(costs))
    tot = int(r_int.sum())
    sup = np.zeros(2 + J + n_jh + L * S, np.int64); sup[src] = tot; sup[snk] = -tot
    mcf.set_nodes_supplies(np.arange(len(sup)), sup)
    st = mcf.solve()
    if st != mcf.OPTIMAL:
        raise RuntimeError(f"min-cost flow status {st}")
    n_x = len(jh_rep)
    fl = mcf.flows(np.arange(first_x - n_x, first_x)) / QW
    slack = mcf.flows(np.arange(J, 2 * J)) / QW
    return jj[jh_rep], h_rep, s_rep, fl, slack

def solve_alloc(rem, width, orig, dl, t, cost, cap):
    """Myopic receding-horizon plan over pending jobs; returns first-hour x0[j, s]."""
    J = len(rem); L = cost.shape[0]
    win = np.clip(dl - t, 1, L)  # overdue work gets a one-hour window
    jj, hh, ss, x, _ = flow_solve(rem, width, orig, np.zeros(J, int), win, cost, cap, np.full(J, BIG))
    x0 = np.zeros((J, S)); f = hh == 0
    np.add.at(x0, (jj[f], ss[f]), x[f])
    return x0


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------
class Policy:
    name = "base"
    migrate = True

    def reset(self, jobs, t0):
        pass

    def act(self, t, P, view):
        raise NotImplementedError


def _edf_fill(order, rem, width, capleft, site_pref):
    """Greedy: for jobs in `order`, place up to width at preferred sites."""
    x = np.zeros((len(rem), S))
    for j in order:
        need = min(rem[j], width[j])
        for s in site_pref[j]:
            if need <= 1e-9:
                break
            q = min(need, capleft[s])
            if q > 0:
                x[j, s] += q
                capleft[s] -= q
                need -= q
    return x


class ASAPLocal(Policy):
    """Carbon-agnostic: run immediately at the origin site, earliest deadline first."""
    name = "ASAP-Local"

    def act(self, t, P, view):
        order = np.argsort(P["dl"], kind="stable")
        pref = [[P["orig"][j]] for j in range(len(order))]
        return _edf_fill(order, P["rem"], P["width"], view["cap"][0].copy(), pref)


class ASAPGreen(Policy):
    """Spatial-only greedy: run immediately at the currently greenest feasible site."""
    name = "Greedy-Spatial"

    def act(self, t, P, view):
        c0 = view["cost_true"][0]  # [o, s]
        order = np.argsort(P["dl"], kind="stable")
        pref = [[s for s in np.argsort(c0[P["orig"][j]]) if c0[P["orig"][j], s] < 1e6]
                for j in range(len(P["rem"]))]
        return _edf_fill(order, P["rem"], P["width"], view["cap"][0].copy(), pref)


class Reservation(Policy):
    """Forecast-based reservation (lowest-window booking at arrival, no re-planning)."""
    name = "Forecast-Reserve"

    def __init__(self, cube_key="pred"):
        self.cube_key = cube_key

    def reset(self, jobs, t0):
        self.book = {}      # job id -> dict[(hour, site)] = amount
        self.used = {}      # (hour, site) -> amount booked

    def act(self, t, P, view):
        cap = view["cap"]
        x = np.zeros((len(P["rem"]), S))
        for k, jid in enumerate(P["id"]):
            if jid not in self.book:  # new arrival: book cheapest forecast slots
                need = P["rem"][k]
                L = max(1, P["dl"][k] - t)
                c = view["cost_fc"][:L, P["orig"][k], :]  # [L, S]
                b = {}
                for flat in np.argsort(c, axis=None):
                    h, s = divmod(int(flat), S)
                    if need <= 1e-9:
                        break
                    if c[h, s] >= 1e6:
                        continue
                    free = cap[h, s] - self.used.get((t + h, s), 0.0)
                    inrow = sum(v for (hh, _), v in b.items() if hh == t + h)
                    q = min(need, free, P["width"][k] - inrow)
                    if q > 1e-9:
                        b[(t + h, s)] = b.get((t + h, s), 0.0) + q
                        self.used[(t + h, s)] = self.used.get((t + h, s), 0.0) + q
                        need -= q
                self.book[jid] = b
        capleft = cap[0].copy()
        for k, jid in enumerate(P["id"]):
            for s in range(S):
                q = min(self.book[jid].get((t, s), 0.0), capleft[s], P["rem"][k] - x[k].sum())
                if q > 0:
                    x[k, s] += q
                    capleft[s] -= q
        # overdue or unbooked residual work: run now greedily (EDF)
        rem2 = P["rem"] - x.sum(1)
        late = [k for k in range(len(rem2)) if rem2[k] > 1e-9 and P["dl"][k] <= t + 1]
        if late:
            w2 = P["width"] - x.sum(1)
            c0 = view["cost_true"][0]
            pref = [[s for s in np.argsort(c0[P["orig"][j]]) if c0[P["orig"][j], s] < 1e6]
                    for j in range(len(rem2))]
            x += _edf_fill(late, rem2, w2, capleft, pref)
        return x


class MPC(Policy):
    """Receding-horizon LP. `cube_key` selects the carbon-intensity view used for
    future hours: 'truth' (clairvoyant), 'pred', 'persist', 'snaive', or 'upper'
    (conformal upper quantile, the proposed risk-adjusted costing)."""

    def __init__(self, cube_key="pred", name=None, migrate=True):
        self.cube_key = cube_key
        self.name = name or f"MPC-{cube_key}"
        self.migrate = migrate

    def act(self, t, P, view):
        return solve_alloc(P["rem"], P["width"], P["orig"], P["dl"], t,
                           view["cost_fc"], view["cap"])


# --------------------------------------------------------------------------
# Simulation loop
# --------------------------------------------------------------------------
def simulate(policy, jobs, t0, truth, cube, migrate=True):
    """Run `policy` on the job stream. cube[t, s, h] is the CI view at issue time t
    for hour t+h (h=0 is the observed current hour). Metrics cover jobs flagged ev."""
    t_first, t_end = int(jobs["a"].min()), int(jobs["d"].max()) + 1
    rem = jobs["w"].copy()
    emis_j = np.zeros(jobs["n"]); mig_j = np.zeros(jobs["n"]); late_j = np.zeros(jobs["n"])
    policy.reset(jobs, t0)
    ct_all = unit_cost(truth, migrate)  # [T, o, s]
    fill = np.nanmean(truth)
    n_dec, wall = 0, 0.0
    for t in range(t_first, t_end):
        ids = np.where((jobs["a"] <= t) & (rem > 1e-9))[0]
        if len(ids) == 0:
            continue
        view = {"cap": capacity(np.arange(t, t + H)), "cost_true": ct_all[t:t + H],
                "cost_fc": unit_cost(np.nan_to_num(cube[t, :, :H].T, nan=fill), migrate)}
        view["cost_fc"][0] = ct_all[t]
        P = dict(id=ids, rem=rem[ids], width=jobs["m"][ids], orig=jobs["o"][ids], dl=jobs["d"][ids])
        tic = time.perf_counter()
        x = np.maximum(policy.act(t, P, view), 0)
        wall += time.perf_counter() - tic; n_dec += 1
        x = np.minimum(x, rem[ids][:, None])
        rem[ids] = np.maximum(rem[ids] - x.sum(1), 0)
        rem[rem < DONE_TOL] = 0.0  # ignore residue from the 0.01 server-hour flow grid
        step = ct_all[t][jobs["o"][ids]]
        assert not np.any((x > 1e-6) & (step >= 1e6)), "migration used when disallowed"
        emis_j[ids] += (x * step).sum(1)
        mig_j[ids] += (x * (1 - np.eye(S))[jobs["o"][ids]]).sum(1)
        # work still outstanding when the deadline passes is recorded as late
        miss = ids[(jobs["d"][ids] == t + 1) & (rem[ids] > DONE_TOL)]
        late_j[miss] = rem[miss]
    ev = jobs["ev"]; total = jobs["w"][ev].sum()
    return dict(emis_t=emis_j[ev].sum() / 1e6, late_frac=late_j[ev].sum() / total,
                late_jobs=float((late_j[ev] > DONE_TOL).mean()),
                unfinished=float(rem[ev].sum()) / total, mig_frac=mig_j[ev].sum() / total,
                work=total, jobs=int(ev.sum()), ms_per_decision=1e3 * wall / max(n_dec, 1))


def oracle(jobs, t0, truth, migrate=True):
    """Offline clairvoyant plan (known arrivals and CI): lower bound on emissions."""
    tf = int(jobs["a"].min()); L = int(jobs["d"].max()) - tf
    ct = unit_cost(truth[tf:tf + L], migrate)
    cap = capacity(np.arange(tf, tf + L))
    jj, hh, ss, x, u = flow_solve(jobs["w"], jobs["m"], jobs["o"], jobs["a"] - tf,
                                  jobs["d"] - jobs["a"], ct, cap, np.full(jobs["n"], BIG))
    ev = jobs["ev"]; total = jobs["w"][ev].sum(); k = ev[jj]
    return dict(emis_t=float((x * ct[hh, jobs["o"][jj], ss])[k].sum()) / 1e6,
                late_frac=float(u[ev].sum()) / total, late_jobs=float((u[ev] > 1e-3).mean()),
                unfinished=float(u[ev].sum()) / total,
                mig_frac=float(x[k & (jobs["o"][jj] != ss)].sum()) / total, work=total,
                jobs=int(ev.sum()), ms_per_decision=np.nan)


# --------------------------------------------------------------------------
# Anticipatory, risk-adjusted MPC (proposed)
# --------------------------------------------------------------------------
WIN_BINS = np.array([1, 2, 4, 6, 9, 13, 18, 24])


def demand_profile(job_sets, t0s, hours=168):
    """Empirical expected arriving work and width per (hour-of-week, origin, window
    bin), estimated from historical job traces. Returns (work[168,S,B], width[...])."""
    B = len(WIN_BINS)
    W = np.zeros((168, S, B)); M = np.zeros((168, S, B))
    for jobs, t0 in zip(job_sets, t0s):
        ev = jobs["ev"]
        jobs = {k: v[ev] for k, v in jobs.items() if isinstance(v, np.ndarray)}
        how = (jobs["a"] - t0) % 168
        win = jobs["d"] - jobs["a"]
        b = np.searchsorted(WIN_BINS, win, side="right") - 1  # largest bin <= window (conservative)
        np.add.at(W, (how, jobs["o"], b), jobs["w"])
        np.add.at(M, (how, jobs["o"], b), jobs["m"])
    n = len(job_sets)
    return W / n, M / n


class AnticipatoryMPC(Policy):
    """Receding-horizon LP over real pending jobs plus 'ghost' jobs representing
    expected arrivals in the next `look` hours, with future carbon costs taken from
    the chosen forecast cube (e.g. conformal upper quantiles)."""

    def __init__(self, profile, kappa=1.0, look=12, name="CARMA", migrate=True, ghost_pen=None):
        self.W, self.M = profile
        self.kappa, self.look, self.name, self.migrate = kappa, look, name, migrate
        self.ghost_pen = ghost_pen

    def reset(self, jobs, t0):
        self.t0 = t0

    def act(self, t, P, view):
        J = len(P["rem"])
        how = (t + np.arange(1, self.look + 1) - self.t0) % 168
        Wg = self.kappa * self.W[how]          # [look, S, B]
        Mg = self.kappa * self.M[how]
        k, o, b = np.nonzero(Wg > 1e-6)
        if len(k) == 0:
            return solve_alloc(P["rem"], P["width"], P["orig"], P["dl"], t, view["cost_fc"], view["cap"])
        rem = np.r_[P["rem"], Wg[k, o, b]]
        width = np.r_[P["width"], Mg[k, o, b]]
        orig = np.r_[P["orig"], o].astype(int)
        rel = np.r_[np.zeros(J, int), k + 1].astype(int)
        dl = np.r_[np.clip(P["dl"] - t, 1, H), np.minimum(k + 1 + WIN_BINS[b], H)].astype(int)
        return solve_general(rem, width, orig, rel, dl, view["cost_fc"], view["cap"], J, self.ghost_pen)


def solve_general(rem, width, orig, rel, dl, cost, cap, n_real, ghost_pen=None):
    """Plan with release times (relative hours). Real jobs (first n_real) carry the
    deadline penalty BIG; ghost jobs a softer one. Returns first-hour x for real jobs."""
    J = len(rem)
    win = np.maximum(dl - rel, 1)
    gp = ghost_pen if ghost_pen is not None else 1.5 * cost[:, np.arange(S), np.arange(S)].max()
    pen = np.r_[np.full(n_real, BIG), np.full(J - n_real, gp)]
    jj, hh, ss, x, _ = flow_solve(rem, width, orig, rel, win, cost, cap, pen)
    x0 = np.zeros((n_real, S)); f = (hh == 0) & (jj < n_real)
    np.add.at(x0, (jj[f], ss[f]), x[f])
    return x0
