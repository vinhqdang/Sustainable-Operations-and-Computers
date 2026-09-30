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
CAP = BASE_CAP.copy()                              # batch servers per site
ORIGIN_P = BASE_ORIGIN_P.copy()                     # share of jobs submitted per site
S = len(SITE_IDS)
DONE_TOL = 0.05  # server-hours treated as complete
BIG = 1.0e4      # penalty (g) per server-hour left unfinished at a deadline


def capacity(hours):
    """Servers available for batch work: capacity net of diurnal interactive load."""
    hod = (np.asarray(hours)[:, None] + TZ[None, :]) % 24          # local hour of each site
    bg = 0.20 + 0.15 * np.sin(2 * np.pi * (hod - 8) / 24)
    return CAP[None, :] * (1 - bg)


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
    m_mean = (64 - 4) / np.log(16.0)
    r_mean = 3.5
    avg_cap = capacity(np.arange(24)).sum(axis=1).mean()
    if not TZ.any():
        idx_week = (hh // 24) % 7  # day index relative to data origin (2022-01-01 = Saturday)
        wd = ((idx_week + 5) % 7) < 5  # Monday..Friday
        diurnal = 1 + 0.5 * np.sin(2 * np.pi * ((hh % 24) - 8) / 24)
        shape = diurnal * np.where(wd, 1.0, 0.7)
        lam = rho * avg_cap / (m_mean * r_mean) * shape / shape.mean()
        n = rng.poisson(lam)
        a = np.repeat(hh, n)
        N = len(a)
        o = rng.choice(S, size=N, p=ORIGIN_P)
    else:
        # sites in different time zones: the arrival rate of every origin follows its own local time
        loc = hh[:, None] + TZ[None, :]
        idx_week = (loc // 24) % 7
        wd = ((idx_week + 5) % 7) < 5
        diurnal = 1 + 0.5 * np.sin(2 * np.pi * ((loc % 24) - 8) / 24)
        shape = diurnal * np.where(wd, 1.0, 0.7)                      # [T, S]
        lam = rho * avg_cap / (m_mean * r_mean) * ORIGIN_P[None, :] * shape / shape.mean(axis=0)[None, :]
        n = rng.poisson(lam)                                          # [T, S]
        a = np.repeat(np.repeat(hh[:, None], S, 1).ravel(), n.ravel())
        o = np.repeat(np.tile(np.arange(S), len(hh)), n.ravel())
        N = len(a)
    m = np.exp(rng.uniform(np.log(4), np.log(64), N))
    r = rng.integers(1, 7, N).astype(float)
    f = np.where(rng.random(N) < urgent, 1.0, rng.uniform(1.5, 4.0, N))
    win = np.minimum(np.ceil(r * f), H).astype(int)
    return dict(a=a, o=o, m=m, w=m * r, d=a + win, n=N, ev=(a >= t0) & (a < t0 + hours))



# --------------------------------------------------------------------------
# Min-cost-flow core. Every allocation problem in this study is a transportation
# problem: source -> job (work) -> job-hour (width) -> hour-site (capacity) -> sink,
# plus a job -> sink slack arc priced at the lateness penalty. Work, widths,
# capacities and costs are quantised to an integer grid (0.01 server-hours, 0.01 gCO2)
# and the quantised network is solved to optimality with the OR-Tools min-cost-flow
# solver (cost-scaling push-relabel).
# --------------------------------------------------------------------------
from ortools.graph.python import min_cost_flow

QW, QC = 100.0, 100.0


def flow_solve(rem, width, orig, rel, win, cost, cap, pen, job_weight=None, backstop=None):
    """rel/win: release offset and window length (hours) of each job within the
    planning grid cost[L, o, s], cap[L, s]. Returns (jj, hh, ss, x) triples and the
    slack per job. With `backstop` = (unit cost per hour [L]), every job-hour may also
    run on a backstop resource of unlimited capacity (Assumption A1); the backstop flow
    per job-hour is then returned as a third element (jj_b, hh_b, xb)."""
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
    # forbidden (origin, site) pairs, e.g. migration in the temporal-only regime, carry a
    # prohibitive cost; their arcs are removed so that feasibility never depends on the
    # objective weights below
    keep = c_arc < 1e6
    jh_rep, s_rep, h_rep, c_arc = jh_rep[keep], s_rep[keep], h_rep[keep], c_arc[keep]
    if job_weight is not None:
        c_arc = c_arc * job_weight[jj[jh_rep]]
    tails.append(jh_n[jh_rep]); heads.append(hs_n[h_rep * S + s_rep])
    caps.append(w_int[jj[jh_rep]]); costs.append(np.rint(np.minimum(c_arc, 1e9) * QC).astype(np.int64))
    first_x = sum(len(t) for t in tails)
    tails.append(hs_n); heads.append(np.full(L * S, snk))
    caps.append(np.floor(cap.ravel() * QW).astype(np.int64)); costs.append(np.zeros(L * S, np.int64))
    first_b = sum(len(t) for t in tails)
    if backstop is not None:
        tails.append(jh_n); heads.append(np.full(n_jh, snk)); caps.append(w_int[jj])
        costs.append(np.rint(np.asarray(backstop)[hh] * QC).astype(np.int64))
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
    if backstop is not None:
        xb = mcf.flows(np.arange(first_b, first_b + n_jh)) / QW
        return jj[jh_rep], h_rep, s_rep, fl, slack, (jj, hh, xb)
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


class MPCProtect(Policy):
    """Myopic MPC with a protection level: planned work may use only a share (1-r) of
    every site's capacity; the protected share r of the current hour is released only
    to jobs whose deadline is at most `tight` hours away (revenue-management style)."""

    def __init__(self, r=0.2, tight=2, name=None, clean_only=True):
        self.r, self.tight, self.clean_only = r, tight, clean_only
        self.name = name or f"MPC-Protect({r})"

    def act(self, t, P, view):
        cap = view["cap"]
        prot = np.full(S, self.r)
        if self.clean_only:  # protect only sites cleaner than the fleet median over the window
            local = view["cost_fc"][:, np.arange(S), np.arange(S)].mean(0)
            prot = np.where(local <= np.median(local), self.r, 0.0)
        x = solve_alloc(P["rem"], P["width"], P["orig"], P["dl"], t, view["cost_fc"], cap * (1 - prot[None, :]))
        left = np.maximum(cap[0] - x.sum(0), 0)
        rem2 = P["rem"] - x.sum(1); w2 = P["width"] - x.sum(1)
        tight = [k for k in np.argsort(P["dl"], kind="stable") if P["dl"][k] - t <= self.tight and rem2[k] > 1e-9]
        if tight:
            c0 = view["cost_true"][0]
            pref = [[s for s in np.argsort(c0[P["orig"][j]]) if c0[P["orig"][j], s] < 1e6]
                    for j in range(len(rem2))]
            x = x + _edf_fill(tight, rem2, w2, left, pref)
        # protected capacity that urgent work did not claim perishes at the end of the
        # hour, so it is released to the remaining work (earliest deadline first)
        rem3 = P["rem"] - x.sum(1); w3 = P["width"] - x.sum(1)
        c0 = view["cost_true"][0]
        rest = [k for k in np.argsort(P["dl"], kind="stable") if rem3[k] > 1e-9]
        if rest and left.sum() > 1e-9:
            pref = [[s for s in np.argsort(c0[P["orig"][j]]) if c0[P["orig"][j], s] < 1e6 and prot[s] > 0]
                    for j in range(len(rem3))]
            x = x + _edf_fill(rest, rem3, np.maximum(w3, 0), left, pref)
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
def simulate(policy, jobs, t0, truth, cube, migrate=True, acct=None):
    """Run `policy` on the job stream. cube[t, s, h] is the CI view at issue time t
    for hour t+h (h=0 is the observed current hour). Metrics cover jobs flagged ev.
    `acct` (default: truth) is the intensity used to account emissions, e.g. a
    marginal emission factor, while decisions are always based on `truth`/`cube`."""
    t_first, t_end = int(jobs["a"].min()), int(jobs["d"].max()) + 1
    rem = jobs["w"].copy()
    emis_j = np.zeros(jobs["n"]); mig_j = np.zeros(jobs["n"]); late_j = np.zeros(jobs["n"])
    emis_host = np.zeros(S)
    policy.reset(jobs, t0)
    ct_all = unit_cost(truth, migrate)  # [T, o, s]
    ca_all = ct_all if acct is None else unit_cost(acct, migrate)
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
        # feasibility of the executed allocation (tolerance: the 0.01 server-hour grid)
        assert np.all(x.sum(0) <= view["cap"][0] + 0.05), "capacity exceeded"
        assert np.all(x.sum(1) <= P["width"] + 0.05), "width exceeded"
        rem[ids] = np.maximum(rem[ids] - x.sum(1), 0)
        rem[rem < DONE_TOL] = 0.0  # ignore residue from the 0.01 server-hour flow grid
        step = ct_all[t][jobs["o"][ids]]
        assert not np.any((x > 1e-6) & (step >= 1e6)), "migration used when disallowed"
        xe = x * ca_all[t][jobs["o"][ids]]
        emis_j[ids] += xe.sum(1)
        emis_host += xe[jobs["ev"][ids]].sum(0)
        mig_j[ids] += (x * (1 - np.eye(S))[jobs["o"][ids]]).sum(1)
        # work still outstanding when the deadline passes is recorded as late
        miss = ids[(jobs["d"][ids] == t + 1) & (rem[ids] > DONE_TOL)]
        late_j[miss] = rem[miss]
    ev = jobs["ev"]; total = jobs["w"][ev].sum()
    by_o = [float(emis_j[ev & (jobs["o"] == o)].sum() / 1e6) for o in range(S)]
    c_late = late_unit_cost(ca_all[t_first:t_end], migrate)
    return dict(emis_by_origin=by_o, emis_by_host=list(emis_host / 1e6), emis_t=emis_j[ev].sum() / 1e6, late_frac=late_j[ev].sum() / total,
                emis_adj_t=(emis_j[ev].sum() + c_late * late_j[ev].sum()) / 1e6,
                late_jobs=float((late_j[ev] > DONE_TOL).mean()),
                unfinished=float(rem[ev].sum()) / total, mig_frac=mig_j[ev].sum() / total,
                work=total, jobs=int(ev.sum()), ms_per_decision=1e3 * wall / max(n_dec, 1))


def late_unit_cost(ct, migrate=True):
    """Charge for late work in the service-adjusted metric: the largest local unit
    cost in the episode (late work is valued as if run at the dirtiest capacity)."""
    return float(ct[:, np.arange(S), np.arange(S)].max())


def oracle(jobs, t0, truth, migrate=True, acct=None):
    """Offline clairvoyant plan (known arrivals and CI). All jobs of the episode must be
    served by their deadlines where capacity allows (unserved work is penalised at BIG),
    but only evaluated jobs (flag ev) carry emission costs in the objective, so the value
    is a lower bound on the evaluated emissions of any schedule that serves every job of
    the episode by its deadline. Migration restrictions are enforced as feasibility
    constraints (see flow_solve)."""
    tf = int(jobs["a"].min()); L = int(jobs["d"].max()) - tf
    ct = unit_cost((truth if acct is None else acct)[tf:tf + L], migrate)
    cap = capacity(np.arange(tf, tf + L))
    ev = jobs["ev"]
    jj, hh, ss, x, u = flow_solve(jobs["w"], jobs["m"], jobs["o"], jobs["a"] - tf,
                                  jobs["d"] - jobs["a"], ct, cap, np.full(jobs["n"], BIG),
                                  job_weight=ev.astype(float))
    total = jobs["w"][ev].sum(); k = ev[jj]
    e = float((x * ct[hh, jobs["o"][jj], ss])[k].sum())
    # benchmark for the service-adjusted metric: evaluated work may be left undone at the
    # same unit charge c_late that the metric applies to late work (boundary work must
    # still be served on time). Any schedule whose boundary work is on time has an
    # on-time part that is feasible here, so this value bounds its adjusted emissions.
    c_late = late_unit_cost(ct, migrate)
    jj2, hh2, ss2, x2, u2 = flow_solve(jobs["w"], jobs["m"], jobs["o"], jobs["a"] - tf,
                                       jobs["d"] - jobs["a"], ct, cap, np.where(ev, c_late, BIG),
                                       job_weight=ev.astype(float))
    e2 = float((x2 * ct[hh2, jobs["o"][jj2], ss2])[ev[jj2]].sum())
    return dict(emis_t=e / 1e6, emis_adj_t=(e2 + c_late * float(u2[ev].sum())) / 1e6,
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
        # ghost jobs are made window-feasible (Assumption A1): width >= work / window
        width[J:] = np.maximum(width[J:], rem[J:] / np.maximum(dl[J:] - rel[J:], 1))
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


# --------------------------------------------------------------------------
# Hindsight LP with duals (cooperative carbon attribution, Section 3.6)
# --------------------------------------------------------------------------
def hindsight_lp(jobs, t0, truth, coalition=None, migrate=True, want_range=False):
    """Offline LP (2)-(5) for the sites in `coalition` (jobs originating there,
    capacity of those sites only). Returns value (tCO2, evaluated jobs only when
    coalition is the grand coalition) and the per-site dual attribution phi[o]."""
    import scipy.sparse as sp
    from scipy.optimize import linprog
    coalition = list(range(S)) if coalition is None else list(coalition)
    inS = np.isin(np.arange(S), coalition)
    tf = int(jobs["a"].min()); L = int(jobs["d"].max()) - tf
    ct = unit_cost(truth[tf:tf + L], migrate)
    cap = capacity(np.arange(tf, tf + L)) * inS[None, :]
    J = jobs["n"]
    w = np.where(inS[jobs["o"]], jobs["w"], 0.0)
    m = np.where(inS[jobs["o"]], jobs["m"], 0.0)
    a = jobs["a"] - tf; win = jobs["d"] - jobs["a"]
    jj = np.repeat(np.arange(J), win * S)
    hh = np.concatenate([np.repeat(a[j] + np.arange(win[j]), S) for j in range(J)])
    ss = np.tile(np.arange(S), win.sum())
    nv = len(jj)
    c = np.minimum(ct[hh, jobs["o"][jj], ss], 1e6)
    cobj = np.r_[c, np.full(J, BIG)]
    Aeq = sp.csr_matrix((np.ones(nv + J), (np.r_[jj, np.arange(J)], np.arange(nv + J))), shape=(J, nv + J))
    A1 = sp.csr_matrix((np.ones(nv), (hh * S + ss, np.arange(nv))), shape=(L * S, nv + J))
    jh = np.r_[0, np.cumsum(win)][jj] + (hh - a[jj])
    A2 = sp.csr_matrix((np.ones(nv), (jh, np.arange(nv))), shape=(int(win.sum()), nv + J))
    b1 = cap.ravel(); b2 = m[np.repeat(np.arange(J), win)]
    res = linprog(cobj, A_ub=sp.vstack([A1, A2]).tocsr(), b_ub=np.r_[b1, b2], A_eq=Aeq, b_eq=w,
                  bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    lam = res.eqlin.marginals                      # work duals (free)
    mu = res.ineqlin.marginals[:L * S].reshape(L, S)  # capacity duals (<= 0)
    nu = res.ineqlin.marginals[L * S:]             # width duals (<= 0)
    nu_job = np.zeros(J); np.add.at(nu_job, np.repeat(np.arange(J), win), nu)
    phi = np.zeros(S)
    for o in coalition:
        sel = jobs["o"] == o
        phi[o] = (w[sel] * lam[sel]).sum() + (m[sel] * nu_job[sel]).sum() + (cap[:, o] * mu[:, o]).sum()
    x = res.x[:nv]
    phys = np.array([(x * c)[jobs["o"][jj] == o].sum() for o in range(S)])   # origin-based
    host = np.array([(x * c)[ss == o].sum() for o in range(S)])              # host-based
    slack = float(res.x[nv:].sum())
    out = dict(value=res.fun / 1e6, emis=float((x * c).sum()) / 1e6, slack=slack,
               phi=phi / 1e6, phys=phys / 1e6, host=host / 1e6)
    if want_range:
        out["phi_range"] = dual_range(cobj, sp.vstack([A1, A2]).tocsr(), Aeq, np.r_[b1, b2], w,
                                      res.fun, jobs, cap, m, win, coalition) / 1e6
    return out


def dual_range(cobj, Aub, Aeq, bub, beq, value, jobs, cap, m, win, coalition, tol=1e-6):
    """Range of each site's dual attribution over the set of optimal dual solutions:
    min/max (b^o)^T y s.t. A^T y <= c, y_ub <= 0, b^T y >= value (1 - tol)."""
    import scipy.sparse as sp
    from scipy.optimize import linprog
    J = Aeq.shape[0]; K = Aub.shape[0]
    AT = sp.hstack([Aeq.T, Aub.T]).tocsr()           # (nv+J) x (J+K)
    b = np.r_[beq, bub]
    bounds = [(None, None)] * J + [(None, 0)] * K
    L = cap.shape[0]
    rng = np.zeros((S, 2))
    rows_w = np.repeat(np.arange(J), win)
    for o in coalition:
        bo = np.zeros(J + K)
        sel = jobs["o"] == o
        bo[:J] = np.where(sel, beq, 0.0)
        capo = np.zeros((L, S)); capo[:, o] = cap[:, o]
        bo[J:J + L * S] = capo.ravel()
        bo[J + L * S:] = np.where(sel[rows_w], bub[L * S:], 0.0)
        for k, sgn in enumerate([1.0, -1.0]):
            r = linprog(sgn * bo, A_ub=sp.vstack([AT, sp.csr_matrix(-b)]).tocsr(),
                        b_ub=np.r_[cobj, -value * (1 - tol)], bounds=bounds, method="highs")
            rng[o, k] = sgn * r.fun if r.status == 0 else np.nan
    return rng


# --------------------------------------------------------------------------
# Scenario-based two-stage lookahead (demand-aware comparator)
# --------------------------------------------------------------------------
class ScenarioMPC(Policy):
    """Two-stage stochastic lookahead with the same demand information as CARMA. At
    each hour, K arrival scenarios for the next `look` hours are drawn from the
    historical job traces from which CARMA's profile is estimated (the jobs of one
    historical week at the same hours of the week). The first-hour allocation of the
    pending real jobs is shared by all scenarios; their future allocation and the
    scenario jobs are planned separately in each scenario, and the expected cost is
    minimised (sample-average approximation, solved as an LP with HiGHS). Scenario
    jobs that cannot be served incur the same soft penalty as CARMA's ghost jobs."""

    def __init__(self, hist_jobs, hist_t0s, K=4, look=12, seed=0, name="Scenario-MPC", ghost_pen=None,
                 period=168):
        self.K, self.look, self.name, self.ghost_pen, self.period = K, look, name, ghost_pen, period
        self.rng = np.random.default_rng(seed)
        self.hist = []
        for jobs, t0 in zip(hist_jobs, hist_t0s):
            ev = jobs["ev"]
            how = (jobs["a"][ev] - t0) % period
            self.hist.append(dict(how=how, o=jobs["o"][ev], w=jobs["w"][ev], m=jobs["m"][ev],
                                  win=(jobs["d"] - jobs["a"])[ev]))

    def reset(self, jobs, t0):
        self.t0 = t0

    def scenario(self, i, t):
        h = self.hist[i]
        rel = (h["how"] - (t - self.t0)) % self.period   # arrival offset from now
        sel = (rel >= 1) & (rel <= self.look)
        return rel[sel], h["o"][sel], h["w"][sel], h["m"][sel], h["win"][sel]

    def act(self, t, P, view):
        import scipy.sparse as sp
        from scipy.optimize import linprog
        cost, cap = view["cost_fc"], view["cap"]
        L = cost.shape[0]
        J = len(P["rem"])
        rwin = np.clip(P["dl"] - t, 1, L)
        gp = self.ghost_pen if self.ghost_pen is not None else 1.5 * cost[:, np.arange(S), np.arange(S)].max()
        scen = [self.scenario(i, t) for i in self.rng.choice(len(self.hist), self.K, replace=False)]
        cols_c, rows, cols, vals = [], [], [], []
        b_eq, b_ub = [], []
        n_eq = n_ub = 0
        eq_r, eq_c, ub_r, ub_c = [], [], [], []
        ncol = 0

        def add_cols(c):
            nonlocal ncol
            c = np.asarray(c, float); idx = ncol + np.arange(len(c)); ncol += len(c); cols_c.append(c)
            return idx

        # first-stage variables: real job j at site s in hour 0
        j0 = np.repeat(np.arange(J), S); s0 = np.tile(np.arange(S), J)
        c0 = cost[0, P["orig"][j0], s0]
        ok0 = c0 < 1e6; j0, s0, c0 = j0[ok0], s0[ok0], c0[ok0]
        x0 = add_cols(c0)
        # shared hour-0 width and capacity rows
        ub_r += [n_ub + j0, n_ub + J + s0]; ub_c += [x0, x0]
        b_ub += [P["width"], cap[0]]; n_ub += J + S
        for (grel, go, gw, gm, gwin) in scen:
            G = len(gw)
            gwin = np.minimum(gwin, L - grel)
            gm = np.maximum(gm, gw / np.maximum(gwin, 1))
            # real jobs, hours 1..win-1
            rj = np.repeat(np.arange(J), np.maximum(rwin - 1, 0) * S)
            rh = np.concatenate([np.repeat(np.arange(1, rwin[j]), S) for j in range(J)]) if J else np.zeros(0, int)
            rs = np.tile(np.arange(S), int(np.maximum(rwin - 1, 0).sum()))
            rc = cost[rh, P["orig"][rj], rs]
            okr = rc < 1e6; rj, rh, rs, rc = rj[okr], rh[okr], rs[okr], rc[okr]
            y = add_cols(rc / self.K)
            # scenario jobs
            gj = np.repeat(np.arange(G), gwin * S)
            gh = np.concatenate([np.repeat(grel[g] + np.arange(gwin[g]), S) for g in range(G)]) if G else np.zeros(0, int)
            gs = np.tile(np.arange(S), int(gwin.sum()))
            gc = cost[gh, go[gj], gs]
            okg = gc < 1e6; gj, gh, gs, gc = gj[okg], gh[okg], gs[okg], gc[okg]
            z = add_cols(gc / self.K)
            u = add_cols(np.full(J, BIG / self.K))
            v = add_cols(np.full(G, gp / self.K))
            # work equalities
            eq_r += [n_eq + j0, n_eq + rj, n_eq + np.arange(J), n_eq + J + gj, n_eq + J + np.arange(G)]
            eq_c += [x0, y, u, z, v]
            b_eq += [P["rem"], gw]; n_eq += J + G
            # width rows (real hours >= 1, scenario hours), capacity rows (hours >= 1)
            wr = n_ub + rj * L + rh
            wg = n_ub + J * L + gj * L + gh
            ub_r += [wr, wg]; ub_c += [y, z]
            b_ub += [np.repeat(P["width"], L), np.repeat(gm, L)]; n_ub += J * L + G * L
            ub_r += [n_ub + rh * S + rs, n_ub + gh * S + gs]; ub_c += [y, z]
            b_ub += [cap.ravel()]; n_ub += L * S
        c = np.concatenate(cols_c)
        Aeq = sp.csr_matrix((np.ones(sum(len(a) for a in eq_c)), (np.concatenate(eq_r), np.concatenate(eq_c))),
                            shape=(n_eq, ncol))
        Aub = sp.csr_matrix((np.ones(sum(len(a) for a in ub_c)), (np.concatenate(ub_r), np.concatenate(ub_c))),
                            shape=(n_ub, ncol))
        res = linprog(c, A_ub=Aub, b_ub=np.concatenate(b_ub), A_eq=Aeq, b_eq=np.concatenate(b_eq),
                      bounds=(0, None), method="highs")
        if res.status != 0:
            raise RuntimeError(res.message)
        x = np.zeros((J, S))
        np.add.at(x, (j0, s0), res.x[x0])
        return x
