"""Numerical check of Theorem 2 in a setting that satisfies its assumptions.

(A1) a backstop resource of unlimited capacity is available in every hour at the
largest unit cost U of the episode; all jobs are window-feasible; real jobs carry the
penalty BIG > U and ghost jobs the penalty U. (A2) each episode is one day (T = 24
hours = H), so every planning problem reaches the end of the episode.

Ghost jobs are the true future jobs of the episode, perturbed in two ways: each future
job is omitted from the predictions with probability q (demand error), and the
carbon-intensity forecast of every future hour receives i.i.d. Gaussian noise with
standard deviation sigma (carbon error). With q = sigma = 0 the predictions are exact.
For every episode we record ALG, OPT, the lateness of real work and the right-hand
side of the smoothness bound, OPT + sum_t [(U - L) d_t + 2 eps_t W_t], where d_t is the
omitted future work visible at t, eps_t the largest future unit-cost error and W_t the
pending work plus the true future work."""
import itertools, multiprocessing as mp
import numpy as np
import pandas as pd
from experiment import TEST, TRUTH, RES
import sim
from sim import S, BIG, capacity, unit_cost, flow_solve, make_jobs

T_EP = 24


def episode_jobs(t0, rho, seed):
    j = make_jobs(t0, rho, seed, hours=T_EP, pre=0, post=0)
    a = j["a"] - t0
    d = np.minimum(j["d"] - t0, T_EP)
    m = np.maximum(j["m"], j["w"] / (d - a))          # window-feasible after truncation
    return dict(a=a, d=d, w=j["w"], m=m, o=j["o"], n=j["n"])


def run(args):
    t0, rho, q, sigma, seed = args
    rng = np.random.default_rng(seed)
    J = episode_jobs(t0, rho, seed)
    ci = TRUTH[t0:t0 + T_EP]
    ct = unit_cost(ci)                                   # [T, o, s]
    cap = capacity(np.arange(t0, t0 + T_EP))
    U = float(ct.max()); L = float(ct.min())
    back = np.full(T_EP, U)
    # clairvoyant optimum with backstop
    out = flow_solve(J["w"], J["m"], J["o"], J["a"], J["d"] - J["a"], ct, cap, np.full(J["n"], BIG), backstop=back)
    jj, hh, ss, x, u, (bj, bh, xb) = out
    opt = float((x * ct[hh, J["o"][jj], ss]).sum() + U * xb.sum() + BIG * u.sum())
    keep = rng.random(J["n"]) >= q                       # jobs that CARMA predicts
    rem = J["w"].copy(); alg = 0.0; bound = opt; late = 0.0; ghost_slack = 0.0
    for t in range(T_EP):
        Lh = T_EP - t
        real = np.where((J["a"] <= t) & (rem > 1e-9))[0]
        fut = np.where(J["a"] > t)[0]
        ghost = fut[keep[fut]]
        noise = rng.normal(0, sigma, (Lh, S)) if sigma > 0 else np.zeros((Lh, S))
        noise[0] = 0.0                                   # the current hour is observed
        chat = unit_cost(np.maximum(ci[t:] + noise, 0))
        eps = float(np.abs(chat[1:] - ct[t + 1:]).max()) if Lh > 1 else 0.0
        d_t = float(J["w"][np.setdiff1d(fut, ghost)].sum())
        W_t = float(rem[real].sum() + J["w"][fut].sum())
        bound += (U - L) * d_t + 2 * eps * W_t
        if len(real) == 0:
            continue
        idx = np.r_[real, ghost]
        r_ = np.r_[rem[real], J["w"][ghost]]
        rel = np.r_[np.zeros(len(real), int), J["a"][ghost] - t]
        win = np.r_[J["d"][real] - t, J["d"][ghost] - J["a"][ghost]]
        pen = np.r_[np.full(len(real), BIG), np.full(len(ghost), U)]
        jj, hh, ss, x, u, (bj, bh, xb) = flow_solve(r_, J["m"][idx], J["o"][idx], rel, win, chat,
                                                    cap[t:], pen, backstop=back[t:])
        ghost_slack += float(u[len(real):].sum())
        # execute the first hour for real jobs (sites and backstop)
        f = (hh == 0) & (jj < len(real))
        fb = (bh == 0) & (bj < len(real))
        done = np.zeros(len(real))
        np.add.at(done, jj[f], x[f]); np.add.at(done, bj[fb], xb[fb])
        alg += float((x[f] * ct[t, J["o"][idx][jj[f]], ss[f]]).sum() + U * xb[fb].sum())
        rem[real] = np.maximum(rem[real] - done, 0)
        rem[rem < 0.02] = 0.0                            # residue of the 0.01 server-hour grid
        due = real[(J["d"][real] == t + 1) & (rem[real] > 0)]
        late += float(rem[due].sum()); rem[due] = 0.0
    return dict(t0=t0, rho=rho, q=q, sigma=sigma, opt=opt / 1e6, alg=alg / 1e6, bound=bound / 1e6,
                late=late, work=float(J["w"].sum()), U=U, L=L)


if __name__ == "__main__":
    specs = [(t0 + 24 * day, rho, q, sg, 1000 * i + day)
             for i, t0 in enumerate(TEST[::4]) for day in (0, 3)
             for rho in (0.5, 0.9) for q, sg in [(0, 0), (0.1, 0), (0.3, 0), (0, 25), (0, 100), (0.3, 100)]]
    with mp.Pool(4) as pool:
        rows = pool.map(run, specs, chunksize=2)
    d = pd.DataFrame(rows)
    d["ratio"] = d.alg / d.opt
    d["holds"] = d.alg <= d.bound * (1 + 1e-6) + 1e-9
    d.to_csv(RES / "verify_theorem2.csv", index=False)
    s = d.groupby(["rho", "q", "sigma"]).agg(episodes=("ratio", "size"), mean_ratio=("ratio", "mean"),
                                             max_ratio=("ratio", "max"), bound_holds=("holds", "mean"),
                                             bound_over_opt=("bound", lambda b: (b / d.loc[b.index, "opt"]).mean()),
                                             late=("late", "max"))
    s.to_csv(RES / "verify_theorem2_summary.csv")
    print(s.round(5).to_string())
