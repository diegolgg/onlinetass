"""Sections 5.1 and 6.1 of the paper.

Changepoint staleness synthetic task: three-state Gaussian HMM with a
single post-changepoint mean shift on the rare-state(s).  Compares six
samplers via SGLD posterior chains and held-out log-predictive density.
Writes CSV summaries to the current working directory.

Run from the repository root: python examples/changepoint_synthetic.py
"""
import sys
from pathlib import Path
# Make src/ importable when this notebook is run from examples/.
_root = Path.cwd()
if (_root / 'src').is_dir() is False and (_root.parent / 'src').is_dir():
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import sys
from pathlib import Path
# Make src/ importable when this notebook is run from examples/.
_root = Path.cwd()
if (_root / 'src').is_dir() is False and (_root.parent / 'src').is_dir():
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None


def normalize(x):
    x = np.asarray(x, dtype=float)
    s = np.sum(x)
    if (not np.isfinite(s)) or s <= 0:
        return np.ones_like(x, dtype=float) / x.size
    return x / s


def normalize_columns(A):
    A = np.asarray(A, dtype=float)
    A = np.abs(A)
    s = np.sum(A, axis=0, keepdims=True)
    s[s <= 0] = 1.0
    return A / s


def paper_transition(kind="one_rare"):
    if kind == "one_rare":
        A = np.array([[0.990, 0.005, 0.4950],
                      [0.005, 0.990, 0.4950],
                      [0.005, 0.005, 0.0100]], dtype=float)
        mu = np.array([-20.0, 0.0, 20.0])
    elif kind == "two_rare":
        A = np.array([[0.9990, 0.1000, 0.1000],
                      [0.0005, 0.9000, 0.0000],
                      [0.0005, 0.0000, 0.9000]], dtype=float)
        mu = np.array([0.0, -20.0, 20.0])
    elif kind == "no_rare":
        A = np.array([[0.990, 0.005, 0.005],
                      [0.005, 0.990, 0.005],
                      [0.005, 0.005, 0.990]], dtype=float)
        mu = np.array([-20.0, 0.0, 20.0])
    else:
        raise ValueError("kind must be one_rare, two_rare, or no_rare")
    return A, mu, np.ones(3)


def stationary_distribution(A, max_iter=100000, tol=1e-14):
    K = A.shape[0]
    p = np.ones(K) / K
    for _ in range(max_iter):
        q = A @ p
        q = normalize(q)
        if np.max(np.abs(q - p)) < tol:
            return q
        p = q
    return p


def simulate_hmm(A, mu, sigmasq, T, pi0=None, seed=1):
    rng = np.random.default_rng(seed)
    K = len(mu)
    if pi0 is None:
        pi0 = np.ones(K) / K
    z = np.zeros(T, dtype=int)
    y = np.zeros(T, dtype=float)
    z[0] = rng.choice(K, p=pi0)
    for t in range(1, T):
        z[t] = rng.choice(K, p=A[:, z[t - 1]])
    for t in range(T):
        y[t] = rng.normal(mu[z[t]], np.sqrt(sigmasq[z[t]]))
    return z, y


def make_paper_scenario(kind="one_rare", Ttrain=10**6, Ttest=10**6, seed=1):
    A, mu, sigmasq = paper_transition(kind)
    pi0 = np.ones(3) / 3.0
    z_all, y_all = simulate_hmm(A, mu, sigmasq, Ttrain + Ttest, pi0=pi0, seed=seed)
    return {
        "kind": kind,
        "A": A,
        "mu": mu,
        "sigmasq": sigmasq,
        "pi0": pi0,
        "z": z_all[:Ttrain],
        "y": y_all[:Ttrain],
        "ztest": z_all[Ttrain:],
        "ytest": y_all[Ttrain:],
    }


def random_transition_parameter(K, pi0, rng):
    A = rng.random((K, K))
    A_hat = rng.random((K, K))
    A = normalize_columns(A)
    A_hat = normalize_columns(A_hat)
    return A, A_hat, pi0.copy()


def matlab_initial_parameters(K, seed=1, sigmasq_init=10.0, pi0_init=None):
    rng = np.random.default_rng(seed)
    if pi0_init is None:
        pi0_init = normalize(rng.random(K))
    A, A_hat, pi0 = random_transition_parameter(K, pi0_init, rng)
    mu = np.zeros(K, dtype=float)
    sigmasq = sigmasq_init * np.ones(K, dtype=float)
    return mu, sigmasq, A, A_hat, pi0


def density_vector(yt, mu, sigmasq):
    sigmasq = np.maximum(sigmasq, 1e-300)
    return np.exp(-0.5 * (np.log(2.0 * np.pi * sigmasq) + (yt - mu) ** 2 / sigmasq))


def approx_pred_matlab(y, t, B, A, pi0, mu, sigmasq):
    T = len(y)
    K = len(mu)
    pipred = pi0.astype(float).copy()
    qmat = np.eye(K)

    if t < B:
        left = range(0, t)
        right = range(t + 1, min(t + B + 1, T))
    elif t > T - B - 2:
        left = range(max(0, t - B), t)
        right = range(t + 1, T)
    else:
        left = range(t - B, t)
        right = range(t + 1, t + B + 1)

    for i in left:
        P = density_vector(y[i], mu, sigmasq)
        pipred = P * (A @ pipred)

    for i in right:
        P = density_vector(y[i], mu, sigmasq)
        qmat = (P[:, None] * A) @ qmat

    qpred = qmat.T @ np.ones(K)
    return pipred, qpred


def block_gradients_matlab(y, block, L, B, A, pi0, mu, sigmasq):
    K = len(mu)
    ll = 2 * L + 1
    lo = block * ll
    hi = min((block + 1) * ll, len(y))
    gmu = np.zeros(K)
    gs2 = np.zeros(K)
    gA = np.zeros((K, K))

    for t in range(lo, hi):
        P = density_vector(y[t], mu, sigmasq)
        pipred, qpred = approx_pred_matlab(y, t, B, A, pi0, mu, sigmasq)
        Apipred = A @ pipred
        den = float(qpred @ (P * Apipred))
        if den <= 0 or not np.isfinite(den):
            den = 1e-300
        resid = y[t] - mu
        gmu += P * qpred * Apipred * resid / (den * sigmasq)
        gs2 += 0.5 * P * qpred * Apipred * (-sigmasq + resid ** 2) / (den * sigmasq ** 2)
        gA += (P * qpred)[:, None] * pipred[None, :] / den

    return gmu, gs2, gA


def prior_gradients(mu, sigmasq, A_hat, mu_prior_var=100.0, ig_a=3.0, ig_b=10.0, alpha=1.0):
    gmu = (0.0 - mu) / mu_prior_var
    gs2 = (ig_a + 1.0) / sigmasq - ig_b / (sigmasq ** 2)
    gA = (alpha - 1.0) / np.maximum(A_hat, 1e-300)
    return gmu, gs2, gA


def kmeans_labels_matlab_style(y, K, seed=1, sort_centres=False):
    if KMeans is None:
        qs = np.quantile(y, np.linspace(0.0, 1.0, K + 1)[1:-1])
        labels = np.digitize(y, qs)
        centers = np.array([np.mean(y[labels == k]) if np.any(labels == k) else 0.0 for k in range(K)])
    else:
        km = KMeans(n_clusters=K, max_iter=1000, n_init=1, random_state=seed)
        labels = km.fit_predict(y.reshape(-1, 1))
        centers = km.cluster_centers_.ravel()
    if sort_centres:
        order = np.argsort(centers)
        remap = np.zeros(K, dtype=int)
        for new, old in enumerate(order):
            remap[old] = new
        labels = remap[labels]
        centers = centers[order]
    return labels.astype(int), centers


def weights_by_clustering_matlab(y, K, L, eta=1e-5, seed=1, sort_centres=True):
    # This mirrors weights_by_clustering.m / weights_alex_clus_leaveoneout.m.
    z, centers = kmeans_labels_matlab_style(y, K, seed=seed, sort_centres=sort_centres)
    ll = 2 * L + 1
    N = len(y) // ll
    y = y[:N * ll]
    z = z[:N * ll]

    ybar = np.zeros(K)
    s2 = np.zeros(K)
    for j in range(K):
        vals = y[z == j]
        ybar[j] = np.mean(vals) if len(vals) else 0.0
        s2[j] = np.var(vals, ddof=1) if len(vals) > 1 else 0.0

    c = np.zeros((N, K))
    ybar_chunk = np.zeros((N, K))
    s2_chunk = np.zeros((N, K))
    for n in range(N):
        idx = slice(n * ll, (n + 1) * ll)
        yn = y[idx]
        zn = z[idx]
        for j in range(K):
            mask = zn == j
            c[n, j] = np.sum(mask)
            if c[n, j] > 0:
                vals = yn[mask]
                ybar_chunk[n, j] = np.mean(vals)
                s2_chunk[n, j] = np.sum((vals - ybar[j]) ** 2) / c[n, j]

    wmu = np.zeros((K, N))
    ws2 = np.zeros((K, N))
    for j in range(K):
        wmu[j] = c[:, j] * np.abs(ybar_chunk[:, j] - ybar[j]) + eta ** 2
        ws2[j] = c[:, j] * (s2[j] + s2_chunk[:, j]) + eta ** 2
        wmu[j] = normalize(wmu[j])
        ws2[j] = normalize(ws2[j])

    xi = np.zeros((N, K, K))
    for n in range(N):
        idx = slice(n * ll, (n + 1) * ll)
        zn = z[idx]
        for old in range(K):
            for new in range(K):
                xi[n, old, new] = np.sum((zn[:-1] == old) & (zn[1:] == new))

    xisum = np.sum(xi, axis=0) + 0.5
    A_map = normalize_columns(xisum.T)
    wA = np.zeros((K, K, N))
    for new in range(K):
        for old in range(K):
            w = xi[:, old, new] / max(A_map[new, old], 1e-300) + 1e-10
            wA[new, old] = normalize(w)

    return {"mu": wmu, "sigmasq": ws2, "A": wA, "labels": z, "centers": centers, "A_map": A_map}


def uniform_weights(K, N):
    return {"mu": np.ones((K, N)) / N,
            "sigmasq": np.ones((K, N)) / N,
            "A": np.ones((K, K, N)) / N}


def transition_grad_IS(y, weights, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng):
    K = len(mu)
    N = len(y) // (2 * L + 1)
    _, _, gA = prior_gradients(mu, sigmasq, A_hat)
    cache = {}
    sampled = []

    def get_block(i):
        if i not in cache:
            cache[i] = block_gradients_matlab(y, i, L, B, A, pi0, mu, sigmasq)
        return cache[i]

    for j in range(K):
        for k in range(K):
            w = weights["A"][j, k]
            mb = rng.choice(N, size=mb_size, replace=True, p=w)
            for i in mb:
                _, _, bA = get_block(int(i))
                gA[j, k] -= bA[j, k] / max(w[i], 1e-300)
                sampled.append(("A", j, k, int(i)))
    gA /= mb_size
    return gA, sampled


def emission_grad_IS(y, weights, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng):
    K = len(mu)
    N = len(y) // (2 * L + 1)
    gmu, gs2, _ = prior_gradients(mu, sigmasq, A_hat)
    cache = {}
    sampled = []

    def get_block(i):
        if i not in cache:
            cache[i] = block_gradients_matlab(y, i, L, B, A, pi0, mu, sigmasq)
        return cache[i]

    for j in range(K):
        w = weights["mu"][j]
        mb = rng.choice(N, size=mb_size, replace=True, p=w)
        for i in mb:
            bmu, _, _ = get_block(int(i))
            gmu[j] -= bmu[j] / max(w[i], 1e-300)
            sampled.append(("mu", j, None, int(i)))

        w = weights["sigmasq"][j]
        mb = rng.choice(N, size=mb_size, replace=True, p=w)
        for i in mb:
            _, bs2, _ = get_block(int(i))
            gs2[j] -= bs2[j] / max(w[i], 1e-300)
            sampled.append(("sigmasq", j, None, int(i)))

    gmu /= mb_size
    gs2 /= mb_size
    return gmu, gs2, sampled


def sgld_update_transition(A_hat, gA, eps, rng):
    A_hat = np.abs(A_hat - eps * (A_hat * gA + 1.0) + np.sqrt(np.maximum(2.0 * eps * A_hat, 1e-300)) * rng.normal(size=A_hat.shape))
    A = normalize_columns(A_hat)
    return A, A_hat


def sgld_update_emission(mu, sigmasq, gmu, gs2, eps, rng):
    old_sigmasq = sigmasq.copy()
    sd = np.sqrt(np.maximum(eps * (2.0 * sigmasq), 1e-300))
    mu = mu - eps * sigmasq * gmu + sd * rng.normal(size=len(mu))
    sigmasq = sigmasq - eps * (sigmasq ** 2 * gs2 + sigmasq) + sd * rng.normal(size=len(mu))
    if np.sum(sigmasq > 0) < len(mu) or np.any(~np.isfinite(sigmasq)):
        sigmasq = old_sigmasq
    return mu, sigmasq


def run_csg_mcmc_is_matlab(y, method="tass", K=3, L=2, B=5, eps=1e-6, n_mcmc=5000,
                           mb_size=10, eta=1e-5, seed=1, init_seed=None,
                           pi0_init=None, sigmasq_init=10.0, sort_centres=True,
                           checkpoint_every=None):
    rng = np.random.default_rng(seed)
    if init_seed is None:
        init_seed = seed + 1000003
    init_rng = np.random.default_rng(init_seed)
    if pi0_init is None:
        pi0_init = normalize(init_rng.random(K))
    mu, sigmasq, A, A_hat, pi0 = matlab_initial_parameters(K, seed=init_seed, sigmasq_init=sigmasq_init, pi0_init=pi0_init)

    ll = 2 * L + 1
    N = len(y) // ll
    y = y[:N * ll]
    if method == "uniform":
        weights = uniform_weights(K, N)
    elif method == "tass":
        weights = weights_by_clustering_matlab(y, K, L, eta=eta, seed=seed, sort_centres=sort_centres)
    else:
        raise ValueError("method must be 'uniform' or 'tass' in exact MATLAB mode")

    chain_mu = np.zeros((n_mcmc + 1, K))
    chain_s2 = np.zeros((n_mcmc + 1, K))
    chain_A = np.zeros((n_mcmc + 1, K, K))
    chain_A_hat = np.zeros((n_mcmc + 1, K, K))
    chain_mu[0] = mu
    chain_s2[0] = sigmasq
    chain_A[0] = A
    chain_A_hat[0] = A_hat

    for itr in range(1, n_mcmc + 1):
        gA, _ = transition_grad_IS(y, weights, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng)
        A, A_hat = sgld_update_transition(A_hat, gA, eps, rng)

        gmu, gs2, _ = emission_grad_IS(y, weights, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng)
        mu, sigmasq = sgld_update_emission(mu, sigmasq, gmu, gs2, eps, rng)

        if itr % 10 == 0:
            A_hat = normalize_columns(A_hat)
            A = normalize_columns(A_hat)

        chain_mu[itr] = mu
        chain_s2[itr] = sigmasq
        chain_A[itr] = A
        chain_A_hat[itr] = A_hat

        if checkpoint_every is not None and (itr % checkpoint_every == 0 or itr == 1):
            print(f"{method}: {itr}/{n_mcmc}, mu={np.round(mu, 3).tolist()}, sigmasq={np.round(sigmasq, 3).tolist()}", flush=True)

    return {"method": method, "mu": chain_mu, "sigmasq": chain_s2, "A": chain_A, "A_hat": chain_A_hat,
            "weights": weights, "pi0": pi0,
            "settings": {"K": K, "L": L, "B": B, "eps": eps, "n_mcmc": n_mcmc, "mb_size": mb_size, "seed": seed,
                         "init_seed": init_seed, "sigmasq_init": sigmasq_init}}


def sort_chain_by_mu(chain):
    mu = chain["mu"]
    s2 = chain["sigmasq"]
    A = chain["A"]
    out_mu = np.zeros_like(mu)
    out_s2 = np.zeros_like(s2)
    out_A = np.zeros_like(A)
    for t in range(len(mu)):
        order = np.argsort(mu[t])
        out_mu[t] = mu[t, order]
        out_s2[t] = s2[t, order]
        out_A[t] = A[t][np.ix_(order, order)]
    out = dict(chain)
    out["mu"] = out_mu
    out["sigmasq"] = out_s2
    out["A"] = out_A
    return out


def posterior_summary(chain, true_mu=None, burn=0.5):
    ch = sort_chain_by_mu(chain)
    start = int(len(ch["mu"]) * burn) if burn < 1 else int(burn)
    samples = ch["mu"][start:]
    rows = []
    for j in range(samples.shape[1]):
        row = {"state": j, "mean": float(np.mean(samples[:, j])),
               "q025": float(np.quantile(samples[:, j], 0.025)),
               "q975": float(np.quantile(samples[:, j], 0.975))}
        if true_mu is not None:
            tm = float(np.sort(true_mu)[j])
            row["true"] = tm
            row["abs_error"] = abs(row["mean"] - tm)
        rows.append(row)
    return pd.DataFrame(rows)


def rare_mean_error(chain, true_mu, rare_sorted_index=-1):
    ch = sort_chain_by_mu(chain)
    tm = np.sort(true_mu)[rare_sorted_index]
    return np.abs(ch["mu"][:, rare_sorted_index] - tm)


def rare_sigmasq_error(chain, true_mu, true_sigmasq, rare_sorted_index=-1):
    ch = sort_chain_by_mu(chain)
    return np.abs(ch["sigmasq"][:, rare_sorted_index] - true_sigmasq[np.argsort(true_mu)[rare_sorted_index]])


def rare_state_log_predictive_curve(chain, ytest, ztest, true_mu, true_sigmasq, rare_state, checkpoints=None, max_hold=200, seed=1):
    ch = sort_chain_by_mu(chain)
    order = np.argsort(true_mu)
    rare_sorted_index = int(np.where(order == rare_state)[0][0])
    rng = np.random.default_rng(seed)
    idx = np.where(ztest == rare_state)[0]
    if len(idx) == 0:
        return pd.DataFrame(columns=["method", "iteration", "mean_log_pred"])
    if len(idx) > max_hold:
        idx = rng.choice(idx, size=max_hold, replace=False)
    yy = ytest[idx]
    if checkpoints is None:
        checkpoints = np.unique(np.linspace(1, len(ch["mu"]) - 1, 50).astype(int))
    rows = []
    for c in checkpoints:
        mus = ch["mu"][:c + 1, rare_sorted_index]
        s2s = np.maximum(ch["sigmasq"][:c + 1, rare_sorted_index], 1e-300)
        vals = []
        for val in yy:
            lp = -0.5 * (np.log(2 * np.pi * s2s) + (val - mus) ** 2 / s2s)
            m = np.max(lp)
            vals.append(m + np.log(np.mean(np.exp(lp - m))))
        rows.append({"method": chain["method"], "iteration": int(c), "mean_log_pred": float(np.mean(vals))})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Online extension and pilot-staleness experiment layer.
# The functions above are the MATLAB-style replication layer.  The functions
# below preserve that layer exactly when method in {uniform, tass},
# dynamic_strength == 0, and pilot_mode == "paper".
# -----------------------------------------------------------------------------


def block_features_1d(y, L, tail_quantile=0.95):
    ll = 2 * L + 1
    N = len(y) // ll
    yy = y[:N * ll]
    qhi = np.quantile(np.abs(yy), tail_quantile)
    X = np.zeros((N, 8), dtype=float)
    for n in range(N):
        vals = yy[n * ll:(n + 1) * ll]
        X[n, 0] = 1.0
        X[n, 1] = np.mean(vals)
        X[n, 2] = np.mean(np.abs(vals))
        X[n, 3] = np.std(vals)
        X[n, 4] = np.mean(vals ** 2)
        X[n, 5] = np.max(np.abs(vals))
        X[n, 6] = np.sum(np.abs(vals) >= qhi)
        X[n, 7] = np.mean(vals < 0.0)
    mu = X[:, 1:].mean(axis=0)
    sd = X[:, 1:].std(axis=0)
    sd[sd <= 1e-12] = 1.0
    X[:, 1:] = (X[:, 1:] - mu) / sd
    return X


class DiscountedRidgeOnlineWeights:
    def __init__(self, features, K, rho=0.98, tau=1.0, lambda_floor=0.05, feedback_eps=1e-8):
        self.W = np.asarray(features, dtype=float)
        self.N, self.p = self.W.shape
        self.K = K
        self.rho = float(rho)
        self.tau = float(tau)
        self.lambda_floor = float(lambda_floor)
        self.feedback_eps = float(feedback_eps)

        self.A_mu = np.tile(self.tau * np.eye(self.p), (K, 1, 1))
        self.b_mu = np.zeros((K, self.p))
        self.beta_mu = np.zeros((K, self.p))

        self.A_s2 = np.tile(self.tau * np.eye(self.p), (K, 1, 1))
        self.b_s2 = np.zeros((K, self.p))
        self.beta_s2 = np.zeros((K, self.p))

        self.A_tr = np.tile(self.tau * np.eye(self.p), (K, K, 1, 1))
        self.b_tr = np.zeros((K, K, self.p))
        self.beta_tr = np.zeros((K, K, self.p))

    def _prob(self, beta):
        score = np.clip(0.5 * (self.W @ beta), -30.0, 30.0)
        raw = np.exp(score)
        p = normalize(raw)
        if self.lambda_floor > 0:
            p = (1.0 - self.lambda_floor) * p + self.lambda_floor / self.N
        return normalize(p)

    def weights(self):
        wmu = np.zeros((self.K, self.N))
        ws2 = np.zeros((self.K, self.N))
        wA = np.zeros((self.K, self.K, self.N))
        for j in range(self.K):
            wmu[j] = self._prob(self.beta_mu[j])
            ws2[j] = self._prob(self.beta_s2[j])
        for j in range(self.K):
            for k in range(self.K):
                wA[j, k] = self._prob(self.beta_tr[j, k])
        return {"mu": wmu, "sigmasq": ws2, "A": wA}

    def update_mu(self, j, block, value):
        x = self.W[block]
        f = np.log(float(value) ** 2 + self.feedback_eps)
        self.A_mu[j] = self.rho * self.A_mu[j] + np.outer(x, x)
        self.b_mu[j] = self.rho * self.b_mu[j] + x * f
        self.beta_mu[j] = np.linalg.solve(self.A_mu[j], self.b_mu[j])

    def update_s2(self, j, block, value):
        x = self.W[block]
        f = np.log(float(value) ** 2 + self.feedback_eps)
        self.A_s2[j] = self.rho * self.A_s2[j] + np.outer(x, x)
        self.b_s2[j] = self.rho * self.b_s2[j] + x * f
        self.beta_s2[j] = np.linalg.solve(self.A_s2[j], self.b_s2[j])

    def update_A(self, j, k, block, value):
        x = self.W[block]
        f = np.log(float(value) ** 2 + self.feedback_eps)
        self.A_tr[j, k] = self.rho * self.A_tr[j, k] + np.outer(x, x)
        self.b_tr[j, k] = self.rho * self.b_tr[j, k] + x * f
        self.beta_tr[j, k] = np.linalg.solve(self.A_tr[j, k], self.b_tr[j, k])


def sorted_pilot_shift(kind, dynamic_strength):
    shift = np.zeros(3, dtype=float)
    s = float(dynamic_strength)
    if kind == "one_rare":
        shift[2] = -s
    elif kind == "two_rare":
        shift[0] = s
        shift[2] = -s
    elif kind == "no_rare":
        shift[:] = 0.0
    return shift


def weights_by_clustering_with_pilot_shift(y, K, L, eta=1e-5, seed=1, sort_centres=True,
                                           pilot_shift=None):
    # This equals weights_by_clustering_matlab when pilot_shift is None or zero.
    z, centers = kmeans_labels_matlab_style(y, K, seed=seed, sort_centres=sort_centres)
    ll = 2 * L + 1
    N = len(y) // ll
    y = y[:N * ll]
    z = z[:N * ll]

    ybar = np.zeros(K)
    s2 = np.zeros(K)
    for j in range(K):
        vals = y[z == j]
        ybar[j] = np.mean(vals) if len(vals) else 0.0
        s2[j] = np.var(vals, ddof=1) if len(vals) > 1 else 0.0

    if pilot_shift is None:
        pilot_ybar = ybar.copy()
    else:
        pilot_ybar = ybar + np.asarray(pilot_shift, dtype=float)

    c = np.zeros((N, K))
    ybar_chunk = np.zeros((N, K))
    s2_chunk = np.zeros((N, K))
    for n in range(N):
        idx = slice(n * ll, (n + 1) * ll)
        yn = y[idx]
        zn = z[idx]
        for j in range(K):
            mask = zn == j
            c[n, j] = np.sum(mask)
            if c[n, j] > 0:
                vals = yn[mask]
                ybar_chunk[n, j] = np.mean(vals)
                s2_chunk[n, j] = np.sum((vals - ybar[j]) ** 2) / c[n, j]

    wmu = np.zeros((K, N))
    ws2 = np.zeros((K, N))
    for j in range(K):
        wmu[j] = c[:, j] * np.abs(ybar_chunk[:, j] - pilot_ybar[j]) + eta ** 2
        ws2[j] = c[:, j] * (s2[j] + s2_chunk[:, j]) + eta ** 2
        wmu[j] = normalize(wmu[j])
        ws2[j] = normalize(ws2[j])

    xi = np.zeros((N, K, K))
    for n in range(N):
        idx = slice(n * ll, (n + 1) * ll)
        zn = z[idx]
        for old in range(K):
            for new in range(K):
                xi[n, old, new] = np.sum((zn[:-1] == old) & (zn[1:] == new))

    xisum = np.sum(xi, axis=0) + 0.5
    A_map = normalize_columns(xisum.T)
    wA = np.zeros((K, K, N))
    for new in range(K):
        for old in range(K):
            w = xi[:, old, new] / max(A_map[new, old], 1e-300) + 1e-10
            wA[new, old] = normalize(w)

    return {"mu": wmu, "sigmasq": ws2, "A": wA, "labels": z, "centers": centers,
            "A_map": A_map, "pilot_ybar": pilot_ybar, "pilot_shift": pilot_shift}


def make_weights_extended(y, method, K, L, eta=1e-5, seed=1, sort_centres=True,
                          kind="one_rare", dynamic_strength=0.0, pilot_mode="paper"):
    ll = 2 * L + 1
    N = len(y) // ll
    if method == "uniform":
        return uniform_weights(K, N)
    if method == "tass":
        if pilot_mode == "paper" and float(dynamic_strength) == 0.0:
            return weights_by_clustering_matlab(y[:N * ll], K, L, eta=eta, seed=seed, sort_centres=sort_centres)
        if pilot_mode == "pilot_shift":
            shift = sorted_pilot_shift(kind, dynamic_strength)
            return weights_by_clustering_with_pilot_shift(y[:N * ll], K, L, eta=eta, seed=seed,
                                                          sort_centres=sort_centres, pilot_shift=shift)
        raise ValueError("pilot_mode must be 'paper' or 'pilot_shift'")
    raise ValueError("method must be uniform, tass, or online")


def transition_grad_online(y, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng,
                           learn=True):
    K = len(mu)
    N = len(y) // (2 * L + 1)
    weights = online.weights()
    _, _, gA = prior_gradients(mu, sigmasq, A_hat)
    cache = {}

    def get_block(i):
        if i not in cache:
            cache[i] = block_gradients_matlab(y, i, L, B, A, pi0, mu, sigmasq)
        return cache[i]

    for j in range(K):
        for k in range(K):
            w = weights["A"][j, k]
            mb = rng.choice(N, size=mb_size, replace=True, p=w)
            for i in mb:
                _, _, bA = get_block(int(i))
                val = bA[j, k]
                gA[j, k] -= val / max(w[i], 1e-300)
                if learn:
                    online.update_A(j, k, int(i), val)
    gA /= mb_size
    return gA


def emission_grad_online(y, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng,
                         learn=True):
    K = len(mu)
    N = len(y) // (2 * L + 1)
    weights = online.weights()
    gmu, gs2, _ = prior_gradients(mu, sigmasq, A_hat)
    cache = {}

    def get_block(i):
        if i not in cache:
            cache[i] = block_gradients_matlab(y, i, L, B, A, pi0, mu, sigmasq)
        return cache[i]

    for j in range(K):
        w = weights["mu"][j]
        mb = rng.choice(N, size=mb_size, replace=True, p=w)
        for i in mb:
            bmu, _, _ = get_block(int(i))
            val = bmu[j]
            gmu[j] -= val / max(w[i], 1e-300)
            if learn:
                online.update_mu(j, int(i), val)

        w = weights["sigmasq"][j]
        mb = rng.choice(N, size=mb_size, replace=True, p=w)
        for i in mb:
            _, bs2, _ = get_block(int(i))
            val = bs2[j]
            gs2[j] -= val / max(w[i], 1e-300)
            if learn:
                online.update_s2(j, int(i), val)

    gmu /= mb_size
    gs2 /= mb_size
    return gmu, gs2


def run_csg_mcmc_is_extended(y, method="tass", K=3, L=2, B=5, eps=1e-6, n_mcmc=5000,
                             mb_size=10, eta=1e-5, seed=1, init_seed=None,
                             pi0_init=None, sigmasq_init=10.0, sort_centres=True,
                             kind="one_rare", dynamic_strength=0.0, pilot_mode="paper",
                             online_rho=0.98, online_tau=1.0, online_lambda_floor=0.05,
                             online_warmup=0, online_feedback_eps=1e-8,
                             checkpoint_every=None):
    # Exact paper/MATLAB path: method in {uniform,tass}, dynamic_strength=0, pilot_mode="paper".
    if method in ("uniform", "tass") and float(dynamic_strength) == 0.0 and pilot_mode == "paper":
        return run_csg_mcmc_is_matlab(y, method=method, K=K, L=L, B=B, eps=eps, n_mcmc=n_mcmc,
                                     mb_size=mb_size, eta=eta, seed=seed, init_seed=init_seed,
                                     pi0_init=pi0_init, sigmasq_init=sigmasq_init,
                                     sort_centres=sort_centres, checkpoint_every=checkpoint_every)

    rng = np.random.default_rng(seed)
    if init_seed is None:
        init_seed = seed + 1000003
    init_rng = np.random.default_rng(init_seed)
    if pi0_init is None:
        pi0_init = normalize(init_rng.random(K))
    mu, sigmasq, A, A_hat, pi0 = matlab_initial_parameters(K, seed=init_seed,
                                                           sigmasq_init=sigmasq_init,
                                                           pi0_init=pi0_init)

    ll = 2 * L + 1
    N = len(y) // ll
    y = y[:N * ll]

    weights = None
    online = None
    if method in ("uniform", "tass"):
        weights = make_weights_extended(y, method, K, L, eta=eta, seed=seed, sort_centres=sort_centres,
                                        kind=kind, dynamic_strength=dynamic_strength,
                                        pilot_mode=pilot_mode)
    elif method == "online":
        features = block_features_1d(y, L)
        online = DiscountedRidgeOnlineWeights(features, K, rho=online_rho, tau=online_tau,
                                              lambda_floor=online_lambda_floor,
                                              feedback_eps=online_feedback_eps)
    else:
        raise ValueError("method must be uniform, tass, or online")

    chain_mu = np.zeros((n_mcmc + 1, K))
    chain_s2 = np.zeros((n_mcmc + 1, K))
    chain_A = np.zeros((n_mcmc + 1, K, K))
    chain_A_hat = np.zeros((n_mcmc + 1, K, K))
    chain_mu[0] = mu
    chain_s2[0] = sigmasq
    chain_A[0] = A
    chain_A_hat[0] = A_hat

    for itr in range(1, n_mcmc + 1):
        if method == "online":
            if itr <= online_warmup:
                uw = uniform_weights(K, N)
                gA, _ = transition_grad_IS(y, uw, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng)
                A, A_hat = sgld_update_transition(A_hat, gA, eps, rng)
                gmu, gs2, _ = emission_grad_IS(y, uw, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng)
            else:
                gA = transition_grad_online(y, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng, learn=True)
                A, A_hat = sgld_update_transition(A_hat, gA, eps, rng)
                gmu, gs2 = emission_grad_online(y, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng, learn=True)
            mu, sigmasq = sgld_update_emission(mu, sigmasq, gmu, gs2, eps, rng)
        else:
            gA, _ = transition_grad_IS(y, weights, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng)
            A, A_hat = sgld_update_transition(A_hat, gA, eps, rng)
            gmu, gs2, _ = emission_grad_IS(y, weights, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng)
            mu, sigmasq = sgld_update_emission(mu, sigmasq, gmu, gs2, eps, rng)

        if itr % 10 == 0:
            A_hat = normalize_columns(A_hat)
            A = normalize_columns(A_hat)

        chain_mu[itr] = mu
        chain_s2[itr] = sigmasq
        chain_A[itr] = A
        chain_A_hat[itr] = A_hat

        if checkpoint_every is not None and (itr % checkpoint_every == 0 or itr == 1):
            print(f"{method}: {itr}/{n_mcmc}, mu={np.round(mu, 3).tolist()}, sigmasq={np.round(sigmasq, 3).tolist()}", flush=True)

    out = {"method": method, "mu": chain_mu, "sigmasq": chain_s2, "A": chain_A, "A_hat": chain_A_hat,
           "weights": weights, "online": online, "pi0": pi0,
           "settings": {"K": K, "L": L, "B": B, "eps": eps, "n_mcmc": n_mcmc,
                        "mb_size": mb_size, "seed": seed, "init_seed": init_seed,
                        "sigmasq_init": sigmasq_init, "kind": kind,
                        "dynamic_strength": dynamic_strength, "pilot_mode": pilot_mode,
                        "online_rho": online_rho, "online_tau": online_tau,
                        "online_lambda_floor": online_lambda_floor,
                        "online_warmup": online_warmup}}
    return out


def run_methods_extended(scenario, methods=("uniform", "tass", "online"), seed=1, init_seed=999,
                         K=3, L=2, B=5, eps=1e-6, n_mcmc=5000, mb_size=10, eta=1e-5,
                         dynamic_strength=0.0, pilot_mode="paper", online_rho=0.98,
                         online_tau=1.0, online_lambda_floor=0.05, online_warmup=0,
                         checkpoint_every=None):
    chains = {}
    for r, method in enumerate(methods):
        chains[method] = run_csg_mcmc_is_extended(
            scenario["y"], method=method, K=K, L=L, B=B, eps=eps, n_mcmc=n_mcmc,
            mb_size=mb_size, eta=eta, seed=seed + r, init_seed=init_seed,
            kind=scenario.get("kind", "one_rare"), dynamic_strength=dynamic_strength,
            pilot_mode=pilot_mode, online_rho=online_rho, online_tau=online_tau,
            online_lambda_floor=online_lambda_floor, online_warmup=online_warmup,
            checkpoint_every=checkpoint_every)
    return chains


def paper_exact_condition(methods, dynamic_strength, pilot_mode):
    return tuple(methods) == ("uniform", "tass") and float(dynamic_strength) == 0.0 and pilot_mode == "paper"


def simulate_hmm_covariate(A, mu, sigmasq, beta, T, pi0=None, seed=1,
                           x_mode="block_trend", block_len=5):
    rng = np.random.default_rng(seed)
    K = len(mu)
    if pi0 is None:
        pi0 = np.ones(K) / K
    z = np.zeros(T, dtype=int)
    x = np.zeros(T, dtype=float)
    y = np.zeros(T, dtype=float)
    z[0] = rng.choice(K, p=pi0)
    for t in range(1, T):
        z[t] = rng.choice(K, p=A[:, z[t - 1]])

    if x_mode == "block_trend":
        N = int(np.ceil(T / block_len))
        xb = rng.normal(size=N)
        # Mild smoothness makes x feature-predictable at block scale.
        for n in range(1, N):
            xb[n] = 0.85 * xb[n - 1] + 0.55 * xb[n]
        x = np.repeat(xb, block_len)[:T]
        x = (x - np.mean(x)) / max(np.std(x), 1e-12)
    elif x_mode == "iid":
        x = rng.normal(size=T)
    else:
        raise ValueError("x_mode must be block_trend or iid")

    for t in range(T):
        m = mu[z[t]] + beta[z[t]] * x[t]
        y[t] = rng.normal(m, np.sqrt(sigmasq[z[t]]))
    return z, x, y


def dynamic_beta_vector(kind="one_rare", dynamic_strength=0.0, roles=None):
    beta = np.zeros(3, dtype=float)
    s = float(dynamic_strength)
    if roles is None:
        roles = {}
    if kind == "one_rare":
        if roles.get(2, "dynamic_informed") == "dynamic_informed":
            beta[2] = s
    elif kind == "two_rare":
        if roles.get(1, "dynamic_informed") == "dynamic_informed":
            beta[1] = -s
        if roles.get(2, "dynamic_informed") == "dynamic_informed":
            beta[2] = s
    elif kind == "no_rare":
        beta[:] = 0.0
    return beta


def make_organic_dynamic_scenario(kind="one_rare", dynamic_strength=0.0, roles=None,
                                  Ttrain=10**6, Ttest=10**6, seed=1,
                                  x_mode="block_trend", L=2):
    A, mu, sigmasq = paper_transition(kind)
    pi0 = np.ones(3) / 3.0
    beta = dynamic_beta_vector(kind, dynamic_strength, roles=roles)
    z_all, x_all, y_all = simulate_hmm_covariate(
        A, mu, sigmasq, beta, Ttrain + Ttest, pi0=pi0, seed=seed,
        x_mode=x_mode, block_len=2 * L + 1)
    return {
        "kind": kind,
        "A": A,
        "mu": mu,
        "sigmasq": sigmasq,
        "beta": beta,
        "pi0": pi0,
        "roles": roles if roles is not None else {},
        "dynamic_strength": float(dynamic_strength),
        "x": x_all[:Ttrain],
        "z": z_all[:Ttrain],
        "y": y_all[:Ttrain],
        "xtest": x_all[Ttrain:],
        "ztest": z_all[Ttrain:],
        "ytest": y_all[Ttrain:],
    }


def density_vector_cov(yt, xt, mu, sigmasq, beta):
    sigmasq = np.maximum(sigmasq, 1e-300)
    mean = mu + beta * xt
    return np.exp(-0.5 * (np.log(2.0 * np.pi * sigmasq) + (yt - mean) ** 2 / sigmasq))


def approx_pred_cov(y, x, t, B, A, pi0, mu, sigmasq, beta):
    T = len(y)
    K = len(mu)
    pipred = pi0.astype(float).copy()
    qmat = np.eye(K)

    if t < B:
        left = range(0, t)
        right = range(t + 1, min(t + B + 1, T))
    elif t > T - B - 2:
        left = range(max(0, t - B), t)
        right = range(t + 1, T)
    else:
        left = range(t - B, t)
        right = range(t + 1, t + B + 1)

    for i in left:
        P = density_vector_cov(y[i], x[i], mu, sigmasq, beta)
        pipred = P * (A @ pipred)

    for i in right:
        P = density_vector_cov(y[i], x[i], mu, sigmasq, beta)
        qmat = (P[:, None] * A) @ qmat

    qpred = qmat.T @ np.ones(K)
    return pipred, qpred


def block_gradients_cov(y, x, block, L, B, A, pi0, mu, sigmasq, beta):
    K = len(mu)
    ll = 2 * L + 1
    lo = block * ll
    hi = min((block + 1) * ll, len(y))
    gmu = np.zeros(K)
    gs2 = np.zeros(K)
    gbeta = np.zeros(K)
    gA = np.zeros((K, K))

    for t in range(lo, hi):
        P = density_vector_cov(y[t], x[t], mu, sigmasq, beta)
        pipred, qpred = approx_pred_cov(y, x, t, B, A, pi0, mu, sigmasq, beta)
        Apipred = A @ pipred
        den = float(qpred @ (P * Apipred))
        if den <= 0 or not np.isfinite(den):
            den = 1e-300
        mean = mu + beta * x[t]
        resid = y[t] - mean
        common = P * qpred * Apipred / den
        gmu += common * resid / sigmasq
        gbeta += common * x[t] * resid / sigmasq
        gs2 += 0.5 * common * (-sigmasq + resid ** 2) / (sigmasq ** 2)
        gA += (P * qpred)[:, None] * pipred[None, :] / den

    return gmu, gs2, gbeta, gA


def prior_gradients_cov(mu, sigmasq, beta, A_hat, mu_prior_var=100.0,
                        beta_prior_var=100.0, ig_a=3.0, ig_b=10.0, alpha=1.0):
    gmu = (0.0 - mu) / mu_prior_var
    gbeta = (0.0 - beta) / beta_prior_var
    gs2 = (ig_a + 1.0) / sigmasq - ig_b / (sigmasq ** 2)
    gA = (alpha - 1.0) / np.maximum(A_hat, 1e-300)
    return gmu, gs2, gbeta, gA


def block_features_yx(y, x, L, tail_quantile=0.95):
    ll = 2 * L + 1
    N = len(y) // ll
    yy = y[:N * ll]
    xx = x[:N * ll]
    qhi = np.quantile(np.abs(yy), tail_quantile)
    X = np.zeros((N, 13), dtype=float)
    for n in range(N):
        ys = yy[n * ll:(n + 1) * ll]
        xs = xx[n * ll:(n + 1) * ll]
        X[n, 0] = 1.0
        X[n, 1] = np.mean(ys)
        X[n, 2] = np.mean(np.abs(ys))
        X[n, 3] = np.std(ys)
        X[n, 4] = np.mean(ys ** 2)
        X[n, 5] = np.max(np.abs(ys))
        X[n, 6] = np.sum(np.abs(ys) >= qhi)
        X[n, 7] = np.mean(ys < 0.0)
        X[n, 8] = np.mean(xs)
        X[n, 9] = np.std(xs)
        X[n, 10] = np.mean(xs * ys)
        X[n, 11] = np.mean(np.abs(xs * ys))
        X[n, 12] = np.max(np.abs(xs * ys))
    muX = X[:, 1:].mean(axis=0)
    sdX = X[:, 1:].std(axis=0)
    sdX[sdX <= 1e-12] = 1.0
    X[:, 1:] = (X[:, 1:] - muX) / sdX
    return X


def weights_by_clustering_cov_pilot(y, x, K, L, eta=1e-5, seed=1, sort_centres=True):
    # Paper-style frozen TASS weights, plus beta-coordinate weights computed at
    # the same K-means pilot.  No manual pilot shift is used.
    z, centers = kmeans_labels_matlab_style(y, K, seed=seed, sort_centres=sort_centres)
    ll = 2 * L + 1
    N = len(y) // ll
    y = y[:N * ll]
    x = x[:N * ll]
    z = z[:N * ll]

    ybar = np.zeros(K)
    s2 = np.zeros(K)
    for j in range(K):
        vals = y[z == j]
        ybar[j] = np.mean(vals) if len(vals) else 0.0
        s2[j] = np.var(vals, ddof=1) if len(vals) > 1 else 1.0

    c = np.zeros((N, K))
    ybar_chunk = np.zeros((N, K))
    s2_chunk = np.zeros((N, K))
    beta_score = np.zeros((N, K))
    for n in range(N):
        idx = slice(n * ll, (n + 1) * ll)
        yn = y[idx]
        xn = x[idx]
        zn = z[idx]
        for j in range(K):
            mask = zn == j
            c[n, j] = np.sum(mask)
            if c[n, j] > 0:
                vals = yn[mask]
                xs = xn[mask]
                ybar_chunk[n, j] = np.mean(vals)
                s2_chunk[n, j] = np.sum((vals - ybar[j]) ** 2) / c[n, j]
                beta_score[n, j] = abs(np.sum(xs * (vals - ybar[j]) / max(s2[j], 1e-12)))

    wmu = np.zeros((K, N))
    ws2 = np.zeros((K, N))
    wbeta = np.zeros((K, N))
    for j in range(K):
        wmu[j] = c[:, j] * np.abs(ybar_chunk[:, j] - ybar[j]) + eta ** 2
        ws2[j] = c[:, j] * (s2[j] + s2_chunk[:, j]) + eta ** 2
        wbeta[j] = beta_score[:, j] + eta ** 2
        wmu[j] = normalize(wmu[j])
        ws2[j] = normalize(ws2[j])
        wbeta[j] = normalize(wbeta[j])

    xi = np.zeros((N, K, K))
    for n in range(N):
        idx = slice(n * ll, (n + 1) * ll)
        zn = z[idx]
        for old in range(K):
            for new in range(K):
                xi[n, old, new] = np.sum((zn[:-1] == old) & (zn[1:] == new))

    xisum = np.sum(xi, axis=0) + 0.5
    A_map = normalize_columns(xisum.T)
    wA = np.zeros((K, K, N))
    for new in range(K):
        for old in range(K):
            w = xi[:, old, new] / max(A_map[new, old], 1e-300) + 1e-10
            wA[new, old] = normalize(w)

    return {"mu": wmu, "sigmasq": ws2, "beta": wbeta, "A": wA,
            "labels": z, "centers": centers, "A_map": A_map,
            "pilot_ybar": ybar}


def uniform_weights_cov(K, N):
    w = uniform_weights(K, N)
    w["beta"] = np.ones((K, N)) / N
    return w


class DiscountedRidgeOnlineWeightsCov(DiscountedRidgeOnlineWeights):
    def __init__(self, features, K, rho=0.98, tau=1.0, lambda_floor=0.05, feedback_eps=1e-8):
        super().__init__(features, K, rho=rho, tau=tau, lambda_floor=lambda_floor, feedback_eps=feedback_eps)
        self.A_beta = np.tile(self.tau * np.eye(self.p), (K, 1, 1))
        self.b_beta = np.zeros((K, self.p))
        self.beta_beta = np.zeros((K, self.p))

    def weights(self):
        w = super().weights()
        wb = np.zeros((self.K, self.N))
        for j in range(self.K):
            wb[j] = self._prob(self.beta_beta[j])
        w["beta"] = wb
        return w

    def update_beta(self, j, block, value):
        Xrow = self.W[block]
        f = np.log(float(value) ** 2 + self.feedback_eps)
        self.A_beta[j] = self.rho * self.A_beta[j] + np.outer(Xrow, Xrow)
        self.b_beta[j] = self.rho * self.b_beta[j] + Xrow * f
        self.beta_beta[j] = np.linalg.solve(self.A_beta[j], self.b_beta[j])


def transition_grad_IS_cov(y, x, weights, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, beta, rng):
    K = len(mu)
    N = len(y) // (2 * L + 1)
    _, _, _, gA = prior_gradients_cov(mu, sigmasq, beta, A_hat)
    cache = {}

    def get_block(i):
        if i not in cache:
            cache[i] = block_gradients_cov(y, x, i, L, B, A, pi0, mu, sigmasq, beta)
        return cache[i]

    for j in range(K):
        for k in range(K):
            w = weights["A"][j, k]
            mb = rng.choice(N, size=mb_size, replace=True, p=w)
            for i in mb:
                _, _, _, bA = get_block(int(i))
                gA[j, k] -= bA[j, k] / max(w[i], 1e-300)
    gA /= mb_size
    return gA


def emission_grad_IS_cov(y, x, weights, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, beta, rng):
    K = len(mu)
    N = len(y) // (2 * L + 1)
    gmu, gs2, gbeta, _ = prior_gradients_cov(mu, sigmasq, beta, A_hat)
    cache = {}

    def get_block(i):
        if i not in cache:
            cache[i] = block_gradients_cov(y, x, i, L, B, A, pi0, mu, sigmasq, beta)
        return cache[i]

    for j in range(K):
        w = weights["mu"][j]
        mb = rng.choice(N, size=mb_size, replace=True, p=w)
        for i in mb:
            bmu, _, _, _ = get_block(int(i))
            gmu[j] -= bmu[j] / max(w[i], 1e-300)

        w = weights["sigmasq"][j]
        mb = rng.choice(N, size=mb_size, replace=True, p=w)
        for i in mb:
            _, bs2, _, _ = get_block(int(i))
            gs2[j] -= bs2[j] / max(w[i], 1e-300)

        w = weights["beta"][j]
        mb = rng.choice(N, size=mb_size, replace=True, p=w)
        for i in mb:
            _, _, bbeta, _ = get_block(int(i))
            gbeta[j] -= bbeta[j] / max(w[i], 1e-300)

    gmu /= mb_size
    gs2 /= mb_size
    gbeta /= mb_size
    return gmu, gs2, gbeta


def transition_grad_online_cov(y, x, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, beta, rng, learn=True):
    K = len(mu)
    N = len(y) // (2 * L + 1)
    weights = online.weights()
    _, _, _, gA = prior_gradients_cov(mu, sigmasq, beta, A_hat)
    cache = {}

    def get_block(i):
        if i not in cache:
            cache[i] = block_gradients_cov(y, x, i, L, B, A, pi0, mu, sigmasq, beta)
        return cache[i]

    for j in range(K):
        for k in range(K):
            w = weights["A"][j, k]
            mb = rng.choice(N, size=mb_size, replace=True, p=w)
            for i in mb:
                _, _, _, bA = get_block(int(i))
                val = bA[j, k]
                gA[j, k] -= val / max(w[i], 1e-300)
                if learn:
                    online.update_A(j, k, int(i), val)
    gA /= mb_size
    return gA


def emission_grad_online_cov(y, x, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, beta, rng, learn=True):
    K = len(mu)
    N = len(y) // (2 * L + 1)
    weights = online.weights()
    gmu, gs2, gbeta, _ = prior_gradients_cov(mu, sigmasq, beta, A_hat)
    cache = {}

    def get_block(i):
        if i not in cache:
            cache[i] = block_gradients_cov(y, x, i, L, B, A, pi0, mu, sigmasq, beta)
        return cache[i]

    for j in range(K):
        w = weights["mu"][j]
        mb = rng.choice(N, size=mb_size, replace=True, p=w)
        for i in mb:
            bmu, _, _, _ = get_block(int(i))
            val = bmu[j]
            gmu[j] -= val / max(w[i], 1e-300)
            if learn:
                online.update_mu(j, int(i), val)

        w = weights["sigmasq"][j]
        mb = rng.choice(N, size=mb_size, replace=True, p=w)
        for i in mb:
            _, bs2, _, _ = get_block(int(i))
            val = bs2[j]
            gs2[j] -= val / max(w[i], 1e-300)
            if learn:
                online.update_s2(j, int(i), val)

        w = weights["beta"][j]
        mb = rng.choice(N, size=mb_size, replace=True, p=w)
        for i in mb:
            _, _, bbeta, _ = get_block(int(i))
            val = bbeta[j]
            gbeta[j] -= val / max(w[i], 1e-300)
            if learn:
                online.update_beta(j, int(i), val)

    gmu /= mb_size
    gs2 /= mb_size
    gbeta /= mb_size
    return gmu, gs2, gbeta


def sgld_update_beta(beta, sigmasq, gbeta, eps, rng):
    sd = np.sqrt(np.maximum(eps * (2.0 * sigmasq), 1e-300))
    beta = beta - eps * sigmasq * gbeta + sd * rng.normal(size=len(beta))
    return beta


def run_csg_mcmc_organic_dynamic(scenario, method="tass", K=3, L=2, B=5, eps=1e-6,
                                 n_mcmc=5000, mb_size=10, eta=1e-5, seed=1,
                                 init_seed=None, pi0_init=None, sigmasq_init=10.0,
                                 sort_centres=True, online_rho=0.98, online_tau=1.0,
                                 online_lambda_floor=0.05, online_warmup=0,
                                 online_feedback_eps=1e-8, checkpoint_every=None):
    y = scenario["y"]
    x = scenario["x"]
    kind = scenario.get("kind", "one_rare")
    rng = np.random.default_rng(seed)
    if init_seed is None:
        init_seed = seed + 1000003
    init_rng = np.random.default_rng(init_seed)
    if pi0_init is None:
        pi0_init = normalize(init_rng.random(K))
    mu, sigmasq, A, A_hat, pi0 = matlab_initial_parameters(K, seed=init_seed,
                                                           sigmasq_init=sigmasq_init,
                                                           pi0_init=pi0_init)
    beta = np.zeros(K, dtype=float)

    ll = 2 * L + 1
    N = len(y) // ll
    y = y[:N * ll]
    x = x[:N * ll]

    weights = None
    online = None
    if method == "uniform":
        weights = uniform_weights_cov(K, N)
    elif method == "tass":
        weights = weights_by_clustering_cov_pilot(y, x, K, L, eta=eta, seed=seed,
                                                  sort_centres=sort_centres)
    elif method == "online":
        features = block_features_yx(y, x, L)
        online = DiscountedRidgeOnlineWeightsCov(features, K, rho=online_rho,
                                                 tau=online_tau,
                                                 lambda_floor=online_lambda_floor,
                                                 feedback_eps=online_feedback_eps)
    else:
        raise ValueError("method must be uniform, tass, or online")

    chain_mu = np.zeros((n_mcmc + 1, K))
    chain_s2 = np.zeros((n_mcmc + 1, K))
    chain_beta = np.zeros((n_mcmc + 1, K))
    chain_A = np.zeros((n_mcmc + 1, K, K))
    chain_A_hat = np.zeros((n_mcmc + 1, K, K))
    chain_mu[0] = mu
    chain_s2[0] = sigmasq
    chain_beta[0] = beta
    chain_A[0] = A
    chain_A_hat[0] = A_hat

    for itr in range(1, n_mcmc + 1):
        if method == "online":
            if itr <= online_warmup:
                uw = uniform_weights_cov(K, N)
                gA = transition_grad_IS_cov(y, x, uw, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, beta, rng)
                A, A_hat = sgld_update_transition(A_hat, gA, eps, rng)
                gmu, gs2, gbeta = emission_grad_IS_cov(y, x, uw, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, beta, rng)
            else:
                gA = transition_grad_online_cov(y, x, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, beta, rng, learn=True)
                A, A_hat = sgld_update_transition(A_hat, gA, eps, rng)
                gmu, gs2, gbeta = emission_grad_online_cov(y, x, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, beta, rng, learn=True)
        else:
            gA = transition_grad_IS_cov(y, x, weights, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, beta, rng)
            A, A_hat = sgld_update_transition(A_hat, gA, eps, rng)
            gmu, gs2, gbeta = emission_grad_IS_cov(y, x, weights, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, beta, rng)

        mu, sigmasq = sgld_update_emission(mu, sigmasq, gmu, gs2, eps, rng)
        beta = sgld_update_beta(beta, sigmasq, gbeta, eps, rng)

        if itr % 10 == 0:
            A_hat = normalize_columns(A_hat)
            A = normalize_columns(A_hat)

        chain_mu[itr] = mu
        chain_s2[itr] = sigmasq
        chain_beta[itr] = beta
        chain_A[itr] = A
        chain_A_hat[itr] = A_hat

        if checkpoint_every is not None and (itr % checkpoint_every == 0 or itr == 1):
            print(f"{method}: {itr}/{n_mcmc}, mu={np.round(mu, 3).tolist()}, beta={np.round(beta, 3).tolist()}", flush=True)

    return {"method": method, "mu": chain_mu, "sigmasq": chain_s2, "beta": chain_beta,
            "A": chain_A, "A_hat": chain_A_hat, "weights": weights, "online": online,
            "pi0": pi0, "settings": {"K": K, "L": L, "B": B, "eps": eps,
                                      "n_mcmc": n_mcmc, "mb_size": mb_size,
                                      "seed": seed, "init_seed": init_seed,
                                      "sigmasq_init": sigmasq_init,
                                      "kind": kind,
                                      "dynamic_strength": scenario.get("dynamic_strength", None),
                                      "organic_dynamic": True}}


def run_methods_organic_dynamic(scenario, methods=("uniform", "tass", "online"), seed=1,
                                init_seed=999, K=3, L=2, B=5, eps=1e-6,
                                n_mcmc=5000, mb_size=10, eta=1e-5,
                                online_rho=0.98, online_tau=1.0,
                                online_lambda_floor=0.05, online_warmup=0,
                                checkpoint_every=None):
    chains = {}
    for r, method in enumerate(methods):
        chains[method] = run_csg_mcmc_organic_dynamic(
            scenario, method=method, K=K, L=L, B=B, eps=eps,
            n_mcmc=n_mcmc, mb_size=mb_size, eta=eta, seed=seed + r,
            init_seed=init_seed, online_rho=online_rho, online_tau=online_tau,
            online_lambda_floor=online_lambda_floor, online_warmup=online_warmup,
            checkpoint_every=checkpoint_every)
    return chains


def sort_chain_by_mu_with_beta(chain):
    out = sort_chain_by_mu(chain)
    if "beta" in chain:
        beta = chain["beta"]
        mu = chain["mu"]
        out_beta = np.zeros_like(beta)
        for t in range(len(mu)):
            order = np.argsort(mu[t])
            out_beta[t] = beta[t, order]
        out["beta"] = out_beta
    return out


def beta_error(chain, true_mu, true_beta, rare_sorted_index=-1):
    ch = sort_chain_by_mu_with_beta(chain)
    order = np.argsort(true_mu)
    true_sorted_beta = true_beta[order]
    return np.abs(ch["beta"][:, rare_sorted_index] - true_sorted_beta[rare_sorted_index])


def rare_state_log_predictive_curve_cov(chain, ytest, xtest, ztest, true_mu, true_sigmasq,
                                        true_beta, rare_state, checkpoints=None, max_hold=200, seed=1):
    ch = sort_chain_by_mu_with_beta(chain)
    order = np.argsort(true_mu)
    rare_sorted_index = int(np.where(order == rare_state)[0][0])
    rng = np.random.default_rng(seed)
    idx = np.where(ztest == rare_state)[0]
    if len(idx) == 0:
        return pd.DataFrame(columns=["method", "iteration", "mean_log_pred"])
    if len(idx) > max_hold:
        idx = rng.choice(idx, size=max_hold, replace=False)
    yy = ytest[idx]
    xx = xtest[idx]
    if checkpoints is None:
        checkpoints = np.unique(np.linspace(1, len(ch["mu"]) - 1, 50).astype(int))
    rows = []
    for c in checkpoints:
        mus = ch["mu"][:c + 1, rare_sorted_index]
        betas = ch["beta"][:c + 1, rare_sorted_index]
        s2s = np.maximum(ch["sigmasq"][:c + 1, rare_sorted_index], 1e-300)
        vals = []
        for val, xv in zip(yy, xx):
            mean = mus + betas * xv
            lp = -0.5 * (np.log(2 * np.pi * s2s) + (val - mean) ** 2 / s2s)
            m = np.max(lp)
            vals.append(m + np.log(np.mean(np.exp(lp - m))))
        rows.append({"method": chain["method"], "iteration": int(c), "mean_log_pred": float(np.mean(vals))})
    return pd.DataFrame(rows)


def component_q_for_current_weights_cov(scenario, chain, iteration, component="beta", state=2, L=2, B=5,
                                        method_weights=None, lambda_floor=0.0):
    # Exact diagnostic: Q_theta(p)=sum_n g_n(theta)^2/p_n for one component.
    y = scenario["y"]
    x = scenario["x"]
    ll = 2 * L + 1
    N = len(y) // ll
    y = y[:N * ll]
    x = x[:N * ll]
    ch = chain
    mu = ch["mu"][iteration]
    sigmasq = ch["sigmasq"][iteration]
    beta = ch["beta"][iteration]
    A = ch["A"][iteration]
    pi0 = ch["pi0"]
    scores = np.zeros(N)
    for n in range(N):
        bmu, bs2, bbeta, bA = block_gradients_cov(y, x, n, L, B, A, pi0, mu, sigmasq, beta)
        if component == "mu":
            scores[n] = bmu[state]
        elif component == "sigmasq":
            scores[n] = bs2[state]
        elif component == "beta":
            scores[n] = bbeta[state]
        else:
            raise ValueError("component must be mu, sigmasq, or beta")
    oracle = normalize(np.abs(scores) + 1e-12)
    if method_weights is None:
        p = np.ones(N) / N
    else:
        p = np.asarray(method_weights, dtype=float)
        if lambda_floor > 0:
            p = (1 - lambda_floor) * normalize(p) + lambda_floor / N
        p = normalize(p)
    Q = float(np.sum(scores ** 2 / np.maximum(p, 1e-300)))
    Qoracle = float(np.sum(scores ** 2 / np.maximum(oracle, 1e-300)))
    return {"Q": Q, "Qoracle": Qoracle, "ratio_to_oracle": Q / Qoracle if Qoracle > 0 else np.nan}


def kind_from_rare_count(rare_count):
    if int(rare_count) == 0:
        return "no_rare"
    if int(rare_count) == 1:
        return "one_rare"
    if int(rare_count) == 2:
        return "two_rare"
    raise ValueError("rare_count must be 0, 1, or 2")


def rare_states_from_kind(kind):
    if kind == "no_rare":
        return []
    if kind == "one_rare":
        return [2]
    if kind == "two_rare":
        return [1, 2]
    raise ValueError("unknown kind")


def rare_directions_from_kind(kind):
    if kind == "no_rare":
        return {}
    if kind == "one_rare":
        return {2: 1.0}
    if kind == "two_rare":
        return {1: -1.0, 2: 1.0}
    raise ValueError("unknown kind")


def information_profile(N, mode="ramp", slope=10.0):
    u = np.linspace(0.0, 1.0, N)
    if mode == "ramp":
        info = u
    elif mode == "sigmoid":
        info = 1.0 / (1.0 + np.exp(-slope * (u - 0.5)))
    elif mode == "switch":
        info = (u >= 0.5).astype(float)
    elif mode == "early":
        info = 1.0 - u
    else:
        raise ValueError("mode must be ramp, sigmoid, switch, or early")
    return info.astype(float)


def block_index_for_time(T, L):
    ll = 2 * L + 1
    return np.arange(T) // ll


def make_timevarying_info_scenario(rare_count=1, delta=0.0, Ttrain=10**6, Ttest=10**6,
                                   seed=1, L=2, info_mode="ramp", info_slope=10.0):
    kind = kind_from_rare_count(rare_count)

    if float(delta) == 0.0:
        scenario = make_paper_scenario(kind=kind, Ttrain=Ttrain, Ttest=Ttest, seed=seed)
        ll = 2 * L + 1
        Ntrain = len(scenario["y"]) // ll
        Ntest = len(scenario["ytest"]) // ll
        scenario["rare_count"] = int(rare_count)
        scenario["rare_states"] = rare_states_from_kind(kind)
        scenario["rare_directions"] = rare_directions_from_kind(kind)
        scenario["delta"] = 0.0
        scenario["info_mode"] = info_mode
        scenario["info_by_block"] = np.zeros(Ntrain)
        scenario["info_by_block_test"] = np.zeros(Ntest)
        scenario["info"] = np.zeros(len(scenario["y"]))
        scenario["infotest"] = np.zeros(len(scenario["ytest"]))
        scenario["effective_mu_train"] = scenario["mu"].copy()
        scenario["effective_mu_test"] = scenario["mu"].copy()
        return scenario

    A, mu, sigmasq = paper_transition(kind)
    pi0 = np.ones(3) / 3.0
    Ttotal = int(Ttrain + Ttest)
    z_all, y_base = simulate_hmm(A, mu, sigmasq, Ttotal, pi0=pi0, seed=seed)

    ll = 2 * L + 1
    Ntotal = int(np.ceil(Ttotal / ll))
    info_block = information_profile(Ntotal, mode=info_mode, slope=info_slope)
    b = np.minimum(block_index_for_time(Ttotal, L), Ntotal - 1)
    info = info_block[b]

    directions = rare_directions_from_kind(kind)
    shift = np.zeros(Ttotal)
    for state, direction in directions.items():
        shift[z_all == state] = float(delta) * float(direction) * info[z_all == state]

    y_all = y_base + shift

    z = z_all[:Ttrain]
    y = y_all[:Ttrain]
    ztest = z_all[Ttrain:]
    ytest = y_all[Ttrain:]
    info_train = info[:Ttrain]
    info_test = info[Ttrain:]
    Ntrain = len(y) // ll
    Ntest = len(ytest) // ll

    effective_mu_train = mu.copy()
    effective_mu_test = mu.copy()
    for state, direction in directions.items():
        mask_train = z == state
        mask_test = ztest == state
        if np.any(mask_train):
            effective_mu_train[state] = mu[state] + float(delta) * direction * float(np.mean(info_train[mask_train]))
        if np.any(mask_test):
            effective_mu_test[state] = mu[state] + float(delta) * direction * float(np.mean(info_test[mask_test]))

    return {
        "kind": kind,
        "rare_count": int(rare_count),
        "A": A,
        "mu": mu,
        "sigmasq": sigmasq,
        "pi0": pi0,
        "z": z,
        "y": y,
        "ztest": ztest,
        "ytest": ytest,
        "delta": float(delta),
        "info_mode": info_mode,
        "info": info_train,
        "infotest": info_test,
        "info_by_block": info_block[:Ntrain],
        "info_by_block_test": info_block[Ntrain:Ntrain + Ntest],
        "rare_states": rare_states_from_kind(kind),
        "rare_directions": directions,
        "effective_mu_train": effective_mu_train,
        "effective_mu_test": effective_mu_test,
    }


def block_features_timevarying(y, L, info_by_block=None, tail_quantile=0.95):
    ll = 2 * L + 1
    N = len(y) // ll
    yy = y[:N * ll]
    qhi = np.quantile(np.abs(yy), tail_quantile)
    X = np.zeros((N, 14), dtype=float)
    if info_by_block is None or len(info_by_block) < N:
        known_info = np.linspace(0.0, 1.0, N)
    else:
        known_info = np.asarray(info_by_block[:N], dtype=float)

    for n in range(N):
        vals = yy[n * ll:(n + 1) * ll]
        t = 0.0 if N == 1 else n / (N - 1)
        X[n, 0] = 1.0
        X[n, 1] = t
        X[n, 2] = known_info[n]
        X[n, 3] = np.mean(vals)
        X[n, 4] = np.mean(np.abs(vals))
        X[n, 5] = np.std(vals)
        X[n, 6] = np.mean(vals ** 2)
        X[n, 7] = np.max(np.abs(vals))
        X[n, 8] = np.sum(np.abs(vals) >= qhi)
        X[n, 9] = np.mean(vals < 0.0)
        X[n, 10] = t * X[n, 4]
        X[n, 11] = t * X[n, 6]
        X[n, 12] = known_info[n] * X[n, 4]
        X[n, 13] = known_info[n] * X[n, 6]

    m = X[:, 1:].mean(axis=0)
    s = X[:, 1:].std(axis=0)
    s[s <= 1e-12] = 1.0
    X[:, 1:] = (X[:, 1:] - m) / s
    return X


def run_csg_mcmc_timevarying_info(scenario, method="tass", K=3, L=2, B=5, eps=1e-6,
                                  n_mcmc=5000, mb_size=10, eta=1e-5, seed=1,
                                  init_seed=None, pi0_init=None, sigmasq_init=10.0,
                                  sort_centres=True, online_rho=0.98, online_tau=1.0,
                                  online_lambda_floor=0.05, online_warmup=0,
                                  online_feedback_eps=1e-8, checkpoint_every=None):
    y = scenario["y"]
    delta = float(scenario.get("delta", 0.0))

    if delta == 0.0 and method in ("uniform", "tass"):
        return run_csg_mcmc_is_matlab(
            y, method=method, K=K, L=L, B=B, eps=eps, n_mcmc=n_mcmc,
            mb_size=mb_size, eta=eta, seed=seed, init_seed=init_seed,
            pi0_init=pi0_init, sigmasq_init=sigmasq_init,
            sort_centres=sort_centres, checkpoint_every=checkpoint_every)

    rng = np.random.default_rng(seed)
    if init_seed is None:
        init_seed = seed + 1000003
    init_rng = np.random.default_rng(init_seed)
    if pi0_init is None:
        pi0_init = normalize(init_rng.random(K))
    mu, sigmasq, A, A_hat, pi0 = matlab_initial_parameters(
        K, seed=init_seed, sigmasq_init=sigmasq_init, pi0_init=pi0_init)

    ll = 2 * L + 1
    N = len(y) // ll
    y = y[:N * ll]

    if method == "uniform":
        weights = uniform_weights(K, N)
        online = None
    elif method == "tass":
        weights = weights_by_clustering_matlab(y, K, L, eta=eta, seed=seed, sort_centres=sort_centres)
        online = None
    elif method == "online":
        features = block_features_timevarying(y, L, info_by_block=scenario.get("info_by_block", None))
        online = DiscountedRidgeOnlineWeights(
            features, K, rho=online_rho, tau=online_tau,
            lambda_floor=online_lambda_floor, feedback_eps=online_feedback_eps)
        weights = None
    else:
        raise ValueError("method must be uniform, tass, or online")

    chain_mu = np.zeros((n_mcmc + 1, K))
    chain_s2 = np.zeros((n_mcmc + 1, K))
    chain_A = np.zeros((n_mcmc + 1, K, K))
    chain_A_hat = np.zeros((n_mcmc + 1, K, K))
    chain_mu[0] = mu
    chain_s2[0] = sigmasq
    chain_A[0] = A
    chain_A_hat[0] = A_hat

    for itr in range(1, n_mcmc + 1):
        if method == "online":
            if itr <= online_warmup:
                uw = uniform_weights(K, N)
                gA, _ = transition_grad_IS(y, uw, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng)
                A, A_hat = sgld_update_transition(A_hat, gA, eps, rng)
                gmu, gs2, _ = emission_grad_IS(y, uw, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng)
            else:
                gA = transition_grad_online(y, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng, learn=True)
                A, A_hat = sgld_update_transition(A_hat, gA, eps, rng)
                gmu, gs2 = emission_grad_online(y, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng, learn=True)
        else:
            gA, _ = transition_grad_IS(y, weights, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng)
            A, A_hat = sgld_update_transition(A_hat, gA, eps, rng)
            gmu, gs2, _ = emission_grad_IS(y, weights, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng)

        mu, sigmasq = sgld_update_emission(mu, sigmasq, gmu, gs2, eps, rng)

        if itr % 10 == 0:
            A_hat = normalize_columns(A_hat)
            A = normalize_columns(A_hat)

        chain_mu[itr] = mu
        chain_s2[itr] = sigmasq
        chain_A[itr] = A
        chain_A_hat[itr] = A_hat

        if checkpoint_every is not None and (itr == 1 or itr % checkpoint_every == 0):
            print(f"{method}: {itr}/{n_mcmc}, mu={np.round(mu, 3).tolist()}, sigmasq={np.round(sigmasq, 3).tolist()}", flush=True)

    return {
        "method": method,
        "mu": chain_mu,
        "sigmasq": chain_s2,
        "A": chain_A,
        "A_hat": chain_A_hat,
        "weights": weights,
        "online": online,
        "pi0": pi0,
        "settings": {
            "K": K, "L": L, "B": B, "eps": eps, "n_mcmc": n_mcmc,
            "mb_size": mb_size, "seed": seed, "init_seed": init_seed,
            "sigmasq_init": sigmasq_init, "kind": scenario.get("kind"),
            "rare_count": scenario.get("rare_count"), "delta": delta,
            "timevarying_info": True,
        },
    }


def run_methods_timevarying_info(scenario, methods=("uniform", "tass", "online"), seed=1,
                                 init_seed=999, K=3, L=2, B=5, eps=1e-6,
                                 n_mcmc=5000, mb_size=10, eta=1e-5,
                                 online_rho=0.98, online_tau=1.0,
                                 online_lambda_floor=0.05, online_warmup=0,
                                 checkpoint_every=None):
    chains = {}
    for r, method in enumerate(methods):
        chains[method] = run_csg_mcmc_timevarying_info(
            scenario, method=method, K=K, L=L, B=B, eps=eps,
            n_mcmc=n_mcmc, mb_size=mb_size, eta=eta, seed=seed + r,
            init_seed=init_seed, online_rho=online_rho, online_tau=online_tau,
            online_lambda_floor=online_lambda_floor, online_warmup=online_warmup,
            checkpoint_every=checkpoint_every)
    return chains


def sorted_state_index(true_mu, state):
    order = np.argsort(true_mu)
    return int(np.where(order == state)[0][0])


def effective_mu_error(chain, effective_mu, state):
    ch = sort_chain_by_mu(chain)
    idx = sorted_state_index(effective_mu, state)
    target = np.sort(effective_mu)[idx]
    return np.abs(ch["mu"][:, idx] - target)


def heldout_indices_by_state_info(scenario, state, max_hold=200, seed=1, info_cut=0.5, group="all"):
    rng = np.random.default_rng(seed)
    ztest = scenario["ztest"]
    info = scenario.get("infotest", np.zeros_like(ztest, dtype=float))
    idx = np.where(ztest == state)[0]
    if group == "high":
        idx = idx[info[idx] >= info_cut]
    elif group == "low":
        idx = idx[info[idx] < info_cut]
    elif group != "all":
        raise ValueError("group must be all, high, or low")
    if len(idx) > max_hold:
        idx = rng.choice(idx, size=max_hold, replace=False)
    return idx


def logmeanexp(a):
    a = np.asarray(a, dtype=float)
    m = np.max(a)
    return float(m + np.log(np.mean(np.exp(a - m))))


def rare_log_pred_timevarying(chain, scenario, state, checkpoints=None, max_hold=200,
                              seed=1, group="all", info_cut=0.5):
    ch = sort_chain_by_mu(chain)
    true_mu = scenario["effective_mu_test"]
    idx_sorted = sorted_state_index(true_mu, state)
    idx = heldout_indices_by_state_info(scenario, state, max_hold=max_hold, seed=seed,
                                        info_cut=info_cut, group=group)
    if len(idx) == 0:
        return pd.DataFrame(columns=["method", "iteration", "state", "group", "mean_log_pred"])
    yy = scenario["ytest"][idx]
    if checkpoints is None:
        checkpoints = np.unique(np.linspace(1, len(ch["mu"]) - 1, 40).astype(int))
    rows = []
    for c in checkpoints:
        mus = ch["mu"][:c + 1, idx_sorted]
        s2s = np.maximum(ch["sigmasq"][:c + 1, idx_sorted], 1e-300)
        vals = []
        for val in yy:
            lp = -0.5 * (np.log(2.0 * np.pi * s2s) + (val - mus) ** 2 / s2s)
            vals.append(logmeanexp(lp))
        rows.append({
            "method": chain["method"],
            "iteration": int(c),
            "state": int(state),
            "group": group,
            "mean_log_pred": float(np.mean(vals)),
            "n_hold": int(len(idx)),
        })
    return pd.DataFrame(rows)


def block_state_counts(z, L, K=3):
    ll = 2 * L + 1
    N = len(z) // ll
    counts = np.zeros((N, K), dtype=int)
    zz = z[:N * ll]
    for n in range(N):
        vals = zz[n * ll:(n + 1) * ll]
        counts[n] = np.bincount(vals, minlength=K)
    return counts


def current_component_weights(chain, scenario, component="mu", state=2, L=2):
    ll = 2 * L + 1
    N = len(scenario["y"]) // ll
    method = chain["method"]
    if method == "uniform":
        return np.ones(N) / N
    if method == "tass":
        return normalize(chain["weights"][component][state])
    if method == "online":
        return normalize(chain["online"].weights()[component][state])
    raise ValueError("unknown method")


def q_diagnostic_timevarying(scenario, chain, iteration, component="mu", state=2, L=2, B=5):
    y = scenario["y"]
    ll = 2 * L + 1
    N = len(y) // ll
    y = y[:N * ll]
    mu = chain["mu"][iteration]
    sigmasq = chain["sigmasq"][iteration]
    A = chain["A"][iteration]
    pi0 = chain["pi0"]
    scores = np.zeros(N)
    for n in range(N):
        bmu, bs2, bA = block_gradients_matlab(y, n, L, B, A, pi0, mu, sigmasq)
        if component == "mu":
            scores[n] = bmu[state]
        elif component == "sigmasq":
            scores[n] = bs2[state]
        else:
            raise ValueError("component must be mu or sigmasq")
    oracle = normalize(np.abs(scores) + 1e-12)
    p = current_component_weights(chain, scenario, component=component, state=state, L=L)
    Q = float(np.sum(scores ** 2 / np.maximum(p, 1e-300)))
    Qoracle = float(np.sum(scores ** 2 / np.maximum(oracle, 1e-300)))
    return {
        "Q": Q,
        "Qoracle": Qoracle,
        "ratio_to_oracle": Q / Qoracle if Qoracle > 0 else np.nan,
        "oracle": oracle,
        "weights": p,
        "scores": scores,
    }


def high_info_rare_mass(scenario, chain, state=2, component="mu", L=2, info_cut=0.5):
    p = current_component_weights(chain, scenario, component=component, state=state, L=L)
    counts = block_state_counts(scenario["z"], L, K=3)
    info = scenario.get("info_by_block", np.zeros(len(p)))[:len(p)]
    mask = (counts[:, state] > 0) & (info >= info_cut)
    if not np.any(mask):
        return 0.0
    return float(np.sum(p[mask]))


def pair_summary_table(scenario, chains, L=2, B=5, max_hold=200, seed=1):
    rows = []
    states = scenario.get("rare_states", [])
    if len(states) == 0:
        states = [2]
    final_it = next(iter(chains.values()))["mu"].shape[0] - 1
    for method, chain in chains.items():
        for state in states:
            err_curve = effective_mu_error(chain, scenario["effective_mu_train"], state)
            qd = q_diagnostic_timevarying(scenario, chain, final_it, component="mu", state=state, L=L, B=B)
            lp_all = rare_log_pred_timevarying(chain, scenario, state, checkpoints=[final_it],
                                               max_hold=max_hold, seed=seed, group="all")
            lp_high = rare_log_pred_timevarying(chain, scenario, state, checkpoints=[final_it],
                                                max_hold=max_hold, seed=seed, group="high")
            lp_low = rare_log_pred_timevarying(chain, scenario, state, checkpoints=[final_it],
                                               max_hold=max_hold, seed=seed, group="low")
            rows.append({
                "rare_count": scenario.get("rare_count"),
                "delta": scenario.get("delta"),
                "state": int(state),
                "method": method,
                "final_effective_mu_error": float(err_curve[-1]),
                "mean_effective_mu_error_second_half": float(np.mean(err_curve[len(err_curve)//2:])),
                "final_log_pred_all": float(lp_all["mean_log_pred"].iloc[-1]) if len(lp_all) else np.nan,
                "final_log_pred_high_info": float(lp_high["mean_log_pred"].iloc[-1]) if len(lp_high) else np.nan,
                "final_log_pred_low_info": float(lp_low["mean_log_pred"].iloc[-1]) if len(lp_low) else np.nan,
                "Q_ratio_to_oracle": float(qd["ratio_to_oracle"]),
                "high_info_rare_mass": high_info_rare_mass(scenario, chain, state=state, L=L),
            })
    return pd.DataFrame(rows)


# Single-changepoint mean-shift scenario patch.
#
# The earlier switch-profile experiment could put the switch at the train/test
# boundary when Ttrain == Ttest, which meant the training set had essentially no
# high-information shifted rare blocks. This patch makes the intervention explicit:
#
#   y_t = mu_{z_t} + delta * d_{z_t} * 1{t >= t_star} * 1{z_t rare} + eps_t,
#
# where t_star is a single changepoint inside the training sequence. The same
# post-changepoint regime then continues into the test sequence. The existing
# delta sweep controls the size of this one mean shift.

CHANGEPOINT_FRACTION = 0.50
CHANGEPOINT_KIND = "mean"


def _block_means_from_time_indicator(indicator, L):
    indicator = np.asarray(indicator, dtype=float)
    ll = 2 * L + 1
    N = len(indicator) // ll
    out = np.zeros(N, dtype=float)
    for n in range(N):
        vals = indicator[n * ll:(n + 1) * ll]
        out[n] = float(np.mean(vals)) if len(vals) else 0.0
    return out


def make_timevarying_info_scenario(rare_count=1, delta=0.0, Ttrain=10**6, Ttest=10**6,
                                   seed=1, L=2, info_mode="changepoint_mean", info_slope=10.0,
                                   changepoint_fraction=None):
    """Construct a one-changepoint rare-state mean-shift scenario.

    The latent HMM transition matrix is unchanged. Only the observation mean of
    the rare state(s) changes after a single training-set changepoint. The
    post-changepoint indicator is stored in `info`/`infotest` and block-averaged
    in `info_by_block`/`info_by_block_test`, so the existing diagnostics for
    high-information rare blocks continue to work.
    """
    kind = kind_from_rare_count(rare_count)
    A, mu, sigmasq = paper_transition(kind)
    pi0 = np.ones(3) / 3.0

    Ttrain = int(Ttrain)
    Ttest = int(Ttest)
    Ttotal = int(Ttrain + Ttest)
    if changepoint_fraction is None:
        changepoint_fraction = globals().get("CHANGEPOINT_FRACTION", 0.50)
    changepoint_fraction = float(changepoint_fraction)
    if not (0.0 < changepoint_fraction < 1.0):
        raise ValueError("changepoint_fraction must lie strictly between 0 and 1")

    # Deliberately place the changepoint inside the training sequence.
    t_star = int(round(changepoint_fraction * Ttrain))
    t_star = max(1, min(Ttrain - 1, t_star))

    z_all, y_base = simulate_hmm(A, mu, sigmasq, Ttotal, pi0=pi0, seed=seed)
    post = (np.arange(Ttotal) >= t_star).astype(float)

    directions = rare_directions_from_kind(kind)
    shift = np.zeros(Ttotal, dtype=float)
    for state, direction in directions.items():
        mask = (z_all == state)
        shift[mask] = float(delta) * float(direction) * post[mask]

    y_all = y_base + shift

    z = z_all[:Ttrain]
    y = y_all[:Ttrain]
    ztest = z_all[Ttrain:]
    ytest = y_all[Ttrain:]
    info_train = post[:Ttrain]
    info_test = post[Ttrain:]

    ll = 2 * L + 1
    Ntrain = len(y) // ll
    Ntest = len(ytest) // ll
    info_by_block = _block_means_from_time_indicator(info_train[:Ntrain * ll], L)
    info_by_block_test = _block_means_from_time_indicator(info_test[:Ntest * ll], L)

    effective_mu_train = mu.copy()
    effective_mu_test = mu.copy()
    for state, direction in directions.items():
        mask_train = (z == state)
        mask_test = (ztest == state)
        if np.any(mask_train):
            effective_mu_train[state] = mu[state] + float(delta) * float(direction) * float(np.mean(info_train[mask_train]))
        if np.any(mask_test):
            effective_mu_test[state] = mu[state] + float(delta) * float(direction) * float(np.mean(info_test[mask_test]))

    return {
        "kind": kind,
        "rare_count": int(rare_count),
        "A": A,
        "mu": mu,
        "sigmasq": sigmasq,
        "pi0": pi0,
        "z": z,
        "y": y,
        "ztest": ztest,
        "ytest": ytest,
        "delta": float(delta),
        "info_mode": "changepoint_mean",
        "change_type": "single_changepoint_mean",
        "changepoint_fraction": changepoint_fraction,
        "changepoint_index_train": int(t_star),
        "info": info_train,
        "infotest": info_test,
        "info_by_block": info_by_block,
        "info_by_block_test": info_by_block_test,
        "rare_states": rare_states_from_kind(kind),
        "rare_directions": directions,
        "effective_mu_train": effective_mu_train,
        "effective_mu_test": effective_mu_test,
    }

print("Single-changepoint mean-shift scenario patch loaded.")


# Stability + timing patch.
# Minimal changes:
# 1. Solve online discounted ridge systems with diagonal jitter and a pinv fallback.
# 2. Record wall-clock seconds in each returned chain.
# 3. Add wall-clock columns to the pair summary table.

from time import perf_counter


def safe_ridge_solve(A, b, jitter=1e-5, max_tries=8):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    A = np.nan_to_num(A, nan=0.0, posinf=1e12, neginf=-1e12)
    b = np.nan_to_num(b, nan=0.0, posinf=1e12, neginf=-1e12)
    I = np.eye(A.shape[0])

    lam = float(jitter)
    for _ in range(max_tries):
        try:
            return np.linalg.solve(A + lam * I, b)
        except np.linalg.LinAlgError:
            lam *= 10.0

    return np.linalg.pinv(A + lam * I) @ b


def _online_update_mu_stable(self, j, block, value):
    x = self.W[block]
    f = np.log(float(value) ** 2 + self.feedback_eps)
    self.A_mu[j] = self.rho * self.A_mu[j] + np.outer(x, x)
    self.b_mu[j] = self.rho * self.b_mu[j] + x * f
    self.beta_mu[j] = safe_ridge_solve(self.A_mu[j], self.b_mu[j], jitter=ONLINE_RIDGE_JITTER if "ONLINE_RIDGE_JITTER" in globals() else 1e-5)


def _online_update_s2_stable(self, j, block, value):
    x = self.W[block]
    f = np.log(float(value) ** 2 + self.feedback_eps)
    self.A_s2[j] = self.rho * self.A_s2[j] + np.outer(x, x)
    self.b_s2[j] = self.rho * self.b_s2[j] + x * f
    self.beta_s2[j] = safe_ridge_solve(self.A_s2[j], self.b_s2[j], jitter=ONLINE_RIDGE_JITTER if "ONLINE_RIDGE_JITTER" in globals() else 1e-5)


def _online_update_A_stable(self, j, k, block, value):
    x = self.W[block]
    f = np.log(float(value) ** 2 + self.feedback_eps)
    self.A_tr[j, k] = self.rho * self.A_tr[j, k] + np.outer(x, x)
    self.b_tr[j, k] = self.rho * self.b_tr[j, k] + x * f
    self.beta_tr[j, k] = safe_ridge_solve(self.A_tr[j, k], self.b_tr[j, k], jitter=ONLINE_RIDGE_JITTER if "ONLINE_RIDGE_JITTER" in globals() else 1e-5)


DiscountedRidgeOnlineWeights.update_mu = _online_update_mu_stable
DiscountedRidgeOnlineWeights.update_s2 = _online_update_s2_stable
DiscountedRidgeOnlineWeights.update_A = _online_update_A_stable


_original_run_csg_mcmc_timevarying_info = run_csg_mcmc_timevarying_info


def run_csg_mcmc_timevarying_info(*args, **kwargs):
    tic = perf_counter()
    out = _original_run_csg_mcmc_timevarying_info(*args, **kwargs)
    elapsed = perf_counter() - tic
    out["wallclock_seconds"] = float(elapsed)
    out["wallclock_minutes"] = float(elapsed / 60.0)
    out.setdefault("settings", {})["wallclock_seconds"] = float(elapsed)
    out.setdefault("settings", {})["wallclock_minutes"] = float(elapsed / 60.0)
    return out


_original_pair_summary_table = pair_summary_table


def pair_summary_table(scenario, chains, L=2, B=5, max_hold=200, seed=1):
    df = _original_pair_summary_table(scenario, chains, L=L, B=B, max_hold=max_hold, seed=seed)
    elapsed = {method: float(chain.get("wallclock_seconds", np.nan)) for method, chain in chains.items()}
    df["wallclock_seconds"] = df["method"].map(elapsed)
    df["wallclock_minutes"] = df["wallclock_seconds"] / 60.0
    return df


_original_run_methods_timevarying_info = run_methods_timevarying_info


def run_methods_timevarying_info(scenario, methods=("uniform", "tass", "online"), seed=1,
                                 init_seed=999, K=3, L=2, B=5, eps=1e-6,
                                 n_mcmc=5000, mb_size=10, eta=1e-5,
                                 online_rho=0.98, online_tau=1.0,
                                 online_lambda_floor=0.05, online_warmup=0,
                                 online_feedback_eps=1e-6,
                                 checkpoint_every=None):
    chains = {}
    for r, method in enumerate(methods):
        chains[method] = run_csg_mcmc_timevarying_info(
            scenario, method=method, K=K, L=L, B=B, eps=eps,
            n_mcmc=n_mcmc, mb_size=mb_size, eta=eta, seed=seed + r,
            init_seed=init_seed, online_rho=online_rho, online_tau=online_tau,
            online_lambda_floor=online_lambda_floor, online_warmup=online_warmup,
            online_feedback_eps=online_feedback_eps,
            checkpoint_every=checkpoint_every)
        if "checkpoint" in globals():
            checkpoint(f"finished method={method}, elapsed={chains[method].get('wallclock_seconds', np.nan):.1f}s")
    return chains

print("Stability/timing patch loaded.")


# Online+TASS patch.
# This adds a fourth method, "online_tass".
# It uses paper-style componentwise TASS weights as an anchor, then learns an online
# multiplicative correction from discounted ridge feedback:
#     p_{n,t}^{online+tass} \propto p_n^{TASS} exp(0.5 w_n^T beta_t), with a floor.

ONLINE_ANCHOR_POWER = 1.0


class AnchoredDiscountedRidgeOnlineWeights:
    def __init__(self, features, K, base_weights, rho=0.95, tau=10.0,
                 lambda_floor=0.10, feedback_eps=1e-6, anchor_power=1.0,
                 ridge_jitter=1e-5):
        self.W = np.asarray(features, dtype=float)
        self.N, self.p = self.W.shape
        self.K = K
        self.base = base_weights
        self.rho = float(rho)
        self.tau = float(tau)
        self.lambda_floor = float(lambda_floor)
        self.feedback_eps = float(feedback_eps)
        self.anchor_power = float(anchor_power)
        self.ridge_jitter = float(ridge_jitter)

        self.A_mu = np.tile(self.tau * np.eye(self.p), (K, 1, 1))
        self.b_mu = np.zeros((K, self.p))
        self.beta_mu = np.zeros((K, self.p))

        self.A_s2 = np.tile(self.tau * np.eye(self.p), (K, 1, 1))
        self.b_s2 = np.zeros((K, self.p))
        self.beta_s2 = np.zeros((K, self.p))

        self.A_tr = np.tile(self.tau * np.eye(self.p), (K, K, 1, 1))
        self.b_tr = np.zeros((K, K, self.p))
        self.beta_tr = np.zeros((K, K, self.p))

    def _prob(self, beta, base):
        score = np.clip(0.5 * (self.W @ beta), -30.0, 30.0)
        base = np.maximum(np.asarray(base, dtype=float), 1e-300)
        raw = np.exp(score) * base ** self.anchor_power
        p = normalize(raw)
        if self.lambda_floor > 0:
            p = (1.0 - self.lambda_floor) * p + self.lambda_floor / self.N
        return normalize(p)

    def weights(self):
        wmu = np.zeros((self.K, self.N))
        ws2 = np.zeros((self.K, self.N))
        wA = np.zeros((self.K, self.K, self.N))

        for j in range(self.K):
            wmu[j] = self._prob(self.beta_mu[j], self.base["mu"][j])
            ws2[j] = self._prob(self.beta_s2[j], self.base["sigmasq"][j])

        for j in range(self.K):
            for k in range(self.K):
                wA[j, k] = self._prob(self.beta_tr[j, k], self.base["A"][j, k])

        return {"mu": wmu, "sigmasq": ws2, "A": wA}

    def update_mu(self, j, block, value):
        x = self.W[block]
        f = np.log(float(value) ** 2 + self.feedback_eps)
        self.A_mu[j] = self.rho * self.A_mu[j] + np.outer(x, x)
        self.b_mu[j] = self.rho * self.b_mu[j] + x * f
        self.beta_mu[j] = safe_ridge_solve(self.A_mu[j], self.b_mu[j], jitter=self.ridge_jitter)

    def update_s2(self, j, block, value):
        x = self.W[block]
        f = np.log(float(value) ** 2 + self.feedback_eps)
        self.A_s2[j] = self.rho * self.A_s2[j] + np.outer(x, x)
        self.b_s2[j] = self.rho * self.b_s2[j] + x * f
        self.beta_s2[j] = safe_ridge_solve(self.A_s2[j], self.b_s2[j], jitter=self.ridge_jitter)

    def update_A(self, j, k, block, value):
        x = self.W[block]
        f = np.log(float(value) ** 2 + self.feedback_eps)
        self.A_tr[j, k] = self.rho * self.A_tr[j, k] + np.outer(x, x)
        self.b_tr[j, k] = self.rho * self.b_tr[j, k] + x * f
        self.beta_tr[j, k] = safe_ridge_solve(self.A_tr[j, k], self.b_tr[j, k], jitter=self.ridge_jitter)


def run_csg_mcmc_timevarying_info(scenario, method="tass", K=3, L=2, B=5, eps=1e-6,
                                  n_mcmc=5000, mb_size=10, eta=1e-5, seed=1,
                                  init_seed=None, pi0_init=None, sigmasq_init=10.0,
                                  sort_centres=True, online_rho=0.98, online_tau=1.0,
                                  online_lambda_floor=0.05, online_warmup=0,
                                  online_feedback_eps=1e-8, checkpoint_every=None):
    tic = perf_counter()
    y = scenario["y"]
    delta = float(scenario.get("delta", 0.0))

    if delta == 0.0 and method in ("uniform", "tass"):
        out = run_csg_mcmc_is_matlab(
            y, method=method, K=K, L=L, B=B, eps=eps, n_mcmc=n_mcmc,
            mb_size=mb_size, eta=eta, seed=seed, init_seed=init_seed,
            pi0_init=pi0_init, sigmasq_init=sigmasq_init,
            sort_centres=sort_centres, checkpoint_every=checkpoint_every)
        elapsed = perf_counter() - tic
        out["wallclock_seconds"] = float(elapsed)
        out["wallclock_minutes"] = float(elapsed / 60.0)
        out.setdefault("settings", {})["wallclock_seconds"] = float(elapsed)
        out.setdefault("settings", {})["wallclock_minutes"] = float(elapsed / 60.0)
        return out

    rng = np.random.default_rng(seed)
    if init_seed is None:
        init_seed = seed + 1000003
    init_rng = np.random.default_rng(init_seed)
    if pi0_init is None:
        pi0_init = normalize(init_rng.random(K))
    mu, sigmasq, A, A_hat, pi0 = matlab_initial_parameters(
        K, seed=init_seed, sigmasq_init=sigmasq_init, pi0_init=pi0_init)

    ll = 2 * L + 1
    N = len(y) // ll
    y = y[:N * ll]

    weights = None
    online = None

    if method == "uniform":
        weights = uniform_weights(K, N)
    elif method == "tass":
        weights = weights_by_clustering_matlab(y, K, L, eta=eta, seed=seed, sort_centres=sort_centres)
    elif method == "online":
        features = block_features_timevarying(y, L, info_by_block=scenario.get("info_by_block", None))
        online = DiscountedRidgeOnlineWeights(
            features, K, rho=online_rho, tau=online_tau,
            lambda_floor=online_lambda_floor, feedback_eps=online_feedback_eps)
    elif method == "online_tass":
        weights = weights_by_clustering_matlab(y, K, L, eta=eta, seed=seed, sort_centres=sort_centres)
        features = block_features_timevarying(y, L, info_by_block=scenario.get("info_by_block", None))
        online = AnchoredDiscountedRidgeOnlineWeights(
            features, K, base_weights=weights, rho=online_rho, tau=online_tau,
            lambda_floor=online_lambda_floor, feedback_eps=online_feedback_eps,
            anchor_power=ONLINE_ANCHOR_POWER,
            ridge_jitter=ONLINE_RIDGE_JITTER if "ONLINE_RIDGE_JITTER" in globals() else 1e-5)
    else:
        raise ValueError("method must be uniform, tass, online, or online_tass")

    chain_mu = np.zeros((n_mcmc + 1, K))
    chain_s2 = np.zeros((n_mcmc + 1, K))
    chain_A = np.zeros((n_mcmc + 1, K, K))
    chain_A_hat = np.zeros((n_mcmc + 1, K, K))
    chain_mu[0] = mu
    chain_s2[0] = sigmasq
    chain_A[0] = A
    chain_A_hat[0] = A_hat

    for itr in range(1, n_mcmc + 1):
        if method in ("online", "online_tass"):
            if itr <= online_warmup:
                if method == "online_tass" and weights is not None:
                    warmup_weights = weights
                else:
                    warmup_weights = uniform_weights(K, N)
                gA, _ = transition_grad_IS(y, warmup_weights, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng)
                A, A_hat = sgld_update_transition(A_hat, gA, eps, rng)
                gmu, gs2, _ = emission_grad_IS(y, warmup_weights, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng)
            else:
                gA = transition_grad_online(y, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng, learn=True)
                A, A_hat = sgld_update_transition(A_hat, gA, eps, rng)
                gmu, gs2 = emission_grad_online(y, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng, learn=True)
        else:
            gA, _ = transition_grad_IS(y, weights, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng)
            A, A_hat = sgld_update_transition(A_hat, gA, eps, rng)
            gmu, gs2, _ = emission_grad_IS(y, weights, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng)

        mu, sigmasq = sgld_update_emission(mu, sigmasq, gmu, gs2, eps, rng)

        if itr % 10 == 0:
            A_hat = normalize_columns(A_hat)
            A = normalize_columns(A_hat)

        chain_mu[itr] = mu
        chain_s2[itr] = sigmasq
        chain_A[itr] = A
        chain_A_hat[itr] = A_hat

        if checkpoint_every is not None and (itr == 1 or itr % checkpoint_every == 0):
            print(f"{method}: {itr}/{n_mcmc}, mu={np.round(mu, 3).tolist()}, sigmasq={np.round(sigmasq, 3).tolist()}", flush=True)

    elapsed = perf_counter() - tic
    return {
        "method": method,
        "mu": chain_mu,
        "sigmasq": chain_s2,
        "A": chain_A,
        "A_hat": chain_A_hat,
        "weights": weights,
        "online": online,
        "pi0": pi0,
        "wallclock_seconds": float(elapsed),
        "wallclock_minutes": float(elapsed / 60.0),
        "settings": {
            "K": K, "L": L, "B": B, "eps": eps, "n_mcmc": n_mcmc,
            "mb_size": mb_size, "seed": seed, "init_seed": init_seed,
            "sigmasq_init": sigmasq_init, "kind": scenario.get("kind"),
            "rare_count": scenario.get("rare_count"), "delta": delta,
            "timevarying_info": True,
            "online_rho": online_rho,
            "online_tau": online_tau,
            "online_lambda_floor": online_lambda_floor,
            "online_warmup": online_warmup,
            "wallclock_seconds": float(elapsed),
            "wallclock_minutes": float(elapsed / 60.0),
        },
    }


def run_methods_timevarying_info(scenario, methods=("uniform", "tass", "online", "online_tass"), seed=1,
                                 init_seed=999, K=3, L=2, B=5, eps=1e-6,
                                 n_mcmc=5000, mb_size=10, eta=1e-5,
                                 online_rho=0.98, online_tau=1.0,
                                 online_lambda_floor=0.05, online_warmup=0,
                                 online_feedback_eps=1e-6,
                                 checkpoint_every=None):
    chains = {}
    for r, method in enumerate(methods):
        chains[method] = run_csg_mcmc_timevarying_info(
            scenario, method=method, K=K, L=L, B=B, eps=eps,
            n_mcmc=n_mcmc, mb_size=mb_size, eta=eta, seed=seed + r,
            init_seed=init_seed, online_rho=online_rho, online_tau=online_tau,
            online_lambda_floor=online_lambda_floor, online_warmup=online_warmup,
            online_feedback_eps=online_feedback_eps,
            checkpoint_every=checkpoint_every)
        if "checkpoint" in globals():
            checkpoint(f"finished method={method}, elapsed={chains[method].get('wallclock_seconds', np.nan):.1f}s")
    return chains


def current_component_weights(chain, scenario, component="mu", state=2, L=2):
    ll = 2 * L + 1
    N = len(scenario["y"]) // ll
    method = chain["method"]
    if method == "uniform":
        return np.ones(N) / N
    if method == "tass":
        return normalize(chain["weights"][component][state])
    if method in ("online", "online_tass"):
        return normalize(chain["online"].weights()[component][state])
    raise ValueError("unknown method")

METHODS = ("uniform", "tass", "online", "online_tass")
ONLINE_WARMUP = 0
print("Online+TASS patch loaded. METHODS=", METHODS)


# Groupwise online DRR patch.
# This adds two methods:
#   "online_group": groupwise multivariate online DRR with no static initialisation;
#   "online_tass_group": groupwise multivariate online DRR whose beta_0 is initialised
#                        from Static-TASS proxy group scores only.
#
# In contrast with the earlier "online_tass" anchored method, static TASS does not
# remain as a multiplicative anchor in "online_tass_group". It only sets beta_0.

GROUPWISE_METHODS = ("online_group", "online_tass_group")
GROUPWISE_NAMES = ("mu", "sigmasq", "A")


def static_tass_group_scores(y, K, L, eta=1e-5, seed=1, sort_centres=True,
                             pilot_mu=None, pilot_sigmasq=None, pilot_A=None):
    """Return Static-TASS proxy group scores a_{n,r}^{static}.

    The construction mirrors the notebook's componentwise Static TASS proxy:
    K-means labels provide blockwise approximate sufficient statistics, and a
    pilot parameter value theta_0 gives the reference around which block-gradient
    magnitudes are approximated.  The returned score for a group r is the norm of
    the corresponding componentwise proxy vector/matrix for that block.
    """
    z, centres = kmeans_labels_matlab_style(y, K, seed=seed, sort_centres=sort_centres)
    ll = 2 * L + 1
    N = len(y) // ll
    y = y[:N * ll]
    z = z[:N * ll]

    ybar = np.zeros(K)
    s2 = np.zeros(K)
    for j in range(K):
        vals = y[z == j]
        ybar[j] = np.mean(vals) if len(vals) else 0.0
        s2[j] = np.var(vals, ddof=1) if len(vals) > 1 else 0.0

    if pilot_mu is None:
        pilot_mu = ybar.copy()
    else:
        pilot_mu = np.asarray(pilot_mu, dtype=float)

    if pilot_sigmasq is None:
        pilot_sigmasq = np.maximum(s2.copy(), eta ** 2)
    else:
        pilot_sigmasq = np.maximum(np.asarray(pilot_sigmasq, dtype=float), eta ** 2)

    c = np.zeros((N, K))
    ybar_chunk = np.zeros((N, K))
    s2_chunk = np.zeros((N, K))
    xi = np.zeros((N, K, K))

    for n in range(N):
        idx = slice(n * ll, (n + 1) * ll)
        yn = y[idx]
        zn = z[idx]
        for j in range(K):
            mask = zn == j
            c[n, j] = np.sum(mask)
            if c[n, j] > 0:
                vals = yn[mask]
                ybar_chunk[n, j] = np.mean(vals)
                s2_chunk[n, j] = np.sum((vals - ybar[j]) ** 2) / c[n, j]
        for old in range(K):
            for new in range(K):
                xi[n, old, new] = np.sum((zn[:-1] == old) & (zn[1:] == new))

    xisum = np.sum(xi, axis=0) + 0.5
    A_map = normalize_columns(xisum.T)
    if pilot_A is not None:
        A_map = normalize_columns(np.asarray(pilot_A, dtype=float))

    mu_proxy = np.zeros((N, K))
    s2_proxy = np.zeros((N, K))
    A_proxy = np.zeros((N, K, K))

    for n in range(N):
        for j in range(K):
            mu_proxy[n, j] = c[n, j] * abs(ybar_chunk[n, j] - pilot_mu[j]) + eta ** 2
            s2_proxy[n, j] = c[n, j] * (pilot_sigmasq[j] + s2_chunk[n, j]) + eta ** 2
        for new in range(K):
            for old in range(K):
                A_proxy[n, new, old] = xi[n, old, new] / max(A_map[new, old], 1e-300) + eta ** 2

    scores = {
        "mu": np.linalg.norm(mu_proxy, axis=1),
        "sigmasq": np.linalg.norm(s2_proxy, axis=1),
        "A": np.linalg.norm(A_proxy.reshape(N, -1), axis=1),
    }
    for key in GROUPWISE_NAMES:
        scores[key] = np.maximum(np.asarray(scores[key], dtype=float), eta ** 2)

    return {
        "scores": scores,
        "labels": z,
        "centres": centres,
        "pilot_mu": pilot_mu,
        "pilot_sigmasq": pilot_sigmasq,
        "pilot_A": A_map,
    }


class GroupwiseDiscountedRidgeOnlineWeights:
    """Multivariate/groupwise discounted ridge regression weights.

    For each group r in {mu, sigmasq, A}, one ridge model predicts
        log(||P_r g_n(theta_t)||^2 + eps)
    from block features w_n.  The same resulting block probability vector is
    used for every scalar parameter inside that group.
    """

    def __init__(self, features, K, rho=0.95, tau=10.0, lambda_floor=0.10,
                 feedback_eps=1e-6, static_scores=None, ridge_jitter=1e-5):
        self.W = np.asarray(features, dtype=float)
        self.N, self.p = self.W.shape
        self.K = int(K)
        self.rho = float(rho)
        self.tau = float(tau)
        self.lambda_floor = float(lambda_floor)
        self.feedback_eps = float(feedback_eps)
        self.ridge_jitter = float(ridge_jitter)
        self.groups = GROUPWISE_NAMES

        self.A = {g: self.tau * np.eye(self.p) for g in self.groups}
        self.b = {g: np.zeros(self.p) for g in self.groups}
        self.beta = {g: np.zeros(self.p) for g in self.groups}
        self.static_scores = static_scores

        if static_scores is not None:
            for g in self.groups:
                self.initialise_group_from_scores(g, static_scores[g])

    def initialise_group_from_scores(self, group, scores):
        scores = np.maximum(np.asarray(scores, dtype=float), 0.0)
        pseudo_response = np.log(scores ** 2 + self.feedback_eps)
        self.A[group] = self.W.T @ self.W + self.tau * np.eye(self.p)
        self.b[group] = self.W.T @ pseudo_response
        self.beta[group] = safe_ridge_solve(self.A[group], self.b[group], jitter=self.ridge_jitter)

    def _prob(self, beta):
        score = np.clip(0.5 * (self.W @ beta), -30.0, 30.0)
        raw = np.exp(score)
        p = normalize(raw)
        if self.lambda_floor > 0:
            p = (1.0 - self.lambda_floor) * p + self.lambda_floor / self.N
        return normalize(p)

    def group_prob(self, group):
        return self._prob(self.beta[group])

    def weights(self):
        p_mu = self.group_prob("mu")
        p_s2 = self.group_prob("sigmasq")
        p_A = self.group_prob("A")
        return {
            "mu": np.tile(p_mu[None, :], (self.K, 1)),
            "sigmasq": np.tile(p_s2[None, :], (self.K, 1)),
            "A": np.tile(p_A[None, None, :], (self.K, self.K, 1)),
            "groups": {"mu": p_mu, "sigmasq": p_s2, "A": p_A},
        }

    def update_group(self, group, block, grad_value):
        x = self.W[int(block)]
        g = np.asarray(grad_value, dtype=float)
        f = np.log(float(np.sum(g ** 2)) + self.feedback_eps)
        self.A[group] = self.rho * self.A[group] + np.outer(x, x)
        self.b[group] = self.rho * self.b[group] + x * f
        self.beta[group] = safe_ridge_solve(self.A[group], self.b[group], jitter=self.ridge_jitter)


def transition_grad_online_groupwise(y, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng,
                                     learn=True):
    K = len(mu)
    N = len(y) // (2 * L + 1)
    _, _, gA = prior_gradients(mu, sigmasq, A_hat)
    cache = {}

    def get_block(i):
        if i not in cache:
            cache[i] = block_gradients_matlab(y, i, L, B, A, pi0, mu, sigmasq)
        return cache[i]

    w = online.group_prob("A")
    mb = rng.choice(N, size=mb_size, replace=True, p=w)
    for i in mb:
        _, _, bA = get_block(int(i))
        gA -= bA / max(w[int(i)], 1e-300)
        if learn:
            online.update_group("A", int(i), bA)
    gA /= mb_size
    return gA


def emission_grad_online_groupwise(y, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng,
                                   learn=True):
    N = len(y) // (2 * L + 1)
    gmu, gs2, _ = prior_gradients(mu, sigmasq, A_hat)
    cache = {}

    def get_block(i):
        if i not in cache:
            cache[i] = block_gradients_matlab(y, i, L, B, A, pi0, mu, sigmasq)
        return cache[i]

    w_mu = online.group_prob("mu")
    mb_mu = rng.choice(N, size=mb_size, replace=True, p=w_mu)
    for i in mb_mu:
        bmu, _, _ = get_block(int(i))
        gmu -= bmu / max(w_mu[int(i)], 1e-300)
        if learn:
            online.update_group("mu", int(i), bmu)

    w_s2 = online.group_prob("sigmasq")
    mb_s2 = rng.choice(N, size=mb_size, replace=True, p=w_s2)
    for i in mb_s2:
        _, bs2, _ = get_block(int(i))
        gs2 -= bs2 / max(w_s2[int(i)], 1e-300)
        if learn:
            online.update_group("sigmasq", int(i), bs2)

    gmu /= mb_size
    gs2 /= mb_size
    return gmu, gs2


# Preserve the previous runner under a name that can still be called explicitly.
run_csg_mcmc_timevarying_info_componentwise = run_csg_mcmc_timevarying_info


def run_csg_mcmc_timevarying_info(scenario, method="tass", K=3, L=2, B=5, eps=1e-6,
                                  n_mcmc=5000, mb_size=10, eta=1e-5, seed=1,
                                  init_seed=None, pi0_init=None, sigmasq_init=10.0,
                                  sort_centres=True, online_rho=0.98, online_tau=1.0,
                                  online_lambda_floor=0.05, online_warmup=0,
                                  online_feedback_eps=1e-8, checkpoint_every=None):
    if method not in GROUPWISE_METHODS:
        return run_csg_mcmc_timevarying_info_componentwise(
            scenario, method=method, K=K, L=L, B=B, eps=eps, n_mcmc=n_mcmc,
            mb_size=mb_size, eta=eta, seed=seed, init_seed=init_seed,
            pi0_init=pi0_init, sigmasq_init=sigmasq_init,
            sort_centres=sort_centres, online_rho=online_rho, online_tau=online_tau,
            online_lambda_floor=online_lambda_floor, online_warmup=online_warmup,
            online_feedback_eps=online_feedback_eps, checkpoint_every=checkpoint_every)

    tic = perf_counter()
    y = scenario["y"]
    delta = float(scenario.get("delta", 0.0))

    rng = np.random.default_rng(seed)
    if init_seed is None:
        init_seed = seed + 1000003
    init_rng = np.random.default_rng(init_seed)
    if pi0_init is None:
        pi0_init = normalize(init_rng.random(K))
    mu, sigmasq, A, A_hat, pi0 = matlab_initial_parameters(
        K, seed=init_seed, sigmasq_init=sigmasq_init, pi0_init=pi0_init)

    ll = 2 * L + 1
    N = len(y) // ll
    y = y[:N * ll]
    features = block_features_timevarying(y, L, info_by_block=scenario.get("info_by_block", None))

    static_proxy = None
    static_scores = None
    if method == "online_tass_group":
        static_proxy = static_tass_group_scores(
            y, K, L, eta=eta, seed=seed, sort_centres=sort_centres,
            pilot_mu=mu, pilot_sigmasq=sigmasq, pilot_A=A)
        static_scores = static_proxy["scores"]

    online = GroupwiseDiscountedRidgeOnlineWeights(
        features, K, rho=online_rho, tau=online_tau,
        lambda_floor=online_lambda_floor, feedback_eps=online_feedback_eps,
        static_scores=static_scores,
        ridge_jitter=ONLINE_RIDGE_JITTER if "ONLINE_RIDGE_JITTER" in globals() else 1e-5)

    chain_mu = np.zeros((n_mcmc + 1, K))
    chain_s2 = np.zeros((n_mcmc + 1, K))
    chain_A = np.zeros((n_mcmc + 1, K, K))
    chain_A_hat = np.zeros((n_mcmc + 1, K, K))
    chain_mu[0] = mu
    chain_s2[0] = sigmasq
    chain_A[0] = A
    chain_A_hat[0] = A_hat

    for itr in range(1, n_mcmc + 1):
        if itr <= online_warmup:
            # Warmup is optional.  It is deliberately independent of the TASS
            # beta_0 initialisation, so static TASS still only affects beta_0.
            warmup_weights = uniform_weights(K, N)
            gA, _ = transition_grad_IS(y, warmup_weights, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng)
            A, A_hat = sgld_update_transition(A_hat, gA, eps, rng)
            gmu, gs2, _ = emission_grad_IS(y, warmup_weights, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng)
        else:
            gA = transition_grad_online_groupwise(
                y, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng, learn=True)
            A, A_hat = sgld_update_transition(A_hat, gA, eps, rng)
            gmu, gs2 = emission_grad_online_groupwise(
                y, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng, learn=True)

        mu, sigmasq = sgld_update_emission(mu, sigmasq, gmu, gs2, eps, rng)

        if itr % 10 == 0:
            A_hat = normalize_columns(A_hat)
            A = normalize_columns(A_hat)

        chain_mu[itr] = mu
        chain_s2[itr] = sigmasq
        chain_A[itr] = A
        chain_A_hat[itr] = A_hat

        if checkpoint_every is not None and (itr == 1 or itr % checkpoint_every == 0):
            print(f"{method}: {itr}/{n_mcmc}, mu={np.round(mu, 3).tolist()}, sigmasq={np.round(sigmasq, 3).tolist()}", flush=True)

    elapsed = perf_counter() - tic
    return {
        "method": method,
        "mu": chain_mu,
        "sigmasq": chain_s2,
        "A": chain_A,
        "A_hat": chain_A_hat,
        "weights": None,
        "online": online,
        "static_group_scores": static_scores,
        "static_group_proxy": static_proxy,
        "pi0": pi0,
        "wallclock_seconds": float(elapsed),
        "wallclock_minutes": float(elapsed / 60.0),
        "settings": {
            "K": K, "L": L, "B": B, "eps": eps, "n_mcmc": n_mcmc,
            "mb_size": mb_size, "seed": seed, "init_seed": init_seed,
            "sigmasq_init": sigmasq_init, "kind": scenario.get("kind"),
            "rare_count": scenario.get("rare_count"), "delta": delta,
            "timevarying_info": True,
            "online_rho": online_rho,
            "online_tau": online_tau,
            "online_lambda_floor": online_lambda_floor,
            "online_warmup": online_warmup,
            "groupwise_drr": True,
            "static_tass_beta0_initialisation": bool(method == "online_tass_group"),
            "wallclock_seconds": float(elapsed),
            "wallclock_minutes": float(elapsed / 60.0),
        },
    }


def run_methods_timevarying_info(scenario, methods=("uniform", "tass", "online", "online_tass", "online_group", "online_tass_group"), seed=1,
                                 init_seed=999, K=3, L=2, B=5, eps=1e-6,
                                 n_mcmc=5000, mb_size=10, eta=1e-5,
                                 online_rho=0.98, online_tau=1.0,
                                 online_lambda_floor=0.05, online_warmup=0,
                                 online_feedback_eps=1e-6,
                                 checkpoint_every=None):
    chains = {}
    for r, method in enumerate(methods):
        chains[method] = run_csg_mcmc_timevarying_info(
            scenario, method=method, K=K, L=L, B=B, eps=eps,
            n_mcmc=n_mcmc, mb_size=mb_size, eta=eta, seed=seed + r,
            init_seed=init_seed, online_rho=online_rho, online_tau=online_tau,
            online_lambda_floor=online_lambda_floor, online_warmup=online_warmup,
            online_feedback_eps=online_feedback_eps,
            checkpoint_every=checkpoint_every)
        if "checkpoint" in globals():
            checkpoint(f"finished method={method}, elapsed={chains[method].get('wallclock_seconds', np.nan):.1f}s")
    return chains


def current_component_weights(chain, scenario, component="mu", state=2, L=2):
    ll = 2 * L + 1
    N = len(scenario["y"]) // ll
    method = chain["method"]
    if method == "uniform":
        return np.ones(N) / N
    if method == "tass":
        return normalize(chain["weights"][component][state])
    if method in ("online", "online_tass"):
        return normalize(chain["online"].weights()[component][state])
    if method in GROUPWISE_METHODS:
        if component not in GROUPWISE_NAMES:
            raise ValueError("component must be one of mu, sigmasq, A")
        return normalize(chain["online"].group_prob(component))
    raise ValueError(f"unknown method: {method}")


METHODS = ("uniform", "tass", "online", "online_tass", "online_group", "online_tass_group")
ONLINE_WARMUP = 0
print("Groupwise online DRR patch loaded. METHODS=", METHODS)


# Corrected global-vs-groupwise online DRR patch.
# Semantics after this patch:
#   online:            one global online DRR model for overall block-gradient norm.
#   online_tass:       same global online DRR, beta_0 initialised from static TASS proxy scores only.
#   online_group:      three groupwise online DRR models for mu, sigmasq, and A.
#   online_tass_group: same groupwise online DRR, beta_0^(r) initialised from static TASS group scores only.
#
# This replaces the earlier accidental comparison in which `online` was scalar-componentwise
# while `online_group` had only three groups, making the groupwise methods artificially faster.

GLOBAL_ONLINE_METHODS = ("online", "online_tass")
GROUPWISE_METHODS = ("online_group", "online_tass_group")
ONLINE_FAMILY_METHODS = GLOBAL_ONLINE_METHODS + GROUPWISE_METHODS


def _block_grad_norm_tuple(bmu, bs2, bA):
    return np.concatenate([np.ravel(bmu), np.ravel(bs2), np.ravel(bA)])


def static_tass_global_scores(y, K, L, eta=1e-5, seed=1, sort_centres=True,
                              pilot_mu=None, pilot_sigmasq=None, pilot_A=None):
    """Return Static-TASS proxy global scores a_n^{static} approximating ||g_n(theta_0)||.

    This is the one-distribution analogue of static_tass_group_scores.  It combines
    the group proxy norms into one approximate full-gradient norm for each block.
    """
    proxy = static_tass_group_scores(
        y, K, L, eta=eta, seed=seed, sort_centres=sort_centres,
        pilot_mu=pilot_mu, pilot_sigmasq=pilot_sigmasq, pilot_A=pilot_A)
    gs = proxy["scores"]
    global_scores = np.sqrt(gs["mu"] ** 2 + gs["sigmasq"] ** 2 + gs["A"] ** 2)
    global_scores = np.maximum(np.asarray(global_scores, dtype=float), eta ** 2)
    proxy["global_scores"] = global_scores
    return proxy


class GlobalDiscountedRidgeOnlineWeights:
    """One-distribution discounted ridge regression for overall gradient importance.

    The response for block n is log(||g_n(theta_t)||^2 + eps), where g_n includes
    the mu, sigmasq, and A block-gradient components.  The resulting probability
    vector is used for both transition and emission minibatches.
    """

    def __init__(self, features, K, rho=0.98, tau=1.0, lambda_floor=0.05,
                 feedback_eps=1e-8, static_scores=None, ridge_jitter=1e-5):
        self.W = np.asarray(features, dtype=float)
        self.N, self.p = self.W.shape
        self.K = int(K)
        self.rho = float(rho)
        self.tau = float(tau)
        self.lambda_floor = float(lambda_floor)
        self.feedback_eps = float(feedback_eps)
        self.ridge_jitter = float(ridge_jitter)
        self.A = self.tau * np.eye(self.p)
        self.b = np.zeros(self.p)
        self.beta = np.zeros(self.p)
        self.static_scores = None
        if static_scores is not None:
            self.initialise_from_scores(static_scores)

    def initialise_from_scores(self, scores):
        scores = np.maximum(np.asarray(scores, dtype=float), 0.0)
        pseudo_response = np.log(scores ** 2 + self.feedback_eps)
        self.A = self.W.T @ self.W + self.tau * np.eye(self.p)
        self.b = self.W.T @ pseudo_response
        self.beta = safe_ridge_solve(self.A, self.b, jitter=self.ridge_jitter)
        self.static_scores = scores

    def prob(self):
        score = np.clip(0.5 * (self.W @ self.beta), -30.0, 30.0)
        p = normalize(np.exp(score))
        if self.lambda_floor > 0:
            p = (1.0 - self.lambda_floor) * p + self.lambda_floor / self.N
        return normalize(p)

    def weights(self):
        p = self.prob()
        return {
            "mu": np.tile(p[None, :], (self.K, 1)),
            "sigmasq": np.tile(p[None, :], (self.K, 1)),
            "A": np.tile(p[None, None, :], (self.K, self.K, 1)),
            "global": p,
        }

    def update(self, block, grad_tuple):
        x = self.W[int(block)]
        g = np.asarray(grad_tuple, dtype=float)
        f = np.log(float(np.sum(g ** 2)) + self.feedback_eps)
        self.A = self.rho * self.A + np.outer(x, x)
        self.b = self.rho * self.b + x * f
        self.beta = safe_ridge_solve(self.A, self.b, jitter=self.ridge_jitter)


def transition_grad_online_global(y, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng,
                                  learn=True):
    N = len(y) // (2 * L + 1)
    _, _, gA = prior_gradients(mu, sigmasq, A_hat)
    cache = {}

    def get_block(i):
        if i not in cache:
            cache[i] = block_gradients_matlab(y, i, L, B, A, pi0, mu, sigmasq)
        return cache[i]

    w = online.prob()
    mb = rng.choice(N, size=mb_size, replace=True, p=w)
    for i in mb:
        bmu, bs2, bA = get_block(int(i))
        gA -= bA / max(w[int(i)], 1e-300)
        if learn:
            online.update(int(i), _block_grad_norm_tuple(bmu, bs2, bA))
    gA /= mb_size
    return gA


def emission_grad_online_global(y, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng,
                                learn=True):
    N = len(y) // (2 * L + 1)
    gmu, gs2, _ = prior_gradients(mu, sigmasq, A_hat)
    cache = {}

    def get_block(i):
        if i not in cache:
            cache[i] = block_gradients_matlab(y, i, L, B, A, pi0, mu, sigmasq)
        return cache[i]

    w = online.prob()
    mb = rng.choice(N, size=mb_size, replace=True, p=w)
    for i in mb:
        bmu, bs2, bA = get_block(int(i))
        gmu -= bmu / max(w[int(i)], 1e-300)
        gs2 -= bs2 / max(w[int(i)], 1e-300)
        if learn:
            online.update(int(i), _block_grad_norm_tuple(bmu, bs2, bA))
    gmu /= mb_size
    gs2 /= mb_size
    return gmu, gs2


# Keep handles to the previous implementations for auditing/ablation.
run_csg_mcmc_timevarying_info_previous = run_csg_mcmc_timevarying_info
run_methods_timevarying_info_previous = run_methods_timevarying_info


def run_csg_mcmc_timevarying_info(scenario, method="tass", K=3, L=2, B=5, eps=1e-6,
                                  n_mcmc=5000, mb_size=10, eta=1e-5, seed=1,
                                  init_seed=None, pi0_init=None, sigmasq_init=10.0,
                                  sort_centres=True, online_rho=0.98, online_tau=1.0,
                                  online_lambda_floor=0.05, online_warmup=0,
                                  online_feedback_eps=1e-8, checkpoint_every=None):
    # Static methods are delegated to the pre-patch runner.
    if method in ("uniform", "tass"):
        return run_csg_mcmc_timevarying_info_componentwise(
            scenario, method=method, K=K, L=L, B=B, eps=eps, n_mcmc=n_mcmc,
            mb_size=mb_size, eta=eta, seed=seed, init_seed=init_seed,
            pi0_init=pi0_init, sigmasq_init=sigmasq_init,
            sort_centres=sort_centres, online_rho=online_rho, online_tau=online_tau,
            online_lambda_floor=online_lambda_floor, online_warmup=online_warmup,
            online_feedback_eps=online_feedback_eps, checkpoint_every=checkpoint_every)

    # Legacy ablations, retained but not in METHODS by default.
    if method == "online_scalar":
        return run_csg_mcmc_timevarying_info_componentwise(
            scenario, method="online", K=K, L=L, B=B, eps=eps, n_mcmc=n_mcmc,
            mb_size=mb_size, eta=eta, seed=seed, init_seed=init_seed,
            pi0_init=pi0_init, sigmasq_init=sigmasq_init,
            sort_centres=sort_centres, online_rho=online_rho, online_tau=online_tau,
            online_lambda_floor=online_lambda_floor, online_warmup=online_warmup,
            online_feedback_eps=online_feedback_eps, checkpoint_every=checkpoint_every)
    if method == "online_tass_anchor":
        return run_csg_mcmc_timevarying_info_componentwise(
            scenario, method="online_tass", K=K, L=L, B=B, eps=eps, n_mcmc=n_mcmc,
            mb_size=mb_size, eta=eta, seed=seed, init_seed=init_seed,
            pi0_init=pi0_init, sigmasq_init=sigmasq_init,
            sort_centres=sort_centres, online_rho=online_rho, online_tau=online_tau,
            online_lambda_floor=online_lambda_floor, online_warmup=online_warmup,
            online_feedback_eps=online_feedback_eps, checkpoint_every=checkpoint_every)

    if method not in ONLINE_FAMILY_METHODS:
        raise ValueError("method must be one of uniform, tass, online, online_tass, online_group, online_tass_group")

    tic = perf_counter()
    y = scenario["y"]
    delta = float(scenario.get("delta", 0.0))

    rng = np.random.default_rng(seed)
    if init_seed is None:
        init_seed = seed + 1000003
    init_rng = np.random.default_rng(init_seed)
    if pi0_init is None:
        pi0_init = normalize(init_rng.random(K))
    mu, sigmasq, A, A_hat, pi0 = matlab_initial_parameters(
        K, seed=init_seed, sigmasq_init=sigmasq_init, pi0_init=pi0_init)

    ll = 2 * L + 1
    N = len(y) // ll
    y = y[:N * ll]
    features = block_features_timevarying(y, L, info_by_block=scenario.get("info_by_block", None))

    static_proxy = None
    static_scores = None
    if method == "online_tass":
        static_proxy = static_tass_global_scores(
            y, K, L, eta=eta, seed=seed, sort_centres=sort_centres,
            pilot_mu=mu, pilot_sigmasq=sigmasq, pilot_A=A)
        static_scores = static_proxy["global_scores"]
        online = GlobalDiscountedRidgeOnlineWeights(
            features, K, rho=online_rho, tau=online_tau,
            lambda_floor=online_lambda_floor, feedback_eps=online_feedback_eps,
            static_scores=static_scores,
            ridge_jitter=ONLINE_RIDGE_JITTER if "ONLINE_RIDGE_JITTER" in globals() else 1e-5)
    elif method == "online":
        online = GlobalDiscountedRidgeOnlineWeights(
            features, K, rho=online_rho, tau=online_tau,
            lambda_floor=online_lambda_floor, feedback_eps=online_feedback_eps,
            static_scores=None,
            ridge_jitter=ONLINE_RIDGE_JITTER if "ONLINE_RIDGE_JITTER" in globals() else 1e-5)
    else:
        if method == "online_tass_group":
            static_proxy = static_tass_group_scores(
                y, K, L, eta=eta, seed=seed, sort_centres=sort_centres,
                pilot_mu=mu, pilot_sigmasq=sigmasq, pilot_A=A)
            static_scores = static_proxy["scores"]
        online = GroupwiseDiscountedRidgeOnlineWeights(
            features, K, rho=online_rho, tau=online_tau,
            lambda_floor=online_lambda_floor, feedback_eps=online_feedback_eps,
            static_scores=static_scores,
            ridge_jitter=ONLINE_RIDGE_JITTER if "ONLINE_RIDGE_JITTER" in globals() else 1e-5)

    chain_mu = np.zeros((n_mcmc + 1, K))
    chain_s2 = np.zeros((n_mcmc + 1, K))
    chain_A = np.zeros((n_mcmc + 1, K, K))
    chain_A_hat = np.zeros((n_mcmc + 1, K, K))
    chain_mu[0] = mu
    chain_s2[0] = sigmasq
    chain_A[0] = A
    chain_A_hat[0] = A_hat

    for itr in range(1, n_mcmc + 1):
        if itr <= online_warmup:
            warmup_weights = uniform_weights(K, N)
            gA, _ = transition_grad_IS(y, warmup_weights, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng)
            A, A_hat = sgld_update_transition(A_hat, gA, eps, rng)
            gmu, gs2, _ = emission_grad_IS(y, warmup_weights, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng)
        elif method in GLOBAL_ONLINE_METHODS:
            gA = transition_grad_online_global(
                y, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng, learn=True)
            A, A_hat = sgld_update_transition(A_hat, gA, eps, rng)
            gmu, gs2 = emission_grad_online_global(
                y, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng, learn=True)
        else:
            gA = transition_grad_online_groupwise(
                y, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng, learn=True)
            A, A_hat = sgld_update_transition(A_hat, gA, eps, rng)
            gmu, gs2 = emission_grad_online_groupwise(
                y, online, mb_size, L, B, A, A_hat, pi0, mu, sigmasq, rng, learn=True)

        mu, sigmasq = sgld_update_emission(mu, sigmasq, gmu, gs2, eps, rng)

        if itr % 10 == 0:
            A_hat = normalize_columns(A_hat)
            A = normalize_columns(A_hat)

        chain_mu[itr] = mu
        chain_s2[itr] = sigmasq
        chain_A[itr] = A
        chain_A_hat[itr] = A_hat

        if checkpoint_every is not None and (itr == 1 or itr % checkpoint_every == 0):
            print(f"{method}: {itr}/{n_mcmc}, mu={np.round(mu, 3).tolist()}, sigmasq={np.round(sigmasq, 3).tolist()}", flush=True)

    elapsed = perf_counter() - tic
    return {
        "method": method,
        "mu": chain_mu,
        "sigmasq": chain_s2,
        "A": chain_A,
        "A_hat": chain_A_hat,
        "weights": None,
        "online": online,
        "static_scores": static_scores,
        "static_proxy": static_proxy,
        "pi0": pi0,
        "wallclock_seconds": float(elapsed),
        "wallclock_minutes": float(elapsed / 60.0),
        "settings": {
            "K": K, "L": L, "B": B, "eps": eps, "n_mcmc": n_mcmc,
            "mb_size": mb_size, "seed": seed, "init_seed": init_seed,
            "sigmasq_init": sigmasq_init, "kind": scenario.get("kind"),
            "rare_count": scenario.get("rare_count"), "delta": delta,
            "timevarying_info": True,
            "online_rho": online_rho,
            "online_tau": online_tau,
            "online_lambda_floor": online_lambda_floor,
            "online_warmup": online_warmup,
            "global_online_drr": bool(method in GLOBAL_ONLINE_METHODS),
            "groupwise_drr": bool(method in GROUPWISE_METHODS),
            "static_tass_beta0_initialisation": bool(method in ("online_tass", "online_tass_group")),
            "wallclock_seconds": float(elapsed),
            "wallclock_minutes": float(elapsed / 60.0),
        },
    }


def run_methods_timevarying_info(scenario, methods=("uniform", "tass", "online", "online_tass", "online_group", "online_tass_group"), seed=1,
                                 init_seed=999, K=3, L=2, B=5, eps=1e-6, n_mcmc=5000,
                                 mb_size=10, eta=1e-5, sigmasq_init=10.0,
                                 sort_centres=True, online_rho=0.98, online_tau=1.0,
                                 online_lambda_floor=0.05, online_warmup=0,
                                 online_feedback_eps=1e-8, checkpoint_every=None):
    chains = {}
    for r, method in enumerate(methods):
        chains[method] = run_csg_mcmc_timevarying_info(
            scenario, method=method, K=K, L=L, B=B, eps=eps,
            n_mcmc=n_mcmc, mb_size=mb_size, eta=eta, seed=seed + 1000 * r,
            init_seed=init_seed, sigmasq_init=sigmasq_init, sort_centres=sort_centres,
            online_rho=online_rho, online_tau=online_tau,
            online_lambda_floor=online_lambda_floor, online_warmup=online_warmup,
            online_feedback_eps=online_feedback_eps, checkpoint_every=checkpoint_every)
        checkpoint(f"finished method={method}, elapsed={chains[method].get('wallclock_seconds', np.nan):.1f}s")
    return chains


def current_component_weights(chain, scenario, component="mu", state=2, L=2):
    N = len(scenario["y"]) // (2 * L + 1)
    method = chain["method"]
    if method == "uniform":
        return np.ones(N) / N
    if method == "tass":
        w = chain.get("weights")
        if w is None:
            return np.ones(N) / N
        if component == "A":
            return normalize(w["A"][state, state])
        return normalize(w[component][state])
    if method in GLOBAL_ONLINE_METHODS:
        return normalize(chain["online"].prob())
    if method in GROUPWISE_METHODS:
        return normalize(chain["online"].group_prob(component))
    if method == "online_scalar":
        w = chain["online"].weights()
        if component == "A":
            return normalize(w["A"][state, state])
        return normalize(w[component][state])
    raise ValueError(f"unknown method: {method}")

METHODS = ("uniform", "tass", "online", "online_tass", "online_group", "online_tass_group")
ONLINE_HPARAM_METHODS = ("online", "online_tass", "online_group", "online_tass_group")
print("Corrected global/groupwise online DRR patch loaded. METHODS=", METHODS)


import matplotlib.pyplot as plt
from time import strftime, perf_counter


def checkpoint(message):
    print(f"[{strftime('%H:%M:%S')}] {message}", flush=True)


# Practical sweep defaults.
# These are intentionally smaller than the paper-scale synthetic experiment.
# They are meant to let the time-varying-info sweep finish in Colab and avoid
# spending hours on every rare_count/delta pair.
FAST = False
PAPER_SCALE = False
ITERATION_MULTIPLIER = 4  # run each method for 4x as many SGLD/MCMC iterations

if FAST:
    TTRAIN = 2500
    TTEST = 2500
    N_MCMC = 120 * ITERATION_MULTIPLIER
    DELTA_GRID = [0.0, 2.0]
    CHECKPOINT_EVERY = 40 * ITERATION_MULTIPLIER
elif PAPER_SCALE:
    TTRAIN = 10**6
    TTEST = 10**6
    N_MCMC = 5000 * ITERATION_MULTIPLIER
    DELTA_GRID = [0.0, 1.0, 2.0, 4.0, 6.0]
    CHECKPOINT_EVERY = 500 * ITERATION_MULTIPLIER
else:
    TTRAIN = 50000
    TTEST = 50000
    N_MCMC = 500 * ITERATION_MULTIPLIER
    DELTA_GRID = [0.0, 1.0, 2.0, 4.0]
    CHECKPOINT_EVERY = 100 * ITERATION_MULTIPLIER

# Main sweep.  rare_count=0 is a useful negative control, but it triples runtime
# for a case where the time-varying rare-information mechanism is mostly inactive.
# Add 0 back if you specifically want that control.
RARE_COUNTS = [1, 2]
METHODS = ("uniform", "tass", "online", "online_tass", "online_group", "online_tass_group")

K = 3
L = 2
B = 5
EPS = 1e-6
MB_SIZE = 5
ETA = 1e-5
SEED = 2026
INIT_SEED = 271828

INFO_MODE = "changepoint_mean"  # single rare-state mean changepoint; passed for metadata.
INFO_SLOPE = 10.0             # unused for changepoint_mean; kept for compatibility.
CHANGEPOINT_FRACTION = 0.50   # changepoint occurs halfway through the training sequence.

# Online ridge settings.  The larger tau/lambda_floor are deliberate stability changes:
# tau keeps the discounted Gram matrix invertible, and lambda_floor preserves exploration.
ONLINE_RHO = 0.95
ONLINE_TAU = 10.0
ONLINE_LAMBDA_FLOOR = 0.10
ONLINE_WARMUP = 25 if FAST else 100
ONLINE_RIDGE_JITTER = 1e-5
ONLINE_FEEDBACK_EPS = 1e-6
MAX_HOLD = 100

# Hyperparameter grids for the existing online DRR parameters.
# These do not add new method parameters; they only rerun the same online
# methods under different discount/ridge/exploration settings.
ONLINE_HPARAM_METHODS = ("online", "online_tass", "online_group", "online_tass_group")

# A targeted grid is usually the right first pass: it is much cheaper than the
# full factorial grid and probes the likely stability directions.
TARGETED_ONLINE_HPARAM_GRID = [
    {"label": "rho0.95_tau10_lam0.10", "online_rho": 0.95, "online_tau": 10.0,  "online_lambda_floor": 0.10},
    {"label": "rho0.90_tau10_lam0.10", "online_rho": 0.90, "online_tau": 10.0,  "online_lambda_floor": 0.10},
    {"label": "rho0.95_tau30_lam0.10", "online_rho": 0.95, "online_tau": 30.0,  "online_lambda_floor": 0.10},
    {"label": "rho0.95_tau100_lam0.10", "online_rho": 0.95, "online_tau": 100.0, "online_lambda_floor": 0.10},
    {"label": "rho0.95_tau10_lam0.20", "online_rho": 0.95, "online_tau": 10.0,  "online_lambda_floor": 0.20},
    {"label": "rho0.95_tau30_lam0.20", "online_rho": 0.95, "online_tau": 30.0,  "online_lambda_floor": 0.20},
    {"label": "rho0.90_tau30_lam0.20", "online_rho": 0.90, "online_tau": 30.0,  "online_lambda_floor": 0.20},
    {"label": "rho0.90_tau100_lam0.20", "online_rho": 0.90, "online_tau": 100.0, "online_lambda_floor": 0.20},
]

# Full factorial option, still only over existing parameters. Turn on below only
# when you are willing to pay the runtime cost.
FULL_ONLINE_RHO_GRID = [0.90, 0.95, 0.98]
FULL_ONLINE_TAU_GRID = [3.0, 10.0, 30.0, 100.0]
FULL_ONLINE_LAMBDA_FLOOR_GRID = [0.10, 0.20, 0.30]
USE_FULL_ONLINE_HPARAM_FACTORIAL = False

if USE_FULL_ONLINE_HPARAM_FACTORIAL:
    ONLINE_HPARAM_GRID = [
        {
            "label": f"rho{rho:g}_tau{tau:g}_lam{lam:g}",
            "online_rho": float(rho),
            "online_tau": float(tau),
            "online_lambda_floor": float(lam),
        }
        for rho in FULL_ONLINE_RHO_GRID
        for tau in FULL_ONLINE_TAU_GRID
        for lam in FULL_ONLINE_LAMBDA_FLOOR_GRID
    ]
else:
    ONLINE_HPARAM_GRID = TARGETED_ONLINE_HPARAM_GRID

checkpoint("configuration ready")
print({
    "FAST": FAST,
    "PAPER_SCALE": PAPER_SCALE,
    "TTRAIN": TTRAIN,
    "TTEST": TTEST,
    "N_MCMC": N_MCMC,
    "ITERATION_MULTIPLIER": ITERATION_MULTIPLIER,
    "MB_SIZE": MB_SIZE,
    "RARE_COUNTS": RARE_COUNTS,
    "DELTA_GRID": DELTA_GRID,
    "INFO_MODE": INFO_MODE,
    "CHANGEPOINT_FRACTION": CHANGEPOINT_FRACTION,
    "METHODS": METHODS,
    "ONLINE_RHO": ONLINE_RHO,
    "ONLINE_TAU": ONLINE_TAU,
    "ONLINE_LAMBDA_FLOOR": ONLINE_LAMBDA_FLOOR,
    "ONLINE_WARMUP": ONLINE_WARMUP,
    "ONLINE_RIDGE_JITTER": ONLINE_RIDGE_JITTER,
    "ONLINE_HPARAM_METHODS": ONLINE_HPARAM_METHODS,
    "USE_FULL_ONLINE_HPARAM_FACTORIAL": USE_FULL_ONLINE_HPARAM_FACTORIAL,
    "N_ONLINE_HPARAM_CONFIGS": len(ONLINE_HPARAM_GRID),
})


RUN_EXACT_SANITY = False

if RUN_EXACT_SANITY:
    checkpoint("building exact paper one-rare scenario")
    scenario = make_timevarying_info_scenario(
        rare_count=1,
        delta=0.0,
        Ttrain=TTRAIN,
        Ttest=TTEST,
        seed=SEED,
        L=L,
        info_mode=INFO_MODE,
        info_slope=INFO_SLOPE,
    )
    checkpoint("running exact paper methods: uniform, tass")
    chains = run_methods_timevarying_info(
        scenario,
        methods=("uniform", "tass"),
        seed=SEED + 10,
        init_seed=INIT_SEED,
        K=K,
        L=L,
        B=B,
        eps=EPS,
        n_mcmc=N_MCMC,
        mb_size=MB_SIZE,
        eta=ETA,
        checkpoint_every=CHECKPOINT_EVERY,
    )
    for method, chain in chains.items():
        print("\n", method)
        display(posterior_summary(chain, true_mu=scenario["mu"], burn=0.5))
else:
    print("Set RUN_EXACT_SANITY = True to run the paper-baseline sanity check.")


RUN_SWEEP = True
METHODS = ("uniform", "tass", "online", "online_tass", "online_group", "online_tass_group")

all_rows = []
all_logpred = []
sweep_chains = {}
sweep_scenarios = {}

if RUN_SWEEP:
    for rare_count in RARE_COUNTS:
        for delta in DELTA_GRID:
            pair_start = perf_counter()
            checkpoint(f"building scenario rare_count={rare_count}, delta={delta}")
            scenario = make_timevarying_info_scenario(
                rare_count=rare_count,
                delta=delta,
                Ttrain=TTRAIN,
                Ttest=TTEST,
                seed=SEED + 1000 * rare_count + int(100 * delta),
                L=L,
                info_mode=INFO_MODE,
                info_slope=INFO_SLOPE,
            )
            key = (rare_count, delta)
            sweep_scenarios[key] = scenario

            train_counts = np.bincount(scenario["z"], minlength=K)
            test_counts = np.bincount(scenario["ztest"], minlength=K)
            checkpoint(
                f"scenario ready kind={scenario['kind']} rare_states={scenario['rare_states']} "
                f"train_counts={train_counts.tolist()} test_counts={test_counts.tolist()} "
                f"effective_mu_train={np.round(scenario['effective_mu_train'], 3).tolist()}"
            )

            checkpoint(f"running methods for rare_count={rare_count}, delta={delta}")
            chains = run_methods_timevarying_info(
                scenario,
                methods=METHODS,
                seed=SEED + 2000 + 100 * rare_count + int(10 * delta),
                init_seed=INIT_SEED,
                K=K,
                L=L,
                B=B,
                eps=EPS,
                n_mcmc=N_MCMC,
                mb_size=MB_SIZE,
                eta=ETA,
                online_rho=ONLINE_RHO,
                online_tau=ONLINE_TAU,
                online_lambda_floor=ONLINE_LAMBDA_FLOOR,
                online_warmup=ONLINE_WARMUP,
                online_feedback_eps=ONLINE_FEEDBACK_EPS,
                checkpoint_every=CHECKPOINT_EVERY,
            )
            sweep_chains[key] = chains

            checkpoint(f"computing diagnostics for rare_count={rare_count}, delta={delta}")
            summary = pair_summary_table(
                scenario,
                chains,
                L=L,
                B=B,
                max_hold=MAX_HOLD,
                seed=SEED + 999,
            )
            summary = summary.assign(
                sweep_kind="main",
                hparam_label=f"rho{ONLINE_RHO:g}_tau{ONLINE_TAU:g}_lam{ONLINE_LAMBDA_FLOOR:g}",
                online_rho=float(ONLINE_RHO),
                online_tau=float(ONLINE_TAU),
                online_lambda_floor=float(ONLINE_LAMBDA_FLOOR),
            )
            all_rows.append(summary)
            display(summary)
            checkpoint(f"finished diagnostics rare_count={rare_count}, delta={delta}, pair_elapsed={perf_counter() - pair_start:.1f}s")

            eval_states = scenario["rare_states"] if len(scenario["rare_states"]) else [2]
            target_state = eval_states[-1]
            checkpoints = np.unique(np.linspace(1, N_MCMC, 30).astype(int))

            # Plot 1: rare/evaluation state mean error.
            plt.figure(figsize=(9, 4.5))
            for method, chain in chains.items():
                err = effective_mu_error(chain, scenario["effective_mu_train"], target_state)
                plt.plot(err, label=method)
            plt.title(f"Mean error trajectory: rare_count={rare_count}, delta={delta}, state={target_state}")
            plt.xlabel("SGLD iteration")
            plt.ylabel("absolute error to effective state mean")
            plt.legend()
            plt.show()

            # Plot 2: held-out log predictive curves.
            lp_frames = []
            for method, chain in chains.items():
                for group in ("all", "high"):
                    lp = rare_log_pred_timevarying(
                        chain,
                        scenario,
                        target_state,
                        checkpoints=checkpoints,
                        max_hold=MAX_HOLD,
                        seed=SEED + 444,
                        group=group,
                    )
                    lp_frames.append(lp)
                    all_logpred.append(lp.assign(
                        rare_count=rare_count,
                        delta=delta,
                        sweep_kind="main",
                        hparam_label=f"rho{ONLINE_RHO:g}_tau{ONLINE_TAU:g}_lam{ONLINE_LAMBDA_FLOOR:g}",
                        online_rho=float(ONLINE_RHO),
                        online_tau=float(ONLINE_TAU),
                        online_lambda_floor=float(ONLINE_LAMBDA_FLOOR),
                    ))
            lp_pair = pd.concat(lp_frames, ignore_index=True) if len(lp_frames) else pd.DataFrame()
            if len(lp_pair):
                for group in ("all", "high"):
                    plt.figure(figsize=(9, 4.5))
                    dfg = lp_pair[lp_pair["group"] == group]
                    for method, dfm in dfg.groupby("method"):
                        plt.plot(dfm["iteration"], dfm["mean_log_pred"], marker="o", label=method)
                    plt.title(f"Held-out log predictive ({group}): rare_count={rare_count}, delta={delta}, state={target_state}")
                    plt.xlabel("SGLD iteration")
                    plt.ylabel("mean log predictive density")
                    plt.legend()
                    plt.show()

            # Plot 3: final Q ratio and mass on high-information rare blocks.
            dft = summary[summary["state"] == target_state].copy()
            plt.figure(figsize=(8, 4.2))
            plt.bar(dft["method"], dft["Q_ratio_to_oracle"])
            plt.title(f"Final second-moment ratio Q/Qoracle: rare_count={rare_count}, delta={delta}, state={target_state}")
            plt.ylabel("Q ratio to oracle")
            plt.show()

            plt.figure(figsize=(8, 4.2))
            plt.bar(dft["method"], dft["high_info_rare_mass"])
            plt.title(f"Mass on high-information rare blocks: rare_count={rare_count}, delta={delta}, state={target_state}")
            plt.ylabel("sampling mass")
            plt.show()

            elapsed = perf_counter() - pair_start
            checkpoint(f"finished rare_count={rare_count}, delta={delta}, elapsed={elapsed:.1f}s")

    sweep_results = pd.concat(all_rows, ignore_index=True) if len(all_rows) else pd.DataFrame()
    logpred_results = pd.concat(all_logpred, ignore_index=True) if len(all_logpred) else pd.DataFrame()
    checkpoint("full sweep complete")
    display(sweep_results)
else:
    print("Set RUN_SWEEP = True to run the sweep.")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def state_rank_from_effective_mu(scenario, state):
    effective_mu = np.asarray(
        scenario.get("effective_mu_train", scenario["mu"]),
        dtype=float,
    )
    true_order = np.argsort(effective_mu)
    return int(np.where(true_order == state)[0][0])


def state_mean_and_variance_errors(chain, scenario, state):
    mu_draws = np.asarray(chain["mu"])
    sigmasq_draws = np.asarray(chain["sigmasq"])

    rank = state_rank_from_effective_mu(scenario, state)

    true_mu = float(np.asarray(
        scenario.get("effective_mu_train", scenario["mu"]),
        dtype=float,
    )[state])

    true_sigmasq = float(np.asarray(scenario["sigmasq"], dtype=float)[state])

    mu_error = np.zeros(mu_draws.shape[0])
    sigmasq_error = np.zeros(sigmasq_draws.shape[0])

    for t in range(mu_draws.shape[0]):
        order = np.argsort(mu_draws[t])
        mu_t = mu_draws[t, order][rank]
        sigmasq_t = sigmasq_draws[t, order][rank]

        mu_error[t] = abs(mu_t - true_mu)
        sigmasq_error[t] = abs(sigmasq_t - true_sigmasq)

    return mu_error, sigmasq_error


def plot_variance_errors_after_sweep(sweep_chains, sweep_scenarios):
    rows = []

    for key in sorted(sweep_chains.keys()):
        rare_count, delta = key
        scenario = sweep_scenarios[key]
        chains = sweep_chains[key]
        states = scenario.get("rare_states", [2])

        for state in states:
            plt.figure(figsize=(10, 5))

            for method, chain in chains.items():
                mu_error, sigmasq_error = state_mean_and_variance_errors(
                    chain,
                    scenario,
                    state,
                )

                plt.plot(sigmasq_error, label=method)

                elapsed = float(chain.get("wallclock_seconds", np.nan))

                rows.append({
                    "rare_count": rare_count,
                    "delta": delta,
                    "state": state,
                    "method": method,
                    "final_sigmasq_error": float(sigmasq_error[-1]),
                    "mean_sigmasq_error_second_half": float(
                        np.mean(sigmasq_error[len(sigmasq_error)//2:])
                    ),
                    "min_sigmasq_error": float(np.min(sigmasq_error)),
                    "wallclock_seconds": elapsed,
                    "wallclock_minutes": elapsed / 60.0 if np.isfinite(elapsed) else np.nan,
                })

            plt.title(
                f"Variance error by iteration "
                f"(rare_count={rare_count}, delta={delta}, state={state})"
            )
            plt.xlabel("SGLD iteration")
            plt.ylabel(r"$|\sigma_k^{2,(t)}-\sigma_{k,\mathrm{true}}^2|$")
            plt.legend()
            plt.grid(alpha=0.25)
            plt.show()

            if all("wallclock_seconds" in chain for chain in chains.values()):
                plt.figure(figsize=(10, 5))

                for method, chain in chains.items():
                    mu_error, sigmasq_error = state_mean_and_variance_errors(
                        chain,
                        scenario,
                        state,
                    )

                    elapsed = float(chain["wallclock_seconds"])
                    wallclock = np.linspace(0.0, elapsed, len(sigmasq_error))

                    plt.plot(wallclock, sigmasq_error, label=method)

                plt.title(
                    f"Variance error by wall-clock time "
                    f"(rare_count={rare_count}, delta={delta}, state={state})"
                )
                plt.xlabel("wall-clock seconds")
                plt.ylabel(r"$|\sigma_k^{2,(t)}-\sigma_{k,\mathrm{true}}^2|$")
                plt.legend()
                plt.grid(alpha=0.25)
                plt.show()

    summary = pd.DataFrame(rows).sort_values(
        ["rare_count", "delta", "state", "method"]
    )

    display(summary)

    plt.figure(figsize=(10, 5))
    for method, df in summary.groupby("method"):
        grouped = (
            df.groupby(["rare_count", "delta"])["final_sigmasq_error"]
            .mean()
            .reset_index()
        )
        label = method
        x = grouped["rare_count"].astype(str) + ", " + grouped["delta"].astype(str)
        plt.plot(x, grouped["final_sigmasq_error"], marker="o", label=label)

    plt.title("Final variance error across sweep")
    plt.xlabel("(rare_count, delta)")
    plt.ylabel(r"mean final $|\sigma_k^2-\sigma_{k,\mathrm{true}}^2|$")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.show()

    return summary


variance_error_summary = plot_variance_errors_after_sweep(
    sweep_chains=sweep_chains,
    sweep_scenarios=sweep_scenarios,
)

variance_error_summary


RUN_ONLINE_HPARAM_SWEEP = False
HYPERPARAM_SWEEP_PLOTS = False

online_hparam_rows = []
online_hparam_logpred_rows = []
online_hparam_chains = {}
online_hparam_scenarios = {}

if RUN_ONLINE_HPARAM_SWEEP:
    checkpoint(
        f"starting online hyperparameter sweep over {len(ONLINE_HPARAM_GRID)} configs "
        f"and methods={ONLINE_HPARAM_METHODS}"
    )

    for cfg_id, cfg in enumerate(ONLINE_HPARAM_GRID):
        cfg_label = cfg["label"]
        cfg_rho = float(cfg["online_rho"])
        cfg_tau = float(cfg["online_tau"])
        cfg_lam = float(cfg["online_lambda_floor"])

        checkpoint(
            f"hparam config {cfg_id + 1}/{len(ONLINE_HPARAM_GRID)}: "
            f"{cfg_label}"
        )

        for rare_count in RARE_COUNTS:
            for delta in DELTA_GRID:
                pair_start = perf_counter()
                scenario_key = (rare_count, delta)

                if scenario_key in sweep_scenarios:
                    scenario = sweep_scenarios[scenario_key]
                else:
                    checkpoint(f"building scenario rare_count={rare_count}, delta={delta}")
                    scenario = make_timevarying_info_scenario(
                        rare_count=rare_count,
                        delta=delta,
                        Ttrain=TTRAIN,
                        Ttest=TTEST,
                        seed=SEED + 1000 * rare_count + int(100 * delta),
                        L=L,
                        info_mode=INFO_MODE,
                        info_slope=INFO_SLOPE,
                    )
                    sweep_scenarios[scenario_key] = scenario

                online_hparam_scenarios[scenario_key] = scenario

                run_key = (cfg_label, rare_count, delta)
                checkpoint(
                    f"running online hparam methods for {cfg_label}, "
                    f"rare_count={rare_count}, delta={delta}"
                )
                chains = run_methods_timevarying_info(
                    scenario,
                    methods=ONLINE_HPARAM_METHODS,
                    seed=SEED + 50000 + 1000 * cfg_id + 100 * rare_count + int(10 * delta),
                    init_seed=INIT_SEED,
                    K=K,
                    L=L,
                    B=B,
                    eps=EPS,
                    n_mcmc=N_MCMC,
                    mb_size=MB_SIZE,
                    eta=ETA,
                    online_rho=cfg_rho,
                    online_tau=cfg_tau,
                    online_lambda_floor=cfg_lam,
                    online_warmup=ONLINE_WARMUP,
                    online_feedback_eps=ONLINE_FEEDBACK_EPS,
                    checkpoint_every=CHECKPOINT_EVERY,
                )
                online_hparam_chains[run_key] = chains

                summary = pair_summary_table(
                    scenario,
                    chains,
                    L=L,
                    B=B,
                    max_hold=MAX_HOLD,
                    seed=SEED + 999,
                ).assign(
                    sweep_kind="online_hparam",
                    hparam_label=cfg_label,
                    online_rho=cfg_rho,
                    online_tau=cfg_tau,
                    online_lambda_floor=cfg_lam,
                )
                online_hparam_rows.append(summary)
                display(summary)

                eval_states = scenario["rare_states"] if len(scenario["rare_states"]) else [2]
                target_state = eval_states[-1]
                checkpoints = np.unique(np.linspace(1, N_MCMC, 30).astype(int))

                for method, chain in chains.items():
                    for group in ("all", "high"):
                        lp = rare_log_pred_timevarying(
                            chain,
                            scenario,
                            target_state,
                            checkpoints=checkpoints,
                            max_hold=MAX_HOLD,
                            seed=SEED + 444,
                            group=group,
                        ).assign(
                            rare_count=rare_count,
                            delta=delta,
                            sweep_kind="online_hparam",
                            hparam_label=cfg_label,
                            online_rho=cfg_rho,
                            online_tau=cfg_tau,
                            online_lambda_floor=cfg_lam,
                        )
                        online_hparam_logpred_rows.append(lp)

                if HYPERPARAM_SWEEP_PLOTS:
                    dft = summary[summary["state"] == target_state].copy()
                    plt.figure(figsize=(8, 4.2))
                    plt.bar(dft["method"], dft["final_log_pred_high_info"].fillna(dft["final_log_pred_all"]))
                    plt.title(
                        f"Selection log-predictive: {cfg_label}, "
                        f"rare_count={rare_count}, delta={delta}, state={target_state}"
                    )
                    plt.ylabel("mean log predictive density")
                    plt.show()

                checkpoint(
                    f"finished {cfg_label}, rare_count={rare_count}, delta={delta}, "
                    f"elapsed={perf_counter() - pair_start:.1f}s"
                )

    online_hparam_results = (
        pd.concat(online_hparam_rows, ignore_index=True)
        if len(online_hparam_rows)
        else pd.DataFrame()
    )
    online_hparam_logpred_results = (
        pd.concat(online_hparam_logpred_rows, ignore_index=True)
        if len(online_hparam_logpred_rows)
        else pd.DataFrame()
    )
    checkpoint("online hyperparameter sweep complete")
    display(online_hparam_results)
else:
    print("Set RUN_ONLINE_HPARAM_SWEEP = True to run the online hyperparameter sweep.")
    print(f"Configured {len(ONLINE_HPARAM_GRID)} candidate settings:")
    display(pd.DataFrame(ONLINE_HPARAM_GRID))


if "online_hparam_results" in globals() and len(online_hparam_results):
    hpdf = online_hparam_results.copy()
    hpdf["selection_log_pred"] = hpdf["final_log_pred_high_info"].where(
        hpdf["final_log_pred_high_info"].notna(),
        hpdf["final_log_pred_all"],
    )

    best_by_scenario_method = (
        hpdf.sort_values("selection_log_pred", ascending=False)
        .groupby(["rare_count", "delta", "state", "method"], as_index=False)
        .head(1)
        .sort_values(["rare_count", "delta", "state", "method"])
    )
    display(best_by_scenario_method)

    aggregate_hparam_summary = (
        hpdf.groupby(["method", "hparam_label", "online_rho", "online_tau", "online_lambda_floor"], as_index=False)
        .agg(
            mean_selection_log_pred=("selection_log_pred", "mean"),
            median_selection_log_pred=("selection_log_pred", "median"),
            min_selection_log_pred=("selection_log_pred", "min"),
            mean_Q_ratio_to_oracle=("Q_ratio_to_oracle", "mean"),
            median_Q_ratio_to_oracle=("Q_ratio_to_oracle", "median"),
            mean_wallclock_seconds=("wallclock_seconds", "mean"),
        )
        .sort_values(["method", "mean_selection_log_pred"], ascending=[True, False])
    )
    display(aggregate_hparam_summary)

    for method, dfm in aggregate_hparam_summary.groupby("method"):
        top = dfm.head(10).copy()
        plt.figure(figsize=(10, 4.5))
        plt.bar(top["hparam_label"], top["mean_selection_log_pred"])
        plt.title(f"Top online hyperparameter settings by mean selection log-predictive: {method}")
        plt.xlabel("hyperparameter setting")
        plt.ylabel("mean selection log predictive density")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
else:
    print("Run the online hyperparameter sweep cell first.")


if "sweep_results" in globals() and len(sweep_results):
    display(sweep_results)

    for rare_count, dfr in sweep_results.groupby("rare_count"):
        # If two rare states are present, keep the positive rare state by default for a clean line plot.
        state = int(dfr["state"].max())
        dfr = dfr[dfr["state"] == state]

        plt.figure(figsize=(9, 4.5))
        for method, dfm in dfr.groupby("method"):
            dfm = dfm.sort_values("delta")
            plt.plot(dfm["delta"], dfm["final_log_pred_high_info"], marker="o", label=method)
        plt.title(f"Final high-info rare log predictive vs delta, rare_count={rare_count}, state={state}")
        plt.xlabel("delta")
        plt.ylabel("final high-info mean log predictive density")
        plt.legend()
        plt.show()

        plt.figure(figsize=(9, 4.5))
        for method, dfm in dfr.groupby("method"):
            dfm = dfm.sort_values("delta")
            plt.plot(dfm["delta"], dfm["Q_ratio_to_oracle"], marker="o", label=method)
        plt.title(f"Final Q/Qoracle vs delta, rare_count={rare_count}, state={state}")
        plt.xlabel("delta")
        plt.ylabel("Q ratio to oracle")
        plt.legend()
        plt.show()

        plt.figure(figsize=(9, 4.5))
        for method, dfm in dfr.groupby("method"):
            dfm = dfm.sort_values("delta")
            plt.plot(dfm["delta"], dfm["high_info_rare_mass"], marker="o", label=method)
        plt.title(f"High-info rare sampling mass vs delta, rare_count={rare_count}, state={state}")
        plt.xlabel("delta")
        plt.ylabel("sampling mass")
        plt.legend()
        plt.show()
else:
    print("Run the sweep cell first.")


## Aggregate wall-clock plots

if "sweep_results" in globals() and len(sweep_results):
    for rare_count, dfr in sweep_results.groupby("rare_count"):
        state = int(dfr["state"].max())
        dfr = dfr[dfr["state"] == state].copy()

        plt.figure(figsize=(9, 4.5))
        for method, dfm in dfr.groupby("method"):
            dfm = dfm.sort_values("delta")
            plt.plot(dfm["delta"], dfm["wallclock_minutes"], marker="o", label=method)
        plt.title(f"Wall-clock minutes vs delta, rare_count={rare_count}, state={state}")
        plt.xlabel("delta")
        plt.ylabel("wall-clock minutes")
        plt.legend()
        plt.show()

        plt.figure(figsize=(9, 4.5))
        for method, dfm in dfr.groupby("method"):
            dfm = dfm.sort_values("delta")
            plt.plot(dfm["wallclock_minutes"], dfm["final_log_pred_high_info"], marker="o", label=method)
        plt.title(f"High-info log predictive vs wall-clock, rare_count={rare_count}, state={state}")
        plt.xlabel("wall-clock minutes")
        plt.ylabel("final high-info mean log predictive density")
        plt.legend()
        plt.show()
else:
    print("Run the sweep cell first.")


wrote_any = False

if "sweep_results" in globals() and len(sweep_results):
    sweep_results.to_csv("timevarying_info_sweep_results.csv", index=False)
    print("Wrote timevarying_info_sweep_results.csv, including wallclock_seconds and wallclock_minutes")
    wrote_any = True

if "logpred_results" in globals() and len(logpred_results):
    logpred_results.to_csv("timevarying_info_logpred_curves.csv", index=False)
    print("Wrote timevarying_info_logpred_curves.csv")
    wrote_any = True

if "online_hparam_results" in globals() and len(online_hparam_results):
    online_hparam_results.to_csv("timevarying_info_online_hparam_sweep_results.csv", index=False)
    print("Wrote timevarying_info_online_hparam_sweep_results.csv")
    wrote_any = True

if "online_hparam_logpred_results" in globals() and len(online_hparam_logpred_results):
    online_hparam_logpred_results.to_csv("timevarying_info_online_hparam_logpred_curves.csv", index=False)
    print("Wrote timevarying_info_online_hparam_logpred_curves.csv")
    wrote_any = True

if not wrote_any:
    print("No sweep results to export yet.")


