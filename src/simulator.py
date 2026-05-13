"""Synthetic K-state Gaussian HMM data generator used by the practical-proxy
sweep (Section 5.2).  Build a spec with ``make_three_state_rare_gaussian_hmm``,
draw a trajectory with ``simulate_hmm_stationary``, and unpack the parameter
pytree with ``params_from_spec``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianHMMSpec:
    transition: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    init_probs: np.ndarray | None = None
    rare_state: int = 2


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    total = float(x.sum())
    if total <= 0.0:
        return np.ones_like(x) / len(x)
    return x / total


def stationary_distribution(A: np.ndarray, max_iter: int = 10_000, tol: float = 1e-12) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    K = A.shape[0]
    pi = np.ones(K, dtype=float) / K
    for _ in range(max_iter):
        pi_next = normalize(pi @ A)
        if np.linalg.norm(pi_next - pi, ord=1) <= tol:
            return pi_next
        pi = pi_next
    return pi


def make_three_state_rare_gaussian_hmm(
    obs_dim: int = 3,
    rare_scale: float = 2.25,
    transition_stickiness: float = 0.94,
    rare_entry_prob: float = 0.03,
    rare_exit_prob: float = 0.20,
) -> GaussianHMMSpec:
    if obs_dim < 2:
        raise ValueError("obs_dim must be at least 2 for the grouped experiments.")

    common_1 = np.linspace(-1.0, 0.4, obs_dim)
    common_2 = np.linspace(0.8, -0.6, obs_dim)
    rare = np.linspace(rare_scale, rare_scale + 0.8, obs_dim)
    means = np.vstack([common_1, common_2, rare])

    covariances = np.stack(
        [
            0.35 * np.eye(obs_dim),
            0.45 * np.eye(obs_dim),
            0.65 * np.eye(obs_dim),
        ]
    )

    stay = float(transition_stickiness)
    enter = float(rare_entry_prob)
    exit_rare = float(rare_exit_prob)

    transition = np.array(
        [
            [stay - 0.03, 1.0 - stay, enter],
            [1.0 - stay, stay - 0.03, enter],
            [0.5 * exit_rare, 0.5 * exit_rare, 1.0 - exit_rare],
        ],
        dtype=float,
    )
    transition = transition / transition.sum(axis=1, keepdims=True)
    init_probs = stationary_distribution(transition)

    return GaussianHMMSpec(
        transition=transition,
        means=means,
        covariances=covariances,
        init_probs=init_probs,
        rare_state=2,
    )


def shift_rare_state_mean(
    spec: GaussianHMMSpec,
    delta: float,
    direction: np.ndarray | None = None,
    rare_state: int | None = None,
) -> GaussianHMMSpec:
    means = np.array(spec.means, copy=True)
    rare_idx = spec.rare_state if rare_state is None else int(rare_state)
    if direction is None:
        direction = np.ones(means.shape[1], dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    means[rare_idx] = means[rare_idx] + float(delta) * direction
    return GaussianHMMSpec(
        transition=np.array(spec.transition, copy=True),
        means=means,
        covariances=np.array(spec.covariances, copy=True),
        init_probs=np.array(spec.init_probs, copy=True) if spec.init_probs is not None else None,
        rare_state=rare_idx,
    )


def simulate_hmm_stationary(spec: GaussianHMMSpec, T: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    K = spec.transition.shape[0]
    pi0 = stationary_distribution(spec.transition) if spec.init_probs is None else normalize(spec.init_probs)

    states = np.zeros(T, dtype=int)
    y = np.zeros((T, spec.means.shape[1]), dtype=float)

    states[0] = int(rng.choice(K, p=pi0))
    y[0] = rng.multivariate_normal(spec.means[states[0]], spec.covariances[states[0]])

    for t in range(1, T):
        states[t] = int(rng.choice(K, p=spec.transition[states[t - 1]]))
        y[t] = rng.multivariate_normal(spec.means[states[t]], spec.covariances[states[t]])

    return y, states


def simulate_hmm_drifting(theta_traj, T: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    spec0 = theta_traj(0)
    K = spec0.transition.shape[0]
    obs_dim = spec0.means.shape[1]

    states = np.zeros(T, dtype=int)
    y = np.zeros((T, obs_dim), dtype=float)

    pi0 = stationary_distribution(spec0.transition) if spec0.init_probs is None else normalize(spec0.init_probs)
    states[0] = int(rng.choice(K, p=pi0))
    y[0] = rng.multivariate_normal(spec0.means[states[0]], spec0.covariances[states[0]])

    for t in range(1, T):
        spec_t = theta_traj(t)
        states[t] = int(rng.choice(K, p=spec_t.transition[states[t - 1]]))
        y[t] = rng.multivariate_normal(spec_t.means[states[t]], spec_t.covariances[states[t]])

    return y, states


def params_from_spec(spec: GaussianHMMSpec) -> dict[str, np.ndarray]:
    return {
        "trans_logits": np.log(np.clip(spec.transition, 1e-12, 1.0)),
        "mu": np.array(spec.means, copy=True),
    }
