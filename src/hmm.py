"""JAX forward-backward and buffered block log-likelihood gradients
(Ma et al. 2017).  Import factories with
``from src.hmm import make_buffered_block_grad_fn, make_full_grad_fn``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax


def stationary_distribution(A: jnp.ndarray, n_iter: int = 200) -> jnp.ndarray:
    K = A.shape[0]

    def body_fun(_, pi):
        pi = pi @ A
        pi = pi / jnp.sum(pi)
        return pi

    pi0 = jnp.ones((K,)) / K
    return lax.fori_loop(0, n_iter, body_fun, pi0)


def emission_loglik_matrix(y: jnp.ndarray, mu: jnp.ndarray, covariances: jnp.ndarray) -> jnp.ndarray:
    inv_cov = jnp.linalg.inv(covariances)
    sign, logdet = jnp.linalg.slogdet(covariances)
    d = y.shape[1]

    def state_logpdf(mu_k, inv_cov_k, logdet_k):
        diff = y - mu_k[None, :]
        mahal = jnp.einsum("td,dd,te->t", diff, inv_cov_k, diff)
        return -0.5 * (d * jnp.log(2.0 * jnp.pi) + logdet_k + mahal)

    logpdf = jax.vmap(state_logpdf, in_axes=(0, 0, 0))(mu, inv_cov, logdet)
    return logpdf.T


def forward_pass(log_pi: jnp.ndarray, log_A: jnp.ndarray, emission_lp: jnp.ndarray):
    log_alpha0 = log_pi + emission_lp[0]

    def step(log_alpha_prev, emission_t):
        log_alpha_t = emission_t + jsp.special.logsumexp(
            log_alpha_prev[:, None] + log_A,
            axis=0,
        )
        return log_alpha_t, log_alpha_t

    _, log_alpha_rest = lax.scan(step, log_alpha0, emission_lp[1:])
    log_alpha = jnp.vstack([log_alpha0[None, :], log_alpha_rest])
    loglik = jsp.special.logsumexp(log_alpha[-1])
    return loglik, log_alpha


def backward_pass(log_A: jnp.ndarray, emission_lp: jnp.ndarray):
    T, K = emission_lp.shape
    log_beta_T = jnp.zeros((K,))

    def step(log_beta_next, emission_next):
        log_beta_t = jsp.special.logsumexp(
            log_A + emission_next[None, :] + log_beta_next[None, :],
            axis=1,
        )
        return log_beta_t, log_beta_t

    _, beta_rev = lax.scan(step, log_beta_T, emission_lp[1:][::-1])
    log_beta = jnp.vstack([beta_rev[::-1], log_beta_T[None, :]])
    return log_beta


def unpack_params(params):
    A = jax.nn.softmax(params["trans_logits"], axis=1)
    pi = stationary_distribution(A)
    mu = params["mu"]
    return A, pi, mu


def log_prior(params, sigma_mu: float = 5.0, sigma_trans: float = 2.0) -> jnp.ndarray:
    mu = params["mu"]
    trans_logits = params["trans_logits"]
    mu_penalty = 0.5 * jnp.sum((mu / sigma_mu) ** 2)
    trans_penalty = 0.5 * jnp.sum((trans_logits / sigma_trans) ** 2)
    return -(mu_penalty + trans_penalty)


def log_likelihood(params, y: jnp.ndarray, covariances: jnp.ndarray) -> jnp.ndarray:
    A, pi, mu = unpack_params(params)
    emission_lp = emission_loglik_matrix(y, mu, covariances)
    log_pi = jnp.log(jnp.clip(pi, 1e-30, 1.0))
    log_A = jnp.log(jnp.clip(A, 1e-30, 1.0))
    loglik, _ = forward_pass(log_pi, log_A, emission_lp)
    return loglik


def log_posterior(params, y: jnp.ndarray, covariances: jnp.ndarray) -> jnp.ndarray:
    return log_likelihood(params, y, covariances) + log_prior(params)


def neg_log_posterior(params, y: jnp.ndarray, covariances: jnp.ndarray) -> jnp.ndarray:
    return -log_posterior(params, y, covariances)


def _block_row(blocks, j: int) -> dict:
    if hasattr(blocks, "iloc"):
        row = blocks.iloc[int(j)]
        return {col: row[col] for col in row.index}
    return dict(blocks[int(j)])


def _forward_from_message(log_A: jnp.ndarray, emission_seg: jnp.ndarray, init_msg: jnp.ndarray, use_transition: bool):
    if use_transition:
        log_alpha0 = emission_seg[0] + jsp.special.logsumexp(init_msg[:, None] + log_A, axis=0)
    else:
        log_alpha0 = init_msg + emission_seg[0]

    if emission_seg.shape[0] == 1:
        return log_alpha0

    def step(log_alpha_prev, emission_t):
        log_alpha_t = emission_t + jsp.special.logsumexp(
            log_alpha_prev[:, None] + log_A,
            axis=0,
        )
        return log_alpha_t, log_alpha_t

    log_alpha_last, _ = lax.scan(step, log_alpha0, emission_seg[1:])
    return log_alpha_last


def _right_message(log_A: jnp.ndarray, emission_right: jnp.ndarray) -> jnp.ndarray:
    if emission_right.shape[0] == 0:
        K = log_A.shape[0]
        return jnp.zeros((K,))

    log_beta_right = backward_pass(log_A, emission_right)
    return jsp.special.logsumexp(
        log_A + emission_right[0][None, :] + log_beta_right[0][None, :],
        axis=1,
    )


def buffered_block_logscore(params, y: jnp.ndarray, block_row, covariances: jnp.ndarray) -> jnp.ndarray:
    A, pi, mu = unpack_params(params)
    log_pi = jnp.log(jnp.clip(pi, 1e-30, 1.0))
    log_A = jnp.log(jnp.clip(A, 1e-30, 1.0))

    left = int(block_row["left"])
    right = int(block_row["right"])
    start = int(block_row["start"])
    end = int(block_row["end"])

    y_buf = y[left : right + 1]
    emission_lp = emission_loglik_matrix(y_buf, mu, covariances)

    start_local = start - left
    end_local = end - left

    if start_local == 0:
        left_msg = lax.stop_gradient(log_pi)
        use_transition = False
    else:
        _, log_alpha_left = forward_pass(log_pi, log_A, emission_lp[:start_local])
        left_msg = lax.stop_gradient(log_alpha_left[-1])
        use_transition = True

    central_emission = emission_lp[start_local : end_local + 1]
    alpha_last = _forward_from_message(log_A, central_emission, left_msg, use_transition=use_transition)

    if end_local == emission_lp.shape[0] - 1:
        return jsp.special.logsumexp(alpha_last)

    right_msg = lax.stop_gradient(_right_message(log_A, emission_lp[end_local + 1 :]))
    return jsp.special.logsumexp(alpha_last + right_msg)


def make_full_grad_fn(covariances: jnp.ndarray):
    return jax.grad(lambda params, y: neg_log_posterior(params, y, covariances))


def make_buffered_block_grad_fn(covariances: jnp.ndarray):
    return jax.grad(lambda params, y, block_row: -buffered_block_logscore(params, y, block_row, covariances))
