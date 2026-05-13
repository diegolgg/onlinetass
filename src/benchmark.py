"""Gradient-RMSE benchmark glue: oracle weight bundles, the six samplers
compared in the paper (``uniform_stochastic_grad``, ``static_group_tass_grad``,
``static_component_tass_grad``, ``online_group_tass_grad``), and
``gradient_rmse(...)`` for the replicate-averaged RMSE evaluator.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .hmm import log_prior
from .samplers import GroupedOnlineTASS


def tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_scale(a, scale: float):
    return jax.tree_util.tree_map(lambda x: x * scale, a)


def tree_zeros_like(a):
    return jax.tree_util.tree_map(jnp.zeros_like, a)


def tree_l2_norm(a) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(a)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def tree_rmse(a, b) -> jnp.ndarray:
    diff = jax.tree_util.tree_map(lambda x, y: x - y, a, b)
    leaves = jax.tree_util.tree_leaves(diff)
    sq_sum = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)
    n_total = sum(int(np.prod(leaf.shape)) for leaf in leaves)
    return jnp.sqrt(sq_sum / max(n_total, 1))


def make_prior_grad_fn():
    return jax.grad(lambda params: -log_prior(params))


def block_grad_cached(params, y, blocks, j: int, block_grad_fn, grad_cache: dict | None = None):
    if grad_cache is None:
        grad_cache = {}
    block_id = int(j)
    if block_id not in grad_cache:
        if hasattr(blocks, "iloc"):
            row = blocks.iloc[block_id]
            block_row = {col: row[col] for col in row.index}
        else:
            block_row = dict(blocks[block_id])
        grad_cache[block_id] = block_grad_fn(params, y, block_row)
    return grad_cache[block_id]


def mean_group_sqnorm(g, state: int) -> float:
    return float(np.sum(np.square(np.asarray(g["mu"][state]))))


def trans_group_sqnorm(g, state: int) -> float:
    return float(np.sum(np.square(np.asarray(g["trans_logits"][state]))))


def compute_oracle_weight_bundle(
    params,
    y,
    blocks,
    K: int,
    block_grad_fn,
    eps: float = 1e-12,
):
    if hasattr(blocks, "__len__"):
        N = len(blocks)
    else:
        raise ValueError("blocks must be an indexable sequence or dataframe")

    mu_shape = np.asarray(params["mu"]).shape
    D = mu_shape[1]

    group_mean = np.zeros((K, N), dtype=float)
    group_trans = np.zeros((K, N), dtype=float)
    component_mean = np.zeros((K, D, N), dtype=float)
    component_trans = np.zeros((K, K, N), dtype=float)

    cache = {}
    for n in range(N):
        g = block_grad_cached(params, y, blocks, n, block_grad_fn, grad_cache=cache)
        mu_g = np.asarray(g["mu"])
        trans_g = np.asarray(g["trans_logits"])
        for k in range(K):
            component_mean[k, :, n] = np.abs(mu_g[k]) + eps
            group_mean[k, n] = np.linalg.norm(mu_g[k], ord=2) + eps
            component_trans[k, :, n] = np.abs(trans_g[k]) + eps
            group_trans[k, n] = np.linalg.norm(trans_g[k], ord=2) + eps

    component_mean /= component_mean.sum(axis=-1, keepdims=True)
    group_mean /= group_mean.sum(axis=-1, keepdims=True)
    component_trans /= component_trans.sum(axis=-1, keepdims=True)
    group_trans /= group_trans.sum(axis=-1, keepdims=True)

    return {
        "component_mean": component_mean,
        "group_mean": group_mean,
        "component_trans": component_trans,
        "group_trans": group_trans,
    }


def uniform_stochastic_grad(params, y, blocks, M: int, rng_key, block_grad_fn, prior_grad_fn=None):
    N = len(blocks)
    probs = np.ones(N, dtype=float) / N
    idx = np.asarray(jax.random.choice(rng_key, a=N, shape=(M,), replace=True), dtype=int)

    total = tree_zeros_like(params)
    cache = {}
    for j in idx:
        g = block_grad_cached(params, y, blocks, int(j), block_grad_fn, grad_cache=cache)
        total = tree_add(total, tree_scale(g, 1.0 / probs[int(j)]))
    total = tree_scale(total, 1.0 / M)

    if prior_grad_fn is not None:
        total = tree_add(total, prior_grad_fn(params))
    return total, {"indices": idx}


def static_group_tass_grad(
    params,
    y,
    blocks,
    mean_probs: np.ndarray,
    trans_probs: np.ndarray,
    M_mean: int,
    M_trans: int,
    rng_key,
    block_grad_fn,
    prior_grad_fn=None,
):
    K = mean_probs.shape[0]
    keys = jax.random.split(rng_key, 2 * K)
    cache = {}

    out = tree_zeros_like(params)
    mu = out["mu"]
    trans_logits = out["trans_logits"]
    sampled = {"mean": {}, "trans": {}}

    for k in range(K):
        idx = np.asarray(
            jax.random.choice(keys[k], a=len(blocks), shape=(M_mean,), replace=True, p=jnp.asarray(mean_probs[k])),
            dtype=int,
        )
        sampled["mean"][k] = idx
        accum = np.zeros_like(np.asarray(params["mu"][k]), dtype=float)
        for j in idx:
            g = block_grad_cached(params, y, blocks, int(j), block_grad_fn, grad_cache=cache)
            accum += np.asarray(g["mu"][k]) / float(mean_probs[k, int(j)])
        mu = mu.at[k].set(jnp.asarray(accum / M_mean))

    for k in range(K):
        idx = np.asarray(
            jax.random.choice(
                keys[K + k],
                a=len(blocks),
                shape=(M_trans,),
                replace=True,
                p=jnp.asarray(trans_probs[k]),
            ),
            dtype=int,
        )
        sampled["trans"][k] = idx
        accum = np.zeros_like(np.asarray(params["trans_logits"][k]), dtype=float)
        for j in idx:
            g = block_grad_cached(params, y, blocks, int(j), block_grad_fn, grad_cache=cache)
            accum += np.asarray(g["trans_logits"][k]) / float(trans_probs[k, int(j)])
        trans_logits = trans_logits.at[k].set(jnp.asarray(accum / M_trans))

    out = {"mu": mu, "trans_logits": trans_logits}
    if prior_grad_fn is not None:
        out = tree_add(out, prior_grad_fn(params))
    return out, sampled


def static_component_tass_grad(
    params,
    y,
    blocks,
    mean_probs: np.ndarray,
    trans_probs: np.ndarray,
    M_mean: int,
    M_trans: int,
    rng_key,
    block_grad_fn,
    prior_grad_fn=None,
):
    K, D, _ = mean_probs.shape
    keys = jax.random.split(rng_key, K * D + K * K)
    cache = {}

    out = tree_zeros_like(params)
    mu = out["mu"]
    trans_logits = out["trans_logits"]
    key_ptr = 0
    sampled = {"mean": {}, "trans": {}}

    for k in range(K):
        for d in range(D):
            idx = np.asarray(
                jax.random.choice(
                    keys[key_ptr],
                    a=len(blocks),
                    shape=(M_mean,),
                    replace=True,
                    p=jnp.asarray(mean_probs[k, d]),
                ),
                dtype=int,
            )
            key_ptr += 1
            sampled["mean"][(k, d)] = idx
            accum = 0.0
            for j in idx:
                g = block_grad_cached(params, y, blocks, int(j), block_grad_fn, grad_cache=cache)
                accum += float(g["mu"][k, d]) / float(mean_probs[k, d, int(j)])
            mu = mu.at[k, d].set(accum / M_mean)

    for j_state in range(K):
        for k_state in range(K):
            idx = np.asarray(
                jax.random.choice(
                    keys[key_ptr],
                    a=len(blocks),
                    shape=(M_trans,),
                    replace=True,
                    p=jnp.asarray(trans_probs[j_state, k_state]),
                ),
                dtype=int,
            )
            key_ptr += 1
            sampled["trans"][(j_state, k_state)] = idx
            accum = 0.0
            for block_id in idx:
                g = block_grad_cached(params, y, blocks, int(block_id), block_grad_fn, grad_cache=cache)
                accum += float(g["trans_logits"][j_state, k_state]) / float(trans_probs[j_state, k_state, int(block_id)])
            trans_logits = trans_logits.at[j_state, k_state].set(accum / M_trans)

    out = {"mu": mu, "trans_logits": trans_logits}
    if prior_grad_fn is not None:
        out = tree_add(out, prior_grad_fn(params))
    return out, sampled


def online_group_tass_grad(
    params,
    y,
    blocks,
    online_model: GroupedOnlineTASS,
    M_mean: int,
    M_trans: int,
    rng_key,
    block_grad_fn,
    prior_grad_fn=None,
    update_models: bool = True,
):
    mean_probs = online_model.mean_probs()
    trans_probs = online_model.trans_probs()
    K = mean_probs.shape[0]
    keys = jax.random.split(rng_key, 2 * K)
    cache = {}

    out = tree_zeros_like(params)
    mu = out["mu"]
    trans_logits = out["trans_logits"]
    sampled = {"mean": {}, "trans": {}}

    for k in range(K):
        idx = np.asarray(
            jax.random.choice(keys[k], a=len(blocks), shape=(M_mean,), replace=True, p=jnp.asarray(mean_probs[k])),
            dtype=int,
        )
        sampled["mean"][k] = idx
        accum = np.zeros_like(np.asarray(params["mu"][k]), dtype=float)
        for block_id in idx:
            g = block_grad_cached(params, y, blocks, int(block_id), block_grad_fn, grad_cache=cache)
            accum += np.asarray(g["mu"][k]) / float(mean_probs[k, int(block_id)])
            if update_models:
                online_model.update_mean(k, int(block_id), mean_group_sqnorm(g, k))
        mu = mu.at[k].set(jnp.asarray(accum / M_mean))

    for k in range(K):
        idx = np.asarray(
            jax.random.choice(
                keys[K + k],
                a=len(blocks),
                shape=(M_trans,),
                replace=True,
                p=jnp.asarray(trans_probs[k]),
            ),
            dtype=int,
        )
        sampled["trans"][k] = idx
        accum = np.zeros_like(np.asarray(params["trans_logits"][k]), dtype=float)
        for block_id in idx:
            g = block_grad_cached(params, y, blocks, int(block_id), block_grad_fn, grad_cache=cache)
            accum += np.asarray(g["trans_logits"][k]) / float(trans_probs[k, int(block_id)])
            if update_models:
                online_model.update_trans(k, int(block_id), trans_group_sqnorm(g, k))
        trans_logits = trans_logits.at[k].set(jnp.asarray(accum / M_trans))

    out = {"mu": mu, "trans_logits": trans_logits}
    if prior_grad_fn is not None:
        out = tree_add(out, prior_grad_fn(params))
    return out, sampled


def gradient_rmse(
    draw_grad_fn: Callable[[jax.Array], tuple[dict, dict]],
    full_grad,
    n_rep: int = 50,
    seed: int = 0,
):
    key = jax.random.PRNGKey(seed)
    errs = []
    for _ in range(n_rep):
        key, subkey = jax.random.split(key)
        g_hat, _ = draw_grad_fn(subkey)
        errs.append(float(tree_rmse(g_hat, full_grad)))
    return float(np.mean(errs)), float(np.std(errs))
