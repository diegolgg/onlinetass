"""Quick-start demo: simulate a K=3 HMM, fit the online learner, and compare
the six samplers' gradient RMSE at the pilot.

Run from the repository root: python tass_online_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from src.benchmark import (
    compute_oracle_weight_bundle,
    gradient_rmse,
    make_prior_grad_fn,
    online_group_tass_grad,
    static_component_tass_grad,
    static_group_tass_grad,
    uniform_stochastic_grad,
)
from src.features import compute_block_features, partition_blocks
from src.hmm import make_buffered_block_grad_fn, make_full_grad_fn
from src.samplers import GroupedOnlineTASS, build_proxy_weight_bundle
from src.simulator import (
    make_three_state_rare_gaussian_hmm,
    params_from_spec,
    simulate_hmm_stationary,
)


def main() -> None:
    seed = 7
    T = 1200
    half_width = 6        # block central length = 13
    buffer = 4
    K = 3
    M_mean = M_trans = 6
    adapt_steps = 8
    n_rep = 15

    # 1. Simulate a stationary three-state Gaussian HMM with one rare state.
    spec = make_three_state_rare_gaussian_hmm(obs_dim=3)
    y_np, _ = simulate_hmm_stationary(spec, T=T, seed=seed)
    y_jax = jnp.asarray(y_np, dtype=jnp.float32)
    covariances = jnp.asarray(spec.covariances, dtype=jnp.float32)
    pilot_params = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in params_from_spec(spec).items()}

    # 2. Build the block partition + block features.
    blocks_df = partition_blocks(T, half_width=half_width, buffer=buffer)
    feats = compute_block_features(y_np, blocks_df)
    feature_cols = [c for c in feats.columns if c != "block_id"]
    W = feats[feature_cols].to_numpy(dtype=float)
    W = (W - W.mean(axis=0, keepdims=True)) / np.maximum(W.std(axis=0, keepdims=True), 1e-8)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    N = len(blocks_df)
    print(f"T={T}  N_blocks={N}  feature_dim={W.shape[1]}")

    # 3. Build static (proxy) and oracle weight bundles, plus the online learner.
    proxy = build_proxy_weight_bundle(y_np, blocks_df, K=K, seed=seed, mix_uniform=1e-3)
    block_grad_fn = make_buffered_block_grad_fn(covariances)
    full_grad_fn = make_full_grad_fn(covariances)
    prior_grad_fn = make_prior_grad_fn()
    oracle = compute_oracle_weight_bundle(pilot_params, y_jax, blocks_df, K=K, block_grad_fn=block_grad_fn)
    online_model = GroupedOnlineTASS(W, K=K, ridge=50.0, discount=0.99, lambda_floor=0.05)

    # 4. A handful of online adapt steps at the pilot.
    print(f"Running {adapt_steps} online adapt steps...")
    key = jax.random.PRNGKey(seed)
    t0 = perf_counter()
    for step in range(adapt_steps):
        key, subkey = jax.random.split(key)
        online_group_tass_grad(
            params=pilot_params, y=y_jax, blocks=blocks_df, online_model=online_model,
            M_mean=M_mean, M_trans=M_trans, rng_key=subkey,
            block_grad_fn=block_grad_fn, prior_grad_fn=prior_grad_fn, update_models=True,
        )
    print(f"  done in {perf_counter() - t0:.1f}s")

    # 5. Gradient-RMSE comparison at the pilot.
    full_grad = full_grad_fn(pilot_params, y_jax)
    samplers = {
        "uniform": lambda key: uniform_stochastic_grad(
            pilot_params, y_jax, blocks_df, M_mean, key, block_grad_fn, prior_grad_fn),
        "static_group_proxy_tass": lambda key: static_group_tass_grad(
            params=pilot_params, y=y_jax, blocks=blocks_df,
            mean_probs=proxy.group_mean, trans_probs=proxy.group_trans,
            M_mean=M_mean, M_trans=M_trans, rng_key=key,
            block_grad_fn=block_grad_fn, prior_grad_fn=prior_grad_fn,
        ),
        "static_component_proxy_tass": lambda key: static_component_tass_grad(
            params=pilot_params, y=y_jax, blocks=blocks_df,
            mean_probs=proxy.component_mean, trans_probs=proxy.component_trans,
            M_mean=max(1, M_mean // 3), M_trans=max(1, M_trans // 3), rng_key=key,
            block_grad_fn=block_grad_fn, prior_grad_fn=prior_grad_fn,
        ),
        "online_group": lambda key: online_group_tass_grad(
            params=pilot_params, y=y_jax, blocks=blocks_df, online_model=online_model,
            M_mean=M_mean, M_trans=M_trans, rng_key=key,
            block_grad_fn=block_grad_fn, prior_grad_fn=prior_grad_fn, update_models=False,
        ),
        "oracle_group": lambda key: static_group_tass_grad(
            params=pilot_params, y=y_jax, blocks=blocks_df,
            mean_probs=oracle["group_mean"], trans_probs=oracle["group_trans"],
            M_mean=M_mean, M_trans=M_trans, rng_key=key,
            block_grad_fn=block_grad_fn, prior_grad_fn=prior_grad_fn,
        ),
    }

    print(f"\nGradient RMSE at the pilot, n_rep={n_rep}:")
    print(f"  {'method':32s}  {'RMSE':>10s}  {'oracle ratio':>14s}")
    rmse_oracle = None
    rows = []
    for name, draw in samplers.items():
        rmse_mean, _ = gradient_rmse(draw, full_grad=full_grad, n_rep=n_rep, seed=seed + 1)
        rows.append((name, rmse_mean))
        if name == "oracle_group":
            rmse_oracle = rmse_mean
    for name, r in rows:
        ratio = r / rmse_oracle if rmse_oracle else float("nan")
        print(f"  {name:32s}  {r:>10.4f}  {ratio:>14.4f}")


if __name__ == "__main__":
    main()
