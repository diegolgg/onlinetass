"""Sections 5.2 and 6.2 of the paper.

Practical-proxy TASS RMSE sweep on the stationary K=3 synthetic HMM.
Sweeps the rare-state mean perturbation delta and compares six samplers'
gradient RMSE against the exact full gradient.  Writes CSVs and three
PNG figures into runs/practical_proxy_sweep/<timestamp>/.

Run from the repository root: python examples/practical_proxy_sweep.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path.cwd().resolve()
# allow running from either the repo root or examples/
if (PROJECT_ROOT / "examples").is_dir() is False and (PROJECT_ROOT.parent / "examples").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmark import (
    compute_oracle_weight_bundle,
    gradient_rmse,
    make_prior_grad_fn,
    online_group_tass_grad,
    static_component_tass_grad,
    static_group_tass_grad,
)
from src.features import compute_block_features, partition_blocks
from src.hmm import make_buffered_block_grad_fn, make_full_grad_fn
from src.samplers import GroupedOnlineTASS, build_proxy_weight_bundle
from src.simulator import (
    make_three_state_rare_gaussian_hmm,
    params_from_spec,
    shift_rare_state_mean,
    simulate_hmm_stationary,
)

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 200)
print("PROJECT_ROOT =", PROJECT_ROOT)


@dataclass
class SweepConfig:
    seed: int = 7
    T: int = 2400
    K: int = 3
    obs_dim: int = 3
    half_width: int = 6
    buffer: int = 4
    delta_grid: tuple[float, ...] = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4)
    group_budget_mean: int = 6
    group_budget_trans: int = 6
    adapt_steps_per_delta: int = 10
    n_rep_rmse: int = 20
    mix_uniform_proxy: float = 1e-3
    online_discount: float = 0.99
    online_ridge: float = 50.0
    online_floor: float = 0.05
    online_feedback_eps: float = 1e-8
    anchor_power: float = 1.0


CFG = SweepConfig()
CFG


def component_budgets(cfg: SweepConfig):
    mean_component = max(1, (cfg.K * cfg.group_budget_mean) // (cfg.K * cfg.obs_dim))
    trans_component = max(1, (cfg.K * cfg.group_budget_trans) // (cfg.K * cfg.K))
    return mean_component, trans_component


def chi_square(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    q = np.maximum(q, eps)
    return float(np.sum((p - q) ** 2 / q))


def mean_staleness(current_oracle: dict, pilot_oracle: dict) -> float:
    pieces = []
    for k in range(current_oracle['group_mean'].shape[0]):
        pieces.append(chi_square(current_oracle['group_mean'][k], pilot_oracle['group_mean'][k]))
        pieces.append(chi_square(current_oracle['group_trans'][k], pilot_oracle['group_trans'][k]))
    return float(np.mean(pieces))


def mean_group_disagreement(current_oracle: dict) -> float:
    pieces = []
    K, D, _ = current_oracle['component_mean'].shape
    for k in range(K):
        g = current_oracle['group_mean'][k]
        for d in range(D):
            pieces.append(chi_square(current_oracle['component_mean'][k, d], g))
        for dest in range(K):
            pieces.append(chi_square(current_oracle['component_trans'][k, dest], current_oracle['group_trans'][k]))
    return float(np.mean(pieces))


def current_group_logscore_errors(params, y_jax, blocks_df, block_grad_fn, online_model, eps: float = 1e-8):
    mean_preds = np.column_stack([model.predict(online_model.W) for model in online_model.mean_models])
    trans_preds = np.column_stack([model.predict(online_model.W) for model in online_model.trans_models])

    mean_true = []
    trans_true = []
    cache = {}
    for n in range(len(blocks_df)):
        g = block_grad_fn(params, y_jax, blocks_df.iloc[n].to_dict())
        mean_true.append([np.log(np.sum(np.square(np.asarray(g['mu'][k]))) + eps) for k in range(CFG.K)])
        trans_true.append([np.log(np.sum(np.square(np.asarray(g['trans_logits'][k]))) + eps) for k in range(CFG.K)])
    mean_true = np.asarray(mean_true)
    trans_true = np.asarray(trans_true)

    return {
        'mean_mae': float(np.mean(np.abs(mean_preds - mean_true))),
        'trans_mae': float(np.mean(np.abs(trans_preds - trans_true))),
    }


RUN_DIR = PROJECT_ROOT / "runs" / "practical_proxy_sweep"
RUN_DIR.mkdir(parents=True, exist_ok=True)


def save_online_model_state(model, path):
    payload = {
        "mean_A": np.stack([m.A for m in model.mean_models]),
        "mean_b": np.stack([m.b for m in model.mean_models]),
        "trans_A": np.stack([m.A for m in model.trans_models]),
        "trans_b": np.stack([m.b for m in model.trans_models]),
    }
    np.savez(path, **payload)


base_spec = make_three_state_rare_gaussian_hmm(obs_dim=CFG.obs_dim)
y_np, z_np = simulate_hmm_stationary(base_spec, T=CFG.T, seed=CFG.seed)

blocks_df = partition_blocks(len(y_np), half_width=CFG.half_width, buffer=CFG.buffer)
block_features_df = compute_block_features(y_np, blocks_df)
feature_cols = [c for c in block_features_df.columns if c != 'block_id']
W_raw = block_features_df[feature_cols].to_numpy(dtype=float)
W_mean = W_raw.mean(axis=0, keepdims=True)
W_std = W_raw.std(axis=0, keepdims=True)
W_std = np.where(W_std < 1e-8, 1.0, W_std)
W_np = (W_raw - W_mean) / W_std
W_np = np.nan_to_num(W_np, nan=0.0, posinf=0.0, neginf=0.0)

y_jax = jnp.asarray(y_np, dtype=jnp.float32)
covariances_jax = jnp.asarray(base_spec.covariances, dtype=jnp.float32)

full_grad_fn = make_full_grad_fn(covariances_jax)
block_grad_fn = make_buffered_block_grad_fn(covariances_jax)
prior_grad_fn = make_prior_grad_fn()

pilot_params = {
    key: jnp.asarray(value, dtype=jnp.float32)
    for key, value in params_from_spec(base_spec).items()
}

proxy_bundle = build_proxy_weight_bundle(
    y_np,
    blocks_df,
    K=CFG.K,
    seed=CFG.seed,
    mix_uniform=CFG.mix_uniform_proxy,
)
pilot_oracle = compute_oracle_weight_bundle(
    pilot_params,
    y_jax,
    blocks_df,
    K=CFG.K,
    block_grad_fn=block_grad_fn,
)

mean_component_budget, trans_component_budget = component_budgets(CFG)

print('n_blocks =', len(blocks_df))
print('feature_dim =', W_np.shape[1])
print('component budgets =', (mean_component_budget, trans_component_budget))
print('feature abs max =', float(np.abs(W_np).max()))


plain_online = GroupedOnlineTASS(
    W_np,
    K=CFG.K,
    ridge=CFG.online_ridge,
    discount=CFG.online_discount,
    lambda_floor=CFG.online_floor,
    feedback_eps=CFG.online_feedback_eps,
)

anchored_online = GroupedOnlineTASS(
    W_np,
    K=CFG.K,
    ridge=CFG.online_ridge,
    discount=CFG.online_discount,
    lambda_floor=CFG.online_floor,
    feedback_eps=CFG.online_feedback_eps,
    base_group_mean=proxy_bundle.group_mean,
    base_group_trans=proxy_bundle.group_trans,
    anchor_power=CFG.anchor_power,
)


def adapt_online_models(params, online_model, n_steps: int, seed: int, label: str = "online"):
    key = jax.random.PRNGKey(seed)
    t0 = perf_counter()
    for step in range(n_steps):
        key, subkey = jax.random.split(key)
        online_group_tass_grad(
            params=params,
            y=y_jax,
            blocks=blocks_df,
            online_model=online_model,
            M_mean=CFG.group_budget_mean,
            M_trans=CFG.group_budget_trans,
            rng_key=subkey,
            block_grad_fn=block_grad_fn,
            prior_grad_fn=prior_grad_fn,
            update_models=True,
        )
        if (step + 1) % max(1, n_steps // 5) == 0 or step == 0 or (step + 1) == n_steps:
            elapsed = perf_counter() - t0
            print(f"[{label}] adapt step {step + 1}/{n_steps}  elapsed={elapsed:.1f}s", flush=True)


def evaluate_at_params(params, online_model, anchored_model, seed: int, verbose: bool = True):
    full_grad = full_grad_fn(params, y_jax)
    oracle = compute_oracle_weight_bundle(
        params,
        y_jax,
        blocks_df,
        K=CFG.K,
        block_grad_fn=block_grad_fn,
    )

    methods = {}
    methods['uniform'] = lambda key: static_group_tass_grad(
        params=params,
        y=y_jax,
        blocks=blocks_df,
        mean_probs=np.ones_like(proxy_bundle.group_mean) / len(blocks_df),
        trans_probs=np.ones_like(proxy_bundle.group_trans) / len(blocks_df),
        M_mean=CFG.group_budget_mean,
        M_trans=CFG.group_budget_trans,
        rng_key=key,
        block_grad_fn=block_grad_fn,
        prior_grad_fn=prior_grad_fn,
    )
    methods['static_group_proxy_tass'] = lambda key: static_group_tass_grad(
        params=params,
        y=y_jax,
        blocks=blocks_df,
        mean_probs=proxy_bundle.group_mean,
        trans_probs=proxy_bundle.group_trans,
        M_mean=CFG.group_budget_mean,
        M_trans=CFG.group_budget_trans,
        rng_key=key,
        block_grad_fn=block_grad_fn,
        prior_grad_fn=prior_grad_fn,
    )
    methods['static_component_proxy_tass'] = lambda key: static_component_tass_grad(
        params=params,
        y=y_jax,
        blocks=blocks_df,
        mean_probs=proxy_bundle.component_mean,
        trans_probs=proxy_bundle.component_trans,
        M_mean=mean_component_budget,
        M_trans=trans_component_budget,
        rng_key=key,
        block_grad_fn=block_grad_fn,
        prior_grad_fn=prior_grad_fn,
    )
    methods['online_group'] = lambda key: online_group_tass_grad(
        params=params,
        y=y_jax,
        blocks=blocks_df,
        online_model=online_model,
        M_mean=CFG.group_budget_mean,
        M_trans=CFG.group_budget_trans,
        rng_key=key,
        block_grad_fn=block_grad_fn,
        prior_grad_fn=prior_grad_fn,
        update_models=False,
    )
    methods['residual_online_group'] = lambda key: online_group_tass_grad(
        params=params,
        y=y_jax,
        blocks=blocks_df,
        online_model=anchored_model,
        M_mean=CFG.group_budget_mean,
        M_trans=CFG.group_budget_trans,
        rng_key=key,
        block_grad_fn=block_grad_fn,
        prior_grad_fn=prior_grad_fn,
        update_models=False,
    )
    methods['oracle_group'] = lambda key: static_group_tass_grad(
        params=params,
        y=y_jax,
        blocks=blocks_df,
        mean_probs=oracle['group_mean'],
        trans_probs=oracle['group_trans'],
        M_mean=CFG.group_budget_mean,
        M_trans=CFG.group_budget_trans,
        rng_key=key,
        block_grad_fn=block_grad_fn,
        prior_grad_fn=prior_grad_fn,
    )
    methods['oracle_component'] = lambda key: static_component_tass_grad(
        params=params,
        y=y_jax,
        blocks=blocks_df,
        mean_probs=oracle['component_mean'],
        trans_probs=oracle['component_trans'],
        M_mean=mean_component_budget,
        M_trans=trans_component_budget,
        rng_key=key,
        block_grad_fn=block_grad_fn,
        prior_grad_fn=prior_grad_fn,
    )

    rows = []
    oracle_group_rmse = None
    oracle_component_rmse = None
    eval_t0 = perf_counter()
    for method_idx, (name, draw_fn) in enumerate(methods.items(), start=1):
        rmse_mean, rmse_std = gradient_rmse(
            draw_grad_fn=draw_fn,
            full_grad=full_grad,
            n_rep=CFG.n_rep_rmse,
            seed=seed,
        )
        if name == 'oracle_group':
            oracle_group_rmse = rmse_mean
        if name == 'oracle_component':
            oracle_component_rmse = rmse_mean
        rows.append({'method': name, 'rmse_mean': rmse_mean, 'rmse_std': rmse_std})
        if verbose:
            elapsed = perf_counter() - eval_t0
            print(f"  method {method_idx}/{len(methods)}: {name:<28} rmse={rmse_mean:.3f}  elapsed={elapsed:.1f}s", flush=True)

    out = pd.DataFrame(rows)
    out['oracle_group_ratio'] = out['rmse_mean'] / oracle_group_rmse
    out['oracle_component_ratio'] = out['rmse_mean'] / oracle_component_rmse

    diagnostics = current_group_logscore_errors(params, y_jax, blocks_df, block_grad_fn, online_model)
    diagnostics_anchor = current_group_logscore_errors(params, y_jax, blocks_df, block_grad_fn, anchored_model)

    return out, oracle, {
        'plain_online_mean_mae': diagnostics['mean_mae'],
        'plain_online_trans_mae': diagnostics['trans_mae'],
        'anchored_online_mean_mae': diagnostics_anchor['mean_mae'],
        'anchored_online_trans_mae': diagnostics_anchor['trans_mae'],
    }


results = []
current_plain = plain_online
current_anchor = anchored_online
sweep_t0 = perf_counter()
run_stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
run_dir = RUN_DIR / run_stamp
run_dir.mkdir(parents=True, exist_ok=True)
print("checkpoint dir:", run_dir)

for step_idx, delta in enumerate(CFG.delta_grid, start=1):
    delta_t0 = perf_counter()
    print(f"\n=== delta {step_idx}/{len(CFG.delta_grid)} : {delta:.3f} ===", flush=True)
    spec_delta = shift_rare_state_mean(base_spec, delta=delta)
    params_delta = {
        key: jnp.asarray(value, dtype=jnp.float32)
        for key, value in params_from_spec(spec_delta).items()
    }

    adapt_seed = CFG.seed + 100 * step_idx
    adapt_online_models(params_delta, current_plain, CFG.adapt_steps_per_delta, seed=adapt_seed, label="plain")
    adapt_online_models(params_delta, current_anchor, CFG.adapt_steps_per_delta, seed=adapt_seed + 17, label="anchored")

    sweep_df, oracle_bundle, diag = evaluate_at_params(
        params_delta,
        online_model=current_plain,
        anchored_model=current_anchor,
        seed=CFG.seed + 1_000 + step_idx,
        verbose=True,
    )
    sweep_df['delta'] = delta
    sweep_df['staleness_group'] = mean_staleness(oracle_bundle, pilot_oracle)
    sweep_df['group_disagreement'] = mean_group_disagreement(oracle_bundle)
    for key, value in diag.items():
        sweep_df[key] = value
    results.append(sweep_df)

    partial_df = pd.concat(results, ignore_index=True)
    partial_df.to_csv(run_dir / 'results_partial.csv', index=False)
    sweep_df.to_csv(run_dir / f'results_delta_{step_idx:02d}.csv', index=False)
    np.savez(run_dir / f'oracle_delta_{step_idx:02d}.npz',
             group_mean=oracle_bundle['group_mean'],
             component_mean=oracle_bundle['component_mean'],
             group_trans=oracle_bundle['group_trans'],
             component_trans=oracle_bundle['component_trans'])
    save_online_model_state(current_plain, run_dir / 'plain_online_state.npz')
    save_online_model_state(current_anchor, run_dir / 'anchored_online_state.npz')

    elapsed_delta = perf_counter() - delta_t0
    print(f"completed delta={delta:.3f} in {elapsed_delta:.1f}s  [checkpoint saved]", flush=True)

results_df = pd.concat(results, ignore_index=True)
results_df.to_csv(run_dir / 'results_final.csv', index=False)
print(f"\nfull sweep finished in {perf_counter() - sweep_t0:.1f}s", flush=True)
results_df.head(12)


# Figure 1: RMSE + group-oracle ratio vs delta
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
for method, sub in results_df.groupby('method'):
    sub = sub.sort_values('delta')
    axes[0].plot(sub['delta'], sub['rmse_mean'], marker='o', label=method)
    axes[1].plot(sub['delta'], sub['oracle_group_ratio'], marker='o', label=method)
axes[0].set_title('Whole-gradient RMSE vs rare-state drift'); axes[0].set_xlabel(r'$\delta$'); axes[0].set_ylabel('RMSE')
axes[1].set_title('RMSE / oracle-group RMSE'); axes[1].set_xlabel(r'$\delta$'); axes[1].set_ylabel('ratio')
axes[1].set_yscale('log'); axes[1].axhline(1.0, color='black', lw=1, ls='--')
axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
fig.savefig(run_dir / 'rmse_and_ratio.png', dpi=160, bbox_inches='tight')
plt.show()

# Figure 2: chi-square staleness + group disagreement
diag = results_df[results_df['method']=='uniform'].sort_values('delta')
fig, axes = plt.subplots(1, 2, figsize=(13, 4.0), constrained_layout=True)
axes[0].plot(diag['delta'], diag['staleness_group'], marker='o', color='#0ea5e9')
axes[0].set_xlabel(r'$\delta$'); axes[0].set_ylabel(r'$\chi^2$(static pilot, oracle)')
axes[0].set_title('Static-pilot staleness (group-averaged)')
axes[1].plot(diag['delta'], diag['group_disagreement'], marker='o', color='#ef4444')
axes[1].set_xlabel(r'$\delta$'); axes[1].set_ylabel(r'$\chi^2$(group vs component oracle)')
axes[1].set_title('Group disagreement')
axes[1].set_ylim(0, max(0.001, float(diag['group_disagreement'].max())*1.6))
fig.savefig(run_dir / 'staleness_and_disagreement.png', dpi=160, bbox_inches='tight')
plt.show()

# Figure 3: online predictor MAE
fig, ax = plt.subplots(figsize=(9, 4.0))
ax.plot(diag['delta'], diag['plain_online_mean_mae'], marker='o', label='plain — mean')
ax.plot(diag['delta'], diag['plain_online_trans_mae'], marker='s', label='plain — trans')
ax.plot(diag['delta'], diag['anchored_online_mean_mae'], marker='o', ls='--', label='anchored — mean')
ax.plot(diag['delta'], diag['anchored_online_trans_mae'], marker='s', ls='--', label='anchored — trans')
ax.set_xlabel(r'$\delta$'); ax.set_ylabel('online predictor MAE (log gradient norm)')
ax.set_title('Online discounted-ridge prediction error along the sweep')
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
fig.savefig(run_dir / 'online_mae.png', dpi=160, bbox_inches='tight')
plt.show()

