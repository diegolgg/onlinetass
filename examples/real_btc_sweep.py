"""Sections 5.3 and 6.3 of the paper.

Real-data gradient-RMSE sweep on five trading days of 5-minute BTC USDT
futures deseasonalized log realized variance.  Reads
data/btc_5min_logrv.parquet and writes CSVs and three PNG figures into
runs/real_btc_sweep/<timestamp>/.

Run from the repository root: python examples/real_btc_sweep.py
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

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 200)
print("PROJECT_ROOT =", PROJECT_ROOT)

DATA_PATH = PROJECT_ROOT / "data" / "btc_5min_logrv.parquet"
data_df_full = pd.read_parquet(DATA_PATH)
# Subset to ~5 days so the per-block JAX gradient JIT trace is reused
# efficiently and the whole sweep runs in <15 min.  Bump T_USE for the
# headline full-series result.
T_USE = 1500
data_df = data_df_full.iloc[:T_USE]
print(f"T = {len(data_df)} (subset; full series has {len(data_df_full)} bars)")
print(f"date range: {data_df.index.min()}  ->  {data_df.index.max()}")
print(f"y stats:    mean={data_df['y'].mean():+.4f}  std={data_df['y'].std():.4f}")
print(f"            q01={data_df['y'].quantile(0.01):+.2f}  q99={data_df['y'].quantile(0.99):+.2f}")

y_np = data_df["y"].to_numpy(dtype=float)

fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
axes[0].plot(data_df.index, data_df["log_rv"], lw=0.4, color="#374151")
axes[0].set_ylabel("log RV (raw)")
axes[0].set_title("BTC USDT futures 5-min log realized variance (subset)")
axes[1].plot(data_df.index, data_df["y"], lw=0.4, color="#1f2937")
axes[1].set_ylabel("y = deseas + standardized")
axes[1].axhline(0, color="k", lw=0.3, alpha=0.5)
fig.tight_layout()
plt.show()


@dataclass
class SweepConfig:
    seed: int = 7
    K: int = 2
    obs_dim: int = 1
    half_width: int = 10            # block central length = 21 (~1.75 hour)
    buffer: int = 6
    delta_grid: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    group_budget_mean: int = 6
    group_budget_trans: int = 6
    adapt_steps_per_delta: int = 6
    n_rep_rmse: int = 50
    mix_uniform_proxy: float = 1e-3
    online_discount: float = 0.99
    online_ridge: float = 50.0
    online_floor: float = 0.05
    online_feedback_eps: float = 1e-8
    anchor_power: float = 1.0
    sigma_low: float = 0.55
    sigma_high: float = 1.6
    perturb_direction: str = "mu_high"

CFG = SweepConfig()
CFG


from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=CFG.K, covariance_type="full", random_state=CFG.seed)
gm.fit(y_np.reshape(-1, 1))
mu_init = np.sort(gm.means_.ravel())
print(f"GM means (sorted): {mu_init.round(3)}")
print(f"GM stds          : {np.sqrt(gm.covariances_.ravel()).round(3)}")

# Build reference parameters in the shapes the src.benchmark expects.
mu_ref = np.array([[mu_init[0]], [mu_init[1]]], dtype=np.float32)          # (K, D=1)
covariances_ref = np.stack([                                                 # (K, D=1, D=1)
    np.array([[CFG.sigma_low ** 2]], dtype=np.float32),
    np.array([[CFG.sigma_high ** 2]], dtype=np.float32),
])
# Persistent transition matrix (calm sticky, stress sticky).
A_ref = np.array([[0.94, 0.06],
                  [0.15, 0.85]], dtype=np.float32)
trans_logits_ref = np.log(np.clip(A_ref, 1e-8, 1.0)).astype(np.float32)

ref_params = {
    "mu": jnp.asarray(mu_ref),
    "trans_logits": jnp.asarray(trans_logits_ref),
}
print("reference mu      :", mu_ref.ravel().round(3))
print("reference sigma   :", np.sqrt(np.diag(covariances_ref.reshape(CFG.K, -1))).round(3))
print("reference A       :\n", A_ref)

blocks_df = partition_blocks(len(y_np), half_width=CFG.half_width, buffer=CFG.buffer)
block_features_df = compute_block_features(y_np, blocks_df)
feature_cols = [c for c in block_features_df.columns if c != "block_id"]
W_raw = block_features_df[feature_cols].to_numpy(dtype=float)
W_mean = W_raw.mean(axis=0, keepdims=True)
W_std = W_raw.std(axis=0, keepdims=True)
W_std = np.where(W_std < 1e-8, 1.0, W_std)
W_np = (W_raw - W_mean) / W_std
W_np = np.nan_to_num(W_np, nan=0.0, posinf=0.0, neginf=0.0)

y_jax = jnp.asarray(y_np[:, None], dtype=jnp.float32)             # (T, D=1)
covariances_jax = jnp.asarray(covariances_ref, dtype=jnp.float32)

full_grad_fn = make_full_grad_fn(covariances_jax)
block_grad_fn = make_buffered_block_grad_fn(covariances_jax)
prior_grad_fn = make_prior_grad_fn()

proxy_bundle = build_proxy_weight_bundle(
    y_np,
    blocks_df,
    K=CFG.K,
    seed=CFG.seed,
    mix_uniform=CFG.mix_uniform_proxy,
)
pilot_oracle = compute_oracle_weight_bundle(
    ref_params, y_jax, blocks_df, K=CFG.K, block_grad_fn=block_grad_fn,
)

print("N blocks =", len(blocks_df))
print("feature dim =", W_np.shape[1])
print("reference full-gradient norm:",
      float(jnp.sqrt(sum(jnp.sum(v ** 2) for v in jax.tree_util.tree_leaves(full_grad_fn(ref_params, y_jax))))))

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

def shift_mu(params, delta: float, coord: str = "mu_high"):
    """Perturb the reference mu vector along a chosen direction.

    coord = 'mu_high'  : push the stress-state mean upward by delta
    coord = 'mu_low'   : push the calm-state mean downward by delta
    coord = 'separation': widen the gap (mu_high += delta, mu_low -= delta)
    """
    mu = np.asarray(params["mu"]).copy()
    if coord == "mu_high":
        mu[1, 0] = mu[1, 0] + float(delta)
    elif coord == "mu_low":
        mu[0, 0] = mu[0, 0] - float(delta)
    elif coord == "separation":
        mu[0, 0] = mu[0, 0] - 0.5 * float(delta)
        mu[1, 0] = mu[1, 0] + 0.5 * float(delta)
    else:
        raise ValueError(coord)
    return {
        "mu": jnp.asarray(mu, dtype=jnp.float32),
        "trans_logits": params["trans_logits"],
    }


mean_component_budget = max(1, (CFG.K * CFG.group_budget_mean) // (CFG.K * CFG.obs_dim))
trans_component_budget = max(1, (CFG.K * CFG.group_budget_trans) // (CFG.K * CFG.K))
print("component budgets =", (mean_component_budget, trans_component_budget))


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
    print(f"  [{label}] adapted for {n_steps} steps in {perf_counter() - t0:.1f}s", flush=True)


def chi_square(p, q, eps=1e-12):
    p = np.asarray(p); q = np.maximum(np.asarray(q), eps)
    return float(np.sum((p - q) ** 2 / q))


def mean_staleness(current, pilot):
    pieces = []
    for k in range(current["group_mean"].shape[0]):
        pieces.append(chi_square(current["group_mean"][k], pilot["group_mean"][k]))
        pieces.append(chi_square(current["group_trans"][k], pilot["group_trans"][k]))
    return float(np.mean(pieces))


def top_decile_mass_for(probs_KN: np.ndarray, gradient_norms_N: np.ndarray) -> float:
    if probs_KN.ndim == 2:
        p_joint = probs_KN.mean(axis=0)
    else:
        p_joint = probs_KN.reshape(-1, probs_KN.shape[-1]).mean(axis=0)
    p_joint = p_joint / p_joint.sum()
    top = np.argsort(gradient_norms_N)[-max(1, len(gradient_norms_N) // 10):]
    return float(p_joint[top].sum())


def evaluate_at_params(params, plain_model, anchor_model, seed: int):
    full_grad = full_grad_fn(params, y_jax)
    oracle = compute_oracle_weight_bundle(
        params, y_jax, blocks_df, K=CFG.K, block_grad_fn=block_grad_fn,
    )
    # combined gradient norm per block (mu + trans groups), for top-decile diagnostic
    grad_norms = np.zeros(len(blocks_df))
    cache = {}
    for n in range(len(blocks_df)):
        row = blocks_df.iloc[n].to_dict()
        g = block_grad_fn(params, y_jax, row)
        cache[n] = g
        grad_norms[n] = float(
            np.sqrt(np.sum(np.square(np.asarray(g["mu"]))) + np.sum(np.square(np.asarray(g["trans_logits"]))))
        )

    samplers = {
        "uniform": lambda key: static_group_tass_grad(
            params=params, y=y_jax, blocks=blocks_df,
            mean_probs=np.ones_like(proxy_bundle.group_mean) / len(blocks_df),
            trans_probs=np.ones_like(proxy_bundle.group_trans) / len(blocks_df),
            M_mean=CFG.group_budget_mean, M_trans=CFG.group_budget_trans,
            rng_key=key, block_grad_fn=block_grad_fn, prior_grad_fn=prior_grad_fn,
        ),
        "static_group_proxy_tass": lambda key: static_group_tass_grad(
            params=params, y=y_jax, blocks=blocks_df,
            mean_probs=proxy_bundle.group_mean, trans_probs=proxy_bundle.group_trans,
            M_mean=CFG.group_budget_mean, M_trans=CFG.group_budget_trans,
            rng_key=key, block_grad_fn=block_grad_fn, prior_grad_fn=prior_grad_fn,
        ),
        "static_component_proxy_tass": lambda key: static_component_tass_grad(
            params=params, y=y_jax, blocks=blocks_df,
            mean_probs=proxy_bundle.component_mean, trans_probs=proxy_bundle.component_trans,
            M_mean=mean_component_budget, M_trans=trans_component_budget,
            rng_key=key, block_grad_fn=block_grad_fn, prior_grad_fn=prior_grad_fn,
        ),
        "online_group": lambda key: online_group_tass_grad(
            params=params, y=y_jax, blocks=blocks_df, online_model=plain_model,
            M_mean=CFG.group_budget_mean, M_trans=CFG.group_budget_trans,
            rng_key=key, block_grad_fn=block_grad_fn, prior_grad_fn=prior_grad_fn, update_models=False,
        ),
        "residual_online_group": lambda key: online_group_tass_grad(
            params=params, y=y_jax, blocks=blocks_df, online_model=anchor_model,
            M_mean=CFG.group_budget_mean, M_trans=CFG.group_budget_trans,
            rng_key=key, block_grad_fn=block_grad_fn, prior_grad_fn=prior_grad_fn, update_models=False,
        ),
        "oracle_group": lambda key: static_group_tass_grad(
            params=params, y=y_jax, blocks=blocks_df,
            mean_probs=oracle["group_mean"], trans_probs=oracle["group_trans"],
            M_mean=CFG.group_budget_mean, M_trans=CFG.group_budget_trans,
            rng_key=key, block_grad_fn=block_grad_fn, prior_grad_fn=prior_grad_fn,
        ),
    }

    # also record each method's *sampling distribution* so we can compute
    # top-decile mass without sampling noise.
    method_probs = {
        "uniform": np.ones_like(proxy_bundle.group_mean) / len(blocks_df),
        "static_group_proxy_tass": proxy_bundle.group_mean,
        "static_component_proxy_tass": proxy_bundle.component_mean.mean(axis=1),
        "online_group": plain_model.mean_probs(),
        "residual_online_group": anchor_model.mean_probs(),
        "oracle_group": oracle["group_mean"],
    }

    rows = []
    oracle_rmse = None
    for name, draw_fn in samplers.items():
        rmse_mean, rmse_std = gradient_rmse(
            draw_grad_fn=draw_fn, full_grad=full_grad, n_rep=CFG.n_rep_rmse, seed=seed,
        )
        if name == "oracle_group":
            oracle_rmse = rmse_mean
        rows.append({
            "method": name,
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "top_decile_mass": top_decile_mass_for(method_probs[name], grad_norms),
        })
        print(f"    {name:30s} rmse={rmse_mean:.4f} (top-decile={rows[-1]['top_decile_mass']:.3f})", flush=True)

    out = pd.DataFrame(rows)
    out["oracle_ratio"] = out["rmse_mean"] / oracle_rmse
    return out, oracle

results = []
sweep_t0 = perf_counter()
RUN_DIR = PROJECT_ROOT / "runs" / "real_btc_sweep"
run_stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
run_dir = RUN_DIR / run_stamp
run_dir.mkdir(parents=True, exist_ok=True)
print("checkpoint dir:", run_dir)

for step_idx, delta in enumerate(CFG.delta_grid, start=1):
    print(f"\n=== delta {step_idx}/{len(CFG.delta_grid)} : {delta:+.2f} ===", flush=True)
    delta_t0 = perf_counter()
    params_delta = shift_mu(ref_params, delta=delta, coord=CFG.perturb_direction)

    adapt_online_models(params_delta, plain_online, CFG.adapt_steps_per_delta,
                        seed=CFG.seed + 100 * step_idx, label="plain")
    adapt_online_models(params_delta, anchored_online, CFG.adapt_steps_per_delta,
                        seed=CFG.seed + 100 * step_idx + 17, label="anchored")

    sweep_df, oracle_bundle = evaluate_at_params(
        params_delta, plain_model=plain_online, anchor_model=anchored_online,
        seed=CFG.seed + 1_000 + step_idx,
    )
    sweep_df["delta"] = delta
    sweep_df["staleness_group"] = mean_staleness(oracle_bundle, pilot_oracle)
    results.append(sweep_df)

    partial_df = pd.concat(results, ignore_index=True)
    partial_df.to_csv(run_dir / "results_partial.csv", index=False)
    print(f"completed delta={delta:+.2f} in {perf_counter() - delta_t0:.1f}s", flush=True)

results_df = pd.concat(results, ignore_index=True)
results_df.to_csv(run_dir / "results_final.csv", index=False)
print(f"\nfull sweep finished in {perf_counter() - sweep_t0:.1f}s")
results_df

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
for method, sub in results_df.groupby("method"):
    sub = sub.sort_values("delta")
    axes[0].plot(sub["delta"], sub["rmse_mean"], marker="o", label=method)
    axes[1].plot(sub["delta"], sub["oracle_ratio"], marker="o", label=method)
axes[0].set_title("Gradient RMSE vs rare-state drift (real BTC log-RV)")
axes[0].set_xlabel("delta (mu_stress shift)"); axes[0].set_ylabel("RMSE")
axes[1].set_title("RMSE / oracle-group RMSE")
axes[1].set_xlabel("delta"); axes[1].set_ylabel("ratio")
axes[1].axhline(1.0, color="black", lw=1, ls="--")
axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.savefig(run_dir / "rmse_and_ratio.png", dpi=160, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(9, 4.5))
for method, sub in results_df.groupby("method"):
    sub = sub.sort_values("delta")
    ax.plot(sub["delta"], sub["top_decile_mass"] * 100, marker="o", label=method)
ax.axhline(10, color="black", lw=1, ls="--", label="uniform (10%)")
ax.set_xlabel("delta"); ax.set_ylabel("sampling mass on top-decile blocks (%)")
ax.set_title("Where each method concentrates its weight as theta drifts")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
fig.savefig(run_dir / "top_decile_mass.png", dpi=160, bbox_inches="tight")
plt.show()

# Staleness diagnostic per the plan: oracle group weights at theta(delta)
# vs at theta_0.  This is the empirical T2.1 evidence on real data.
fig, ax = plt.subplots(figsize=(9, 4.5))
stale = results_df[results_df["method"] == "oracle_group"].sort_values("delta")
ax.plot(stale["delta"], stale["staleness_group"], marker="o", color="#0ea5e9")
ax.set_xlabel("delta"); ax.set_ylabel(r"$\chi^2(p^*_\theta, p^*_{\theta_0})$")
ax.set_title("How quickly oracle weights move away from pilot (staleness)")
plt.tight_layout()
fig.savefig(run_dir / "staleness.png", dpi=160, bbox_inches="tight")
plt.show()
