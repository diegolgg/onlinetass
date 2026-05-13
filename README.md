# Online Group-Specific TASS for SG-MCMC on HMMs

Python/JAX implementation of online group-specific TASS, an adaptive
extension of static TASS-HMM
([Ou, Sen, Young, Dunson, 2024](https://arxiv.org/abs/1810.13431))
for buffered stochastic-gradient MCMC on hidden Markov models.

## Install

```bash
pip install -r requirements.txt
python tass_online_demo.py
```

Dependencies: JAX, NumPy, pandas, scikit-learn, matplotlib, pyarrow.
Python 3.10+.

## Repository layout

| Path | What it is |
|------|-----------|
| `src/hmm.py` | JAX forward–backward, buffered block log-likelihood gradients (`make_full_grad_fn`, `make_buffered_block_grad_fn`). |
| `src/features.py` | Block partitioning + per-block features (`partition_blocks`, `compute_block_features`). |
| `src/simulator.py` | Synthetic K-state Gaussian HMM generator (`make_three_state_rare_gaussian_hmm`, `simulate_hmm_stationary`, `params_from_spec`). |
| `src/samplers.py` | Proxy K-means weights and the online learner (`build_proxy_weight_bundle`, `DiscountedRidge`, `GroupedOnlineTASS`). |
| `src/benchmark.py` | The six samplers compared in the paper (`uniform_stochastic_grad`, `static_group_tass_grad`, `static_component_tass_grad`, `online_group_tass_grad`), the oracle bundle (`compute_oracle_weight_bundle`), and the RMSE evaluator (`gradient_rmse`). |
| `src/fetch_tardis.py`, `src/build_btc_1s.py`, `src/data_prep.py` | Real-data pipeline: download Tardis CSVs → 1-second mid-price parquet → 5-minute deseasonalized log-RV parquet. |
| `examples/changepoint_synthetic.py` | Sections 5.1 / 6.1.  Self-contained — does not import from `src/`. |
| `examples/practical_proxy_sweep.py` | Sections 5.2 / 6.2.  Uses `src/`. |
| `examples/real_btc_sweep.py` | Sections 5.3 / 6.3.  Uses `src/`. |
| `tass_online_demo.py` | Quick-start.  Uses `src/`. |
| `run_all.sh` | End-to-end orchestrator. |

## API reference

The online estimator is one function call: `online_group_tass_grad`.

```python
g_hat, info = online_group_tass_grad(
    params=params,
    y=y_jax,
    blocks=blocks_df,
    online_model=GroupedOnlineTASS(W, K=K),
    M_mean=6, M_trans=6,
    rng_key=key,
    block_grad_fn=make_buffered_block_grad_fn(covariances),
    prior_grad_fn=make_prior_grad_fn(),
    update_models=True,
)
```

Arguments:

| Variable | Explanation |
|----------|-------------|
| `params` | Parameter pytree with keys `mu`, `trans_logits` (and possibly more) holding the current value of θ. |
| `y` | JAX array, the observation sequence of shape `(T,)` or `(T, D)`. |
| `blocks` | pandas DataFrame produced by `partition_blocks(T, half_width, buffer)`, with columns `block_id, center, start, end, left, right, block_length, buffer`. |
| `online_model` | `GroupedOnlineTASS` instance holding per-group discounted-ridge state. |
| `M_mean`, `M_trans` | Mini-batch size for the emission-mean and transition-row groups (blocks per group per call). |
| `rng_key` | JAX PRNG key used for the block draws. |
| `block_grad_fn` | Returned by `make_buffered_block_grad_fn(covariances)`; computes the buffered block log-likelihood gradient. |
| `prior_grad_fn` | Returned by `make_prior_grad_fn()`; computes the log-prior gradient. |
| `update_models` | If `True`, the per-group ridge predictors are updated with the new feedback. |

Returns:

| Variable | Explanation |
|----------|-------------|
| `g_hat` | Stochastic gradient estimate (pytree with the same shape as `params`). |
| `info["mean"][k]` | Array of block indices sampled for the emission-mean group of state `k`. |
| `info["trans"][k]` | Array of block indices sampled for the transition-row group of state `k`. |

`GroupedOnlineTASS` constructor defaults are
`ridge=5.0`, `discount=0.995`, `lambda_floor=0.05`, `feedback_eps=1e-8`;
the paper experiments override `ridge=50.0` and `discount=0.99` (see
the [Parameter tuning](#parameter-tuning) section below).

| Hyperparameter | Default | Role |
|----------------|---------|------|
| `discount` ρ | `0.995` | Sufficient-statistic decay for the ridge recursion (smaller ρ tracks faster but is noisier). |
| `ridge` τ | `5.0` | ℓ₂ penalty on β̂ in the discounted ridge fit. |
| `lambda_floor` λ | `0.05` | Uniform-floor mixing: final probability is `(1 − λ) · softmax(w·β) + λ/N`. |
| `feedback_eps` ε | `1e-8` | Numerical floor inside `log(‖g‖² + ε)` so empty blocks don't NaN. |

## Minimal example

```python
import jax, jax.numpy as jnp
import numpy as np

from src.simulator import (
    make_three_state_rare_gaussian_hmm, simulate_hmm_stationary, params_from_spec,
)
from src.features import partition_blocks, compute_block_features
from src.hmm import make_buffered_block_grad_fn, make_full_grad_fn
from src.samplers import GroupedOnlineTASS, build_proxy_weight_bundle
from src.benchmark import (
    online_group_tass_grad, static_group_tass_grad, gradient_rmse,
    compute_oracle_weight_bundle, make_prior_grad_fn,
)

# 1.  Simulate a stationary K=3 Gaussian HMM with one rare state.
spec = make_three_state_rare_gaussian_hmm(obs_dim=3)
y_np, _ = simulate_hmm_stationary(spec, T=2000, seed=0)
params = {k: jnp.asarray(v) for k, v in params_from_spec(spec).items()}
covariances = jnp.asarray(spec.covariances, dtype=jnp.float32)
y_jax = jnp.asarray(y_np, dtype=jnp.float32)

# 2.  Block partition + features.
blocks_df = partition_blocks(2000, half_width=6, buffer=4)
W = compute_block_features(y_np, blocks_df).drop(columns=["block_id"]).to_numpy()
W = (W - W.mean(axis=0)) / np.maximum(W.std(axis=0), 1e-8)

# 3.  Static proxy + online ridge bundles.
proxy = build_proxy_weight_bundle(y_np, blocks_df, K=3, seed=0, mix_uniform=1e-3)
online_model = GroupedOnlineTASS(W, K=3, ridge=50.0, discount=0.99, lambda_floor=0.05)
block_grad_fn = make_buffered_block_grad_fn(covariances)
prior_grad_fn = make_prior_grad_fn()

# 4.  Adapt the online learner for a few SGLD-like steps.
key = jax.random.PRNGKey(0)
for _ in range(10):
    key, sk = jax.random.split(key)
    online_group_tass_grad(
        params=params, y=y_jax, blocks=blocks_df, online_model=online_model,
        M_mean=6, M_trans=6, rng_key=sk,
        block_grad_fn=block_grad_fn, prior_grad_fn=prior_grad_fn, update_models=True,
    )

# 5.  Compare gradient-RMSE at the pilot against the exact full gradient.
full_grad = make_full_grad_fn(covariances)(params, y_jax)
rmse_online, _ = gradient_rmse(
    lambda k: online_group_tass_grad(
        params=params, y=y_jax, blocks=blocks_df, online_model=online_model,
        M_mean=6, M_trans=6, rng_key=k,
        block_grad_fn=block_grad_fn, prior_grad_fn=prior_grad_fn, update_models=False,
    ),
    full_grad=full_grad, n_rep=20, seed=1,
)
print("Online TASS gradient RMSE at the pilot:", rmse_online)
```

The runnable version is `tass_online_demo.py` at the repository root.

## Paper experiments

Three reproduction scripts live in `examples/`.  Each writes a fresh
timestamped subdirectory under `runs/...` containing CSV results and
PNG figures.

| File | Paper section | Imports from `src/` | What it produces |
|------|---------------|---------------------|------------------|
| `examples/changepoint_synthetic.py` | 5.1 / 6.1 | none — self-contained | Held-out log-predictive density on the 1-rare and 2-rare changepoint regimes (Tables 1, 2; Figures 1, 2). |
| `examples/practical_proxy_sweep.py` | 5.2 / 6.2 | `simulator`, `features`, `hmm`, `samplers`, `benchmark` | Oracle-ratio sweep on the stationary K=3 HMM (Table 3; Figures 3, 4, 5). |
| `examples/real_btc_sweep.py`         | 5.3 / 6.3 | `features`, `hmm`, `samplers`, `benchmark` | Oracle-ratio sweep on real BTC log-RV (Table 4; Figures 6, 7, 8). |

Wall-clock budgets on a modern desktop CPU are roughly a few minutes
for the changepoint script, ~4 h for the K=3 synthetic sweep, and ~3 h
for the real-data sweep.  Each script's top-of-file docstring names the
corresponding paper sections.

The changepoint script does its own SGLD chain sampling and held-out
predictive density evaluation inline (Colab-derived development
history), which is why it does not share the `src/` machinery.  The two
gradient-RMSE sweeps are built on `src/` end to end.

## Full pipeline

`run_all.sh` chains every step together.  Two common patterns:

**Reuse shipped data (no API key needed).**

```bash
./run_all.sh
```

Runs `tass_online_demo.py` → practical-proxy sweep → real-BTC sweep →
changepoint script, all against `data/btc_5min_logrv.parquet` that
ships with the repository.  Total wall-clock about 7 hours on a modern
desktop CPU; can be restricted with `--skip` (e.g.
`./run_all.sh --skip practical real` for just the demo + changepoint
script, ~30 min total).

**Rebuild data from raw Tardis CSVs first.**

```bash
cp .env.example .env       # fill in TARDIS_API_KEY
./run_all.sh --fetch 2026-04-01 2026-04-30
```

Adds three pipeline steps before the experiments: `fetch_tardis`
downloads daily `book_snapshot_5` CSVs, `build_btc_1s` aggregates them
to a 1-second mid-price parquet, and `data_prep` produces the
deseasonalized 5-minute log-RV series consumed by the real-data sweep.
`.env` is gitignored and never committed; the script never prints the
key.

## Parameter tuning

The defaults
`GroupedOnlineTASS(ridge=5.0, discount=0.995, lambda_floor=0.05, feedback_eps=1e-8)`
together with the block hyperparameters used in the paper experiments
(`ridge=50.0`, `discount=0.99`) were chosen by the rules of thumb
below.  Each group's hyperparameters could in principle be tuned
independently, but for all paper experiments we use the same value for
every group.

**Block geometry (`half_width` h, `buffer` B).**  `h` controls the
trade-off between sub-sampling efficiency (small h is cheap per block)
and the bias of the buffered forward–backward approximation (small h
means more boundary artefacts).  Pick the smallest h such that the
within-buffer state transitions are reasonably approximated.  The
paper uses `h=10` (block length 21) for real BTC at 5-minute bars and
`h=6` (block length 13) for the synthetic K=3 HMM.  The buffer `B`
should be a few times the slowest mixing time of the latent chain:
`B=6` for the K=2 real-data experiment (`A_{00} ≈ 0.97`) and `B=4` for
the K=3 synthetic.

**Online ridge regression.**

- `ridge` τ — Too small makes the ridge ill-conditioned on high-dimensional
  block features; too large biases the predictor toward zero, in which
  case online TASS reduces to uniform sampling.  We tuned this on the
  synthetic K=3 sweep and use τ=50 throughout the paper.
- `discount` ρ ∈ (0,1) — How aggressively old sufficient statistics
  decay.  Smaller ρ tracks parameter drift faster but raises Monte
  Carlo variance; larger ρ is more conservative but slower to adapt.
  The paper uses ρ=0.99 (effective horizon ≈ 100 SGLD steps).  When the
  staleness diagnostic `staleness_group` in the run CSVs grows fast,
  prefer a smaller ρ.
- `lambda_floor` λ — The uniform-mixing floor.  Provides a finite-variance
  guarantee on the inverse-probability estimator and acts as
  exploration noise so the ridge cannot collapse onto a single block.
  λ=0.05 is the right order of magnitude; values much smaller risk
  pathologically peaked sampling distributions.
- `feedback_eps` ε — Avoids `log 0` when a sampled block has zero
  contribution to the group gradient.  Should be a few orders of
  magnitude below the smallest non-zero block-gradient norm; ε=1e-8
  is fine for all paper experiments.

**Static-TASS proxy (`build_proxy_weight_bundle`).**  The K-means proxy
uses `K` equal to the number of latent states and a `mix_uniform=1e-3`
safety floor on the resulting block probabilities.  Larger
`mix_uniform` shrinks the proxy toward uniform; useful when K-means
produces near-degenerate labels on noisy data.

**Adapt steps S per parameter update.**  Number of online ridge
updates per change in θ.  In a real SGLD chain this would be 1 (one
ridge update per SGLD step); the gradient-RMSE protocol in the paper
uses S=6 (real BTC, §5.3) or S=10 (synthetic K=3, §5.2) to amortize
the JIT trace and to let the online learner converge before each new
θ value is evaluated.  If `plain_online_mean_mae` in the run CSVs
stays large after a sweep, increase S.

**Mini-batch size M per group.**  Number of block gradients evaluated
per call per group.  Larger M reduces variance but raises wall-clock
roughly linearly.  We use `M_mean = M_trans = 6` for K ∈ {2, 3} so
each group's block draws cover ~10% of available blocks per call.
For larger K, scale M so the effective per-state minibatch is at
least 2.

## Notes

- The forward–backward recursion in `src/hmm.py` is JAX-jitted; the
  first call per `(half_width, buffer, K)` configuration pays a
  compile cost of 30–60 s.  Subsequent calls reuse the compiled cache.
- `examples/real_btc_sweep.py` expects `data/btc_5min_logrv.parquet`
  to exist.  A small (~400 KB) copy of the preprocessed parquet ships
  with the repository.  To rebuild it from scratch, populate `.env` at
  the repo root with `TARDIS_API_KEY=...` (template in `.env.example`)
  and run

  ```bash
  python -m src.fetch_tardis --start 2026-04-01 --end 2026-04-30
  python -m src.build_btc_1s --start 2026-04-01 --end 2026-04-30 \
                             --out data/btc_1s.parquet
  python -m src.data_prep --src data/btc_1s.parquet \
                          --out data/btc_5min_logrv.parquet
  ```

  The Tardis key is never committed; `.env` is in `.gitignore`.
- If `jax.random.choice` raises a "non-positive probabilities" error,
  raise `mix_uniform` or `lambda_floor` slightly; this usually means
  the online ridge has temporarily collapsed onto a near-zero block.
