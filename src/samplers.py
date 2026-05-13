"""Block-sampling weight constructions for buffered SG-MCMC on HMMs:
K-means proxy weights (``build_proxy_weight_bundle``), the discounted
ridge learner (``DiscountedRidge``), and the per-group bundle used in
the paper (``GroupedOnlineTASS``).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    total = float(x.sum())
    if total <= 0.0:
        return np.ones_like(x) / len(x)
    return x / total


def safe_ridge_solve(A: np.ndarray, b: np.ndarray, jitter: float = 1e-6, max_tries: int = 8) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    eye = np.eye(A.shape[0], dtype=float)
    for power in range(max_tries):
        try:
            return np.linalg.solve(A + (10.0**power) * jitter * eye, b)
        except np.linalg.LinAlgError:
            continue
    return np.linalg.pinv(A) @ b


def kmeans_labels_matlab_style(
    y: np.ndarray,
    K: int,
    seed: int = 0,
    sort_centres: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)
    y_scaled = (y - y_mean) / y_std
    y_scaled = np.clip(y_scaled, -20.0, 20.0)
    y_scaled = np.nan_to_num(y_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    model = KMeans(
        n_clusters=K,
        init="random",
        n_init=20,
        random_state=seed,
        algorithm="lloyd",
    )
    labels = model.fit_predict(y_scaled)
    centers_scaled = np.asarray(model.cluster_centers_, dtype=float)
    centers = centers_scaled * y_std + y_mean

    if not sort_centres:
        return labels, centers

    order = np.argsort(np.linalg.norm(centers, axis=1))
    inverse = np.zeros(K, dtype=int)
    inverse[order] = np.arange(K)
    relabelled = inverse[labels]
    return relabelled, centers[order]


def _mix_and_normalize(weights: np.ndarray, mix_uniform: float, eps: float) -> np.ndarray:
    weights = np.maximum(np.asarray(weights, dtype=float), eps)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    if mix_uniform > 0.0:
        weights = (1.0 - mix_uniform) * weights + mix_uniform / weights.shape[-1]
        weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights


def _prepare_block_rows(blocks: pd.DataFrame | list[dict]) -> list[dict]:
    if isinstance(blocks, pd.DataFrame):
        return blocks.to_dict(orient="records")
    return list(blocks)


@dataclass
class ProxyWeightBundle:
    labels: np.ndarray
    centers: np.ndarray
    component_mean: np.ndarray
    group_mean: np.ndarray
    component_trans: np.ndarray
    group_trans: np.ndarray
    diagnostics: pd.DataFrame


def build_proxy_weight_bundle(
    y: np.ndarray,
    blocks: pd.DataFrame | list[dict],
    K: int,
    seed: int = 0,
    mix_uniform: float = 1e-3,
    eps: float = 1e-8,
) -> ProxyWeightBundle:
    X = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    T, d = X.shape

    labels, centers = kmeans_labels_matlab_style(X, K=K, seed=seed, sort_centres=True)
    block_rows = _prepare_block_rows(blocks)
    N = len(block_rows)

    global_mean = np.zeros((K, d), dtype=float)
    for k in range(K):
        mask = labels == k
        if np.any(mask):
            global_mean[k] = X[mask].mean(axis=0)

    global_trans_counts = np.zeros((K, K), dtype=float)
    for t in range(1, T):
        global_trans_counts[labels[t - 1], labels[t]] += 1.0
    global_row_probs = np.zeros((K, K), dtype=float)
    for j in range(K):
        row_total = global_trans_counts[j].sum()
        if row_total > 0.0:
            global_row_probs[j] = global_trans_counts[j] / row_total
        else:
            global_row_probs[j] = np.ones(K, dtype=float) / K

    component_mean = np.zeros((K, d, N), dtype=float)
    group_mean = np.zeros((K, N), dtype=float)
    component_trans = np.zeros((K, K, N), dtype=float)
    group_trans = np.zeros((K, N), dtype=float)
    diag_rows: list[dict[str, float | int]] = []

    for n, block in enumerate(block_rows):
        start = int(block["start"])
        end = int(block["end"])
        idx = np.arange(start, end + 1)
        X_block = X[idx]
        z_block = labels[idx]

        for k in range(K):
            mask = z_block == k
            count_k = int(mask.sum())
            if count_k > 0:
                mean_block = X_block[mask].mean(axis=0)
                diff = np.abs(mean_block - global_mean[k])
                component_mean[k, :, n] = count_k * diff
                group_mean[k, n] = count_k * np.linalg.norm(diff, ord=2)
            diag_rows.append(
                {
                    "block_id": int(block["block_id"]),
                    "state": k,
                    "family": "mean",
                    "count": count_k,
                    "group_score": float(group_mean[k, n]),
                }
            )

        for j in range(K):
            row_counts = np.zeros(K, dtype=float)
            origin_count = 0.0
            for t in range(start + 1, end + 1):
                prev_state = labels[t - 1]
                curr_state = labels[t]
                if prev_state == j:
                    origin_count += 1.0
                    row_counts[curr_state] += 1.0

            if origin_count > 0.0:
                local_row = row_counts / origin_count
                diff = np.abs(local_row - global_row_probs[j])
                component_trans[j, :, n] = origin_count * diff
                group_trans[j, n] = origin_count * np.linalg.norm(diff, ord=2)
            diag_rows.append(
                {
                    "block_id": int(block["block_id"]),
                    "state": j,
                    "family": "transition",
                    "count": int(origin_count),
                    "group_score": float(group_trans[j, n]),
                }
            )

    component_mean = _mix_and_normalize(component_mean, mix_uniform=mix_uniform, eps=eps)
    group_mean = _mix_and_normalize(group_mean, mix_uniform=mix_uniform, eps=eps)
    component_trans = _mix_and_normalize(component_trans, mix_uniform=mix_uniform, eps=eps)
    group_trans = _mix_and_normalize(group_trans, mix_uniform=mix_uniform, eps=eps)

    return ProxyWeightBundle(
        labels=labels,
        centers=centers,
        component_mean=component_mean,
        group_mean=group_mean,
        component_trans=component_trans,
        group_trans=group_trans,
        diagnostics=pd.DataFrame(diag_rows),
    )


@dataclass
class DiscountedRidge:
    p: int
    ridge: float = 5.0
    discount: float = 0.995
    jitter: float = 1e-6
    target_clip: tuple[float, float] = (-20.0, 20.0)
    coef_clip: float = 100.0
    feature_clip: float = 20.0

    def __post_init__(self) -> None:
        self.A = self.ridge * np.eye(self.p, dtype=float)
        self.b = np.zeros(self.p, dtype=float)

    def update(self, x: np.ndarray, y: float) -> None:
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = float(np.clip(y, self.target_clip[0], self.target_clip[1]))
        self.A = self.discount * self.A + np.outer(x, x)
        self.b = self.discount * self.b + x * y
        self.A = np.nan_to_num(self.A, nan=0.0, posinf=1e12, neginf=-1e12)
        self.b = np.nan_to_num(self.b, nan=0.0, posinf=1e12, neginf=-1e12)

    def coef(self) -> np.ndarray:
        beta = safe_ridge_solve(self.A, self.b, jitter=self.jitter)
        beta = np.nan_to_num(beta, nan=0.0, posinf=self.coef_clip, neginf=-self.coef_clip)
        return np.clip(beta, -self.coef_clip, self.coef_clip)

    def predict(self, X: np.ndarray) -> np.ndarray:
        beta = self.coef()
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, -self.feature_clip, self.feature_clip)
        beta = np.clip(beta, -self.coef_clip, self.coef_clip)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            pred = X @ beta
        return np.nan_to_num(pred, nan=0.0, posinf=self.target_clip[1], neginf=self.target_clip[0])


class GroupedOnlineTASS:
    def __init__(
        self,
        features: np.ndarray,
        K: int,
        ridge: float = 5.0,
        discount: float = 0.995,
        lambda_floor: float = 0.05,
        feedback_eps: float = 1e-8,
        clip_pred: tuple[float, float] = (-20.0, 20.0),
        clip_feedback: tuple[float, float] = (-20.0, 20.0),
        base_group_mean: np.ndarray | None = None,
        base_group_trans: np.ndarray | None = None,
        anchor_power: float = 1.0,
    ) -> None:
        self.W = np.asarray(features, dtype=float)
        self.N, self.p = self.W.shape
        self.K = int(K)
        self.lambda_floor = float(lambda_floor)
        self.feedback_eps = float(feedback_eps)
        self.clip_pred = clip_pred
        self.clip_feedback = clip_feedback
        self.anchor_power = float(anchor_power)

        self.mean_models = [
            DiscountedRidge(
                self.p,
                ridge=ridge,
                discount=discount,
                target_clip=clip_feedback,
            )
            for _ in range(K)
        ]
        self.trans_models = [
            DiscountedRidge(
                self.p,
                ridge=ridge,
                discount=discount,
                target_clip=clip_feedback,
            )
            for _ in range(K)
        ]

        self.base_group_mean = None if base_group_mean is None else np.asarray(base_group_mean, dtype=float)
        self.base_group_trans = None if base_group_trans is None else np.asarray(base_group_trans, dtype=float)

    def _probs_from_model(self, model: DiscountedRidge, base: np.ndarray | None = None) -> np.ndarray:
        pred = np.clip(model.predict(self.W), self.clip_pred[0], self.clip_pred[1])
        raw = np.exp(0.5 * pred)
        if base is not None:
            raw = raw * np.maximum(base, 1e-12) ** self.anchor_power
        probs = normalize(raw)
        if self.lambda_floor > 0.0:
            probs = (1.0 - self.lambda_floor) * probs + self.lambda_floor / self.N
        return normalize(probs)

    def mean_probs(self) -> np.ndarray:
        probs = np.zeros((self.K, self.N), dtype=float)
        for k in range(self.K):
            base = None if self.base_group_mean is None else self.base_group_mean[k]
            probs[k] = self._probs_from_model(self.mean_models[k], base=base)
        return probs

    def trans_probs(self) -> np.ndarray:
        probs = np.zeros((self.K, self.N), dtype=float)
        for k in range(self.K):
            base = None if self.base_group_trans is None else self.base_group_trans[k]
            probs[k] = self._probs_from_model(self.trans_models[k], base=base)
        return probs

    def update_mean(self, state: int, block_id: int, sqnorm: float) -> None:
        target = np.log(float(sqnorm) + self.feedback_eps)
        target = float(np.clip(target, self.clip_feedback[0], self.clip_feedback[1]))
        self.mean_models[int(state)].update(self.W[int(block_id)], target)

    def update_trans(self, state: int, block_id: int, sqnorm: float) -> None:
        target = np.log(float(sqnorm) + self.feedback_eps)
        target = float(np.clip(target, self.clip_feedback[0], self.clip_feedback[1]))
        self.trans_models[int(state)].update(self.W[int(block_id)], target)
