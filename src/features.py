"""Block partitioning and per-block features for the online ridge predictor.
Use ``partition_blocks(T, half_width, buffer)`` to get block indices and
``compute_block_features(series, blocks_df)`` to get the feature matrix
consumed by ``GroupedOnlineTASS``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def partition_blocks(T: int, half_width: int, buffer: int) -> pd.DataFrame:
    central_width = 2 * int(half_width) + 1
    step = central_width
    start_center = half_width + buffer
    stop_center = T - half_width - buffer

    rows: list[dict[str, int]] = []
    block_id = 0
    for center in range(start_center, stop_center, step):
        start = center - half_width
        end = center + half_width
        left = start - buffer
        right = end + buffer
        if left < 0 or right >= T:
            continue
        rows.append(
            {
                "block_id": block_id,
                "center": center,
                "start": start,
                "end": end,
                "left": left,
                "right": right,
                "block_length": central_width,
                "buffer": int(buffer),
            }
        )
        block_id += 1

    return pd.DataFrame(rows)


def prepare_block_rows(blocks: pd.DataFrame | list[dict]) -> list[dict]:
    if isinstance(blocks, pd.DataFrame):
        return blocks.to_dict(orient="records")
    return list(blocks)


def compute_block_features(
    series: np.ndarray,
    block_indices: pd.DataFrame | list[dict],
    add_intercept: bool = True,
) -> pd.DataFrame:
    X = np.asarray(series, dtype=float)
    if X.ndim == 1:
        X = X[:, None]

    rows = []
    for block in prepare_block_rows(block_indices):
        start = int(block["start"])
        end = int(block["end"])
        sub = X[start : end + 1]

        row: dict[str, float | int] = {"block_id": int(block["block_id"])}
        if add_intercept:
            row["W_intercept"] = 1.0

        mean = sub.mean(axis=0)
        std = sub.std(axis=0, ddof=0)
        maxabs = np.abs(sub).max(axis=0)
        pos_share = (sub > 0.0).mean(axis=0)
        l2 = np.linalg.norm(sub, axis=1)

        for j in range(sub.shape[1]):
            row[f"W_mean__dim{j}"] = float(mean[j])
            row[f"W_std__dim{j}"] = float(std[j])
            row[f"W_maxabs__dim{j}"] = float(maxabs[j])
            row[f"W_posshare__dim{j}"] = float(pos_share[j])

        row["W_l2_mean"] = float(l2.mean())
        row["W_l2_max"] = float(l2.max())
        row["W_l2_std"] = float(l2.std(ddof=0))

        rows.append(row)

    return pd.DataFrame(rows)
