"""Build the 5-minute deseasonalized log-realized-variance series consumed by
``examples/real_btc_sweep.py``.

Run: python -m src.data_prep --src data/btc_1s.parquet --out data/btc_5min_logrv.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_logrv_series(mid: pd.Series) -> pd.DataFrame:
    """Return the four-column log-RV parquet built from 1-second mid prices."""
    ret_1s = np.log(mid).diff().dropna()
    rv_5m = ret_1s.pow(2).resample("5min", label="right", closed="right").sum()
    rv_5m = rv_5m[rv_5m > 0].copy()
    log_rv = np.log(rv_5m + 1e-16)

    seasonal = log_rv.groupby(log_rv.index.hour).transform("mean")
    log_rv_deseas = log_rv - seasonal
    y = (log_rv_deseas - log_rv_deseas.mean()) / log_rv_deseas.std()

    return pd.concat(
        {"rv": rv_5m, "log_rv": log_rv, "log_rv_deseas": log_rv_deseas, "y": y},
        axis=1,
    ).dropna()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True, type=Path,
                        help="Input 1-second parquet with a 'mid' column.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output 5-min log-RV parquet.")
    args = parser.parse_args()

    mid = pd.read_parquet(args.src)["mid"]
    out = build_logrv_series(mid)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"Wrote {args.out}  T={len(out)}  date {out.index.min()} -> {out.index.max()}")
    print(out.describe().T[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
