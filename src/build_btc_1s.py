"""Aggregate the daily Tardis CSVs in ``data/raw/`` into a single 1-second
mid-price parquet.

Run: python -m src.build_btc_1s --start 2026-04-01 --end 2026-04-30 --out data/btc_1s.parquet
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_OUT = REPO_ROOT / "data" / "btc_1s.parquet"

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
GRID_FREQ = "1s"


def _load_raw_day(path: Path) -> pd.DataFrame:
    """Read one daily Tardis CSV, keep only the timestamp and top-level prices."""
    df = pd.read_csv(path, compression="gzip")
    df["ts"] = pd.to_datetime(df["timestamp"], unit="us", utc=True)
    return df[["ts", "asks[0].price", "bids[0].price"]].set_index("ts").sort_index()


def _resample_to_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Take the last snapshot per 1-second bucket and forward-fill ≤10 s of gaps."""
    return df.resample(GRID_FREQ).last().ffill(limit=10)


def _compute_mid(df: pd.DataFrame) -> pd.DataFrame:
    mid = 0.5 * (df["asks[0].price"] + df["bids[0].price"])
    return pd.DataFrame({"mid": mid}, index=df.index)


def _filter_paths(paths: list[Path], start: _dt.date | None, end: _dt.date | None) -> list[Path]:
    if start is None and end is None:
        return paths
    out: list[Path] = []
    for p in paths:
        m = DATE_RE.search(p.name)
        if not m:
            continue
        d = _dt.date.fromisoformat(m.group(1))
        if start and d < start:
            continue
        if end and d > end:
            continue
        out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default=None, help="Inclusive start date, YYYY-MM-DD.")
    ap.add_argument("--end", default=None, help="Inclusive end date, YYYY-MM-DD.")
    ap.add_argument("--out", default=str(DEFAULT_OUT), type=Path,
                    help="Output parquet path.")
    a = ap.parse_args()

    start = _dt.date.fromisoformat(a.start) if a.start else None
    end = _dt.date.fromisoformat(a.end) if a.end else None

    raw_paths = sorted(RAW_DIR.glob("binance-futures_book_snapshot_5_*_BTCUSDT.csv.gz"))
    raw_paths = _filter_paths(raw_paths, start, end)
    if not raw_paths:
        raise SystemExit(
            f"No raw files in {RAW_DIR} matching range.  "
            "Run `python -m src.fetch_tardis --start ... --end ...` first."
        )
    print(f"Using {len(raw_paths)} raw files"
          + (f" from {start}" if start else "")
          + (f" to {end}" if end else ""))

    parts: list[pd.DataFrame] = []
    for p in raw_paths:
        print(f"  loading {p.name}")
        parts.append(_compute_mid(_resample_to_grid(_load_raw_day(p))))

    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out)
    print(f"\nWrote {a.out}  rows={len(df):,}  "
          f"range {df.index.min()} -> {df.index.max()}")


if __name__ == "__main__":
    main()
