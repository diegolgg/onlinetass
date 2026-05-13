"""Download daily Binance BTCUSDT perp ``book_snapshot_5`` CSVs from Tardis
into ``data/raw/``.  Reads ``TARDIS_API_KEY`` from ``.env`` at the repo
root (or from the environment); see ``.env.example``.

Run: python -m src.fetch_tardis --start 2026-04-01 --end 2026-04-30
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import os
import urllib.error
from pathlib import Path

# Repository root: this file lives in src/, so two parents up is the repo.
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

EXCHANGE = "binance-futures"
SYMBOL = "BTCUSDT"
DATA_TYPES = ["book_snapshot_5"]


def _load_env(path: Path) -> dict[str, str]:
    """Minimal .env parser — no python-dotenv dependency."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _api_key() -> str:
    """Return the Tardis API key from os.environ or from .env at the repo root.

    Looks up keys in this order:

    1. ``$TARDIS_API_KEY`` in the process environment.
    2. ``$TARDIS_KEY`` (legacy name) in the process environment.
    3. ``TARDIS_API_KEY`` in ``<repo root>/.env``.
    4. ``TARDIS_KEY`` in ``<repo root>/.env``.

    Raises :class:`SystemExit` with a helpful message if none of these is set.
    """
    env = _load_env(REPO_ROOT / ".env")
    key = (
        os.environ.get("TARDIS_API_KEY")
        or os.environ.get("TARDIS_KEY")
        or env.get("TARDIS_API_KEY")
        or env.get("TARDIS_KEY")
    )
    if not key:
        raise SystemExit(
            "No Tardis API key found.  Set TARDIS_API_KEY in your environment "
            f"or write\n\n    TARDIS_API_KEY=your-key-here\n\nto {REPO_ROOT / '.env'}."
        )
    return key


async def _download_day(day: datetime.date, api_key: str) -> None:
    from tardis_dev.datasets.download import (
        default_file_name,
        default_timeout,
        download_async,
    )

    fname = f"{EXCHANGE}_{DATA_TYPES[0]}_{day.isoformat()}_{SYMBOL}.csv.gz"
    out = RAW_DIR / fname
    if out.exists():
        print(f"  [skip] {fname} already present")
        return

    print(f"  [dl]   {fname}")
    next_day = day + datetime.timedelta(days=1)
    try:
        await download_async(
            exchange=EXCHANGE,
            data_types=DATA_TYPES,
            symbols=[SYMBOL],
            from_date=day.isoformat(),
            to_date=next_day.isoformat(),
            format="csv",
            api_key=api_key,
            download_dir=str(RAW_DIR),
            get_filename=default_file_name,
            timeout=default_timeout,
            download_url_base="datasets.tardis.dev",
            concurrency=2,
            http_proxy=None,
        )
    except urllib.error.HTTPError as exc:
        print(f"    HTTP {exc.code}: {exc.msg}")
    except Exception as exc:  # noqa: BLE001 — bubble any backend hiccup up to stderr
        print(f"    ERROR: {exc}")


async def _main_async(start: datetime.date, end: datetime.date) -> None:
    key = _api_key()
    day = start
    while day <= end:
        await _download_day(day, key)
        day += datetime.timedelta(days=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", required=True, help="Inclusive start date, YYYY-MM-DD.")
    p.add_argument("--end", required=True, help="Inclusive end date, YYYY-MM-DD.")
    a = p.parse_args()

    s = datetime.date.fromisoformat(a.start)
    e = datetime.date.fromisoformat(a.end)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {EXCHANGE} {SYMBOL} book_snapshot_5 from {s} to {e}")
    print(f"  output dir: {RAW_DIR}")
    asyncio.run(_main_async(s, e))
    print("Done.")


if __name__ == "__main__":
    main()
