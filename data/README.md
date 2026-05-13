# Data

This directory holds the preprocessed observation series consumed by
`examples/real_btc_sweep.py`.

## `btc_5min_logrv.parquet`

Five-minute deseasonalized log realized variance of the Binance BTC USDT
perpetual futures contract, April 2026.  Built by the pipeline in
`src/data_prep.py`; column schema:

| column | description |
|--------|-------------|
| `rv` | realized variance per 5-minute bar (sum of squared 1-second log returns) |
| `log_rv` | raw log realized variance |
| `log_rv_deseas` | hour-of-day demeaned log realized variance |
| `y` | globally standardized `log_rv_deseas` — the HMM observation |

T = 8602 rows, indexed by 5-minute UTC timestamps.

## Regenerating from scratch

The upstream raw `book_snapshot_5` CSVs and the 1-second mid-price
parquet are not redistributed.  The full rebuild has three steps; each
is callable as a `python -m src.<...>` module.

1. **Download raw Tardis snapshots.**  Set `TARDIS_API_KEY` in `.env`
   at the repository root (see `.env.example`), then

   ```bash
   python -m src.fetch_tardis --start 2026-04-01 --end 2026-04-30
   ```

   Raw daily CSVs land in `data/raw/` and are gitignored.  The Tardis
   API key is read from `.env` or from `$TARDIS_API_KEY` in the
   environment; it never enters the repository.

2. **Aggregate to a 1-second mid-price parquet.**

   ```bash
   python -m src.build_btc_1s --start 2026-04-01 --end 2026-04-30 \
                              --out data/btc_1s.parquet
   ```

   The output parquet has a 1-second timestamped index and a single
   `mid` column.

3. **Build the 5-minute log-RV series consumed by the experiment.**

   ```bash
   python -m src.data_prep --src data/btc_1s.parquet \
                           --out data/btc_5min_logrv.parquet
   ```

   The pipeline drops the leading NaN, computes 5-minute realized
   variance, discards any bar with zero realized variance, takes logs,
   removes the hour-of-day seasonal mean, and standardizes the result
   to zero mean and unit variance.
