#!/usr/bin/env bash
# Run the TASS-Online pipeline end-to-end.
#
# Usage
# -----
#   ./run_all.sh                                  # use shipped parquet, all 3 experiments
#   ./run_all.sh --fetch 2026-04-01 2026-04-30    # also fetch + rebuild data first
#   ./run_all.sh --skip practical real            # skip selected experiments
#
# Available step names for --skip:
#   demo         tass_online_demo.py                  (~2 min)
#   practical    examples/practical_proxy_sweep.py    (~4 h)
#   real        examples/real_btc_sweep.py            (~3 h)
#   changepoint  examples/changepoint_synthetic.py    (~30 min)
#
# Outputs land under runs/<experiment>/<timestamp>/.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

FETCH_START=""
FETCH_END=""
SKIP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fetch)
            FETCH_START="$2"
            FETCH_END="$3"
            shift 3
            ;;
        --skip)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SKIP="$SKIP $1"
                shift
            done
            ;;
        -h|--help)
            sed -n '2,17p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

skip_step() {
    [[ " $SKIP " == *" $1 "* ]]
}

step() {
    echo
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

# ---------- optional data rebuild ----------
if [[ -n "$FETCH_START" ]]; then
    step "Step 1/4 — fetch raw Tardis snapshots ($FETCH_START to $FETCH_END)"
    python -m src.fetch_tardis --start "$FETCH_START" --end "$FETCH_END"

    step "Step 2/4 — aggregate to 1-second mid prices"
    python -m src.build_btc_1s --start "$FETCH_START" --end "$FETCH_END" \
        --out data/btc_1s.parquet

    step "Step 3/4 — build 5-minute deseasonalized log-RV series"
    python -m src.data_prep --src data/btc_1s.parquet \
        --out data/btc_5min_logrv.parquet
else
    echo
    echo "Skipping data rebuild — using shipped data/btc_5min_logrv.parquet."
    if [[ ! -f data/btc_5min_logrv.parquet ]]; then
        echo "  ERROR: data/btc_5min_logrv.parquet not present.  Rerun with" >&2
        echo "         --fetch <YYYY-MM-DD> <YYYY-MM-DD> after setting" >&2
        echo "         TARDIS_API_KEY in .env." >&2
        exit 1
    fi
fi

# ---------- experiments ----------
if ! skip_step demo; then
    step "Step 4a — quick-start demo (tass_online_demo.py)"
    python tass_online_demo.py
fi

if ! skip_step practical; then
    step "Step 4b — practical-proxy TASS RMSE sweep (Sections 5.2 / 6.2)"
    python examples/practical_proxy_sweep.py
fi

if ! skip_step real; then
    step "Step 4c — real-data BTC log-RV sweep (Sections 5.3 / 6.3)"
    python examples/real_btc_sweep.py
fi

if ! skip_step changepoint; then
    step "Step 4d — changepoint staleness synthetic task (Sections 5.1 / 6.1)"
    python examples/changepoint_synthetic.py
fi

echo
echo "================================================================"
echo "  All requested steps complete."
echo "  Inspect outputs under: $REPO/runs/"
echo "================================================================"
