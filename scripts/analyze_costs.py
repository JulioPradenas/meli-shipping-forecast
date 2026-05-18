"""Batch cost-sensitive analysis for Phase 7.

Trains LightGBM (tuned) and SeasonalNaive on each fold, generates
predictions, and computes asymmetric-cost metrics across a sweep of
cost ratios. Persists results as CSVs for the Phase 7 notebook.

Per the Phase 7 design decisions:

1. **Cost ratios swept**: 1x, 2x, 3x, 5x (c_under / c_over).
2. **Primary metric**: expected_gain of LightGBM vs SeasonalNaive.
3. **Dimensions**: by fold + zoom on Black Friday and Día dos Namorados
   event windows (+/- 7 days).
4. **Threshold tuning**: both without tuning and with the optimal
   multiplicative offset, reported in parallel.

Outputs (CSV) in data/processed/phase7/:

* gain_by_fold.csv: expected gain per (fold, ratio, tuning_strategy).
* gain_in_events.csv: same but restricted to BF / DdN windows.
* optimal_alpha.csv: best alpha per (fold, ratio).
* predictions_per_fold.csv: all predictions for the notebook to load.

Usage:

    python scripts/analyze_costs.py
    # Or with custom output dir:
    python scripts/analyze_costs.py --output-dir data/processed/phase7_v2
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from shipping_forecast.evaluation import (
    asymmetric_cost,
    expected_gain,
    optimal_threshold_multiplier,
    time_series_split,
)
from shipping_forecast.features import HolidayFeatures
from shipping_forecast.models import LightGBMForecaster, SeasonalNaiveForecaster

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "shipping.db"
BEST_PARAMS_PATH = PROJECT_ROOT / "data" / "processed" / "best_lgbm_params.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "phase7"

# Cost ratios to sweep. c_over is fixed at 1.0 USD/unit; c_under varies.
COST_RATIOS = (1.0, 2.0, 3.0, 5.0)

# Event windows (Phase 7.3): +/- 7 days around the event date.
EVENT_WINDOWS = {
    "black_friday_2017": ("2017-11-17", "2017-12-01"),  # BF was 2017-11-24
    "dia_dos_namorados_2018": ("2018-06-05", "2018-06-19"),  # DdN: 2018-06-12
}

# Alpha grid for threshold tuning (mirrors evaluation.cost_metrics default).
ALPHA_GRID = (
    -0.20,
    -0.15,
    -0.10,
    -0.05,
    0.0,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
)

# Fixed LightGBM params (the search-fixed ones; tuned params loaded from JSON).
FIXED_LGB_PARAMS = {
    "objective": "regression_l1",
    "n_estimators": 1000,
    "verbose": -1,
    "random_state": 42,
    "bagging_freq": 5,
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_panel(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load the daily shipments panel from SQLite."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"Shipping DB not found at {db_path}. Run scripts/load_data.py first."
        )
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT shipment_date, customer_state, n_shipments "
            "FROM fact_daily_shipments_by_state "
            "ORDER BY customer_state, shipment_date",
            conn,
            parse_dates=["shipment_date"],
        )
    return df


def load_tuned_lgb_params(path: Path = BEST_PARAMS_PATH) -> dict:
    """Combine fixed + tuned params for the LightGBM model."""
    with open(path) as f:
        tuning_result = json.load(f)
    return {**FIXED_LGB_PARAMS, **tuning_result["best_params"]}


# ---------------------------------------------------------------------------
# Predictions per fold
# ---------------------------------------------------------------------------


def generate_predictions(
    df: pd.DataFrame,
    lgb_params: dict,
) -> pd.DataFrame:
    """For each fold, fit both models and return predictions joined with
    truth and is_operational flag.

    Returns:
        DataFrame with columns:
            fold_id, shipment_date, customer_state, n_shipments,
            y_pred_lgb, y_pred_seasn, is_operational.
    """
    folds = time_series_split(df)
    all_preds = []

    for fold_idx, fold in enumerate(folds, start=1):
        train, test = fold.train, fold.test
        horizon = test["shipment_date"].nunique()

        logger.info(f"Fold {fold_idx}: fitting models (horizon={horizon} days)")

        # LightGBM
        lgb = LightGBMForecaster(params=lgb_params)
        lgb.fit(train)
        lgb_preds = lgb.predict(train, horizon=horizon)

        # SeasonalNaive
        sn = SeasonalNaiveForecaster(season=7)
        sn.fit(train)
        sn_preds = sn.predict(train, horizon=horizon)

        # Merge predictions with truth from test
        merged = test.merge(
            lgb_preds.rename(columns={"y_pred": "y_pred_lgb"}),
            on=["shipment_date", "customer_state"],
        ).merge(
            sn_preds.rename(columns={"y_pred": "y_pred_seasn"}),
            on=["shipment_date", "customer_state"],
        )

        # Add is_operational flag (filter applied downstream)
        merged = HolidayFeatures(sort_col="shipment_date").transform(merged)
        merged["fold_id"] = fold_idx

        all_preds.append(
            merged[
                [
                    "fold_id",
                    "shipment_date",
                    "customer_state",
                    "n_shipments",
                    "y_pred_lgb",
                    "y_pred_seasn",
                    "is_operational",
                ]
            ]
        )

    return pd.concat(all_preds, ignore_index=True)


# ---------------------------------------------------------------------------
# Cost analysis
# ---------------------------------------------------------------------------


def compute_gain_by_fold(predictions: pd.DataFrame) -> pd.DataFrame:
    """Expected gain per (fold, cost_ratio, tuning_strategy).

    Strategies:
        * 'no_tuning': use y_pred_lgb directly.
        * 'tuned': use y_pred_lgb * (1 + best_alpha) per (fold, ratio).
    """
    op = predictions[predictions["is_operational"] == 1].copy()
    rows = []

    for fold_id, fold_df in op.groupby("fold_id"):
        yt = fold_df["n_shipments"].to_numpy(dtype=float)
        yp_lgb = fold_df["y_pred_lgb"].to_numpy(dtype=float)
        yp_sn = fold_df["y_pred_seasn"].to_numpy(dtype=float)

        for ratio in COST_RATIOS:
            c_under, c_over = ratio, 1.0

            # No tuning
            gain_raw = expected_gain(yt, yp_lgb, yp_sn, c_under, c_over)
            cost_lgb_raw = asymmetric_cost(yt, yp_lgb, c_under, c_over)
            cost_sn = asymmetric_cost(yt, yp_sn, c_under, c_over)

            # With tuning: find best alpha for LightGBM
            best_alpha, cost_lgb_tuned = optimal_threshold_multiplier(
                yt, yp_lgb, c_under, c_over, alpha_grid=ALPHA_GRID
            )
            yp_lgb_tuned = np.maximum(yp_lgb * (1.0 + best_alpha), 0.0)
            gain_tuned = expected_gain(yt, yp_lgb_tuned, yp_sn, c_under, c_over)

            rows.append(
                {
                    "fold_id": int(fold_id),
                    "cost_ratio": ratio,
                    "c_under": c_under,
                    "c_over": c_over,
                    "gain_no_tuning": gain_raw,
                    "gain_tuned": gain_tuned,
                    "best_alpha": best_alpha,
                    "cost_lgb_no_tuning": cost_lgb_raw,
                    "cost_lgb_tuned": cost_lgb_tuned,
                    "cost_seasn": cost_sn,
                    "tuning_uplift": gain_tuned - gain_raw,
                }
            )

    return pd.DataFrame(rows)


def compute_gain_in_events(predictions: pd.DataFrame) -> pd.DataFrame:
    """Same as compute_gain_by_fold but restricted to event windows."""
    op = predictions[predictions["is_operational"] == 1].copy()
    op["shipment_date"] = pd.to_datetime(op["shipment_date"])
    rows = []

    for event_name, (start, end) in EVENT_WINDOWS.items():
        event_df = op[
            (op["shipment_date"] >= pd.Timestamp(start))
            & (op["shipment_date"] <= pd.Timestamp(end))
        ]
        if event_df.empty:
            logger.warning(f"No data for event window {event_name}")
            continue

        yt = event_df["n_shipments"].to_numpy(dtype=float)
        yp_lgb = event_df["y_pred_lgb"].to_numpy(dtype=float)
        yp_sn = event_df["y_pred_seasn"].to_numpy(dtype=float)

        for ratio in COST_RATIOS:
            c_under, c_over = ratio, 1.0

            gain_raw = expected_gain(yt, yp_lgb, yp_sn, c_under, c_over)
            best_alpha, _ = optimal_threshold_multiplier(
                yt, yp_lgb, c_under, c_over, alpha_grid=ALPHA_GRID
            )
            yp_lgb_tuned = np.maximum(yp_lgb * (1.0 + best_alpha), 0.0)
            gain_tuned = expected_gain(yt, yp_lgb_tuned, yp_sn, c_under, c_over)

            rows.append(
                {
                    "event": event_name,
                    "n_days": event_df["shipment_date"].nunique(),
                    "n_rows": len(event_df),
                    "cost_ratio": ratio,
                    "c_under": c_under,
                    "c_over": c_over,
                    "gain_no_tuning": gain_raw,
                    "gain_tuned": gain_tuned,
                    "best_alpha": best_alpha,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write CSV outputs (default: data/processed/phase7)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase logging verbosity (-v: INFO, -vv: DEBUG)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level={0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(
            args.verbose, logging.DEBUG
        ),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info("Loading data and tuned LightGBM params")
    df = load_panel()
    lgb_params = load_tuned_lgb_params()

    logger.info("Generating predictions per fold (this takes ~30s)")
    predictions = generate_predictions(df, lgb_params)
    predictions.to_csv(output_dir / "predictions_per_fold.csv", index=False)
    logger.info(f"Predictions saved: {output_dir / 'predictions_per_fold.csv'}")

    logger.info("Computing gain by fold")
    gain_by_fold = compute_gain_by_fold(predictions)
    gain_by_fold.to_csv(output_dir / "gain_by_fold.csv", index=False)

    logger.info("Computing gain in event windows")
    gain_events = compute_gain_in_events(predictions)
    gain_events.to_csv(output_dir / "gain_in_events.csv", index=False)

    elapsed = time.time() - t0
    print(f"\n✓ Analysis complete in {elapsed:.1f}s")
    print(f"  Predictions: {len(predictions)} rows across {predictions['fold_id'].nunique()} folds")
    print(f"  Output: {output_dir}/")
    print()
    print("Summary — gain_by_fold (LightGBM vs SeasonalNaive, USD):")
    print(
        gain_by_fold.pivot_table(
            index="fold_id",
            columns="cost_ratio",
            values="gain_tuned",
            aggfunc="first",
        )
        .round(0)
        .to_string()
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
