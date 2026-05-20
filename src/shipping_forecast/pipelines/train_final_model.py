"""Train the final production LightGBM model with a validation holdout.

This pipeline produces two models in a single run:

1. **Evaluation model**: trained on data up to ``TRAIN_END_EVAL``
   (matching Phase 6's tuning folds 1+2+3). Its predictions on the
   reserved holdout window (Phase 6's fold 4) yield honest out-of-sample
   metrics — these are reported in ``/model/info`` of the API service.

2. **Production model**: trained on the full available data range.
   This is the model that is persisted to ``artifacts/lightgbm_final.joblib``
   and loaded at API service startup.

The asymmetry is intentional: the production model uses every available
data point for better forecasts, while the evaluation model exists solely
to produce metrics that were not contaminated by Optuna's tuning. The
metrics from the evaluation model are embedded as ``extra_metadata`` in
the production model's JSON sidecar, with an explicit ``evaluation_note``
documenting the lineage.

Usage::

    # Full flow: train both, persist production with eval metrics
    python -m shipping_forecast.pipelines.train_final_model

    # Skip evaluation model (faster, no holdout metrics produced)
    python -m shipping_forecast.pipelines.train_final_model --mode production

    # Only validate metrics, do not persist
    python -m shipping_forecast.pipelines.train_final_model --mode evaluation

    # Fast mode for CI (smaller dataset, reduced params)
    python -m shipping_forecast.pipelines.train_final_model --fast-retrain
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shipping_forecast.data.queries import load_panel
from shipping_forecast.evaluation import wape
from shipping_forecast.models import ConformalForecaster, LightGBMForecaster

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "shipping.db"
BEST_PARAMS_PATH = PROJECT_ROOT / "data" / "processed" / "best_lgbm_params.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts"

# Holdout window matches Phase 6's fold 4 — the only fold Optuna NEVER saw.
# Using these exact dates guarantees that the WAPE we report on the holdout
# is an honest out-of-sample metric, not a contaminated one.
TRAIN_END_EVAL = pd.Timestamp("2018-06-30")
HOLDOUT_START = pd.Timestamp("2018-07-01")
HOLDOUT_END = pd.Timestamp("2018-08-31")
HOLDOUT_HORIZON_DAYS = (HOLDOUT_END - HOLDOUT_START).days + 1  # 62 days

# Data cutoff: the raw dataset has 11 days (2018-09-01 to 2018-09-11) of
# essentially empty rows (3 total shipments across 297 state-days). These
# are artifacts of the data collection cutoff, not real demand. Including
# them would teach the model a spurious collapse-to-zero pattern at the
# end of the horizon. We filter them out here.
DATA_CUTOFF = pd.Timestamp("2018-08-31")

# Conformal prediction settings: alpha=0.1 yields 90% nominal coverage,
# matching the lower_90/upper_90 fields in the response schema.
# calibration_days=60 is the empirical sweet spot documented in
# ConformalForecaster's docstring: it gave 89.2% empirical coverage at
# alpha=0.1 on the MELI holdout, vs 81.6% at calib=30 and 86.7% at calib=90.
CONFORMAL_ALPHA = 0.1
CONFORMAL_CALIBRATION_DAYS = 60

# Fast retrain params for CI: a tiny LightGBM that trains in <30s. Tests
# of the API service validate response shape, not model accuracy, so a
# weak model is acceptable here.
FAST_RETRAIN_PARAMS: dict[str, Any] = {
    "num_leaves": 15,
    "learning_rate": 0.1,
    "n_estimators": 50,
    "min_child_samples": 20,
}
FAST_RETRAIN_HISTORY_DAYS = 180

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_best_params(json_path: Path) -> dict[str, Any]:
    """Load hyperparameters tuned by Optuna in Phase 6.3.

    Args:
        json_path: Path to ``best_lgbm_params.json`` produced by
            :mod:`scripts.tune_lightgbm`.

    Returns:
        The ``best_params`` sub-dict, ready to pass to
        :class:`LightGBMForecaster` constructor.

    Raises:
        FileNotFoundError: If ``json_path`` does not exist.
        KeyError: If the JSON does not contain a ``best_params`` key.
    """
    if not json_path.exists():
        raise FileNotFoundError(
            f"Best params JSON not found at {json_path}. Run scripts/tune_lightgbm.py first."
        )
    with open(json_path) as f:
        data = json.load(f)
    if "best_params" not in data:
        raise KeyError(
            f"Expected 'best_params' key in {json_path}, got keys: {sorted(data.keys())}"
        )
    best_params: dict[str, Any] = data["best_params"]
    return best_params


def split_train_holdout(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the panel into training and holdout sets by date.

    Args:
        df: The full panel DataFrame, must contain a ``shipment_date`` column.
        train_end: Last date included in the training set (inclusive).
        holdout_start: First date of the holdout window (inclusive).
        holdout_end: Last date of the holdout window (inclusive).

    Returns:
        Tuple ``(df_train, df_holdout)``:
        - ``df_train``: rows with ``shipment_date <= train_end``.
        - ``df_holdout``: rows with ``holdout_start <= shipment_date <= holdout_end``.

        Both DataFrames preserve the original sort order.

    Raises:
        ValueError: If either resulting DataFrame is empty, or if there is
            a gap between ``train_end`` and ``holdout_start`` (off-by-one bugs).
    """
    if (holdout_start - train_end).days != 1:
        raise ValueError(
            f"Expected holdout_start to be exactly 1 day after train_end. "
            f"Got train_end={train_end.date()}, holdout_start={holdout_start.date()}."
        )

    df_train = df[df["shipment_date"] <= train_end].copy()
    df_holdout = df[
        (df["shipment_date"] >= holdout_start) & (df["shipment_date"] <= holdout_end)
    ].copy()

    if df_train.empty:
        raise ValueError(f"Train set is empty after splitting at {train_end.date()}.")
    if df_holdout.empty:
        raise ValueError(
            f"Holdout set is empty for window [{holdout_start.date()}, {holdout_end.date()}]."
        )

    return df_train, df_holdout


def load_panel_with_cutoff(db_path: Path, cutoff: pd.Timestamp = DATA_CUTOFF) -> pd.DataFrame:
    """Load the panel and filter to the trustworthy date range.

    Wraps :func:`shipping_forecast.data.queries.load_panel` with a date
    cutoff to exclude post-cutoff rows known to be data-collection artifacts.

    Args:
        db_path: Path to the SQLite database.
        cutoff: Inclusive upper bound on ``shipment_date``. Defaults to
            ``DATA_CUTOFF`` (2018-08-31).

    Returns:
        DataFrame filtered to ``shipment_date <= cutoff``.
    """
    df = load_panel(db_path)
    return df[df["shipment_date"] <= cutoff].copy()


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute holdout metrics: WAPE, MAE, RMSE.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values. Must be aligned with ``y_true``.

    Returns:
        Dict with keys ``wape``, ``mae``, ``rmse``, all floats rounded
        to 4 decimal places for readable JSON output.

    Raises:
        ValueError: If the inputs have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} rows, y_pred has {len(y_pred)}."
        )

    error = y_true.to_numpy() - y_pred.to_numpy()
    mae = float(np.abs(error).mean())
    rmse = float(np.sqrt(np.square(error).mean()))
    wape_value = float(wape(y_true.to_numpy(), y_pred.to_numpy()))

    return {
        "wape": round(wape_value, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
    }


def train_evaluation_model(
    df_train: pd.DataFrame,
    df_holdout: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, float]:
    """Train a LightGBM model on the pre-holdout window and score it.

    This model exists ONLY to produce honest out-of-sample metrics. It is
    not persisted to disk. The metrics it produces are embedded in the
    production model's metadata as a separate field, with documentation
    explaining the lineage.

    Note on rigor: unlike ``scripts/tune_lightgbm.py::evaluate_fold``,
    this function does NOT pass ``eval_set`` to ``model.fit()``. Phase 6's
    reported ``holdout_wape=0.4694`` used early stopping on the holdout
    itself, which leaks the iteration count (not features) from the
    holdout into the model. The number this function produces (~0.5156)
    is the strict out-of-sample metric: the holdout is never touched
    during training. This is the metric the API reports in /model/info.

    Args:
        df_train: Panel with ``shipment_date <= TRAIN_END_EVAL``.
        df_holdout: Panel with ``HOLDOUT_START <= shipment_date <= HOLDOUT_END``.
            Used only for scoring; the model never sees these rows.
        params: LightGBM hyperparameters from Phase 6.3's Optuna run.

    Returns:
        Metrics dict from :func:`compute_metrics`, computed on the
        holdout window.
    """
    logger.info(
        "Training evaluation model on %d rows (up to %s)",
        len(df_train),
        df_train["shipment_date"].max().date(),
    )
    model = LightGBMForecaster(params=params)
    model.fit(df_train)

    logger.info(
        "Predicting %d days ahead for evaluation against holdout",
        HOLDOUT_HORIZON_DAYS,
    )
    predictions = model.predict(df_train, horizon=HOLDOUT_HORIZON_DAYS)

    # Align predictions with holdout by (date, state). predict() returns
    # one row per (date, state) for the future horizon; the holdout has
    # the same shape with true values.
    merged = df_holdout.merge(
        predictions,
        on=["shipment_date", "customer_state"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(df_holdout):
        raise RuntimeError(
            f"Prediction/holdout alignment failed: holdout has {len(df_holdout)} rows, "
            f"merged has {len(merged)} rows. Check date ranges and state coverage."
        )

    metrics = compute_metrics(merged["n_shipments"], merged["y_pred"])
    logger.info(
        "Evaluation metrics on holdout (%s to %s): WAPE=%.4f, MAE=%.2f, RMSE=%.2f",
        df_holdout["shipment_date"].min().date(),
        df_holdout["shipment_date"].max().date(),
        metrics["wape"],
        metrics["mae"],
        metrics["rmse"],
    )
    return metrics


def train_production_model(
    df_full: pd.DataFrame,
    params: dict[str, Any],
) -> ConformalForecaster:
    """Train the production model: LightGBM wrapped with conformal calibration.

    The production model is a ``ConformalForecaster`` that wraps a
    ``LightGBMForecaster`` to provide both point predictions and
    distribution-free 90% prediction intervals. The wrapper re-fits the
    base LightGBM on all but the last ``CONFORMAL_CALIBRATION_DAYS`` days
    of the input, then calibrates the interval margins from the empirical
    residuals on the held-out calibration window.

    This is the model that gets persisted to
    ``artifacts/lightgbm_final.joblib`` and served by the API.

    Args:
        df_full: The full panel after applying ``DATA_CUTOFF``.
        params: LightGBM hyperparameters from Phase 6.3's Optuna run.

    Returns:
        A fitted ``ConformalForecaster`` ready to ``.save()``. The point
        forecasts (``y_pred``) come from the base LightGBM and are identical
        to what an unwrapped model would produce; the addition is the
        ``y_lower`` and ``y_upper`` columns representing the 90% interval.
    """
    logger.info(
        "Training production model on %d rows (up to %s)",
        len(df_full),
        df_full["shipment_date"].max().date(),
    )
    base = LightGBMForecaster(params=params)
    wrapper = ConformalForecaster(
        base_model=base,
        alpha=CONFORMAL_ALPHA,
        calibration_days=CONFORMAL_CALIBRATION_DAYS,
    )
    wrapper.fit(df_full)
    logger.info(
        "Production model fitted (base=%s, alpha=%.2f, calibration_days=%d, "
        "empirical_coverage=%.3f).",
        type(wrapper.base_model).__name__,
        wrapper.alpha,
        wrapper.calibration_days,
        wrapper.empirical_coverage_,
    )
    return wrapper


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Orchestrate the training pipeline.

    Args:
        argv: Optional argument list (for testing). If None, reads sys.argv.

    Returns:
        Exit code: 0 on success.
    """
    parser = argparse.ArgumentParser(
        description="Train the final LightGBM model with honest holdout evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["evaluation", "production", "both"],
        default="both",
        help=(
            "evaluation: train eval model and report metrics (no persistence). "
            "production: train and persist production model (no metrics). "
            "both (default): full flow, persists production with eval metrics embedded."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for persisted artifacts. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--fast-retrain",
        action="store_true",
        help=(
            "Fast mode for CI: use a smaller dataset (last 180 days) and a "
            "lightweight LightGBM. The resulting model is intentionally weak; "
            "use only when testing API code, not when serving real predictions."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Step 1: load data and params -----------------------------------
    df = load_panel_with_cutoff(DB_PATH)

    if args.fast_retrain:
        cutoff_date = df["shipment_date"].max() - pd.Timedelta(days=FAST_RETRAIN_HISTORY_DAYS)
        df = df[df["shipment_date"] >= cutoff_date].copy()
        params = FAST_RETRAIN_PARAMS
        logger.info(
            "Fast retrain mode: trimmed to %d rows (last %d days), using simplified params.",
            len(df),
            FAST_RETRAIN_HISTORY_DAYS,
        )
    else:
        params = load_best_params(BEST_PARAMS_PATH)

    # --- Step 2: evaluation model ---------------------------------------
    eval_metrics: dict[str, float] | None = None
    if args.mode in ("evaluation", "both"):
        df_train, df_holdout = split_train_holdout(df, TRAIN_END_EVAL, HOLDOUT_START, HOLDOUT_END)
        eval_metrics = train_evaluation_model(df_train, df_holdout, params)
        logger.info("Evaluation complete. Metrics: %s", eval_metrics)

    # If only evaluation requested, we are done.
    if args.mode == "evaluation":
        return 0

    # --- Step 3: production model ---------------------------------------
    prod_model = train_production_model(df, params)

    # --- Step 4: persist with embedded eval metrics ---------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "lightgbm_final"

    from shipping_forecast.models import LightGBMForecaster

    base = prod_model.base_model
    assert isinstance(base, LightGBMForecaster), (
        f"Expected LightGBMForecaster, got {type(base).__name__}"
    )
    extra_metadata: dict[str, Any] = {
        "phase": "8.1",
        "version": "lgbm-v1.1.0",
        "data_cutoff": DATA_CUTOFF.date().isoformat(),
        "fast_retrain": args.fast_retrain,
        # Fields propagated from the base LightGBMForecaster. These are
        # needed by the /v1/predict endpoint (states validation, last_train_date
        # cutoff) and /v1/model/info (Phase 8.4). Without them the API has to
        # touch model internals at request time, which we explicitly want to avoid.
        "last_train_date": df["shipment_date"].max().date().isoformat(),
        "n_features": len(base.feature_names_),
        "feature_names": list(base.feature_names_),
        "n_groups": len(base.state_avg_volume_),
        "groups": sorted(base.state_avg_volume_.keys()),
    }
    if eval_metrics is not None:
        extra_metadata["evaluation_metrics"] = {
            "window_start": HOLDOUT_START.date().isoformat(),
            "window_end": HOLDOUT_END.date().isoformat(),
            "n_days": HOLDOUT_HORIZON_DAYS,
            **eval_metrics,
        }
        extra_metadata["evaluation_note"] = (
            "Metrics computed by a sibling model trained only up to "
            f"{TRAIN_END_EVAL.date().isoformat()} (Phase 6 fold 3 boundary). "
            "This production model was trained on the full dataset including "
            "the holdout window. The evaluation metrics reflect the model's "
            "expected out-of-sample performance with strict no-eval-set-leakage "
            "methodology (cf. Phase 6's holdout_wape=0.4694, which used "
            "eval_set leakage for early stopping)."
        )

    prod_model.save(output_path, extra_metadata=extra_metadata)
    logger.info(
        "Production model persisted to %s.joblib (and %s.json sidecar).",
        output_path,
        output_path,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
