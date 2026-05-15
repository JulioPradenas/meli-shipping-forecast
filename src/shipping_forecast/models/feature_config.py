"""Canonical feature pipeline configuration for LightGBM forecasting.

This module is the single source of truth for **which features the
production LightGBM model uses** and **how raw data is transformed into
modellable rows**. Versioning this configuration in git means changes
to the feature set are auditable and reproducible.

Design decisions (documented in handoff and MODELS_BASELINE_SUMMARY.md):

* **Feature set**: lag-7/14, rolling-7 mean/std, calendar, holidays,
  events (BF + DdN), trend, and volume tier. Total of ~22 features.
* **Volume tier as feature**: rather than training separate models per
  tier, a single LightGBM ingests ``volume_tier`` as a categorical
  feature. Justification: see Phase 5 baseline summary, decision 4.
* **Leakage prevention strategy**: features are computed on
  ``concat(train, test)`` but the per-state volume statistics are
  computed on ``train`` only. This implements the "optimistic"
  evaluation strategy (decision B in Phase 6 design):  the test set
  uses real (not predicted) lag values from previous test days, which
  simulates a system that retrains daily with yesterday's actuals.
* **Operational filter**: rows with ``is_operational=0`` (Sundays,
  holidays) are dropped from both train AND test before modelling.
  WAPE is reported only on days where prediction is non-trivial.

If any of these decisions need to change, this module is the place.
"""

from __future__ import annotations

import pandas as pd

from shipping_forecast.evaluation import Fold
from shipping_forecast.features import (
    CalendarFeatures,
    EventFeatures,
    FeaturePipeline,
    HolidayFeatures,
    LagFeatures,
    RollingFeatures,
    TrendFeatures,
    VolumeFeatures,
)

# Column conventions. Centralised to avoid magic strings across the model layer.
TARGET_COL = "n_shipments"
GROUP_COL = "customer_state"
DATE_COL = "shipment_date"

# Columns added by the pipeline that should NEVER be treated as model features
# (they're either the target, the group identifier, the date, or derived flags
# used for filtering / not for training).
NON_FEATURE_COLS: frozenset[str] = frozenset(
    {
        TARGET_COL,
        GROUP_COL,
        DATE_COL,
        # HolidayFeatures adds these; we keep them for filtering/inspection
        # but they're not LightGBM features in the production model.
        "is_operational",
        "day_type",
    }
)


def build_default_pipeline(state_avg_volume: dict[str, float]) -> FeaturePipeline:
    """Build the canonical FeaturePipeline used by LightGBMForecaster.

    Args:
        state_avg_volume: Mapping ``state -> mean daily shipments`` computed
            **on the training set only**. The caller is responsible for
            ensuring no leakage from test data.

    Returns:
        A configured :class:`FeaturePipeline` ready to apply.

    Raises:
        ValueError: If ``state_avg_volume`` is empty (propagated from
            ``VolumeFeatures`` validation).
    """
    return FeaturePipeline(
        [
            LagFeatures(lags=[7, 14]),
            RollingFeatures(windows=[7], stats=["mean", "std"]),
            CalendarFeatures(
                features=[
                    "day_of_week",
                    "month",
                    "is_weekend",
                    "is_month_start",
                    "is_month_end",
                ]
            ),
            HolidayFeatures(),
            EventFeatures(),
            TrendFeatures(features=["days_since_start", "year_progress"]),
            VolumeFeatures(state_avg_volume=state_avg_volume),
        ]
    )


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that should be passed to LightGBM as features.

    Excludes the target, group identifier, date, and any derived flag
    columns listed in :data:`NON_FEATURE_COLS`.

    Args:
        df: DataFrame after the pipeline has been applied.

    Returns:
        Ordered list of feature column names (preserves DataFrame order).
    """
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def prepare_fold_data(
    fold: Fold,
    pipeline: FeaturePipeline | None = None,
    filter_operational: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """Adapt a Fold into modellable train/test sets with features applied.

    Steps:
        1. Compute ``state_avg_volume`` from the training set (no leakage).
        2. If no pipeline is provided, build the default one with those stats.
        3. Apply the pipeline to ``concat(train, test)``.
        4. Optionally filter ``is_operational=False`` rows from both splits.
        5. Drop rows with NaN in lag/rolling columns (first ~14 days of train).
        6. Separate ``X`` / ``y`` for each split, using
           :func:`get_feature_columns` to choose feature columns.

    Args:
        fold: A :class:`~shipping_forecast.evaluation.Fold` from the CV split.
            Must contain ``train`` and ``test`` DataFrames with at least
            :data:`DATE_COL`, :data:`GROUP_COL`, and :data:`TARGET_COL`.
        pipeline: Optional pre-built pipeline. If ``None`` (default), the
            canonical pipeline is built using stats from ``fold.train``.
            Passing a custom pipeline is mainly useful for tests or
            sensitivity analysis.
        filter_operational: If ``True`` (default), drops rows where
            ``is_operational == 0`` from both train and test. Phase 6
            decision A: report WAPE only on days where prediction is
            non-trivial.

    Returns:
        ``(X_train, y_train, X_test, y_test, feature_names)`` where the
        first four are aligned to their respective rows and ``feature_names``
        is the ordered list of feature columns (identical in X_train and X_test).

    Raises:
        ValueError: If train or test is empty after filtering / NaN drop.
    """
    train, test = fold.train, fold.test

    # 1. Compute volume stats from train only (anti-leakage).
    state_stats = VolumeFeatures.compute_stats_from_train(
        train, group_col=GROUP_COL, target_col=TARGET_COL
    )

    # 2. Build default pipeline if none given.
    if pipeline is None:
        pipeline = build_default_pipeline(state_stats)

    # 3. Apply pipeline to concat(train, test).
    # We need a flag column to split them back out cleanly afterwards.
    train_marked = train.assign(_split="train")
    test_marked = test.assign(_split="test")
    combined = pd.concat([train_marked, test_marked], ignore_index=True)
    combined = combined.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)

    transformed = pipeline.transform(combined)

    # 4. Filter is_operational if requested.
    if filter_operational:
        if "is_operational" not in transformed.columns:
            raise RuntimeError(
                "filter_operational=True but pipeline did not add 'is_operational'. "
                "Ensure HolidayFeatures is in the pipeline."
            )
        transformed = transformed[transformed["is_operational"] == 1]

    # 5. Split back into train/test and drop the internal marker column.
    train_out = transformed[transformed["_split"] == "train"].drop(columns="_split")
    test_out = transformed[transformed["_split"] == "test"].drop(columns="_split")

    # 6. Compute feature columns AFTER dropping _split, then drop NaN rows
    # (start-of-series in train have NaN lag_14).
    feature_cols = get_feature_columns(train_out)
    train_out = train_out.dropna(subset=feature_cols)
    test_out = test_out.dropna(subset=feature_cols)

    if train_out.empty:
        raise ValueError(
            "Training set is empty after feature application + filtering. "
            "Check that the fold has enough history for the largest lag."
        )
    if test_out.empty:
        raise ValueError(
            "Test set is empty after feature application + filtering. "
            "Check that the fold's test period contains operational days."
        )

    X_train = train_out[feature_cols].reset_index(drop=True)
    y_train = train_out[TARGET_COL].reset_index(drop=True)
    X_test = test_out[feature_cols].reset_index(drop=True)
    y_test = test_out[TARGET_COL].reset_index(drop=True)

    return X_train, y_train, X_test, y_test, feature_cols
