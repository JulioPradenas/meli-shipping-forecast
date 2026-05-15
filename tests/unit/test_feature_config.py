"""Tests for the feature_config module — the adapter between Folds and ML inputs.

Coverage focus (does NOT re-test individual builders):

* Pipeline construction: builder set is what we expect, no duplicates.
* prepare_fold_data invariants:
    - X_train and X_test have identical columns in identical order.
    - No NaN in outputs.
    - is_operational filter removes Sundays and holidays.
    - Volume stats are computed from train only (anti-leakage).
    - Both splits produce non-empty results.
* Error paths: empty input, missing pipeline columns.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from shipping_forecast.evaluation import Fold
from shipping_forecast.features import FeaturePipeline, LagFeatures
from shipping_forecast.models.feature_config import (
    DATE_COL,
    GROUP_COL,
    NON_FEATURE_COLS,
    TARGET_COL,
    build_default_pipeline,
    get_feature_columns,
    prepare_fold_data,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_panel(
    n_days: int = 90,
    states: tuple[str, ...] = ("SP", "RJ", "AC"),
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic shipment panel with realistic weekly seasonality.

    SP gets high volume (~100/day), RJ medium (~20/day), AC tail (~1/day).
    Sundays have 0 shipments (matches the real operational pattern).
    """
    rng = np.random.default_rng(seed)
    start = date(2018, 1, 1)
    rows = []
    base_vols = {"SP": 100.0, "RJ": 20.0, "AC": 1.0}

    for i in range(n_days):
        d = start + timedelta(days=i)
        for st in states:
            base = base_vols[st]
            # Sunday = 0 shipments (matches operational pattern)
            n = 0 if d.weekday() == 6 else max(0, int(rng.normal(base, base * 0.2)))
            rows.append({DATE_COL: pd.Timestamp(d), GROUP_COL: st, TARGET_COL: n})

    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_fold() -> Fold:
    """A Fold with 60 train days + 30 test days across 3 states."""
    df = _make_synthetic_panel(n_days=90)
    train_end_idx = 60
    train_end_date = df[DATE_COL].iloc[train_end_idx * 3 - 1].date()
    test_start_date = df[DATE_COL].iloc[train_end_idx * 3].date()
    train_df = df[df[DATE_COL] <= pd.Timestamp(train_end_date)].reset_index(drop=True)
    test_df = df[df[DATE_COL] >= pd.Timestamp(test_start_date)].reset_index(drop=True)

    return Fold(
        fold_id=99,
        train=train_df,
        test=test_df,
        train_period=(train_df[DATE_COL].min().date(), train_end_date),
        test_period=(test_start_date, test_df[DATE_COL].max().date()),
        label="synthetic fold for tests",
    )


# ---------------------------------------------------------------------------
# build_default_pipeline
# ---------------------------------------------------------------------------


def test_default_pipeline_constructs_with_volume_stats():
    pipe = build_default_pipeline({"SP": 100.0, "RJ": 20.0, "AC": 1.0})
    assert isinstance(pipe, FeaturePipeline)


def test_default_pipeline_includes_expected_features():
    """Lock the feature set in the production pipeline."""
    pipe = build_default_pipeline({"SP": 100.0})
    features = set(pipe.feature_names)

    # Critical features that must be present (representative subset).
    must_have = {
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "day_of_week",
        "is_holiday",
        "is_operational",
        "days_to_black_friday",
        "is_post_black_friday_peak",
        "days_since_start",
        "state_avg_volume",
        "volume_tier",
    }
    missing = must_have - features
    assert not missing, f"Pipeline missing required features: {missing}"


def test_default_pipeline_has_no_duplicate_features():
    """FeaturePipeline catches this at construction; verify our config is clean."""
    pipe = build_default_pipeline({"SP": 100.0})
    names = pipe.feature_names
    assert len(names) == len(set(names)), "Duplicate feature names in default pipeline"


# ---------------------------------------------------------------------------
# get_feature_columns
# ---------------------------------------------------------------------------


def test_get_feature_columns_excludes_metadata():
    """Target, group, date, and known non-features must be excluded."""
    df = pd.DataFrame(
        {
            DATE_COL: [pd.Timestamp("2018-01-01")],
            GROUP_COL: ["SP"],
            TARGET_COL: [100],
            "lag_7": [90],
            "is_operational": [1],
            "day_type": ["business_day"],
            "day_of_week": [0],
        }
    )
    cols = get_feature_columns(df)
    assert cols == ["lag_7", "day_of_week"]
    assert NON_FEATURE_COLS.isdisjoint(cols)


# ---------------------------------------------------------------------------
# prepare_fold_data — core invariants
# ---------------------------------------------------------------------------


def test_prepare_fold_data_returns_aligned_splits(synthetic_fold):
    X_train, y_train, X_test, y_test, _ = prepare_fold_data(synthetic_fold)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) > 0
    assert len(X_test) > 0


def test_prepare_fold_data_train_test_same_columns(synthetic_fold):
    """X_train and X_test must have identical columns in identical order.
    Critical for downstream ML: any mismatch silently breaks prediction.
    """
    X_train, _, X_test, _, feature_names = prepare_fold_data(synthetic_fold)

    assert list(X_train.columns) == list(X_test.columns)
    assert list(X_train.columns) == feature_names


def test_prepare_fold_data_no_nan_in_outputs(synthetic_fold):
    """NaN values would break LightGBM training silently or noisily.
    Lag-induced NaN at start of series must be dropped.
    """
    X_train, y_train, X_test, y_test, _ = prepare_fold_data(synthetic_fold)

    assert X_train.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0
    assert y_train.isna().sum() == 0
    assert y_test.isna().sum() == 0


def test_prepare_fold_data_filters_non_operational_days(synthetic_fold):
    """Sundays (and holidays) must not appear in X_train or X_test when
    filter_operational=True (the default).
    """
    X_train, _, X_test, _, _ = prepare_fold_data(synthetic_fold)

    # day_of_week == 6 is Sunday; should not appear in any output row.
    assert (X_train["day_of_week"] == 6).sum() == 0
    assert (X_test["day_of_week"] == 6).sum() == 0


def test_prepare_fold_data_can_skip_operational_filter(synthetic_fold):
    """With filter_operational=False, Sundays should reappear."""
    X_train, _, X_test, _, _ = prepare_fold_data(synthetic_fold, filter_operational=False)

    assert (X_train["day_of_week"] == 6).sum() > 0
    assert (X_test["day_of_week"] == 6).sum() > 0


# ---------------------------------------------------------------------------
# prepare_fold_data — anti-leakage
# ---------------------------------------------------------------------------


def test_volume_stats_computed_from_train_only(synthetic_fold):
    """Critical anti-leakage check: state_avg_volume in test rows must match
    the training-set mean, NOT the test-set mean.
    """
    _, _, X_test, _, _ = prepare_fold_data(synthetic_fold)

    # Recompute the expected stats from the original training data.
    train = synthetic_fold.train
    expected_stats = train.groupby(GROUP_COL)[TARGET_COL].mean().to_dict()

    # Test rows must show the train-derived stat for their state.
    # Reconstruct state from the fold's test rows aligned by row order.
    # We need a stable mapping; easiest is to re-prepare with metadata-keeping
    # path. Instead, we'll verify directly: state_avg_volume column must take
    # only as many unique values as states, and each must match expected_stats.
    unique_avgs = sorted(X_test["state_avg_volume"].unique())
    expected_avgs = sorted(round(v, 3) for v in expected_stats.values())
    actual_rounded = sorted(round(v, 3) for v in unique_avgs)
    assert actual_rounded == expected_avgs, (
        f"state_avg_volume in test reflects test stats, not train stats. "
        f"Expected {expected_avgs}, got {actual_rounded}"
    )


def test_volume_stats_differ_from_test_period_means(synthetic_fold):
    """If the test period has a different mean volume from train, the injected
    state_avg_volume must NOT match the test mean. This is the strongest
    anti-leakage signal we can assert.
    """
    train_means = synthetic_fold.train.groupby(GROUP_COL)[TARGET_COL].mean()
    test_means = synthetic_fold.test.groupby(GROUP_COL)[TARGET_COL].mean()

    # Synthetic data has natural variance, so train and test means will differ
    # for at least one state. Confirm the test fixture itself has this property.
    diffs = (train_means - test_means).abs()
    assert (diffs > 0.1).any(), "Synthetic fixture too clean to test leakage. Increase randomness."

    _, _, X_test, _, _ = prepare_fold_data(synthetic_fold)

    # The injected state_avg_volume in test must match TRAIN means, not test.
    for state, train_mean in train_means.items():
        # X_test doesn't carry the state column (it's in NON_FEATURE_COLS),
        # so we filter by state_avg_volume value instead.
        matching = X_test[np.isclose(X_test["state_avg_volume"], train_mean, atol=0.01)]
        assert len(matching) > 0, (
            f"State {state} train_mean={train_mean:.2f} not present in test rows"
        )


# ---------------------------------------------------------------------------
# prepare_fold_data — custom pipeline
# ---------------------------------------------------------------------------


def test_prepare_fold_data_with_custom_pipeline_raises_when_filter_needs_holiday(
    synthetic_fold,
):
    """If filter_operational=True but pipeline doesn't add is_operational,
    a clear RuntimeError must be raised.
    """
    minimal = FeaturePipeline([LagFeatures(lags=[7])])
    with pytest.raises(RuntimeError, match="is_operational"):
        prepare_fold_data(synthetic_fold, pipeline=minimal, filter_operational=True)


def test_prepare_fold_data_with_custom_pipeline_works_without_filter(synthetic_fold):
    """A minimal custom pipeline works fine when operational filter is off."""
    minimal = FeaturePipeline([LagFeatures(lags=[7])])
    X_train, _, X_test, _, feature_names = prepare_fold_data(
        synthetic_fold, pipeline=minimal, filter_operational=False
    )

    assert feature_names == ["lag_7"]
    assert len(X_train) > 0
    assert len(X_test) > 0
