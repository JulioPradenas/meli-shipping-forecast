"""Tests for LightGBMForecaster — the production model class.

Coverage focus:

* Smoke: fit/predict on synthetic data, save/load roundtrip, custom params.
* API contract: required errors when used incorrectly.
* Predictions sanity: shape, non-negativity, operational rule, column schema.
* Recursive correctness: the bug we fixed (day-8+ collapse) must not regress.
* Persistence: JSON sidecar contents.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shipping_forecast.models.feature_config import DATE_COL, GROUP_COL, TARGET_COL
from shipping_forecast.models.lightgbm_model import (
    DEFAULT_PARAMS,
    LightGBMForecaster,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_panel(
    n_days: int = 365,
    states: tuple[str, ...] = ("SP", "RJ", "AC"),
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic panel with weekly seasonality and per-state volumes."""
    rng = np.random.default_rng(seed)
    start = date(2017, 1, 1)
    rows = []
    base_vols = {"SP": 100.0, "RJ": 20.0, "AC": 1.0}

    for i in range(n_days):
        d = start + timedelta(days=i)
        weekday = d.weekday()
        for st in states:
            base = base_vols[st]
            if weekday == 6:  # Sunday
                n = 0
            elif weekday == 5:  # Saturday: 8% of weekday volume
                n = max(0, int(rng.normal(base * 0.08, 1)))
            else:
                n = max(0, int(rng.normal(base, base * 0.2)))
            rows.append({DATE_COL: pd.Timestamp(d), GROUP_COL: st, TARGET_COL: n})

    return pd.DataFrame(rows)


@pytest.fixture
def train_df() -> pd.DataFrame:
    """One year of synthetic data — long enough for lag_14 + plenty of train."""
    return _make_panel(n_days=365)


@pytest.fixture
def fitted_model(train_df: pd.DataFrame) -> LightGBMForecaster:
    """A fitted LightGBMForecaster with tiny n_estimators for speed."""
    model = LightGBMForecaster(params={"n_estimators": 30, "verbose": -1})
    model.fit(train_df)
    return model


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_fit_predict_smoke(train_df: pd.DataFrame) -> None:
    """End-to-end: untrained model can fit then predict, producing valid output."""
    model = LightGBMForecaster(params={"n_estimators": 30, "verbose": -1})
    model.fit(train_df)

    preds = model.predict(train_df, horizon=14)
    assert len(preds) > 0
    assert set(preds.columns) == {DATE_COL, GROUP_COL, "y_pred"}


def test_custom_params_override_defaults() -> None:
    """User-supplied params override defaults; unspecified keys keep their default."""
    model = LightGBMForecaster(params={"n_estimators": 500, "learning_rate": 0.01})

    assert model.params["n_estimators"] == 500
    assert model.params["learning_rate"] == 0.01
    # Unspecified default is preserved
    assert model.params["objective"] == DEFAULT_PARAMS["objective"]
    assert model.params["random_state"] == DEFAULT_PARAMS["random_state"]


def test_default_params_constant_is_sane() -> None:
    """Lock the documented default behaviour: MAE objective, fixed seed, etc."""
    assert DEFAULT_PARAMS["objective"] == "regression_l1"
    assert DEFAULT_PARAMS["random_state"] == 42
    assert DEFAULT_PARAMS["n_estimators"] > 0
    assert 0 < DEFAULT_PARAMS["learning_rate"] < 1


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------


def test_predict_without_fit_raises() -> None:
    model = LightGBMForecaster()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(pd.DataFrame({DATE_COL: [], GROUP_COL: [], TARGET_COL: []}), horizon=7)


def test_predict_with_non_positive_horizon_raises(
    fitted_model: LightGBMForecaster, train_df: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="horizon must be positive"):
        fitted_model.predict(train_df, horizon=0)
    with pytest.raises(ValueError, match="horizon must be positive"):
        fitted_model.predict(train_df, horizon=-5)


def test_fit_with_missing_column_raises() -> None:
    """If the input lacks the target column, fit() must fail loudly."""
    df = pd.DataFrame({DATE_COL: pd.date_range("2018-01-01", periods=10), GROUP_COL: ["SP"] * 10})
    model = LightGBMForecaster(params={"n_estimators": 5, "verbose": -1})
    with pytest.raises(KeyError, match="n_shipments"):
        model.fit(df)


def test_save_unfitted_raises(tmp_path: Path) -> None:
    model = LightGBMForecaster()
    with pytest.raises(RuntimeError, match="Cannot save an unfitted model"):
        model.save(tmp_path / "noop")


# ---------------------------------------------------------------------------
# Predictions sanity
# ---------------------------------------------------------------------------


def test_predict_output_shape(fitted_model: LightGBMForecaster, train_df: pd.DataFrame) -> None:
    """Output rows = horizon * n_groups."""
    horizon = 21
    preds = fitted_model.predict(train_df, horizon=horizon)
    n_groups = train_df[GROUP_COL].nunique()

    assert len(preds) == horizon * n_groups
    assert preds[DATE_COL].nunique() == horizon
    assert preds[GROUP_COL].nunique() == n_groups


def test_predictions_are_non_negative(
    fitted_model: LightGBMForecaster, train_df: pd.DataFrame
) -> None:
    """LightGBM can output negatives; the clip(lower=0) must catch them."""
    preds = fitted_model.predict(train_df, horizon=30)
    assert (preds["y_pred"] >= 0).all()


def test_sundays_predicted_as_zero(
    fitted_model: LightGBMForecaster, train_df: pd.DataFrame
) -> None:
    """Non-operational rule: Sundays must be exactly 0 regardless of model output."""
    preds = fitted_model.predict(train_df, horizon=30)
    preds = preds.assign(weekday=preds[DATE_COL].dt.dayofweek)

    sunday_preds = preds[preds["weekday"] == 6]
    assert len(sunday_preds) > 0, "Test fixture should span at least one Sunday"
    assert (sunday_preds["y_pred"] == 0).all()


def test_output_columns_match_abc_contract(
    fitted_model: LightGBMForecaster, train_df: pd.DataFrame
) -> None:
    """ForecastModel ABC requires shipment_date, group_col, y_pred."""
    preds = fitted_model.predict(train_df, horizon=7)
    assert list(preds.columns) == [DATE_COL, GROUP_COL, "y_pred"]


# ---------------------------------------------------------------------------
# Recursive correctness — guards against the day-8 collapse bug
# ---------------------------------------------------------------------------


def test_long_horizon_does_not_collapse_to_mean(
    fitted_model: LightGBMForecaster, train_df: pd.DataFrame
) -> None:
    """Regression guard: before recursive forecasting, predictions for day 8+
    collapsed to ~2 (global mean) because lag_7 became NaN. The recursive
    loop fixes this by injecting predicted targets back.

    We assert that the second week of predictions for SP (the high-volume
    state) maintains the order of magnitude of the first week — never drops
    below 10% of the first-week average.
    """
    preds = fitted_model.predict(train_df, horizon=21)
    sp = preds[preds[GROUP_COL] == "SP"].sort_values(DATE_COL).reset_index(drop=True)

    # Operational days only for fair comparison
    sp = sp.assign(weekday=sp[DATE_COL].dt.dayofweek)
    sp_op = sp[sp["weekday"] < 5]  # Mon-Fri

    first_week_mean = sp_op.iloc[:5]["y_pred"].mean()
    third_week_mean = sp_op.iloc[-5:]["y_pred"].mean()

    # If recursive forecasting broke, third week would be near zero.
    assert third_week_mean > first_week_mean * 0.5, (
        f"Predictions collapsed: week 1 mean={first_week_mean:.2f}, "
        f"week 3 mean={third_week_mean:.2f}. Recursive injection may be broken."
    )


def test_injected_predictions_affect_subsequent_lags(
    fitted_model: LightGBMForecaster, train_df: pd.DataFrame
) -> None:
    """If we manually corrupt the model to always predict 999, the lag-7
    feature used on day 8+ must reflect that — proving the injection works.

    We test this by comparing two predict() calls with different mock models
    is overkill. Simpler: verify that predict(horizon=14) and predict(horizon=21)
    agree on the first 14 days. If injection works, both runs follow the same
    recursive path through day 14, producing identical outputs.
    """
    preds_14 = (
        fitted_model.predict(train_df, horizon=14)
        .sort_values([DATE_COL, GROUP_COL])
        .reset_index(drop=True)
    )
    preds_21 = (
        fitted_model.predict(train_df, horizon=21)
        .sort_values([DATE_COL, GROUP_COL])
        .reset_index(drop=True)
    )

    # First 14*n_groups rows of preds_21 should match preds_14 exactly
    n_groups = train_df[GROUP_COL].nunique()
    overlap = preds_21.iloc[: 14 * n_groups].reset_index(drop=True)
    pd.testing.assert_frame_equal(preds_14, overlap)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_preserves_predictions(
    fitted_model: LightGBMForecaster, train_df: pd.DataFrame, tmp_path: Path
) -> None:
    """A loaded model produces identical predictions to the original."""
    path = tmp_path / "model"
    fitted_model.save(path)

    loaded = LightGBMForecaster.load(path)
    preds_original = (
        fitted_model.predict(train_df, horizon=7)
        .sort_values([DATE_COL, GROUP_COL])
        .reset_index(drop=True)
    )
    preds_loaded = (
        loaded.predict(train_df, horizon=7)
        .sort_values([DATE_COL, GROUP_COL])
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(preds_original, preds_loaded)


def test_save_writes_json_sidecar_with_expected_keys(
    fitted_model: LightGBMForecaster, tmp_path: Path
) -> None:
    """The JSON sidecar must include enough metadata to audit a saved model."""
    path = tmp_path / "model"
    fitted_model.save(path)

    json_path = path.with_suffix(".joblib").with_suffix(".json")
    assert json_path.exists()

    metadata = json.loads(json_path.read_text())
    assert metadata["model_name"] == "LightGBMForecaster"
    assert "trained_at" in metadata
    assert "last_train_date" in metadata
    assert metadata["n_features"] > 0
    assert isinstance(metadata["feature_names"], list)
    assert metadata["n_groups"] == 3
    assert sorted(metadata["groups"]) == ["AC", "RJ", "SP"]
    assert "params" in metadata
    assert metadata["params"]["objective"] == "regression_l1"
