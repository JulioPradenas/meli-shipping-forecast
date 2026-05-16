"""Tests for ConformalForecaster — split conformal prediction wrapper.

Coverage:

* Smoke: fit/predict with a cheap base model, output schema, save/load.
* API contract: invalid alpha / calibration_days, predict without fit,
  missing required columns.
* Conformal correctness: empirical coverage on calibration is 1 - alpha
  by construction, intervals are asymmetric, y_lower is clipped to 0.
* Anti-leakage: the base model is fitted on the non-calibration portion
  only; the calibration split is based on date, not row order.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shipping_forecast.models import ConformalForecaster, SeasonalNaiveForecaster
from shipping_forecast.models.feature_config import DATE_COL, GROUP_COL, TARGET_COL

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_panel(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic panel with weekly seasonality and 3 states of differing volume."""
    rng = np.random.default_rng(seed)
    start = date(2018, 1, 1)
    states = ["SP", "RJ", "AC"]
    base = {"SP": 100.0, "RJ": 20.0, "AC": 1.0}
    rows = []
    for i in range(n_days):
        d = start + timedelta(days=i)
        wd = d.weekday()
        for st in states:
            b = base[st]
            if wd == 6:
                n = 0
            elif wd == 5:
                n = max(0, int(rng.normal(b * 0.08, 1)))
            else:
                n = max(0, int(rng.normal(b, b * 0.2)))
            rows.append({DATE_COL: pd.Timestamp(d), GROUP_COL: st, TARGET_COL: n})
    return pd.DataFrame(rows)


@pytest.fixture
def train_df() -> pd.DataFrame:
    """200 days, large enough for a 30-day calib split and meaningful fit."""
    return _make_panel(n_days=200)


@pytest.fixture
def fitted_conformal(train_df: pd.DataFrame) -> ConformalForecaster:
    """A ConformalForecaster wrapping SeasonalNaive (fast, deterministic)."""
    base = SeasonalNaiveForecaster(season=7)
    conf = ConformalForecaster(base_model=base, alpha=0.1, calibration_days=30)
    conf.fit(train_df)
    return conf


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_fit_predict_smoke(train_df: pd.DataFrame) -> None:
    """End-to-end: wrapper fits and predicts without errors."""
    conf = ConformalForecaster(
        base_model=SeasonalNaiveForecaster(season=7),
        alpha=0.1,
        calibration_days=30,
    )
    conf.fit(train_df)
    preds = conf.predict(train_df, horizon=14)
    assert len(preds) > 0


def test_predict_output_columns_include_intervals(fitted_conformal, train_df) -> None:
    """Output must have point + interval columns per the ABC contract."""
    preds = fitted_conformal.predict(train_df, horizon=7)
    assert set(preds.columns) >= {DATE_COL, GROUP_COL, "y_pred", "y_lower", "y_upper"}


def test_save_load_roundtrip_preserves_predictions(
    fitted_conformal: ConformalForecaster, train_df: pd.DataFrame, tmp_path: Path
) -> None:
    """A reloaded wrapper produces identical predictions."""
    path = tmp_path / "conformal"
    fitted_conformal.save(path)

    loaded = ConformalForecaster.load(path)
    a = (
        fitted_conformal.predict(train_df, horizon=7)
        .sort_values([DATE_COL, GROUP_COL])
        .reset_index(drop=True)
    )
    b = (
        loaded.predict(train_df, horizon=7)
        .sort_values([DATE_COL, GROUP_COL])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(a, b)


def test_save_json_sidecar_includes_calibration_metadata(
    fitted_conformal: ConformalForecaster, tmp_path: Path
) -> None:
    """The JSON sidecar must allow auditing the calibration."""
    path = tmp_path / "conformal"
    fitted_conformal.save(path)

    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["model_name"] == "ConformalForecaster"
    assert metadata["alpha"] == 0.1
    assert metadata["nominal_coverage"] == 0.9
    assert metadata["calibration_days"] == 30
    assert "lower_offset" in metadata
    assert "upper_offset" in metadata
    assert metadata["interval_width_at_pred_zero"] >= 0
    assert metadata["base_model_class"] == "SeasonalNaiveForecaster"


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------


def test_invalid_alpha_raises() -> None:
    base = SeasonalNaiveForecaster(season=7)
    with pytest.raises(ValueError, match="alpha must be in"):
        ConformalForecaster(base_model=base, alpha=0)
    with pytest.raises(ValueError, match="alpha must be in"):
        ConformalForecaster(base_model=base, alpha=1)
    with pytest.raises(ValueError, match="alpha must be in"):
        ConformalForecaster(base_model=base, alpha=-0.1)


def test_invalid_calibration_days_raises() -> None:
    base = SeasonalNaiveForecaster(season=7)
    with pytest.raises(ValueError, match="calibration_days must be positive"):
        ConformalForecaster(base_model=base, calibration_days=0)
    with pytest.raises(ValueError, match="calibration_days must be positive"):
        ConformalForecaster(base_model=base, calibration_days=-5)


def test_predict_without_fit_raises() -> None:
    conf = ConformalForecaster(base_model=SeasonalNaiveForecaster(season=7))
    with pytest.raises(RuntimeError, match="not fitted"):
        conf.predict(pd.DataFrame({DATE_COL: [], GROUP_COL: [], TARGET_COL: []}), horizon=7)


def test_fit_with_missing_column_raises() -> None:
    """Missing required column produces a clear KeyError."""
    df = pd.DataFrame({DATE_COL: pd.date_range("2018-01-01", periods=50), GROUP_COL: ["SP"] * 50})
    conf = ConformalForecaster(base_model=SeasonalNaiveForecaster(season=7))
    with pytest.raises(KeyError, match="n_shipments"):
        conf.fit(df)


def test_save_unfitted_raises(tmp_path: Path) -> None:
    conf = ConformalForecaster(base_model=SeasonalNaiveForecaster(season=7))
    with pytest.raises(RuntimeError, match="Cannot save"):
        conf.save(tmp_path / "noop")


# ---------------------------------------------------------------------------
# Conformal correctness
# ---------------------------------------------------------------------------


def test_empirical_coverage_on_calib_matches_nominal(fitted_conformal) -> None:
    """By construction, the fraction of calibration residuals inside
    [lower_offset_, upper_offset_] is exactly 1 - alpha (give or take a
    single-sample boundary). We allow a small tolerance for the discrete
    quantile estimator on finite samples."""
    nominal = 1 - fitted_conformal.alpha
    assert abs(fitted_conformal.empirical_coverage_ - nominal) < 0.05


def test_intervals_can_be_asymmetric(fitted_conformal) -> None:
    """The two offsets are stored separately, so the wrapper is capable of
    producing asymmetric intervals. We assert this by checking that
    abs(lower) and abs(upper) are not forced to be equal."""
    # If the base model has any bias at all on the synthetic data, the
    # quantiles should differ in magnitude.
    assert fitted_conformal.lower_offset_ != -fitted_conformal.upper_offset_


def test_y_lower_clipped_to_zero(fitted_conformal, train_df) -> None:
    """Shipments are non-negative; y_lower must never be < 0 in output."""
    preds = fitted_conformal.predict(train_df, horizon=14)
    assert (preds["y_lower"] >= 0).all()


def test_y_lower_le_y_pred_le_y_upper(fitted_conformal, train_df) -> None:
    """The interval must contain the point estimate (modulo clipping)."""
    preds = fitted_conformal.predict(train_df, horizon=14)
    # y_upper >= y_pred always (additive positive margin)
    assert (preds["y_upper"] >= preds["y_pred"]).all()
    # y_lower <= y_pred WHERE not clipped to 0; if y_lower=0, y_pred can
    # legitimately be smaller (e.g. y_pred=0 on Sundays).
    not_clipped = preds["y_lower"] > 0
    if not_clipped.any():
        assert (preds.loc[not_clipped, "y_lower"] <= preds.loc[not_clipped, "y_pred"]).all()


# ---------------------------------------------------------------------------
# Anti-leakage
# ---------------------------------------------------------------------------


def test_calibration_split_excludes_last_n_days_from_fit(train_df: pd.DataFrame) -> None:
    """The base model must NOT be fitted on the calibration days.

    Strategy: build a wrapper, fit it, and check that the base model's
    internal last-train-date (when available) is calib_start - 1.
    For SeasonalNaiveForecaster, we use _last_date.
    """
    last_date = train_df[DATE_COL].max()
    calib_days = 30
    expected_fit_end = last_date - pd.Timedelta(days=calib_days)

    base = SeasonalNaiveForecaster(season=7)
    conf = ConformalForecaster(base_model=base, calibration_days=calib_days)
    conf.fit(train_df)

    # SeasonalNaiveForecaster stores _last_date as the final date it saw.
    assert pd.Timestamp(base._last_date) == expected_fit_end


def test_too_short_train_raises_clear_error() -> None:
    """If calibration_days >= train length, fit must fail loudly."""
    tiny = _make_panel(n_days=20)
    conf = ConformalForecaster(
        base_model=SeasonalNaiveForecaster(season=7),
        calibration_days=30,
    )
    with pytest.raises(ValueError, match="no rows remain"):
        conf.fit(tiny)
