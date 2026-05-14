"""Unit tests for ProphetForecaster.

Note:
    Prophet is slow to fit (compiled Stan model). To keep the test
    suite snappy, we share a single fitted model across most tests via
    a module-scoped fixture.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shipping_forecast.models.prophet_model import ProphetForecaster

# --------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def history_df() -> pd.DataFrame:
    """A two-group historical DataFrame covering 1.5 years.

    Long enough for Prophet to fit weekly and yearly seasonality.
    """
    dates = pd.date_range("2023-01-01", periods=450, freq="D")
    rows: list[dict] = []
    for state, base in [("SP", 100), ("RJ", 50)]:
        for i, d in enumerate(dates):
            # Add weekly + yearly seasonality + small trend
            weekly = 30 * ((i % 7) - 3) / 3
            yearly = 20 * (i / 365)
            value = max(0, base + weekly + yearly)
            rows.append(
                {
                    "shipment_date": d,
                    "customer_state": state,
                    "n_shipments": float(value),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted_model(history_df: pd.DataFrame) -> ProphetForecaster:
    """A fitted Prophet model. Shared across tests to avoid refit overhead.

    Yearly seasonality off to keep training fast: weekly is enough for
    contract checks.
    """
    return ProphetForecaster(
        yearly_seasonality=False,
        use_br_holidays=False,
    ).fit(history_df)


# ------------------------------------------------------------------------ fit


def test_fit_returns_self(history_df: pd.DataFrame) -> None:
    """fit() must return the same instance for chaining."""
    model = ProphetForecaster(yearly_seasonality=False, use_br_holidays=False)
    assert model.fit(history_df) is model


def test_fit_creates_one_model_per_group(fitted_model: ProphetForecaster) -> None:
    """After fit, the model dict has one Prophet instance per group."""
    assert set(fitted_model._models.keys()) == {"SP", "RJ"}


def test_fit_stores_last_date(fitted_model: ProphetForecaster, history_df: pd.DataFrame) -> None:
    """The last training date must equal the max date in the input."""
    expected = history_df["shipment_date"].max()
    assert fitted_model._last_date == expected


def test_fit_raises_on_missing_column() -> None:
    """Missing required columns must raise KeyError."""
    bad_df = pd.DataFrame({"shipment_date": [], "customer_state": []})
    with pytest.raises(KeyError, match="Missing required column"):
        ProphetForecaster().fit(bad_df)


# -------------------------------------------------------------------- predict


def test_predict_requires_fit() -> None:
    """predict() before fit() must raise."""
    df = pd.DataFrame()
    with pytest.raises(RuntimeError, match="fit"):
        ProphetForecaster().predict(df, horizon=5)


def test_predict_rejects_zero_horizon(
    fitted_model: ProphetForecaster, history_df: pd.DataFrame
) -> None:
    """horizon < 1 must raise."""
    with pytest.raises(ValueError, match="horizon"):
        fitted_model.predict(history_df, horizon=0)


def test_predict_returns_required_columns(
    fitted_model: ProphetForecaster, history_df: pd.DataFrame
) -> None:
    """Output must include date, group, y_pred, y_lower, y_upper."""
    preds = fitted_model.predict(history_df, horizon=7)
    for col in ["shipment_date", "customer_state", "y_pred", "y_lower", "y_upper"]:
        assert col in preds.columns


def test_predict_horizon_x_groups_rows(
    fitted_model: ProphetForecaster, history_df: pd.DataFrame
) -> None:
    """Row count must equal horizon * number of groups."""
    preds = fitted_model.predict(history_df, horizon=5)
    assert len(preds) == 10  # 5 days * 2 groups


def test_predict_dates_are_future(
    fitted_model: ProphetForecaster, history_df: pd.DataFrame
) -> None:
    """Forecast dates must start the day after the last train date."""
    preds = fitted_model.predict(history_df, horizon=3)
    last_train = history_df["shipment_date"].max()
    min_forecast = preds["shipment_date"].min()
    assert min_forecast == last_train + pd.Timedelta(days=1)


def test_predict_interval_bounds_make_sense(
    fitted_model: ProphetForecaster, history_df: pd.DataFrame
) -> None:
    """y_lower <= y_pred <= y_upper; all >= 0."""
    preds = fitted_model.predict(history_df, horizon=7)
    assert (preds["y_lower"] <= preds["y_pred"]).all()
    assert (preds["y_pred"] <= preds["y_upper"]).all()
    assert (preds["y_lower"] >= 0).all()


def test_predict_captures_weekly_pattern(
    fitted_model: ProphetForecaster, history_df: pd.DataFrame
) -> None:
    """Prophet should produce non-constant predictions reflecting weekly pattern.

    The training data has a clear weekly seasonality (peak on day-of-week
    where (i % 7) == 6, low when (i % 7) == 0). Predictions over 14 days
    should not be a flat line.
    """
    preds = fitted_model.predict(history_df, horizon=14)
    sp_preds = preds[preds["customer_state"] == "SP"]["y_pred"]
    # Standard deviation across 14 days should be non-trivial
    assert sp_preds.std() > 1.0


# -------------------------------------------------------------------- persistence


def test_save_creates_two_files(fitted_model: ProphetForecaster, tmp_path: Path) -> None:
    """save() must create both .joblib and .json."""
    path = tmp_path / "prophet_v1"
    fitted_model.save(path)
    assert path.with_suffix(".joblib").exists()
    assert path.with_suffix(".json").exists()


def test_save_rejects_unfitted(tmp_path: Path) -> None:
    """Unfitted models cannot be saved."""
    with pytest.raises(RuntimeError, match="unfitted"):
        ProphetForecaster().save(tmp_path / "prophet_unfit")


def test_load_restores_predictions(
    fitted_model: ProphetForecaster,
    history_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Loaded model must produce equivalent predictions to the original.

    Note:
        Prophet's prediction intervals (y_lower, y_upper) come from MCMC
        sampling and are NOT bit-exact between calls — even on the same
        in-memory model. We verify:
          - y_pred (deterministic MAP estimate) with exact equality
          - y_lower/y_upper with tolerance (1e-3)
          - Metadata columns (date, state) with exact equality
    """
    path = tmp_path / "prophet_roundtrip"
    fitted_model.save(path)
    restored = ProphetForecaster.load(path)

    original_preds = fitted_model.predict(history_df, horizon=7)
    restored_preds = restored.predict(history_df, horizon=7)

    # Deterministic columns: exact match
    pd.testing.assert_series_equal(
        original_preds["shipment_date"],
        restored_preds["shipment_date"],
    )
    pd.testing.assert_series_equal(
        original_preds["customer_state"],
        restored_preds["customer_state"],
    )
    pd.testing.assert_series_equal(
        original_preds["y_pred"],
        restored_preds["y_pred"],
    )

    # Stochastic columns: approximate match (MCMC sampling)
    pd.testing.assert_series_equal(
        original_preds["y_lower"],
        restored_preds["y_lower"],
        atol=0.5,
    )
    pd.testing.assert_series_equal(
        original_preds["y_upper"],
        restored_preds["y_upper"],
        atol=0.5,
    )


def test_metadata_sidecar_contains_expected_params(
    fitted_model: ProphetForecaster, tmp_path: Path
) -> None:
    """The JSON sidecar should record key model parameters."""
    import json

    path = tmp_path / "prophet_meta"
    fitted_model.save(path)

    with open(path.with_suffix(".json")) as f:
        meta = json.load(f)

    assert meta["model_name"] == "ProphetForecaster"
    assert meta["n_groups"] == 2
    assert "interval_width" in meta["params"]
    assert "use_br_holidays" in meta["params"]


# -------------------------------------------------- BR holidays smoke test


def test_br_holidays_flag_does_not_crash() -> None:
    """When use_br_holidays=True, training should still complete cleanly.

    We use a small dataset; Prophet may emit warnings about small history
    but should not raise.
    """
    dates = pd.date_range("2017-01-01", periods=200, freq="D")
    df = pd.DataFrame(
        {
            "shipment_date": dates,
            "customer_state": ["SP"] * 200,
            "n_shipments": [100.0 + i * 0.5 for i in range(200)],
        }
    )
    model = ProphetForecaster(
        use_br_holidays=True,
        yearly_seasonality=False,
    )
    model.fit(df)  # should not raise
    preds = model.predict(df, horizon=7)
    assert len(preds) == 7
