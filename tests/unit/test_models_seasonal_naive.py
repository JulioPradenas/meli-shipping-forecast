"""Unit tests for SeasonalNaiveForecaster."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shipping_forecast.models.seasonal_naive import SeasonalNaiveForecaster

# --------------------------------------------------------------------- fixtures


@pytest.fixture
def history_df() -> pd.DataFrame:
    """A 14-day single-group DataFrame with distinct values per day.

    Last 7 values (which will become the forecasts for horizon=1..7):
    [200, 210, 220, 230, 240, 250, 260]
    """
    return pd.DataFrame(
        {
            "shipment_date": pd.date_range("2024-01-01", periods=14),
            "customer_state": ["SP"] * 14,
            "n_shipments": [
                100,
                110,
                120,
                130,
                140,
                150,
                160,
                200,
                210,
                220,
                230,
                240,
                250,
                260,
            ],
        }
    )


@pytest.fixture
def two_group_df() -> pd.DataFrame:
    """Two groups, 14 days each. Last 7 of SP=[200..260], RJ=[80..140]."""
    return pd.DataFrame(
        {
            "shipment_date": pd.concat(
                [pd.Series(pd.date_range("2024-01-01", periods=14))] * 2
            ).reset_index(drop=True),
            "customer_state": ["SP"] * 14 + ["RJ"] * 14,
            "n_shipments": [
                100,
                110,
                120,
                130,
                140,
                150,
                160,
                200,
                210,
                220,
                230,
                240,
                250,
                260,
                40,
                45,
                50,
                55,
                60,
                65,
                70,
                80,
                90,
                100,
                110,
                120,
                130,
                140,
            ],
        }
    )


# ----------------------------------------------------------------------- init


def test_init_rejects_invalid_season() -> None:
    """season < 1 must raise at construction."""
    with pytest.raises(ValueError, match="season must be"):
        SeasonalNaiveForecaster(season=0)
    with pytest.raises(ValueError, match="season must be"):
        SeasonalNaiveForecaster(season=-1)


# ------------------------------------------------------------------------ fit


def test_fit_returns_self(history_df: pd.DataFrame) -> None:
    """fit() must return the same instance for chaining."""
    model = SeasonalNaiveForecaster()
    assert model.fit(history_df) is model


def test_fit_stores_last_7_values(history_df: pd.DataFrame) -> None:
    """After fit, the model must memorise the last `season` values per group."""
    model = SeasonalNaiveForecaster(season=7).fit(history_df)
    assert model._last_season_values["SP"] == [200, 210, 220, 230, 240, 250, 260]


def test_fit_stores_last_date(history_df: pd.DataFrame) -> None:
    """After fit, the model must know the last training date."""
    model = SeasonalNaiveForecaster().fit(history_df)
    assert model._last_date == pd.Timestamp("2024-01-14")


def test_fit_raises_when_group_has_fewer_than_season_obs() -> None:
    """A group with fewer than `season` observations must raise."""
    df = pd.DataFrame(
        {
            "shipment_date": pd.date_range("2024-01-01", periods=5),
            "customer_state": ["SP"] * 5,
            "n_shipments": [100, 110, 120, 130, 140],
        }
    )
    with pytest.raises(ValueError, match="need at least"):
        SeasonalNaiveForecaster(season=7).fit(df)


def test_fit_raises_on_missing_column() -> None:
    """Missing required columns must raise KeyError."""
    bad_df = pd.DataFrame({"shipment_date": [], "customer_state": []})
    with pytest.raises(KeyError, match="Missing required column"):
        SeasonalNaiveForecaster().fit(bad_df)


# -------------------------------------------------------------------- predict


def test_predict_requires_fit() -> None:
    """Calling predict before fit must raise RuntimeError."""
    df = pd.DataFrame()
    with pytest.raises(RuntimeError, match="fit"):
        SeasonalNaiveForecaster().predict(df, horizon=5)


def test_predict_rejects_zero_horizon(history_df: pd.DataFrame) -> None:
    """horizon < 1 must raise ValueError."""
    model = SeasonalNaiveForecaster().fit(history_df)
    with pytest.raises(ValueError, match="horizon"):
        model.predict(history_df, horizon=0)


def test_predict_uses_last_7_days_within_horizon_7(
    history_df: pd.DataFrame,
) -> None:
    """For horizon=7, predictions must equal the last 7 training values."""
    model = SeasonalNaiveForecaster(season=7).fit(history_df)
    preds = model.predict(history_df, horizon=7)
    assert preds["y_pred"].tolist() == [200, 210, 220, 230, 240, 250, 260]


def test_predict_cycles_for_horizon_greater_than_season(
    history_df: pd.DataFrame,
) -> None:
    """For horizon > 7, the prediction pattern must repeat cyclically."""
    model = SeasonalNaiveForecaster(season=7).fit(history_df)
    preds = model.predict(history_df, horizon=10)
    # Days 1-7: [200..260]; days 8-10: [200, 210, 220] (cycle)
    expected = [200, 210, 220, 230, 240, 250, 260, 200, 210, 220]
    assert preds["y_pred"].tolist() == expected


def test_predict_per_group(two_group_df: pd.DataFrame) -> None:
    """Each group's predictions must use that group's history independently."""
    model = SeasonalNaiveForecaster(season=7).fit(two_group_df)
    preds = model.predict(two_group_df, horizon=7)

    sp = preds[preds["customer_state"] == "SP"]["y_pred"].tolist()
    rj = preds[preds["customer_state"] == "RJ"]["y_pred"].tolist()

    assert sp == [200, 210, 220, 230, 240, 250, 260]
    assert rj == [80, 90, 100, 110, 120, 130, 140]


def test_predict_forecast_dates_are_contiguous_future(
    history_df: pd.DataFrame,
) -> None:
    """Forecast dates start the day after the last train date."""
    model = SeasonalNaiveForecaster().fit(history_df)
    preds = model.predict(history_df, horizon=3)
    expected_dates = pd.date_range("2024-01-15", periods=3).tolist()
    assert preds["shipment_date"].tolist() == expected_dates


def test_predict_returns_required_columns(history_df: pd.DataFrame) -> None:
    """Output must include all standard columns."""
    model = SeasonalNaiveForecaster().fit(history_df)
    preds = model.predict(history_df, horizon=3)
    for col in ["shipment_date", "customer_state", "y_pred", "y_lower", "y_upper"]:
        assert col in preds.columns


def test_predict_interval_bounds_make_sense(history_df: pd.DataFrame) -> None:
    """y_lower <= y_pred <= y_upper, y_lower >= 0."""
    model = SeasonalNaiveForecaster().fit(history_df)
    preds = model.predict(history_df, horizon=3)
    assert (preds["y_lower"] <= preds["y_pred"]).all()
    assert (preds["y_pred"] <= preds["y_upper"]).all()
    assert (preds["y_lower"] >= 0).all()


# ----------------------------------------------- season other than 7


def test_custom_season() -> None:
    """A monthly-ish season (30 days) must work just like weekly."""
    df = pd.DataFrame(
        {
            "shipment_date": pd.date_range("2024-01-01", periods=60),
            "customer_state": ["SP"] * 60,
            "n_shipments": list(range(60)),
        }
    )
    model = SeasonalNaiveForecaster(season=30).fit(df)
    # Last 30 values = [30, 31, ..., 59]
    preds = model.predict(df, horizon=3)
    assert preds["y_pred"].tolist() == [30, 31, 32]


# -------------------------------------------------------------------- persistence


def test_save_creates_two_files(history_df: pd.DataFrame, tmp_path: Path) -> None:
    """save() must create both .joblib and .json."""
    model = SeasonalNaiveForecaster().fit(history_df)
    path = tmp_path / "snaive_v1"
    model.save(path)
    assert path.with_suffix(".joblib").exists()
    assert path.with_suffix(".json").exists()


def test_save_rejects_unfitted(tmp_path: Path) -> None:
    """Saving an unfitted model must raise RuntimeError."""
    model = SeasonalNaiveForecaster()
    with pytest.raises(RuntimeError, match="unfitted"):
        model.save(tmp_path / "snaive_unfit")


def test_load_restores_predictions(history_df: pd.DataFrame, tmp_path: Path) -> None:
    """A loaded model must produce identical predictions."""
    original = SeasonalNaiveForecaster().fit(history_df)
    original_preds = original.predict(history_df, horizon=7)

    path = tmp_path / "snaive_roundtrip"
    original.save(path)
    restored = SeasonalNaiveForecaster.load(path)
    restored_preds = restored.predict(history_df, horizon=7)

    pd.testing.assert_frame_equal(original_preds, restored_preds)


def test_metadata_sidecar_contains_season(history_df: pd.DataFrame, tmp_path: Path) -> None:
    """The JSON sidecar must include the season parameter."""
    import json

    model = SeasonalNaiveForecaster(season=7).fit(history_df)
    path = tmp_path / "snaive_meta"
    model.save(path)

    with open(path.with_suffix(".json")) as f:
        meta = json.load(f)

    assert meta["model_name"] == "SeasonalNaiveForecaster"
    assert meta["params"]["season"] == 7
