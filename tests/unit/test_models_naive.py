"""Unit tests for NaiveForecaster."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shipping_forecast.models.naive import NaiveForecaster

# --------------------------------------------------------------------- fixtures


@pytest.fixture
def history_df() -> pd.DataFrame:
    """A small two-group historical DataFrame.

    SP last value = 190, RJ last value = 95.
    """
    return pd.DataFrame(
        {
            "shipment_date": pd.concat(
                [pd.Series(pd.date_range("2024-01-01", periods=10))] * 2
            ).reset_index(drop=True),
            "customer_state": ["SP"] * 10 + ["RJ"] * 10,
            "n_shipments": [
                100,
                110,
                120,
                130,
                140,
                150,
                160,
                170,
                180,
                190,
                50,
                55,
                60,
                65,
                70,
                75,
                80,
                85,
                90,
                95,
            ],
        }
    )


# --------------------------------------------------------------------- fit


def test_fit_returns_self(history_df: pd.DataFrame) -> None:
    """fit() must return the same instance for chaining."""
    model = NaiveForecaster()
    assert model.fit(history_df) is model


def test_fit_stores_last_value_per_group(history_df: pd.DataFrame) -> None:
    """After fit, the model must know the last value of each group."""
    model = NaiveForecaster().fit(history_df)
    assert model._last_values == {"SP": 190.0, "RJ": 95.0}


def test_fit_stores_last_date(history_df: pd.DataFrame) -> None:
    """After fit, the model must know the latest training date."""
    model = NaiveForecaster().fit(history_df)
    assert model._last_date == pd.Timestamp("2024-01-10")


def test_fit_raises_on_missing_column() -> None:
    """Missing required columns must raise KeyError."""
    bad_df = pd.DataFrame({"shipment_date": [], "customer_state": []})
    with pytest.raises(KeyError, match="Missing required column"):
        NaiveForecaster().fit(bad_df)


# --------------------------------------------------------------------- predict


def test_predict_requires_fit() -> None:
    """Calling predict before fit must raise RuntimeError."""
    df = pd.DataFrame()
    model = NaiveForecaster()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(df, horizon=5)


def test_predict_rejects_zero_horizon(history_df: pd.DataFrame) -> None:
    """horizon < 1 must raise ValueError."""
    model = NaiveForecaster().fit(history_df)
    with pytest.raises(ValueError, match="horizon"):
        model.predict(history_df, horizon=0)


def test_predict_returns_dataframe_with_required_columns(
    history_df: pd.DataFrame,
) -> None:
    """The output DataFrame must include date, group, y_pred, y_lower, y_upper."""
    model = NaiveForecaster().fit(history_df)
    preds = model.predict(history_df, horizon=3)

    assert "shipment_date" in preds.columns
    assert "customer_state" in preds.columns
    assert "y_pred" in preds.columns
    assert "y_lower" in preds.columns
    assert "y_upper" in preds.columns


def test_predict_horizon_x_groups_rows(history_df: pd.DataFrame) -> None:
    """Output row count = horizon * n_groups."""
    model = NaiveForecaster().fit(history_df)
    preds = model.predict(history_df, horizon=5)
    # 5 days x 2 groups (SP, RJ) = 10 rows
    assert len(preds) == 10


def test_predict_uses_last_value_per_group(history_df: pd.DataFrame) -> None:
    """Every forecast row for a group must equal that group's last value."""
    model = NaiveForecaster().fit(history_df)
    preds = model.predict(history_df, horizon=3)

    sp_preds = preds[preds["customer_state"] == "SP"]["y_pred"]
    rj_preds = preds[preds["customer_state"] == "RJ"]["y_pred"]

    assert (sp_preds == 190.0).all()
    assert (rj_preds == 95.0).all()


def test_predict_forecast_dates_are_contiguous_and_future(
    history_df: pd.DataFrame,
) -> None:
    """Forecast dates must be the day after the last train date, contiguous."""
    model = NaiveForecaster().fit(history_df)
    preds = model.predict(history_df, horizon=3)

    sp_dates = sorted(preds[preds["customer_state"] == "SP"]["shipment_date"])
    expected = pd.date_range("2024-01-11", periods=3, freq="D").tolist()
    assert sp_dates == expected


def test_predict_interval_bounds_make_sense(history_df: pd.DataFrame) -> None:
    """y_lower <= y_pred <= y_upper, and y_lower >= 0."""
    model = NaiveForecaster().fit(history_df)
    preds = model.predict(history_df, horizon=3)

    assert (preds["y_lower"] <= preds["y_pred"]).all()
    assert (preds["y_pred"] <= preds["y_upper"]).all()
    assert (preds["y_lower"] >= 0).all()


# --------------------------------------------------------------------- save / load


def test_save_creates_two_files(history_df: pd.DataFrame, tmp_path: Path) -> None:
    """save() must create both a .joblib and a .json file."""
    model = NaiveForecaster().fit(history_df)
    path = tmp_path / "naive_v1"
    model.save(path)

    assert path.with_suffix(".joblib").exists()
    assert path.with_suffix(".json").exists()


def test_save_rejects_unfitted_model(tmp_path: Path) -> None:
    """Saving an unfitted model must raise RuntimeError."""
    model = NaiveForecaster()
    with pytest.raises(RuntimeError, match="unfitted"):
        model.save(tmp_path / "naive_unfit")


def test_load_restores_predictions(history_df: pd.DataFrame, tmp_path: Path) -> None:
    """A loaded model must produce identical predictions to the original."""
    original = NaiveForecaster().fit(history_df)
    original_preds = original.predict(history_df, horizon=3)

    path = tmp_path / "naive_roundtrip"
    original.save(path)
    restored = NaiveForecaster.load(path)
    restored_preds = restored.predict(history_df, horizon=3)

    pd.testing.assert_frame_equal(original_preds, restored_preds)


def test_load_rejects_wrong_class(history_df: pd.DataFrame, tmp_path: Path) -> None:
    """Loading a file that contains a different class must raise TypeError."""
    import joblib

    # Save a non-NaiveForecaster object at the expected path
    bogus = {"this": "is not a model"}
    path = tmp_path / "wrong_type"
    joblib.dump(bogus, path.with_suffix(".joblib"))

    with pytest.raises(TypeError, match="expected NaiveForecaster"):
        NaiveForecaster.load(path)


# --------------------------------------------------------------------- metadata


def test_metadata_sidecar_contains_expected_keys(history_df: pd.DataFrame, tmp_path: Path) -> None:
    """The JSON sidecar must include model name, train date, params, etc."""
    import json

    model = NaiveForecaster().fit(history_df)
    path = tmp_path / "naive_meta"
    model.save(path)

    with open(path.with_suffix(".json")) as f:
        meta = json.load(f)

    assert meta["model_name"] == "NaiveForecaster"
    assert meta["last_train_date"] == "2024-01-10T00:00:00"
    assert meta["n_groups"] == 2
    assert set(meta["groups"]) == {"SP", "RJ"}
    assert meta["params"]["interval_width"] == 1.28
