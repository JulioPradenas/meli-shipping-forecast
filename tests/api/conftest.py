"""Test fixtures for the API test suite.

Instead of letting the lifespan load or train a real model (which needs
the SQLite DB and the artifacts on disk — neither present in CI), these
fixtures inject a lightweight mock model into app.state directly. This
keeps API tests fast, hermetic, and independent of data/model files.

The mock mimics the minimal surface the endpoints touch:
  - app.state.model: an object with a .predict() returning a small
    DataFrame and a .base_model attribute.
  - app.state.model_info: a dict with the keys the endpoints read.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient


def _mock_predict(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return a tiny deterministic prediction frame for two states."""
    dates = pd.date_range("2018-09-01", periods=horizon, freq="D")
    rows = []
    for d in dates:
        for state in ("SP", "RJ"):
            rows.append(
                {
                    "shipment_date": d,
                    "customer_state": state,
                    "y_pred": 100.0,
                    "y_lower": 80.0,
                    "y_upper": 120.0,
                }
            )
    return pd.DataFrame(rows)


def _mock_history_panel() -> pd.DataFrame:
    """A tiny stand-in for the historical panel the endpoint injects.

    The mock model's predict() ignores its content (it returns a fixed
    frame regardless), so this only needs to be a non-empty DataFrame
    with the expected columns to satisfy the dependency.
    """
    return pd.DataFrame(
        {
            "shipment_date": pd.to_datetime(["2018-08-30", "2018-08-31"]),
            "customer_state": ["SP", "RJ"],
            "n_shipments": [100, 50],
        }
    )


@pytest.fixture
def mock_model_info() -> dict[str, Any]:
    """Minimal model_info dict matching what endpoints read."""
    return {
        "version": "test-v0",
        "trained_at": "2026-01-01T00:00:00+00:00",
        "last_train_date": "2018-08-31",
        "data_cutoff": "2018-08-31",
        "n_features": 23,
        "n_groups": 2,
        "groups": ["RJ", "SP"],
        "evaluation_metrics": {
            "window_start": "2018-07-01",
            "window_end": "2018-08-31",
            "n_days": 62,
            "wape": 0.5156,
            "mae": 4.04,
            "rmse": 14.22,
        },
        "evaluation_note": "Test evaluation note.",
    }


@pytest.fixture
def client(mock_model_info: dict[str, Any]) -> TestClient:
    """TestClient with a mock model injected, bypassing real model loading.

    We build the client WITHOUT triggering the heavy lifespan loader by
    overriding the app.state after construction. Because TestClient as a
    context manager runs the lifespan, we instead set state on the app and
    avoid the context-manager form for these tests.
    """
    from shipping_forecast.api.app import app

    base = SimpleNamespace(
        feature_names_=["f"] * 23,
        state_avg_volume_={"SP": 1.0, "RJ": 1.0},
    )
    model = SimpleNamespace(predict=_mock_predict, base_model=base)

    app.state.model = model
    app.state.model_info = mock_model_info
    app.state.history_panel = _mock_history_panel()

    # Do NOT use TestClient(app) as a context manager here: that would run
    # the lifespan and try to load/train a real model. Plain instantiation
    # uses the state we just set.
    return TestClient(app)


@pytest.fixture
def client_no_model() -> TestClient:
    """TestClient with NO model loaded, for testing 503 paths.

    Because the FastAPI app is a module-level singleton, app.state may
    carry a model set by another test. This fixture explicitly removes
    model and model_info from app.state so the dependencies (get_model,
    get_model_info) hit their 503 branch.

    Like the `client` fixture, it does NOT use TestClient as a context
    manager, so the lifespan loader never runs and cannot repopulate the
    state.
    """
    from shipping_forecast.api.app import app

    # Explicitly clear any state a prior test may have left behind.
    for attr in ("model", "model_info", "history_panel"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)

    return TestClient(app)
