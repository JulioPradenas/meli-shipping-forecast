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


@pytest.fixture
def mock_model_info() -> dict[str, Any]:
    """Minimal model_info dict matching what endpoints read."""
    return {
        "version": "test-v0",
        "last_train_date": "2018-08-31",
        "n_features": 23,
        "n_groups": 2,
        "groups": ["RJ", "SP"],
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

    # Do NOT use TestClient(app) as a context manager here: that would run
    # the lifespan and try to load/train a real model. Plain instantiation
    # uses the state we just set.
    return TestClient(app)
