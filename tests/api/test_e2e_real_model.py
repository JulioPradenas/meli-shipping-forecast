"""End-to-end tests against the REAL model and database.

Unlike the mock-based tests (test_predict.py, test_model_info.py), these
exercise the full stack: the lifespan loads the real ConformalForecaster
from artifacts/lightgbm_final.joblib and the real historical panel from
the SQLite DB, then runs actual predictions.

These tests are SKIPPED automatically when the model artifact or the DB
is not present (e.g. in CI, which has neither). To run them locally:

    python scripts/load_data.py    # builds data/processed/shipping.db
    make train-model               # builds artifacts/lightgbm_final.{joblib,json}
    uv run pytest tests/api/test_e2e_real_model.py -v

This mirrors the skip-if-no-DB pattern already used by
tests/integration/test_features_pipeline_e2e.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "lightgbm_final.joblib"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "shipping.db"

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and DB_PATH.exists()),
    reason="Real model artifact and/or SQLite DB not found. "
    "Run `make train-model` and `python scripts/load_data.py` to enable E2E tests.",
)


@pytest.fixture
def real_client() -> TestClient:
    """TestClient with the REAL model + panel loaded into app.state.

    Rather than driving the lifespan via the context-manager form (which
    caused generator-teardown conflicts), we load the real artifacts the
    same way the lifespan does and inject them into app.state directly.
    This exercises the real model and real historical panel without the
    lifespan's startup/shutdown cycle.
    """

    from shipping_forecast.api.app import _load_artifacts, app
    from shipping_forecast.pipelines.train_final_model import (
        DATA_CUTOFF,
        load_panel_with_cutoff,
    )
    from shipping_forecast.pipelines.train_final_model import (
        DB_PATH as TRAIN_DB_PATH,
    )

    model, model_info = _load_artifacts(MODEL_PATH)
    app.state.model = model
    app.state.model_info = model_info
    app.state.history_panel = load_panel_with_cutoff(TRAIN_DB_PATH, DATA_CUTOFF)

    return TestClient(app)


def test_e2e_health_ok(real_client: TestClient) -> None:
    """Health reports the real model is loaded."""
    response = real_client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "lgbm-v1.1.0"


def test_e2e_model_info_real_metadata(real_client: TestClient) -> None:
    """Model info reports the real training metadata and honest metrics."""
    response = real_client.get("/v1/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["last_train_date"] == "2018-08-31"
    assert body["n_groups"] == 27
    assert body["evaluation_metrics"]["wape"] == 0.5156


def test_e2e_predict_returns_real_predictions(real_client: TestClient) -> None:
    """A real predict request returns well-formed, non-negative predictions."""
    body = {
        "start_date": "2018-09-01",
        "end_date": "2018-09-03",
        "states": ["SP", "RJ"],
    }
    response = real_client.post("/v1/predict", json=body)
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "lgbm-v1.1.0"
    # 3 days x 2 states = 6 predictions
    assert data["metadata"]["n_predictions"] == 6
    for p in data["predictions"]:
        assert p["point"] >= 0.0
        assert p["lower_90"] >= 0.0
        assert p["upper_90"] >= p["lower_90"]
        # recommended sits between point and upper for alpha > 0
        assert p["recommended"] >= p["point"]


def test_e2e_predict_all_states(real_client: TestClient) -> None:
    """Omitting states predicts all 27 Brazilian states."""
    body = {"start_date": "2018-09-01", "end_date": "2018-09-01"}
    response = real_client.post("/v1/predict", json=body)
    assert response.status_code == 200
    states = {p["state"] for p in response.json()["predictions"]}
    assert len(states) == 27
