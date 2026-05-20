"""Tests for the POST /v1/predict endpoint using the mock model fixture.

The mock model (see conftest.py) returns a deterministic frame for SP/RJ
with y_pred=100, y_lower=80, y_upper=120 for every requested day. The
mock model_info reports last_train_date=2018-08-31 and groups=[RJ, SP].
These fixed values let us assert exact numbers.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

VALID_BODY = {
    "start_date": "2018-09-01",
    "end_date": "2018-09-02",
    "states": ["SP"],
}


def test_predict_happy_path_returns_200(client: TestClient) -> None:
    """A valid request returns 200 with the expected envelope."""
    response = client.post("/v1/predict", json=VALID_BODY)
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "test-v0"
    assert "predictions" in body
    assert "metadata" in body


def test_predict_returns_one_row_per_day_per_state(client: TestClient) -> None:
    """2 days x 1 state = 2 predictions."""
    response = client.post("/v1/predict", json=VALID_BODY)
    body = response.json()
    assert body["metadata"]["n_predictions"] == 2
    assert len(body["predictions"]) == 2
    assert all(p["state"] == "SP" for p in body["predictions"])


def test_predict_state_filter_excludes_others(client: TestClient) -> None:
    """Requesting only RJ excludes SP from the mock's SP/RJ output."""
    body_rj = {**VALID_BODY, "states": ["RJ"]}
    response = client.post("/v1/predict", json=body_rj)
    preds = response.json()["predictions"]
    assert all(p["state"] == "RJ" for p in preds)


def test_predict_no_states_returns_all(client: TestClient) -> None:
    """Omitting states returns all the mock's states (SP and RJ)."""
    body = {"start_date": "2018-09-01", "end_date": "2018-09-01"}
    response = client.post("/v1/predict", json=body)
    preds = response.json()["predictions"]
    states = {p["state"] for p in preds}
    assert states == {"SP", "RJ"}


def test_predict_intervals_off_nulls_bounds(client: TestClient) -> None:
    """include_intervals=False sets lower_90/upper_90 to null."""
    body = {**VALID_BODY, "include_intervals": False}
    response = client.post("/v1/predict", json=body)
    for p in response.json()["predictions"]:
        assert p["lower_90"] is None
        assert p["upper_90"] is None


def test_predict_cost_aware_off_nulls_recommended(client: TestClient) -> None:
    """include_cost_aware=False sets recommended/alpha_used/cost_ratio_used to null."""
    body = {**VALID_BODY, "include_cost_aware": False}
    response = client.post("/v1/predict", json=body)
    for p in response.json()["predictions"]:
        assert p["recommended"] is None
        assert p["alpha_used"] is None
        assert p["cost_ratio_used"] is None
    meta = response.json()["metadata"]
    assert meta["alpha_source"] == "server_default"


def test_predict_alpha_override_reported_in_metadata(client: TestClient) -> None:
    """An explicit alpha is echoed in alpha_used with source 'request'."""
    body = {**VALID_BODY, "alpha": 1.0}
    response = client.post("/v1/predict", json=body)
    meta = response.json()["metadata"]
    assert meta["alpha_source"] == "request"
    for p in response.json()["predictions"]:
        assert p["alpha_used"] == 1.0


def test_predict_alpha_one_makes_recommended_equal_upper(client: TestClient) -> None:
    """With alpha=1.0, recommended == y_upper (mock upper is 120)."""
    body = {**VALID_BODY, "alpha": 1.0}
    response = client.post("/v1/predict", json=body)
    for p in response.json()["predictions"]:
        assert p["recommended"] == p["upper_90"]


def test_predict_unknown_state_returns_422(client: TestClient) -> None:
    """States not known to the model return 422 with the valid list."""
    body = {**VALID_BODY, "states": ["XX", "ZZ"]}
    response = client.post("/v1/predict", json=body)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "XX" in detail or "ZZ" in detail


def test_predict_start_date_in_past_returns_422(client: TestClient) -> None:
    """start_date on or before last_train_date (2018-08-31) returns 422."""
    body = {"start_date": "2018-07-01", "end_date": "2018-07-10"}
    response = client.post("/v1/predict", json=body)
    assert response.status_code == 422
    assert "last_train_date" in response.json()["detail"]


def test_predict_horizon_over_90_days_returns_422(client: TestClient) -> None:
    """A window longer than 90 days is rejected by the Pydantic schema."""
    body = {"start_date": "2018-09-01", "end_date": "2019-01-01"}
    response = client.post("/v1/predict", json=body)
    assert response.status_code == 422


def test_predict_alpha_out_of_range_returns_422(client: TestClient) -> None:
    """alpha outside [-2, 2] is rejected by the Pydantic schema."""
    body = {**VALID_BODY, "alpha": 5.0}
    response = client.post("/v1/predict", json=body)
    assert response.status_code == 422


def test_predict_cost_ratio_out_of_range_returns_422(client: TestClient) -> None:
    """cost_ratio outside [0.5, 10] is rejected by the Pydantic schema."""
    body = {**VALID_BODY, "cost_ratio": 50.0}
    response = client.post("/v1/predict", json=body)
    assert response.status_code == 422


def test_predict_end_before_start_returns_422(client: TestClient) -> None:
    """end_date before start_date is rejected by the model_validator."""
    body = {"start_date": "2018-09-10", "end_date": "2018-09-01"}
    response = client.post("/v1/predict", json=body)
    assert response.status_code == 422
