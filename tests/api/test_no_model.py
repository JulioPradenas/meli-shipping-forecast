"""Tests for the 503 paths when no model is loaded.

These use the client_no_model fixture, which strips model and model_info
from app.state so the get_model / get_model_info dependencies hit their
503 branch.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_503_without_model(client_no_model: TestClient) -> None:
    """Health is a readiness probe: 503 when no model is loaded."""
    response = client_no_model.get("/v1/health")
    assert response.status_code == 503


def test_predict_returns_503_without_model(client_no_model: TestClient) -> None:
    """Predict cannot serve without a model: 503."""
    body = {"start_date": "2018-09-01", "end_date": "2018-09-02"}
    response = client_no_model.post("/v1/predict", json=body)
    assert response.status_code == 503


def test_model_info_returns_503_without_model(client_no_model: TestClient) -> None:
    """Model info cannot be returned without a model: 503."""
    response = client_no_model.get("/v1/model/info")
    assert response.status_code == 503
