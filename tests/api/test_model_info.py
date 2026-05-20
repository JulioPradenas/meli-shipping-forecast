"""Tests for the GET /v1/model/info endpoint using the mock model fixture."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_model_info_returns_200(client: TestClient) -> None:
    """A loaded model returns 200 with the public metadata."""
    response = client.get("/v1/model/info")
    assert response.status_code == 200


def test_model_info_exposes_expected_fields(client: TestClient) -> None:
    """The response surfaces version, dates, states, and metrics."""
    response = client.get("/v1/model/info")
    body = response.json()
    assert body["model_version"] == "test-v0"
    assert body["last_train_date"] == "2018-08-31"
    assert body["n_groups"] == 2
    assert set(body["groups"]) == {"SP", "RJ"}
    assert "evaluation_metrics" in body
    assert "wape" in body["evaluation_metrics"]


def test_model_info_hides_internal_fields(client: TestClient) -> None:
    """Internal conformal fields are not exposed in the public contract."""
    response = client.get("/v1/model/info")
    body = response.json()
    for hidden in ("lower_offset", "upper_offset", "calibration_days", "params"):
        assert hidden not in body
