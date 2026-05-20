"""Tests for the /v1/health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    """Health is a liveness probe; it returns 200 OK.

    Uses the mock-model client fixture so it does not depend on a real
    model or the SQLite database being present.
    """
    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "test-v0"
