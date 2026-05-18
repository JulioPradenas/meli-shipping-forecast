"""Tests del endpoint /v1/health."""

from fastapi.testclient import TestClient

from shipping_forecast.api.app import app


def test_health_returns_ok():
    """En 8.0 /health siempre retorna OK. En 8.4 se agregan checks reales."""
    with TestClient(app) as client:
        response = client.get("/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
