"""FastAPI dependency providers for v1 endpoints.

Dependency injection lets endpoints declare what they need (the model,
the settings) without reaching into ``app.state`` directly. This makes
endpoints easier to test (mocks can be injected via
``app.dependency_overrides``) and isolates the lifespan loader's
responsibility (writing to app.state) from the endpoint's
responsibility (reading and acting on it).

Providers exposed here:
  - get_settings(): returns the application Settings instance.
  - get_model(): returns the loaded ConformalForecaster, or raises 503
    if the lifespan has not loaded it yet.
  - get_model_info(): returns the metadata dict loaded alongside the model.

The model and model_info are populated by the lifespan in app.py during
startup (Phase 8.4). Until that lands, get_model raises 503.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request, status

from shipping_forecast.api.settings import get_settings
from shipping_forecast.models import ConformalForecaster

__all__ = ["get_model", "get_model_info", "get_settings"]


def get_model(request: Request) -> ConformalForecaster:
    """Return the model loaded at startup, or raise 503 if not ready.

    The lifespan in app.py is responsible for setting app.state.model.
    If a request arrives before startup completes, or if startup failed
    to load the model, callers get a clear 503 instead of a cryptic
    AttributeError.
    """
    model: ConformalForecaster | None = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. The service is still starting up or load failed.",
        )
    return model


def get_model_info(request: Request) -> dict[str, Any]:
    """Return the model metadata dict, or raise 503 if not ready.

    Companion to get_model: same source (app.state), same error path.
    The metadata is the parsed JSON sidecar from artifacts/lightgbm_final.json.
    """
    info: dict[str, Any] | None = getattr(request.app.state, "model_info", None)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is not loaded.",
        )
    return info
