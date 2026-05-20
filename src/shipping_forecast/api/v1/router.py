"""Router del API v1.

Endpoints exposed:
  - GET  /health: liveness probe (full health check arrives in 8.4)
  - POST /predict: produce point + (optional) interval + (optional)
    cost-aware predictions for a date range and (optional) subset of states

The /model/info endpoint is added in 8.4.
"""

from __future__ import annotations

import datetime
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, status

from shipping_forecast.api.logging_config import get_logger
from shipping_forecast.api.settings import Settings
from shipping_forecast.api.v1.dependencies import (
    get_model,
    get_model_info,
    get_settings,
)
from shipping_forecast.api.v1.schemas import (
    ModelInfoResponse,
    PredictMetadata,
    PredictRequest,
    PredictResponse,
)
from shipping_forecast.api.v1.services import (
    build_model_info_response,
    map_predictions_to_response,
    resolve_cost_params,
)
from shipping_forecast.data.queries import load_panel
from shipping_forecast.models import ConformalForecaster
from shipping_forecast.pipelines.train_final_model import DATA_CUTOFF, DB_PATH

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["v1"])


@router.get("/health")
def health(request: Request) -> dict[str, str | bool]:
    """Liveness + readiness probe.

    Returns 200 with model info when the model is loaded and ready to
    serve. Returns 503 when the model is not loaded (startup not complete
    or load failed). A health check that returned OK without a usable
    model would be misleading, so readiness is part of this probe.
    """
    model = getattr(request.app.state, "model", None)
    model_info = getattr(request.app.state, "model_info", None)
    if model is None or model_info is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. The service is starting up or load failed.",
        )
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": model_info["version"],
    }


@router.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    model: Annotated[ConformalForecaster, Depends(get_model)],
    model_info: Annotated[dict[str, Any], Depends(get_model_info)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> PredictResponse:
    """Produce predictions for a date range and optional subset of states.

    Validation order (fail fast):
      1. Schema: Pydantic enforces date range (<=90 days), alpha (-2 to 2),
         cost_ratio (0.5 to 10) before this function even runs.
      2. Runtime: start_date must be strictly after the model's
         last_train_date (no backtesting via /predict).
      3. Runtime: states must be a subset of model_info["groups"]
         (no unknown states).

    On success, runs ``model.predict(df_history, horizon)`` to get y_pred,
    y_lower, y_upper, then maps to a list of Prediction schemas.
    """
    # --- 2. Validate runtime: start_date vs model's training cutoff ----
    last_train_date = datetime.date.fromisoformat(model_info["last_train_date"])
    if request.start_date <= last_train_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"start_date ({request.start_date}) must be strictly after the "
                f"model's last_train_date ({last_train_date}). The /predict endpoint "
                f"does not support backtesting; use a date in the future."
            ),
        )

    # --- 3. Validate runtime: states are known to the model ------------
    known_states: set[str] = set(model_info["groups"])
    if request.states:
        unknown = set(request.states) - known_states
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Unknown states: {sorted(unknown)}. Valid states: {sorted(known_states)}."
                ),
            )

    # --- 4. Resolve cost-aware parameters from request + Settings ------
    alpha, alpha_source, cost_ratio, cost_ratio_source = resolve_cost_params(request, settings)

    # --- 5. Compute the prediction horizon -----------------------------
    horizon_days = (request.end_date - last_train_date).days

    # --- 6. Load training panel and predict ----------------------------
    # The model needs the historical panel up to last_train_date as input
    # to predict() (it uses it to compute lag features for the future).
    df_history = load_panel(DB_PATH)
    df_history = df_history[df_history["shipment_date"] <= DATA_CUTOFF].copy()
    df_predictions = model.predict(df_history, horizon=horizon_days)

    # --- 7. Filter to the requested date window -----------------------
    start_ts = pd.Timestamp(request.start_date)
    end_ts = pd.Timestamp(request.end_date)
    df_window = df_predictions[
        (df_predictions["shipment_date"] >= start_ts) & (df_predictions["shipment_date"] <= end_ts)
    ].copy()

    # --- 8. Map DataFrame -> list[Prediction] --------------------------
    predictions = map_predictions_to_response(
        df_window,
        states_filter=request.states,
        include_intervals=request.include_intervals,
        include_cost_aware=request.include_cost_aware,
        alpha=alpha,
        cost_ratio=cost_ratio,
    )

    # --- 9. Build response with metadata -------------------------------
    response = PredictResponse(
        model_version=model_info["version"],
        predictions=predictions,
        metadata=PredictMetadata(
            predicted_at=datetime.datetime.now(datetime.UTC),
            n_predictions=len(predictions),
            alpha_source=alpha_source or "server_default",
            cost_ratio_source=cost_ratio_source or "server_default",
        ),
    )

    logger.info(
        "prediction_completed",
        n_predictions=len(predictions),
        horizon_days=horizon_days,
        states_requested=len(request.states) if request.states else "all",
        include_intervals=request.include_intervals,
        include_cost_aware=request.include_cost_aware,
        alpha_used=alpha,
        alpha_source=alpha_source,
        cost_ratio_used=cost_ratio,
        cost_ratio_source=cost_ratio_source,
        model_version=model_info["version"],
    )

    return response


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(
    info: Annotated[dict[str, Any], Depends(get_model_info)],
) -> ModelInfoResponse:
    """Return metadata about the currently loaded model.

    Surfaces the subset of model metadata consumers care about: version,
    training date range, known states, and honest holdout metrics. The
    full sidecar contains more (internal conformal offsets, hyperparameters)
    but those are deliberately not exposed to keep the contract minimal.

    Returns 503 (via the get_model_info dependency) if no model is loaded.
    """
    return build_model_info_response(info)
