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
from fastapi import APIRouter, Depends, HTTPException, status

from shipping_forecast.api.settings import Settings
from shipping_forecast.api.v1.dependencies import (
    get_model,
    get_model_info,
    get_settings,
)
from shipping_forecast.api.v1.schemas import (
    PredictMetadata,
    PredictRequest,
    PredictResponse,
)
from shipping_forecast.api.v1.services import (
    map_predictions_to_response,
    resolve_cost_params,
)
from shipping_forecast.data.queries import load_panel
from shipping_forecast.models import ConformalForecaster
from shipping_forecast.pipelines.train_final_model import DATA_CUTOFF, DB_PATH

router = APIRouter(prefix="/v1", tags=["v1"])


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe.

    En 8.0 retorna siempre OK. En 8.4 chequea que el modelo esté cargado
    y retorna 503 si no lo está.
    """
    return {"status": "ok"}


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
    return PredictResponse(
        model_version=model_info["version"],
        predictions=predictions,
        metadata=PredictMetadata(
            predicted_at=datetime.datetime.now(datetime.UTC),
            n_predictions=len(predictions),
            alpha_source=alpha_source or "server_default",
            cost_ratio_source=cost_ratio_source or "server_default",
        ),
    )
