"""Business logic for v1 endpoints.

This module is for pure functions that the router calls to do work that
is not just HTTP plumbing or dependency injection. Examples:

  - Resolving cost-aware parameters from request + Settings (each value
    can come from either source; we track provenance for traceability).
  - Mapping the wrapper's prediction DataFrame to a list of Prediction
    schemas (transformations between internal representations and the
    API contract live here).

Keeping this logic out of router.py means we can unit-test it with
plain function calls instead of spinning up a TestClient.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from shipping_forecast.api.settings import Settings
from shipping_forecast.api.v1.schemas import Prediction, PredictRequest

ParamSource = Literal["request", "server_default"]


def resolve_cost_params(
    request: PredictRequest,
    settings: Settings,
) -> tuple[float | None, ParamSource | None, float | None, ParamSource | None]:
    """Resolve alpha and cost_ratio from request + Settings + flags.

    The resolution honors three things:
      1. If the request set the value explicitly, use it (source="request").
      2. Else, use the Settings default (source="server_default").
      3. If include_cost_aware is False, both alpha and cost_ratio are
         set to None (and their sources to None) because they are not
         meaningfully applied. Reporting a default that was never used
         would be misleading.

    Args:
        request: The validated PredictRequest from the endpoint.
        settings: The application Settings instance.

    Returns:
        A 4-tuple ``(alpha, alpha_source, cost_ratio, cost_ratio_source)``.
        When ``include_cost_aware`` is False, all four are None.
        When True, alpha and cost_ratio are always non-None floats, and
        their sources are always non-None ParamSource literals.
    """
    if not request.include_cost_aware:
        return None, None, None, None

    if request.alpha is not None:
        alpha: float = request.alpha
        alpha_source: ParamSource = "request"
    else:
        alpha = settings.default_alpha
        alpha_source = "server_default"

    if request.cost_ratio is not None:
        cost_ratio: float = request.cost_ratio
        cost_ratio_source: ParamSource = "request"
    else:
        cost_ratio = settings.default_cost_ratio
        cost_ratio_source = "server_default"

    return alpha, alpha_source, cost_ratio, cost_ratio_source


def map_predictions_to_response(
    df: pd.DataFrame,
    states_filter: list[str] | None,
    include_intervals: bool,
    include_cost_aware: bool,
    alpha: float | None,
    cost_ratio: float | None,
) -> list[Prediction]:
    """Convert the wrapper's prediction DataFrame to a list of Prediction schemas.

    Applies four transformations:
      1. Optional filter to a subset of states.
      2. Optional inclusion of conformal bounds (lower_90, upper_90).
      3. Optional inclusion of cost-aware ``recommended`` value, computed
         as ``y_pred + alpha * asymmetric_margin``. The margin is the
         distance from y_pred to the bound on the side of the shift:
            - alpha > 0 (penalize under-prediction): margin = y_upper - y_pred
            - alpha < 0 (penalize over-prediction): margin = y_pred - y_lower
            - alpha == 0: recommended == y_pred (no shift).
         This honors the asymmetric nature of the conformal interval.
      4. Carries alpha/cost_ratio to the alpha_used/cost_ratio_used fields,
         keeping them None when include_cost_aware is False.

    Args:
        df: DataFrame from ``model.predict()`` with columns
            ``[shipment_date, customer_state, y_pred, y_lower, y_upper]``.
        states_filter: If non-empty, only rows whose customer_state is in
            this list are kept. If None or empty, all rows pass through.
        include_intervals: When False, lower_90/upper_90 are set to None
            in the output.
        include_cost_aware: When False, recommended/alpha_used/cost_ratio_used
            are set to None.
        alpha: The resolved alpha. Only used when include_cost_aware=True.
        cost_ratio: The resolved cost_ratio. Only carried for reporting.

    Returns:
        A list of Prediction schemas, one per (date, state) row, in the
        same order as the input DataFrame.
    """
    if states_filter:
        df = df[df["customer_state"].isin(states_filter)].copy()

    predictions: list[Prediction] = []
    for record in df.to_dict(orient="records"):
        y_pred = float(record["y_pred"])
        y_lower = float(record["y_lower"])
        y_upper = float(record["y_upper"])

        if include_cost_aware and alpha is not None:
            margin = y_upper - y_pred if alpha >= 0 else y_pred - y_lower
            recommended: float | None = max(0.0, y_pred + alpha * margin)
        else:
            recommended = None

        ship_date = record["shipment_date"]
        if hasattr(ship_date, "date"):
            ship_date = ship_date.date()

        predictions.append(
            Prediction(
                date=ship_date,
                state=str(record["customer_state"]),
                point=y_pred,
                lower_90=y_lower if include_intervals else None,
                upper_90=y_upper if include_intervals else None,
                recommended=recommended,
                alpha_used=alpha if include_cost_aware else None,
                cost_ratio_used=cost_ratio if include_cost_aware else None,
            )
        )
    return predictions
