"""FastAPI app for the shipping demand forecasting service.

Service entrypoint. The lifespan loads the persisted model and metadata
at startup so the /v1/predict endpoint can serve requests without
re-training on every request. Two modes:

  1. Default (MELI_API_AUTO_TRAIN=false): the joblib at the configured
     path must exist or startup fails fast with a clear message.
  2. CI/dev (MELI_API_AUTO_TRAIN=true): if the joblib does not exist,
     the lifespan invokes the training pipeline (with --fast-retrain to
     keep CI under 30s) and then loads the freshly produced artifacts.

After successful load, app.state has:
  - app.state.model: the ConformalForecaster instance.
  - app.state.model_info: the parsed JSON sidecar dict.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI

from shipping_forecast.api.settings import get_settings
from shipping_forecast.api.v1.router import router as v1_router
from shipping_forecast.models import ConformalForecaster

logger = logging.getLogger(__name__)


def _load_artifacts(model_path: Path) -> tuple[ConformalForecaster, dict]:
    """Load the joblib and its JSON sidecar from disk.

    Both files are expected to coexist: ``model_path`` for the pickled
    wrapper, and ``model_path.with_suffix('.json')`` for the metadata.
    """
    sidecar_path = model_path.with_suffix(".json")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model joblib not found at {model_path}. "
            f"Run `make train-model` to generate it, or set "
            f"MELI_API_AUTO_TRAIN=true to have the lifespan train one."
        )
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"Model JSON sidecar not found at {sidecar_path}. "
            f"The joblib at {model_path} appears orphaned; regenerate both with `make train-model`."
        )

    model = joblib.load(model_path)
    if not isinstance(model, ConformalForecaster):
        raise TypeError(
            f"Loaded model is {type(model).__name__}, expected ConformalForecaster. "
            f"Regenerate with `make train-model` (Phase 8.2.5+)."
        )
    with open(sidecar_path) as f:
        model_info = json.load(f)
    return model, model_info


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model eagerly at startup; clean nothing on shutdown.

    Failure modes:
      - joblib missing + auto_train=False: FileNotFoundError, service does
        not start (visible at deploy time, not at first request).
      - joblib missing + auto_train=True: invoke train_final_model.main
        with --fast-retrain and then load.
      - joblib exists but is not a ConformalForecaster (legacy 8.1 artifact):
        TypeError, asks the user to re-run training.
    """
    settings = get_settings()
    model_path = Path(settings.model_path)

    if not model_path.exists() and settings.auto_train:
        logger.warning(
            "Model joblib missing at %s; auto_train=True, invoking training pipeline.",
            model_path,
        )
        from shipping_forecast.pipelines.train_final_model import main as train_main

        exit_code = train_main(["--fast-retrain", "--output-dir", str(model_path.parent)])
        if exit_code != 0:
            raise RuntimeError(f"Auto-train failed with exit code {exit_code}.")

    model, model_info = _load_artifacts(model_path)
    logger.info(
        "Model loaded: %s wrapping %s, last_train_date=%s, n_groups=%d, version=%s",
        type(model).__name__,
        type(model.base_model).__name__,
        model_info["last_train_date"],
        model_info["n_groups"],
        model_info["version"],
    )

    app.state.model = model
    app.state.model_info = model_info

    yield

    # No cleanup needed: joblib and dict GC normally when the process exits.


app = FastAPI(
    title="MELI Shipping Forecast API",
    description="Forecasting de demanda de envíos LATAM con cost-aware predictions",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(v1_router)
