"""FastAPI app para el servicio de forecasting de shipping demand.

Punto de entrada del servicio. La carga del modelo se hace eager al
startup vía lifespan context manager (se completa en 8.4).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from shipping_forecast.api.v1.router import router as v1_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carga el modelo al startup. Placeholder hasta 8.4."""
    # TODO 8.4: cargar modelo de disco o reentrenar si no existe
    # app.state.model = load_model_from_disk_or_retrain(settings)
    yield
    # cleanup si hace falta


app = FastAPI(
    title="MELI Shipping Forecast API",
    description="Forecasting de demanda de envíos LATAM con cost-aware predictions",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(v1_router)
