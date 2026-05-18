"""Router del API v1.

En 8.0 solo expone /health. Los endpoints /predict y /model/info se agregan
en 8.3 y 8.4 respectivamente.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/v1", tags=["v1"])


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe.

    En 8.0 retorna siempre OK. En 8.4 chequea que el modelo esté cargado
    y retorna 503 si no lo está.
    """
    return {"status": "ok"}
