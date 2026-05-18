"""Configuración del servicio API leída de variables de entorno.

Variables soportadas (prefijo MELI_API_):
  - MELI_API_DEFAULT_ALPHA: shift aplicado al modo cost-aware (default 0.65)
  - MELI_API_DEFAULT_COST_RATIO: ratio de costos under/over (default 3.0)
  - MELI_API_MODEL_PATH: ruta al joblib del modelo
  - MELI_API_FAST_RETRAIN: si True, reentrena con sample reducido (para CI)
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración del servicio.

    Los defaults reflejan los hallazgos de Fase 7:
    - alpha=0.65 es el óptimo en steady state con cost_ratio=3x
    - cost_ratio=3.0 es el ratio business típico
    """

    default_alpha: float = Field(default=0.65, ge=-2.0, le=2.0)
    default_cost_ratio: float = Field(default=3.0, ge=0.5, le=10.0)
    model_path: str = "artifacts/lightgbm_final.joblib"
    fast_retrain: bool = False

    model_config = SettingsConfigDict(
        env_prefix="MELI_API_",
        env_file=".env",
        extra="ignore",
    )


def get_settings() -> Settings:
    """Factory para inyección de dependencias en FastAPI.

    No es singleton: cada llamada lee env vars frescas. Tests pueden
    sobrescribir con app.dependency_overrides[get_settings].
    """
    return Settings()
