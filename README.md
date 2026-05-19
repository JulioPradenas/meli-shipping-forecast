# Sistema de Forecasting de Demanda de Envíos

[![CI](https://github.com/JulioPradenas/meli-shipping-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/JulioPradenas/meli-shipping-forecast/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sistema end-to-end de forecasting de demanda de envíos para e-commerce LATAM, orientado a la planificación de capacidad operacional.

## Problema

Las plataformas de e-commerce en LATAM operan redes logísticas continentales. La planificación de capacidad operacional (personal de bodega, flota, espacio en centros de distribución) depende críticamente de predecir cuántos paquetes se procesarán por región y por día.

- **Sub-predecir** → colapso operacional, atrasos, mala experiencia del cliente
- **Sobre-predecir** → costo operacional innecesario (personal ocioso, flota subutilizada)

## Objetivo

Dado el registro histórico de envíos por región, predecir el volumen diario de paquetes por estado con un horizonte de 7 a 90 días, con suficiente precisión para informar decisiones de capacidad. El sistema va más allá de la precisión promedio: contempla explícitamente el costo asimétrico de sub-predecir vs sobre-predecir.

## Aspectos destacados

- **285 tests passing**, CI verde, codebase modular OOP distribuido en 39 archivos fuente.
- **Evaluación cost-sensitive**: se detectó que la precisión promedio (WAPE) es engañosa bajo costos asimétricos. Se construyó un pipeline completo de análisis costo-vs-tuning que cuantifica la ganancia en dólares a lo largo de distintos ratios de costo (1x a 5x).
- **Auditoría de rigor estadístico**: se identificó un leakage sutil de `eval_set` en el WAPE de holdout reportado en Fase 6 (0.4694). Se re-evaluó honestamente sin ese atajo (0.5156) y la trazabilidad quedó documentada explícitamente en el metadata sidecar del modelo.
- **Forecasting recursivo**: el modelo LightGBM maneja correctamente las features de lag en horizontes largos inyectando las predicciones de vuelta en el DataFrame de trabajo, evitando el clásico bug de colapso al promedio.

## Stack técnico

- **Lenguaje**: Python 3.11
- **Gestor de dependencias**: uv
- **Datos**: pandas, SQLAlchemy, pyarrow
- **ML**: scikit-learn, LightGBM, statsmodels, Prophet
- **Tuning de hiperparámetros**: Optuna (TPE + MedianPruner)
- **Interpretabilidad**: SHAP
- **Tracking de experimentos**: MLflow
- **API**: FastAPI (schemas Pydantic v2, carga del modelo gestionada por lifespan)
- **Dashboard**: Streamlit (planificado)
- **Infraestructura**: Docker, GitHub Actions
- **Calidad**: pytest, ruff, mypy, pre-commit

## Estructura del proyecto
meli-shipping-forecast/
├── src/shipping_forecast/
│   ├── api/                 # Servicio FastAPI (Fase 8)
│   │   └── v1/              # Endpoints versionados + schemas Pydantic
│   ├── config/              # Configuración (Pydantic Settings)
│   ├── data/                # Ingesta de datos, queries y loader SQLite
│   ├── evaluation/          # Métricas, cross-validation, cost-sensitive
│   ├── features/            # Feature engineering OOP (lags, calendar, eventos)
│   ├── models/              # SeasonalNaive, LightGBM, Conformal, Prophet
│   ├── pipelines/           # Entrenamiento final con holdout honesto
│   └── utils/               # Logging y helpers
├── tests/                   # 285 tests unitarios + integración
├── notebooks/               # EDA + tuning + análisis cost-sensitive
├── sql/                     # Scripts DDL y queries analíticas
├── scripts/                 # Tuning con Optuna, carga inicial, análisis batch
└── .github/workflows/       # CI/CD

## Setup

### Prerequisitos

- Python 3.11
- [uv](https://github.com/astral-sh/uv) (gestor moderno de dependencias)
- Usuarios de macOS: `brew install libomp` (requerido por LightGBM)

### Instalación

```bash
git clone https://github.com/JulioPradenas/meli-shipping-forecast.git
cd meli-shipping-forecast
uv venv --python 3.11
source .venv/bin/activate
uv sync --all-extras
```

### Comandos comunes

```bash
make test          # Correr tests con coverage
make lint          # Ruff linter (sin auto-fix)
make typecheck     # Mypy
make check         # Todas las quality checks (lint + typecheck + test)
make fix           # Auto-formatear código + correr todas las checks
make train-model   # Entrenar y persistir el modelo LightGBM final
```

## Estado del proyecto

### Fases completadas

- [x] **Fase 1**: Setup del proyecto, tooling, CI/CD
- [x] **Fase 2**: Ingesta de datos del dataset Olist a SQLite, SQL avanzado (window functions para lags/rolling stats)
- [x] **Fase 3**: EDA temporal (estacionalidad, eventos, anomalías por estado)
- [x] **Fase 4**: Feature engineering en OOP (LagFeatures, CalendarFeatures, EventFeatures, etc., pipeline composable)
- [x] **Fase 5**: Modelos baseline (Naive, SeasonalNaive con WAPE=0.4355 promedio en CV folds 1-3)
- [x] **Fase 6**: Modelo LightGBM tuneado con Optuna (100 trials, 8 hiperparámetros, MedianPruner). WAPE CV mean = 0.4042
- [x] **Fase 7**: Evaluación cost-sensitive. Pipeline batch que computa ganancia esperada bajo costos asimétricos para 4 cost ratios y 4 folds. Notebook con análisis y 3 plots
- [x] **Fase 8 (parcial)**: API productiva con FastAPI

### Fase 8 — En progreso

- [x] **8.0**: Scaffolding del servicio FastAPI con endpoint `/v1/health`
- [x] **8.1**: Pipeline `make train-model` con doble modelo (evaluación + producción) y holdout honesto sin leakage de `eval_set`. WAPE honesto = 0.5156
- [x] **8.2**: Schemas Pydantic v2 para el endpoint `/v1/predict` (PredictRequest, Prediction, PredictMetadata, PredictResponse, ModelInfoResponse) con validators de horizonte ≤90 días, alpha en [-2, +2], cost_ratio en [0.5, 10]
- [ ] **8.3**: Implementación del endpoint `/v1/predict` con resolución de alpha/cost_ratio y conformal intervals
- [ ] **8.4**: Endpoints `/v1/health` y `/v1/model/info` conectados al modelo cargado vía lifespan
- [ ] **8.5**: Logging estructurado con structlog + middleware de request_id
- [ ] **8.6**: Tests del API con modelo mock (corren en cada PR)
- [ ] **8.7**: Tests E2E con modelo real (job nightly separado de CI)
- [ ] **8.8**: Notebook de ejemplo de uso del API
- [ ] **8.9**: README de la API + cierre de Fase 8

### Fases pendientes

- [ ] **Fase 9**: Dashboard Streamlit + deploy en Streamlit Cloud

## Métricas clave del modelo actual

Modelo en producción: **LightGBM tuneado con Optuna**, 23 features, 27 estados brasileños.

| Métrica | Valor | Notas |
|---|---|---|
| WAPE (holdout 2018-07 a 2018-08, 62 días) | 0.5156 | Strict, sin leakage de `eval_set` |
| MAE (mismo holdout) | 4.04 | Mean Absolute Error en unidades de paquetes/día |
| RMSE (mismo holdout) | 14.22 | Root Mean Squared Error |
| CV WAPE (folds 1-3) | 0.4042 | Usado por Optuna durante tuning |
| Best Optuna trial | #36 de 100 | TPE sampler con MedianPruner |

El modelo se persiste con metadata trazable: timestamp UTC de entrenamiento, fechas del holdout, métricas honestas, y nota explicando la metodología sin atajos estadísticos.

## Licencia

MIT
