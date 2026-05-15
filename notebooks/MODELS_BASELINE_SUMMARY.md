# Baseline Models Summary

> **5 min read**. Resumen ejecutivo del notebook [`02_baseline_models.ipynb`](02_baseline_models.ipynb).
> Establece la línea base para el forecasting diario de envíos por estado brasileño,
> evaluando 3 modelos contra una métrica cost-sensitive (WAPE) en 4 contextos
> temporales distintos.

---

## TL;DR

**SeasonalNaive (predecir el valor de hace 7 días) es la línea base a vencer**:
WAPE promedio **0.44**, ganando 3 de 4 folds sin entrenamiento, sin features, sin
hiperparámetros. Prophet (0.49) solo gana en Black Friday gracias a holidays
explícitos. Naive (0.83) es operacionalmente peligroso. La principal lección:
**cualquier modelo más complejo (Fase 6: LightGBM) debe superar a SeasonalNaive en
al menos 3 de 4 folds para justificar su complejidad**.

---

## Setup experimental

**Datos**: 16,416 filas (608 días × 27 estados), 97,304 envíos totales,
periodo 2017-01-01 a 2018-08-31.

**Validación temporal**: expanding window, 4 folds elegidos para cubrir
contextos operacionales distintos:

| Fold | Periodo de test | Contexto |
|---|---|---|
| 1 | nov 2017 | Black Friday 2017 |
| 2 | feb 2018 | Steady state post-BF |
| 3 | jun 2018 | Día dos Namorados 2018 |
| 4 | jul-ago 2018 | Holdout final (steady state) |

**Métricas**: WAPE como primaria (robusta a ceros, interpretable como % de
error sobre volumen), complementada con MAE, RMSE, Bias y descomposición por
ventana de evento (±7 días).

**Modelos evaluados**:
- **Naive**: predice el último valor observado por gru
cat > notebooks/MODELS_BASELINE_SUMMARY.md <<'EOF'
# Baseline Models Summary

> **5 min read**. Resumen ejecutivo del notebook [`02_baseline_models.ipynb`](02_baseline_models.ipynb).
> Establece la línea base para el forecasting diario de envíos por estado brasileño,
> evaluando 3 modelos contra una métrica cost-sensitive (WAPE) en 4 contextos
> temporales distintos.

---

## TL;DR

**SeasonalNaive (predecir el valor de hace 7 días) es la línea base a vencer**:
WAPE promedio **0.44**, ganando 3 de 4 folds sin entrenamiento, sin features, sin
hiperparámetros. Prophet (0.49) solo gana en Black Friday gracias a holidays
explícitos. Naive (0.83) es operacionalmente peligroso. La principal lección:
**cualquier modelo más complejo (Fase 6: LightGBM) debe superar a SeasonalNaive en
al menos 3 de 4 folds para justificar su complejidad**.

---

## Setup experimental

**Datos**: 16,416 filas (608 días × 27 estados), 97,304 envíos totales,
periodo 2017-01-01 a 2018-08-31.

**Validación temporal**: expanding window, 4 folds elegidos para cubrir
contextos operacionales distintos:

| Fold | Periodo de test | Contexto |
|---|---|---|
| 1 | nov 2017 | Black Friday 2017 |
| 2 | feb 2018 | Steady state post-BF |
| 3 | jun 2018 | Día dos Namorados 2018 |
| 4 | jul-ago 2018 | Holdout final (steady state) |

**Métricas**: WAPE como primaria (robusta a ceros, interpretable como % de
error sobre volumen), complementada con MAE, RMSE, Bias y descomposición por
ventana de evento (±7 días).

**Modelos evaluados**:
- **Naive**: predice el último valor observado por grupo
- **SeasonalNaive**: predice el valor de hace 7 días (`season=7`)
- **Prophet**: con holidays brasileños explícitos (BF, Christmas, Día dos Namorados)

---

## Resultados principales

### Ranking global

| Modelo | WAPE promedio | Folds ganados | Bias | Veredicto |
|---|---|---|---|---|
| Naive | 0.83 | 0/4 | — | Descartado (WAPE > 1.0 en Navidad) |
| Prophet | 0.49 | 1/4 (BF) | +0.67 | Sobre-predice, solo brilla con holidays |
| **SeasonalNaive** | **0.44** | **3/4** | −0.48 | **Línea base a vencer** |

### WAPE por fold × modelo

| Modelo | Fold 1 (BF) | Fold 2 (steady) | Fold 3 (DdN) | Fold 4 (holdout) | Promedio |
|---|---|---|---|---|---|
| Naive | 0.753 | 0.654 | 0.929 | 0.985 | 0.83 |
| Prophet | **0.510** | 0.387 | 0.517 | 0.530 | 0.49 |
| SeasonalNaive | 0.546 | **0.364** | **0.397** | **0.456** | **0.44** |

---

## Cinco hallazgos clave

### 1. Estacionalidad semanal domina la señal modelable

Lag-7 explica la mayor parte de la varianza predecible. SeasonalNaive gana sin
ningún tipo de entrenamiento, lo cual implica que cualquier modelo que no
capture esta estacionalidad (Naive) fundamentalmente no funciona en este dominio.
Es el primer feature obligatorio para cualquier modelo de Fase 6.

### 2. Conocer un evento no es suficiente para predecirlo

Prophet conoce Black Friday como holiday explícito y aun así **empata con
SeasonalNaive en la ventana ±7 días** (0.586 vs 0.599). La ventaja agregada
de Prophet en el fold 1 viene de fuera de la ventana del evento, no de modelar
mejor el evento en sí.

**Implicancia**: holidays explícitos son necesarios pero no suficientes. El
valor real está en features de ventana (`days_to_event`, `is_post_event_peak`)
que capturen la dinámica pre/post-evento, no solo el día.

### 3. Navidad es el peor escenario para todos los modelos

WAPE en ventana ±7 días de Christmas 2017:

| Modelo | WAPE Christmas |
|---|---|
| Naive | **1.06** (errores > ventas reales) |
| Prophet | 0.63 |
| SeasonalNaive | 0.63 |

Razón: Navidad genera un periodo extendido de operación reducida (≥1 semana)
que ningún baseline captura. Es **donde las features de Fase 4 deberían marcar
la mayor diferencia** en Fase 6 (`is_christmas_shutdown_window`).

### 4. Bias direccional opuesto entre modelos

- **Prophet sobre-predice**: bias +0.67 envíos/día (overshoot sistemático)
- **SeasonalNaive sub-predice**: bias −0.48 envíos/día

Un ensemble simple (promedio o stacking) podría tener bias ≈ 0 con WAPE
intermedio o mejor. Es material directo para Fase 6.

### 5. El error está dominado por volumen, no por dificultad estructural

Análisis por estado en el fold 4 (holdout final):

- **Correlación Spearman(volumen, WAPE) = −0.867** (p < 1e-08)
- **Brecha SP → AC = 8.6×** en WAPE
- WAPE > 1.0 en estados del long tail (AC, AP, RR) **no es falla del modelo**:
  es ruido Poisson irreductible para procesos con λ ≈ 0.1 envíos/día.

Tres clusters identificados:

| Cluster | Estados | Vol. (envíos/día) | WAPE típico |
|---|---|---|---|
| Core | SP, RJ, MG, PR, RS, SC | >10 | 0.35 – 0.46 |
| Mid | BA, DF, GO, ES, PE, CE | 2 – 10 | 0.50 – 0.80 |
| Long tail | AC, AP, RR, AL, RN, MA | <2 | 0.95 – 3.00 |

**Implicancia**: la granularidad estado-día no es adecuada para ~6 estados del
long tail. Conviene agregar por región (Norte/Nordeste) para esos casos en Fase 6.

---

## Decisiones de diseño documentadas

| Decisión | Elección | Justificación |
|---|---|---|
| API de modelos | `fit(df)` + `predict(df, horizon, group_col)` → DataFrame | Homogénea entre Naive/Prophet/LGBM, fácil de testear |
| Predicción | Punto (`y_pred`) + intervalo opcional (`y_lower`, `y_upper`) | Suficiente para baseline; conformal prediction en Fase 6 |
| Métrica primaria | WAPE | Robusta a ceros, interpretable, comparable entre estados |
| Validación | Expanding window, 4 folds con contextos distintos | Cada fold cubre un régimen operacional diferente |
| Persistencia | joblib + JSON sidecar con metadata | Modelos reproducibles + auditoría de training |
| Análisis por evento | Ventana ±7 días alrededor de fecha del evento | Captura efectos pre/post sin contaminar con steady state |

---

## Hacia Fase 6: LightGBM con Optuna

### Targets concretos a superar

| Métrica | SeasonalNaive (a vencer) | Target LightGBM |
|---|---|---|
| WAPE global | 0.44 | < 0.35 |
| WAPE Christmas window | 0.63 | < 0.45 |
| WAPE estados core | 0.42 | < 0.30 |
| Bias direccional | −0.48 | abs < 0.10 |

### Features prioritarias (combinando lecciones de Fase 4 + baselines)

1. **Lag-7 + lag-14** — lo que hace fuerte a SeasonalNaive
2. **Holidays brasileños explícitos** — lo que ayuda a Prophet
3. **Features de ventana de evento** (Fase 4): `days_to_black_friday`,
   `days_to_dia_dos_namorados`, `is_post_event_peak`,
   `is_christmas_shutdown_window`
4. **Volume tier categórico** + `state_avg_volume` — permite que el modelo
   diferencie regímenes core/mid/tail
5. **Filtro `is_operational=True`** antes de modelar — los días no-operacionales
   con 0 envíos no aportan señal y degradan el WAPE

### Regla de decisión

> Un modelo de Fase 6 que no supere a SeasonalNaive en al menos 3 de 4 folds
> **no justifica su complejidad adicional** y debe ser descartado, sin importar
> qué tan sofisticada sea su arquitectura.

---

## Reproducibilidad

**Correr el notebook**:

```bash
make install                          # uv sync + pre-commit install
jupyter notebook notebooks/02_baseline_models.ipynb
```

**Tiempo de ejecución**: ~10 segundos para los 3 modelos × 4 folds (12 entrenamientos).

**Dependencias críticas**: `scikit-learn`, `prophet`, `pandas`, `pyarrow`. El
proyecto usa Python 3.11 gestionado por `uv`.

**Tests asociados**: 200 pasando, cobertura 98% global, 100% en `models/`,
`evaluation/`, `features/`. Correr con `make test`.

---

*Notebook ejecutado: holdout final jul-ago 2018. Próxima fase: LightGBM con
Optuna (tuning bayesiano) + análisis de cost-matrix asimétrica.*
