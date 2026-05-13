# EDA Summary — Análisis Temporal de Envíos

> Resumen ejecutivo de la Fase 3 del proyecto. Lectura completa en ~5 minutos.

---

## Contexto

Una plataforma de e-commerce LATAM necesita predecir el volumen diario de envíos por región para planificar la capacidad operacional (bodega, flota, distribución). Sub-predecir genera retrasos; sobre-predecir genera costo operativo innecesario.

Este documento resume los hallazgos del análisis temporal exploratorio que informa el diseño del modelo predictivo.

**Notebook fuente**: [`01_eda_temporal.ipynb`](./01_eda_temporal.ipynb)

---

## Dataset

| Atributo | Valor |
|---|---|
| Fuente | Brazilian E-Commerce Public Dataset by Olist (Kaggle) |
| Cobertura original | 2016-10-08 a 2018-09-11 (704 días) |
| Cobertura modelable | **2017-01-01 a 2018-08-31 (608 días, 99.7% del volumen)** |
| Granularidad | 1 fila por (día, estado) |
| Tabla principal | `fact_daily_shipments_by_state` (19,008 filas) |
| Estados | 27 (todos los brasileños) |
| Definición de envío | `order_status ∈ {delivered, shipped}` y `order_delivered_carrier_date IS NOT NULL` |

---

## Hallazgos clave

### 1. El dataset tiene un período de "ramp-up" descartable

Octubre-diciembre 2016 y septiembre 2018 tienen densidad insuficiente (242, 32, 2 envíos mensuales respectivamente al inicio). El periodo modelable se acota a 2017-01-01 → 2018-08-31, reteniendo el 99.7% del volumen total con 608 días útiles.

### 2. Domingos = 0 envíos por diseño operacional

De los 87 domingos del período modelable, **85 tienen exactamente 0 envíos**. No es ausencia de demanda, es ausencia de operación: bodegas y transportistas no operan domingos en Brasil. **El modelo debe respetar esta restricción** (forzar 0 en predicciones, no predecir libremente).

### 3. Calendario operacional brasileño ≠ calendario de la librería `holidays`

La librería `holidays` solo cubre festivos federales obligatorios. Pero **Carnaval (lunes y martes antes de Miércoles de Ceniza) y Corpus Christi (jueves después de Pascua) son días no operativos** aunque sean facultativos. Se calculan vía el algoritmo Anonymous Gregorian para Pascua, sin agregar dependencias.

### 4. Sábado es operacionalmente distinto

| Día | Mediana de envíos |
|---|---|
| Lunes-Viernes | 200+ |
| **Sábado** | **8** |

Sábado funciona al ~8% del volumen de un día hábil. Se trata como **una cuarta categoría** (`saturday`), no como día hábil normal. Esto evita que el modelo aprenda "un día anómalo cada semana" en sus lags.

### 5. Patrón día-de-la-semana decreciente

| Día | Promedio | % vs peak |
|---|---|---|
| Lunes | 241.5 | 93% |
| **Martes** | **258.9** | **100%** |
| Miércoles | 229.8 | 89% |
| Jueves | 209.1 | 81% |
| Viernes | 202.6 | 78% |
| Sábado | 20.3 | 8% |

El lunes/martes procesa el inventario acumulado del fin de semana. El stock disminuye hacia el viernes. **`day_of_week` es feature obligatorio.**

### 6. Crecimiento ~13x en días hábiles durante 19 meses

Lunes promedio: 31 envíos (ene-17) → 406 envíos (ago-18). El crecimiento "limpio" en días operativos es de 13x. La tendencia es **no lineal** con un cambio de régimen abrupto en noviembre 2017.

### 7. Black Friday 2017 cambia el régimen permanentemente

Antes de BF 2017: la serie oscilaba entre 100-280 envíos diarios. Después: se estabiliza en 380-450, sin retorno al nivel anterior. **Modelos lineales (ARIMA simple) sin variable de cambio estructural van a fallar en el post-BF.**

### 8. El pico operativo es 3-4 días DESPUÉS del evento

| Fecha | Día | Envíos | Notas |
|---|---|---|---|
| 2017-11-24 | Viernes (BF) | ~325 | Día de la **venta**, no del **envío** |
| 2017-11-27 | Lunes | ~670 | Inicio del pico operativo |
| **2017-11-28** | **Martes** | **~707** | **Pico absoluto del dataset** |
| 2017-11-29 | Miércoles | ~567 | Decaimiento |

El cliente compra el viernes 24, pero el paquete sale de bodega el lunes 27 / martes 28. La definición operativa (`delivered_carrier_date`) captura cuándo se procesa el envío, no cuándo se vende. **Feature relevante**: `days_since_event`, no solo `is_event_day`.

### 9. Junio tiene un patrón propio (Día dos Namorados)

En junio de ambos años (2017 y 2018), el patrón se invierte: **lunes >> martes**. Coincide con el Día dos Namorados (12 de junio), segundo evento comercial más importante de Brasil después de Black Friday. Requiere feature explícita.

---

## Decisiones de diseño

### Periodo modelable

- **Train+validation**: 2017-01-01 → 2018-08-31 (608 días)
- **Días operacionales**: 503 hábiles + 85 sábados = 588 días con operación
- **Días forzados a 0**: 87 domingos + 18 feriados = 105 días sin operación

### Clasificación de tipos de día (4 categorías)

| Categoría | Días | Mediana | Tratamiento en modelo |
|---|---|---|---|
| `business_day` | 503 | 207.5 | Predicción normal |
| `saturday` | 85 | 8.0 | Predicción restringida |
| `holiday` | 18 | 0.0 | Forzar a 0 |
| `sunday` | 87 | 0.0 | Forzar a 0 |

### Definiciones operativas

| Término | Definición |
|---|---|
| **Envío** | Pedido con `order_status ∈ {delivered, shipped}` y `order_delivered_carrier_date IS NOT NULL` |
| **Día hábil** | Lunes-viernes que no sea feriado nacional, Carnaval ni Corpus Christi |
| **Día operativo** | Día hábil o sábado |
| **Periodo modelable** | 2017-01-01 a 2018-08-31 |

---

## Implicancias para modelado

### Features sugeridas (a implementar en Fase 4)

| Familia | Features | Justificación |
|---|---|---|
| Calendario | `day_of_week`, `month`, `day_type`, `is_business_day`, `is_saturday` | Captura estacionalidad semanal y mensual |
| Lags | `lag_1`, `lag_7`, `lag_14`, `lag_28` (ya en SQL) | `lag_7` será el predictor dominante |
| Rolling | `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_28`, `rolling_std_7` (ya en SQL) | Tendencia local |
| Tendencia | `days_since_start`, `month_index` | Captura el crecimiento 13x no lineal |
| Eventos | `is_black_friday_window`, `is_dia_dos_namorados_window`, `days_to_black_friday`, `is_post_event_peak` | Captura picos operativos pre/post eventos |

### Familia de modelos recomendada

**LightGBM con features categóricas** es el candidato principal por tres razones:

1. **Maneja no-linealidades**: el cambio de régimen post-BF lo absorbe naturalmente
2. **Soporta categóricas nativamente**: `day_type`, `month` sin one-hot
3. **Maneja interacciones**: ej. `month=11 × day_of_week=tuesday` (martes post-BF)

Modelos secundarios para baseline:
- **Seasonal Naive (lag-7)**: piso mínimo
- **Prophet con regressores**: comparativa interpretable

### Validación

- **No `train_test_split` aleatorio**: splits cronológicos obligatorios
- **Time series cross-validation con ventanas crecientes**: 4-5 folds
- **Test set holdout final**: 2018-07-01 → 2018-08-31 (último 10% temporal)

### Métricas

| Métrica | Uso |
|---|---|
| **WAPE** | Métrica principal, robusta a ceros |
| **MAE en días hábiles** | Interpretabilidad operacional |
| **RMSE** | Penaliza errores grandes (críticos en operación logística) |
| **% de error en eventos** | WAPE solo en ventanas de Black Friday / Día dos Namorados |

---

## Próximos pasos

| Fase | Entregable |
|---|---|
| **4** | `FeatureBuilder` POO con todas las families de features |
| **5** | Baselines (Naive, Seasonal Naive, Prophet) |
| **6** | LightGBM con Optuna |
| **7** | Evaluación con análisis de errores por segmento |
| **8** | API REST con FastAPI |
| **9** | Dashboard Streamlit + deployment |

---

*Última actualización: Mayo 2026 — Julio Pradenas*
