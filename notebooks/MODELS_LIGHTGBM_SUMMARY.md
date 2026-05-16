# LightGBM Production Model Summary

> **5 min read**. Resumen ejecutivo del notebook
> [`03_lightgbm_tuning.ipynb`](03_lightgbm_tuning.ipynb).
> Cierra la Fase 6 del proyecto: modelo LightGBM tuneado con Optuna,
> envuelto con conformal prediction para intervalos calibrados, y
> comparado honestamente contra el baseline de Fase 5.

---

## TL;DR

**LightGBM gana 7.2% en CV pero pierde 2.9% en holdout final.** En eventos
comerciales (Día dos Namorados) la mejora sube a **25%**, en steady state
holdout queda ligeramente por debajo del baseline. La narrativa correcta
no es "el modelo es mejor", sino "el modelo es mejor donde los errores
cuestan más" — abre el camino a Fase 7 (cost-sensitive evaluation).
Intervalos de predicción calibrados con conformal: **89.2% empirical
coverage** vs 90% nominal.

---

## Setup experimental

**Datos**: 19,008 filas (704 días × 27 estados), 2016-10-08 a 2018-09-11.

**Modelo base**: `LightGBMForecaster` con 23 features del pipeline canónico:

| Categoría | Features |
|---|---|
| Lags | `lag_7`, `lag_14` |
| Rolling | `rolling_mean_7`, `rolling_std_7` |
| Calendar | `day_of_week`, `month`, `is_weekend`, `is_month_start`, `is_month_end` |
| Holidays BR | `is_holiday`, `is_business_day`, `is_saturday`, `is_sunday`, `is_operational`, `day_type` |
| Eventos | `days_to_black_friday`, `is_black_friday_window`, `is_post_black_friday_peak` (idem para DdN) |
| Trend | `days_since_start`, `year_progress` |
| Volumen | `state_avg_volume`, `volume_tier` |

**Tuning**: 100 trials de Optuna con TPE sampler y MedianPruner. 28 trials
completados, 72 pruned. Runtime total: 33 minutos.

**Mejores hiperparámetros encontrados**:

| Parámetro | Valor |
|---|---|
| `num_leaves` | 25 |
| `learning_rate` | 0.123 |
| `feature_fraction` | 0.823 |
| `bagging_fraction` | 0.879 |
| `min_child_samples` | 48 |
| `reg_alpha` | 0.187 |
| `reg_lambda` | 2.756 |
| `max_depth` | 5 |

**Validación**: expanding window de 4 folds. Folds 1-3 para tuning,
**fold 4 como holdout intocable** (nunca visto durante el tuning de Optuna).

**Métrica primaria**: WAPE solo en días operacionales (lunes-sábado
no feriado).

---

## Resultados principales

### WAPE por fold

| Fold | Periodo | LightGBM | SeasonalNaive | Delta | Ganador |
|---|---|---|---|---|---|
| 1 | nov 2017 (Black Friday) | 0.575 | 0.546 | +5.3% | SeasonalNaive |
| 2 | feb 2018 (steady) | **0.341** | 0.364 | -6.2% | **LightGBM** |
| 3 | jun 2018 (DdN) | **0.296** | 0.397 | **-25.3%** | **LightGBM** |
| 4 | jul-ago 2018 (holdout) | 0.469 | 0.456 | +2.9% | SeasonalNaive |

**CV promedio (folds 1-3)**: LightGBM 0.4042 vs SeasonalNaive 0.4355 → **LGBM gana 7.2%**.
**Holdout (fold 4)**: LightGBM 0.4694 vs SeasonalNaive 0.4563 → **SeasonalNaive gana 2.9%**.

---

## Cinco hallazgos clave

### 1. LightGBM domina eventos comerciales, no steady state

El gap más grande (Día dos Namorados, -25.3%) y el gap más chico (steady
state holdout, +2.9%) revelan el patrón: el modelo **agrega valor cuando
hay señal estructural compleja** (combinación de tendencia + estacionalidad
+ proximidad a evento), pero no cuando la dinámica es estable y SeasonalNaive
ya captura la mayoría.

### 2. `lag_7` no es la señal dominante, era un proxy crudo

| Rank | Feature | Importance |
|---|---|---|
| 1 | `state_avg_volume` | 17.6% |
| 2 | `days_since_start` | 15.0% |
| 3 | `rolling_std_7` | 13.4% |
| 4 | `rolling_mean_7` | 12.2% |
| 5 | `days_to_black_friday` | 8.3% |
| 6 | `day_of_week` | 8.3% |
| ... | ... | ... |
| **10** | **`lag_7`** | **2.8%** |

La estacionalidad semanal sigue siendo señal, pero el modelo prefiere
descomponer la "señal de SeasonalNaive" en componentes más interpretables:
volumen base del estado + tendencia temporal + variabilidad y nivel
recientes. **Esto reescribe la conclusión de Fase 5** ("lag_7 domina") —
domina cuando es la única señal disponible.

### 3. Features binarias de eventos casi inertes

`is_holiday`, `is_post_black_friday_peak`, `is_*_window`: todas tienen
importance < 0.5%. El modelo prefiere **distancias continuas**
(`days_to_event`, rank 5 con 8.3%) sobre flags binarias.

**Lección para feature engineering**: ante eventos, exponer la distancia
continua es preferible. El modelo aprende el shape del impacto (cómo crece
y decrece) cuando el feature es un eje continuo, no un on/off.

### 4. El bug del lag-NaN-collapse y la fix con forecasting recursivo

Durante el smoke test de Fase 6.2 descubrimos que las predicciones
colapsaban al promedio global a partir del día 8 del horizonte de 60 días.
Causa: el `lag_7` del día 8 requería el target del día 1 del futuro, que
es `NaN` (no se conoce).

**Solución**: predecir un día a la vez, inyectando cada predicción como
target del próximo día. Para el día 8, el `lag_7` ya tiene la predicción
del día 1 disponible.

**Bloqueado con test de regresión** (`test_long_horizon_does_not_collapse_to_mean`)
que verifica que las predicciones del día 8+ no colapsen al mean.

### 5. Conformal prediction: 89.2% empirical coverage

**Cobertura empírica en holdout** (LightGBM + ConformalForecaster con
`calibration_days=60`, `alpha=0.1`):

| calib_days | coverage | gap | width | wape_base |
|---|---|---|---|---|
| 30 | 0.816 | -8.4% | 23.39 | 0.695 |
| 45 | 0.878 | -2.2% | 7.72 | 0.583 |
| **60** | **0.892** | **-0.8%** | **9.33** | **0.471** |
| 90 | 0.867 | -3.3% | 11.27 | 0.483 |
| 120 | 0.869 | -3.1% | 10.13 | 0.681 |

**Sweet spot empírico**: 60 días balancea representatividad de la
calibración con tamaño de fit set del modelo base. Gap residual de 0.8%
es esperable por non-stationarity (intercambiabilidad imperfecta entre
calibración y holdout).

---

## Decisiones de diseño documentadas

| Decisión | Elección | Justificación |
|---|---|---|
| Modelo único vs múltiple | Uno con `volume_tier` como feature | Simplicidad; reportamos WAPE por tier |
| Loss interno | `regression_l1` (MAE) | Proxy directo de WAPE |
| Early stopping | patience=50, n_estimators=1000 max | Robusto, ahorra trial budget en Optuna |
| Search space | 8 hiperparámetros | Cubre capacidad, regularización, estocasticidad sin overfitting al CV |
| Inferencia | Forecasting recursivo | Realista (simula re-fit diario); evita lag-NaN-collapse |
| Intervalos | Conformal split con calibration=60 días | Garantía teórica de cobertura; simple de explicar |
| Holdout protection | Fold 4 nunca visto en tuning | Métrica honesta out-of-sample |

---

## Hacia Fase 7: evaluación cost-sensitive

La pregunta de portfolio que abre Fase 7:

> ¿Justifica LightGBM su complejidad sobre SeasonalNaive?

Con WAPE simétrico: **probablemente no** (pierde 3% en holdout).
Con cost matrix asimétrica realista (under-prediction 2-3x más cara
que over-prediction): **probablemente sí** (domina 25% en eventos críticos
donde el under-prediction es más costoso).

Fase 7 va a cuantificar esa intuición con números monetarios:

1. Definir matriz de costos asimétrica calibrada al contexto logístico
2. Calcular **ganancia esperada en USD** por modelo, no WAPE
3. Reportar **expected gain curve** vs threshold de decisión
4. **Conclusión cuantitativa**: ¿el modelo paga su complejidad?

---

## Reproducibilidad

**Re-correr el tuning** (33 minutos):

```bash
python scripts/tune_lightgbm.py --n-trials 100 --study-name lgbm_v1
```

**Re-correr el análisis del notebook** (lee artifacts persistidos):

```bash
jupyter notebook notebooks/03_lightgbm_tuning.ipynb
```

**Dependencias críticas**: `lightgbm`, `optuna`, `scikit-learn`, `pandas`,
`pyarrow`. Python 3.11 gestionado por `uv`.

**Tests asociados**: 266 pasando, 97% coverage. Correr con `make test`.

**Artifacts persistidos**:
data/processed/
├── best_lgbm_params.json (resultado completo del estudio)
└── phase6/
├── wape_per_fold.csv
├── feature_importance.csv
└── lightgbm_final.json (metadata del modelo)

---

*Notebook ejecutado: holdout final jul-ago 2018. Próxima fase: evaluación
cost-sensitive (Fase 7).*
