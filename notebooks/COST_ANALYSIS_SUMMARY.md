# Cost-Sensitive Evaluation Summary

> **5 min read**. Resumen ejecutivo del notebook
> [`04_cost_analysis.ipynb`](04_cost_analysis.ipynb).
> Cierra la Fase 7 del proyecto: evaluación de LightGBM bajo costos
> asimétricos realistas y descubrimiento del valor del threshold tuning.

---

## TL;DR

WAPE simétrico decía que LightGBM **no justifica su complejidad** (pierde
2.9% en holdout vs SeasonalNaive). Pero WAPE asume que sub-predecir y
sobre-predecir cuestan lo mismo, lo cual es falso en logística.

**Con costos asimétricos realistas** (`c_under = 3x c_over`):

- **Sin tuning**: LightGBM pierde en **0/4 folds** ($-9,203 USD total)
- **Con threshold tuning** (`y_pred * (1 + α)` óptimo): LightGBM gana en
  **4/4 folds** ($14,791 USD ahorrados)
- **A cost ratio 5x**: el uplift total del tuning es **$52,653 USD**

**Conclusión**: el modelo justifica su complejidad cuando los costos son
asimétricos y se aplica un ajuste cost-aware de un solo parámetro.

---

## Setup experimental

**Datos**: 8,208 predicciones (4 folds × ~60 días × 27 estados).

**Modelos comparados**:
- LightGBM tuneado (Fase 6.3)
- SeasonalNaive (baseline)

**Configuración de costos**:
- `c_over = 1.0` USD/envío fijo
- `c_under ∈ {1.0, 2.0, 3.0, 5.0}` USD/envío
- Cost ratio = `c_under / c_over` ∈ {1x, 2x, 3x, 5x}

**Métrica primaria**:
expected_gain = costo_total(SeasonalNaive) - costo_total(LightGBM)
Positivo = LightGBM ahorra dinero. Negativo = LightGBM cuesta más.

**Threshold tuning**:
- Grid de `α ∈ [-0.20, +1.00]` con 18 valores
- Para cada (fold, cost_ratio), busca el `α` que minimiza el costo
- Predicción ajustada: `y_pred_tuned = max(y_pred * (1 + α), 0)`

---

## Resultados principales

### Expected gain por (fold, cost_ratio), con threshold tuning

| Fold | Contexto | 1x | 2x | 3x | 5x |
|---|---|---|---|---|---|
| 1 | Black Friday 2017 | +$1,664 | +$6,342 | **+$11,880** | **+$23,233** |
| 2 | Steady state | -$370 | +$174 | **+$1,854** | **+$6,539** |
| 3 | Día dos Namorados 2018 | +$298 | +$31 | +$349 | +$1,725 |
| 4 | Holdout jul-ago 2018 | -$639 | -$454 | **+$708** | **+$4,347** |
| **Total** | | **+$953** | **+$6,093** | **+$14,791** | **+$35,844** |

**Sin tuning** (la misma tabla):

| Fold | 1x | 2x | 3x | 5x |
|---|---|---|---|---|
| 1 (BF) | -$153 | -$455 | -$757 | -$1,361 |
| 2 (steady) | -$729 | -$2,615 | -$4,502 | -$8,274 |
| 3 (DdN) | +$64 | +$31 | -$2 | -$67 |
| 4 (holdout) | -$778 | -$2,360 | -$3,942 | -$7,105 |
| **Total** | **-$1,596** | **-$5,399** | **-$9,203** | **-$16,807** |

**Uplift del threshold tuning** (con − sin):

| Cost ratio | Uplift total |
|---|---|
| 1x | $2,549 |
| 2x | $11,492 |
| 3x | $23,993 |
| **5x** | **$52,653** |

---

## Cinco hallazgos clave

### 1. La hipótesis inicial estaba equivocada

Inicialmente esperábamos que costos asimétricos favorecieran al modelo
más sofisticado. **Pasó lo opuesto**: sin tuning, LightGBM pierde peor
cuanto más asimétrico es el costo, porque su sesgo a sub-predecir se
penaliza más cuando `c_under` crece.

### 2. El threshold tuning lo cambia todo

Un ajuste de un solo parámetro escalar (`α`) genera un cambio **cualitativo**
en el ranking. A cost ratio 3x, el modelo pasa de perder en 4/4 folds a
ganar en 4/4 folds.

Para portfolio, esto es **el insight clave**: a veces el modelo no necesita
ser cambiado, sólo **post-procesado** con conocimiento del costo del negocio.

### 3. El `α` óptimo varía dramáticamente por contexto

| Fold | α óptimo (ratio 3x) | Interpretación |
|---|---|---|
| Black Friday | **+1.00** | Sub-predice 100%, modelo no aprendió el shape |
| Steady state | +0.60 | Sub-predice moderadamente |
| Día dos Namorados | 0.00 | Ya está calibrado, no necesita ajuste |
| Holdout | +0.70 | Sub-predice |

**Esto cambia el deployment**: el `α` debería ser configurable en producción
(parámetro del request, o default por contexto detectado), no hardcodeado.

### 4. El valor económico se concentra en peaks subestimados

Comparación en ventanas de evento (±7 días) a cost ratio 3x:

| Evento | n_días | Gain con tuning | α óptimo |
|---|---|---|---|
| **Black Friday 2017** | 13 | **$3,396** | +1.00 |
| Día dos Namorados 2018 | 13 | $61 | 0.00 |

**55x más valor en BF que en DdN.** Esto **invierte** una narrativa de
Fase 6 donde DdN era "el mejor fold" del modelo en WAPE. Con evaluación
económica, el valor está donde el modelo más sub-predice.

### 5. El extremo del grid en BF es un hallazgo, no un bug

El `α óptimo` en Black Friday alcanza el límite del grid (`+1.00`) en
todos los cost ratios. Esto significa que el modelo es **estructuralmente
pesimista** en BF.

No extendimos el grid más allá de `+1.00` porque:

- Conceptualmente, `α > 1.0` es predecir **más del doble** de lo que
  dice el modelo: ya no es "ajuste fino", es "el modelo está mal".
- La conclusión correcta es: **el modelo necesita re-entrenamiento con
  más ejemplos de Black Friday**, no más ajuste post-hoc.

Reportar el límite del enfoque es **más valioso que refinar un número**
artificial.

---

## El insight central de Fase 7

> **WAPE simétrico mide precisión promedio. Cost-sensitive evaluation
> mide valor económico. Un modelo "menos preciso pero mejor calibrado
> direccionalmente" puede valer más que uno "más preciso pero sesgado".**

Implicaciones operacionales:

1. **No siempre hay que retrenar** cuando los costos cambian. A veces basta
   con re-calibrar `α`.
2. **El monitoring debe ser cost-aware**: un modelo con WAPE estable puede
   estar perdiendo plata si su sesgo direccional cambia.
3. **El `α` debe ser configurable en producción**, no hardcodeado en el
   código del modelo.

---

## Decisiones de diseño documentadas

| Decisión | Elección | Justificación |
|---|---|---|
| Cost ratio | Sweep 1x/2x/3x/5x | Cubre el rango realista en logística |
| Métrica primaria | `expected_gain` vs SeasonalNaive | Responde directamente la pregunta de portfolio |
| Métrica secundaria | `total_cost` por modelo | Da magnitud absoluta |
| Threshold tuning | Multiplicador `(1+α)` con grid -0.20 a +1.00 | Simple y deployable |
| Dimensiones | Por fold + zoom en BF/DdN | Aprovecha la asimetría detectada en Fase 6 |
| Visualización | Iterativo: 1 plot por insight | Narrativa progresiva, no dashboard |

---

## Hacia Fase 8: productivización

Fase 8 va a llevar este modelo (LightGBM tuneado + conformal intervals +
threshold tuning configurable) a un servicio FastAPI.

**Endpoints planeados**:

| Endpoint | Función |
|---|---|
| `POST /predict` | Punto + intervalo + ajuste cost-aware con `α` configurable |
| `GET /health` | Status del servicio |
| `GET /alpha-defaults` | Valores recomendados de `α` por contexto |

**Decisión de diseño clave para Fase 8**:

> El `α` va a ser un parámetro del request, no del modelo. El equipo de
> operaciones lo configura según los costos del momento sin redeploy.

---

## Reproducibilidad

**Re-correr el análisis batch** (30 segundos):

```bash
python scripts/analyze_costs.py -v
```

**Re-correr el análisis del notebook** (lee artifacts persistidos):

```bash
jupyter notebook notebooks/04_cost_analysis.ipynb
```

**Tests asociados**: 282 pasando, 97% coverage. Correr con `make test`.

**Artifacts persistidos**:
data/processed/phase7/
├── gain_by_fold.csv (16 filas: 4 folds × 4 cost ratios)
├── gain_in_events.csv (8 filas: 2 eventos × 4 cost ratios)
└── predictions_per_fold.csv (8208 filas: predicciones crudas)

---

*Notebook ejecutado: holdout final jul-ago 2018. Próxima fase: productivización (Fase 8 FastAPI + monitoring).*
