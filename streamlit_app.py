"""MELI Shipping Forecast — Dashboard Streamlit.

Dashboard interactivo para explorar predicciones de demanda de envíos
por estado brasileño. Carga el ConformalForecaster directamente
(sin capa de API) y usa caché de Streamlit para evitar recargar
el modelo en cada interacción.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "lightgbm_final.joblib"

st.set_page_config(
    page_title="MELI Shipping Forecast",
    page_icon="📦",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Carga del modelo (cacheado — se ejecuta una vez por sesión de servidor)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Cargando modelo...")
def load_model():
    """Carga (o entrena) el ConformalForecaster de producción."""
    from shipping_forecast.api.app import _load_artifacts
    from shipping_forecast.pipelines.train_final_model import main as train_main

    if not MODEL_PATH.exists():
        with st.spinner("Primera ejecución: entrenando modelo (fast-retrain, ~30s)..."):
            train_main(["--fast-retrain", "--output-dir", str(MODEL_PATH.parent)])

    model, model_info = _load_artifacts(MODEL_PATH)
    return model, model_info


@st.cache_resource(show_spinner="Cargando datos históricos...")
def load_history():
    """Carga el panel histórico una vez (necesario para model.predict())."""
    from shipping_forecast.pipelines.train_final_model import (
        DATA_CUTOFF,
        load_panel_with_cutoff,
    )
    from shipping_forecast.pipelines.train_final_model import (
        DB_PATH as TRAIN_DB_PATH,
    )

    return load_panel_with_cutoff(TRAIN_DB_PATH, DATA_CUTOFF)


# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------

st.title("📦 MELI Shipping Forecast")
st.caption(
    "Pronóstico de demanda de envíos por estado brasileño · "
    "LightGBM + Predicción Conformal · Intervalos calibrados al 90%"
)

model, model_info = load_model()
history = load_history()

# ---------------------------------------------------------------------------
# Sidebar — controles
# ---------------------------------------------------------------------------

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/MercadoLibre.svg/320px-MercadoLibre.svg.png",
    width=160,
)
st.sidebar.markdown("### 📦 Shipping Forecast")
st.sidebar.caption("Proyecto de portfolio · Data Science · ML en producción")
st.sidebar.divider()

st.sidebar.header("⚙️ Configuración del pronóstico")

todos_estados = sorted(model_info["groups"])
estados_seleccionados = st.sidebar.multiselect(
    "Estados",
    options=todos_estados,
    default=["SP", "RJ", "MG"],
    help="Selecciona uno o más estados brasileños para pronosticar.",
)

last_train = datetime.date.fromisoformat(model_info["last_train_date"])
min_inicio = last_train + datetime.timedelta(days=1)
max_inicio = last_train + datetime.timedelta(days=60)

fecha_inicio = st.sidebar.date_input(
    "Fecha de inicio",
    value=min_inicio,
    min_value=min_inicio,
    max_value=max_inicio,
)

horizonte = st.sidebar.slider(
    "Horizonte de pronóstico (días)",
    min_value=7,
    max_value=90,
    value=30,
    step=1,
)
fecha_fin = fecha_inicio + datetime.timedelta(days=horizonte - 1)
st.sidebar.caption(f"Período: {fecha_inicio} → {fecha_fin}")

alpha = st.sidebar.slider(
    "Alpha (asimetría de costo)",
    min_value=-1.0,
    max_value=1.0,
    value=0.65,
    step=0.05,
    help=(
        "α > 0: penaliza subpredicción (desplaza 'recomendado' hacia el límite superior). "
        "α = 0: recomendado = pronóstico puntual. "
        "α < 0: penaliza sobrepredicción (desplaza hacia el límite inferior)."
    ),
)

st.sidebar.divider()
st.sidebar.subheader("📊 Resumen del modelo")
st.sidebar.metric("Versión", model_info["version"])
st.sidebar.metric("Último dato", model_info["last_train_date"])
st.sidebar.metric("WAPE holdout", f"{model_info['evaluation_metrics']['wape']:.4f}")
st.sidebar.metric("Estados", model_info["n_groups"])
st.sidebar.divider()
st.sidebar.markdown(
    "[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)]"
    "(https://github.com/JulioPradenas/meli-shipping-forecast)"
)

# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------

if not estados_seleccionados:
    st.warning("⚠️ Selecciona al menos un estado en el panel lateral.")
    st.stop()

with st.spinner("Calculando pronóstico..."):
    last_train_ts = pd.Timestamp(last_train)
    horizonte_dias = (pd.Timestamp(fecha_fin) - last_train_ts).days
    df_pred = model.predict(history, horizon=horizonte_dias)
    df_pred = df_pred[
        (df_pred["shipment_date"] >= pd.Timestamp(fecha_inicio))
        & (df_pred["shipment_date"] <= pd.Timestamp(fecha_fin))
        & (df_pred["customer_state"].isin(estados_seleccionados))
    ].copy()

    if alpha >= 0:
        df_pred["recomendado"] = df_pred["y_pred"] + alpha * (
            df_pred["y_upper"] - df_pred["y_pred"]
        )
    else:
        df_pred["recomendado"] = df_pred["y_pred"] + alpha * (
            df_pred["y_pred"] - df_pred["y_lower"]
        )
    df_pred["recomendado"] = df_pred["recomendado"].clip(lower=0)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_resumen, tab_grafico, tab_tabla, tab_modelo = st.tabs(
    [
        "🏠 Resumen del proyecto",
        "📈 Gráfico de pronóstico",
        "📋 Tabla de datos",
        "🔬 Info del modelo",
    ]
)

# ── Tab 1: Resumen del proyecto ──────────────────────────────────────────────
with tab_resumen:
    st.subheader("¿Qué hace este proyecto?")
    st.markdown(
        """
        Este proyecto construye un sistema de **pronóstico de demanda de envíos** para los
        27 estados de Brasil, usando datos históricos de transacciones de e-commerce de Olist.
        El modelo predice el volumen diario de paquetes por estado con un horizonte de hasta
        90 días, entregando **intervalos de confianza calibrados al 90%** y recomendaciones
        de capacidad **ajustadas por el costo asimétrico** de sub-predecir vs sobre-predecir.
        """
    )

    st.divider()
    st.subheader("📌 Métricas clave")
    k1, k2, k3, k4, k5 = st.columns(5)
    metricas = model_info["evaluation_metrics"]
    k1.metric(
        "WAPE holdout",
        f"{metricas['wape']:.4f}",
        help="Weighted Absolute Percentage Error en holdout estricto (sin leakage)",
    )
    k2.metric("MAE", f"{metricas['mae']:.2f} paquetes/día")
    k3.metric("Estados brasileños", model_info["n_groups"])
    k4.metric("Features engineered", model_info["n_features"])
    k5.metric(
        "Cobertura empírica",
        "89.9%",
        help="Cobertura real del intervalo conformal en la ventana de calibración",
    )

    st.divider()
    st.subheader("🏆 Comparación de modelos")
    st.markdown(
        "Todos los modelos evaluados en el **mismo holdout estricto** (jul-ago 2018, 62 días)."
    )

    df_modelos = pd.DataFrame(
        [
            {
                "Modelo": "Naive (media histórica)",
                "WAPE CV": "—",
                "WAPE Holdout": "0.83",
                "Intervalos": "No",
                "Veredicto": "❌ Descartado",
            },
            {
                "Modelo": "SeasonalNaive",
                "WAPE CV": "0.44",
                "WAPE Holdout": "0.44",
                "Intervalos": "No",
                "Veredicto": "✅ Baseline fuerte",
            },
            {
                "Modelo": "Prophet",
                "WAPE CV": "~0.52",
                "WAPE Holdout": "~0.52",
                "Intervalos": "Sí (no calibrados)",
                "Veredicto": "❌ Superado por baseline",
            },
            {
                "Modelo": "LightGBM sin tuning",
                "WAPE CV": "~0.44",
                "WAPE Holdout": "~0.48",
                "Intervalos": "No",
                "Veredicto": "🔶 Bueno en CV, débil en holdout",
            },
            {
                "Modelo": "LightGBM + Optuna (con leakage)",
                "WAPE CV": "0.40",
                "WAPE Holdout": "0.469",
                "Intervalos": "No",
                "Veredicto": "🔶 Mejor CV, leakage en eval_set",
            },
            {
                "Modelo": "LightGBM + Conformal (producción) ⭐",
                "WAPE CV": "0.40",
                "WAPE Holdout": "0.5156*",
                "Intervalos": "Sí (90% calibrado)",
                "Veredicto": "✅ En producción",
            },
        ]
    )
    st.dataframe(df_modelos, width="stretch", hide_index=True)
    st.caption(
        "\\* WAPE 0.5156 es la métrica **honesta** sin leakage de eval_set. "
        "El LightGBM tuneado con eval_set reportaba 0.469 porque el early stopping "
        "veía datos del holdout durante el entrenamiento."
    )

    st.divider()
    st.subheader("🔑 Importancia de features (Top 15)")

    base_model = model.base_model
    feature_names = base_model.feature_names_
    importances = base_model.model_.feature_importances_
    df_imp = (
        pd.DataFrame({"feature": feature_names, "importancia": importances})
        .sort_values("importancia", ascending=True)
        .tail(15)
    )

    fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
    ax_imp.barh(df_imp["feature"], df_imp["importancia"], color="#2196F3", alpha=0.85)
    ax_imp.set_xlabel("Importancia (gain)")
    ax_imp.set_title("Top 15 features por importancia (gain)")
    ax_imp.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_imp)
    plt.close(fig_imp)

    st.divider()
    st.subheader("🛠️ Stack tecnológico")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Modelado**")
        st.markdown("- LightGBM + Optuna\n- Predicción Conformal\n- scikit-learn\n- SHAP")
    with c2:
        st.markdown("**Servicio**")
        st.markdown(
            "- FastAPI + Pydantic v2\n- structlog (JSON logs)\n- Docker\n- GitHub Actions CI/CD"
        )
    with c3:
        st.markdown("**Presentación**")
        st.markdown("- Streamlit Cloud\n- Jupyter notebooks\n- MLflow (experiment tracking)")

# ── Tab 2: Gráfico de pronóstico ─────────────────────────────────────────────
with tab_grafico:
    st.subheader(f"Demanda de envíos — {fecha_inicio} al {fecha_fin}")
    n_estados = len(estados_seleccionados)
    fig, axes = plt.subplots(n_estados, 1, figsize=(12, 4 * n_estados), sharex=True)
    if n_estados == 1:
        axes = [axes]

    for ax, estado in zip(axes, sorted(estados_seleccionados), strict=False):
        df_s = df_pred[df_pred["customer_state"] == estado]
        ax.fill_between(
            df_s["shipment_date"],
            df_s["y_lower"],
            df_s["y_upper"],
            alpha=0.25,
            color="#2196F3",
            label="Intervalo 90%",
        )
        ax.plot(
            df_s["shipment_date"],
            df_s["y_pred"],
            color="#555555",
            linewidth=1.5,
            linestyle="--",
            label="Pronóstico puntual",
        )
        ax.plot(
            df_s["shipment_date"],
            df_s["recomendado"],
            color="#F44336",
            linewidth=2,
            label=f"Recomendado (α={alpha})",
        )
        ax.set_title(f"Estado: {estado}", fontsize=11)
        ax.set_ylabel("Envíos / día")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    axes[-1].set_xlabel("Fecha")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        f"**¿Qué es alpha (α)?** Con α={alpha}, el valor recomendado se ubica al "
        f"{int(abs(alpha) * 100)}% del camino entre el pronóstico puntual y el "
        f"{'límite superior' if alpha >= 0 else 'límite inferior'} del intervalo. "
        "Esto refleja el costo asimétrico de errar: quedarse sin capacidad (subpredicción) "
        "suele ser más costoso que sobre-provisionar."
    )

# ── Tab 3: Tabla de datos ─────────────────────────────────────────────────────
with tab_tabla:
    st.subheader("Resultados del pronóstico")
    df_display = df_pred.rename(
        columns={
            "shipment_date": "Fecha",
            "customer_state": "Estado",
            "y_pred": "Puntual",
            "y_lower": "Límite inferior 90%",
            "y_upper": "Límite superior 90%",
            "recomendado": f"Recomendado (α={alpha})",
        }
    )[
        [
            "Fecha",
            "Estado",
            "Puntual",
            "Límite inferior 90%",
            "Límite superior 90%",
            f"Recomendado (α={alpha})",
        ]
    ].copy()
    df_display["Fecha"] = df_display["Fecha"].dt.date
    st.dataframe(
        df_display.style.format(
            {
                "Puntual": "{:.1f}",
                "Límite inferior 90%": "{:.1f}",
                "Límite superior 90%": "{:.1f}",
                f"Recomendado (α={alpha})": "{:.1f}",
            }
        ),
        width="stretch",
    )
    csv = df_display.to_csv(index=False)
    st.download_button("⬇️ Descargar CSV", csv, "pronostico.csv", "text/csv")

# ── Tab 4: Info del modelo ────────────────────────────────────────────────────
with tab_modelo:
    st.subheader("Metadatos del modelo")
    col1, col2, col3 = st.columns(3)
    col1.metric("Versión", model_info["version"])
    col2.metric("Último dato de entrenamiento", model_info["last_train_date"])
    col3.metric("Corte de datos", model_info["data_cutoff"])

    st.divider()
    st.subheader("Métricas de evaluación (holdout sin leakage)")
    metricas = model_info["evaluation_metrics"]
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("WAPE", f"{metricas['wape']:.4f}")
    mc2.metric("MAE", f"{metricas['mae']:.3f}")
    mc3.metric("RMSE", f"{metricas['rmse']:.3f}")
    st.caption(
        f"Ventana holdout: {metricas['window_start']} → {metricas['window_end']} "
        f"({metricas['n_days']} días). "
        "Evaluado en un modelo hermano entrenado solo hasta el límite del holdout "
        "(sin leakage de eval_set)."
    )

    st.divider()
    st.subheader("Estados conocidos por el modelo")
    st.write(", ".join(sorted(model_info["groups"])))

    st.divider()
    st.subheader("Nota metodológica")
    st.info(
        "Las métricas fueron calculadas por un modelo hermano entrenado solo hasta "
        "el 30 de junio de 2018 (límite del fold 3 de la Fase 6). El modelo de "
        "producción fue entrenado con el dataset completo incluyendo la ventana de "
        "holdout. Las métricas reflejan el rendimiento esperado fuera de muestra con "
        "metodología estricta sin leakage de eval_set (a diferencia del fold 3 de "
        "la Fase 6 cuyo WAPE=0.4694 usó leakage en el early stopping)."
    )
