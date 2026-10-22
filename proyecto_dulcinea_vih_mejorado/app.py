from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 1. Configuración Institucional y Título Descriptivo
st.set_page_config(
    page_title="ALERTA-TAR Antioquia | Monitoreo Cohorte VIH",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
DIR_SILVER = BASE_DIR / "data" / "silver"
DIR_GOLD = BASE_DIR / "data" / "gold"


@st.cache_data
def cargar_datos_silver():
    ruta_parquet = DIR_SILVER / "cohorte_vih_silver.parquet"
    if ruta_parquet.exists():
        df = pd.read_parquet(ruta_parquet)
    else:
        ruta_csv = DIR_SILVER / "cohorte_vih_silver.csv"
        df = pd.read_csv(ruta_csv)

    # Ingesta estricta de probabilidades calibradas desde Gold
    ruta_pred = DIR_GOLD / "prediccion_riesgo_abandono.csv"
    if ruta_pred.exists():
        df_pred = pd.read_csv(
            ruta_pred,
            usecols=[
                "Numero de Identificacion",
                "Probabilidad_Abandono_%",
                "Nivel_Riesgo_Estimado",
            ],
        )
        df["Numero de Identificacion"] = df["Numero de Identificacion"].astype(str)
        df_pred["Numero de Identificacion"] = df_pred["Numero de Identificacion"].astype(str)
        df = df.merge(df_pred, on="Numero de Identificacion", how="left")
    else:
        # En caso de no existir Gold, no inventar fórmula manual
        df["Probabilidad_Abandono_%"] = np.nan
        df["Nivel_Riesgo_Estimado"] = "Pendiente de Ejecución Gold"

    return df


df = cargar_datos_silver()

# Encabezado institucional
st.title("🛡️ ALERTA-TAR Antioquia: Sistema de Alerta Temprana en Adherencia al TAR")
st.markdown(
    "**Gobernación de Antioquia — Secretaría Seccional de Salud y Protección Social**  \n"
    "*Articulación de Datos SIVIGILA, Cuenta de Alto Costo (CAC) y Cohorte VIH. Responsable: Ing. Claudia Cifuentes.*"
)
st.caption(
    "⚠️ **Nota Metodológica:** Los datos actuales son sintéticos (Prueba de Concepto - Pipeline Validado). No aptos para inferencia clínica directa."
)
st.markdown("---")

# Filtros laterales
st.sidebar.header("🔍 Filtros Territoriales")
subregiones_opt = sorted(df["Subregion"].dropna().unique().tolist())
subregiones_sel = st.sidebar.multiselect(
    "Subregión de Residencia:",
    options=subregiones_opt,
    default=subregiones_opt,
)
df_filtrado = df[df["Subregion"].isin(subregiones_sel)]

municipios_opt = sorted(df_filtrado["Municipio residencia"].dropna().unique().tolist())
municipios_sel = st.sidebar.multiselect(
    "Municipio de Residencia:",
    options=municipios_opt,
    default=municipios_opt,
)
df_filtrado = df_filtrado[df_filtrado["Municipio residencia"].isin(municipios_sel)]

estados_opt = df_filtrado["Estado_Seguimiento_VIH"].unique().tolist()
estados_sel = st.sidebar.multiselect(
    "Estado de Seguimiento Clínico:",
    options=estados_opt,
    default=estados_opt,
)
df_filtrado = df_filtrado[df_filtrado["Estado_Seguimiento_VIH"].isin(estados_sel)]
st.sidebar.markdown("---")
st.sidebar.caption(f"📊 Registros Activos Filtrados: **{len(df_filtrado):,}**")

# Pestañas Principales
tab1, tab2, tab3 = st.tabs(
    [
        "📈 Monitoreo Epidemiológico",
        "🔮 Modelo Predictivo y Priorización",
        "📋 Gestión Operativa y Campo",
    ]
)

with tab1:
    st.subheader("Distribución Territorial y Clínica de la Cohorte")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_pie = px.pie(
            df_filtrado,
            names="Estado_Seguimiento_VIH",
            color="Estado_Seguimiento_VIH",
            color_discrete_map={
                "En TAR - Virológicamente Controlado": "#2ecc71",
                "En TAR - Con Carga Viral Detectable": "#f39c12",
                "Abandono / Sin TAR": "#e74c3c",
            },
            hole=0.4,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_c2:
        df_abandono_mun = (
            df_filtrado[df_filtrado["Estado_Seguimiento_VIH"] == "Abandono / Sin TAR"]
            .groupby("Municipio residencia")
            .size()
            .reset_index(name="Casos_Abandono")
            .sort_values(by="Casos_Abandono", ascending=True)
        )
        fig_bar = px.bar(
            df_abandono_mun,
            y="Municipio residencia",
            x="Casos_Abandono",
            orientation="h",
            color="Casos_Abandono",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.subheader("Estratificación de Riesgo para Priorización de Contacto (Top-K)")
    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        st.markdown("**Distribución por Nivel de Riesgo Calibrado**")
        fig_risk = px.histogram(
            df_filtrado,
            x="Nivel_Riesgo_Estimado",
            color="Nivel_Riesgo_Estimado",
            color_discrete_map={
                "Riesgo Bajo": "#2ecc71",
                "Riesgo Medio": "#f39c12",
                "Riesgo Alto": "#e74c3c",
            },
            text_auto=True,
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    with col_p2:
        st.info(
            "ℹ️ **Criterio de Priorización Operativa:**\n\n"
            "El modelo no debe emplearse como un clasificador clínico individual ad-hoc, sino como un **motor de ordenamiento (ranking)** "
            "para enfocar la capacidad de búsqueda activa en el 10% o 20% de pacientes con mayor probabilidad de interrupción (Top-K Recall)."
        )
        st.markdown(
            """
        - **Riesgo Alto (Top 10%):** Llamada o visita de seguimiento preventivo antes de vencer la fórmula.
        - **Riesgo Medio (Top 20-30%):** Recordatorio vía mensajería de texto o farmacia.
        - **Ya sin TAR:** No requiere predicción; pasa a búsqueda activa inmediata en campo.
        """
        )

with tab3:
    st.subheader("Búsqueda Activa Territorial en Campo")
    filtro_estado = st.radio(
        "Filtrar lista operativa:",
        options=["Solo Abandono / Sin TAR", "Riesgo Alto Estimado", "Toda la Cohorte"],
        horizontal=True,
    )
    if filtro_estado == "Solo Abandono / Sin TAR":
        df_tabla = df_filtrado[df_filtrado["Estado_Seguimiento_VIH"] == "Abandono / Sin TAR"]
    elif filtro_estado == "Riesgo Alto Estimado":
        df_tabla = df_filtrado[df_filtrado["Nivel_Riesgo_Estimado"] == "Riesgo Alto"]
    else:
        df_tabla = df_filtrado

    cols_mostrar = [
        "TD",
        "Numero de Identificacion",
        "Municipio residencia",
        "Subregion",
        "Edad",
        "Estado_Seguimiento_VIH",
        "Probabilidad_Abandono_%",
        "Nivel_Riesgo_Estimado",
        "Nombre IPS/ESE primaria",
    ]
    cols_existentes = [c for c in cols_mostrar if c in df_tabla.columns]
    st.dataframe(df_tabla[cols_existentes], use_container_width=True, hide_index=True)

    csv_data = df_tabla[cols_existentes].to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 Descargar Listado para Equipos Extramurales (CSV)",
        data=csv_data,
        file_name="Listado_Priorizacion_TAR_Antioquia.csv",
        mime="text/csv",
    )
