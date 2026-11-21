"""Prototipo Streamlit: alerta temprana de excesos de dengue en Bolívar.

Pestaña "Mapa de alertas": selección de fecha, proyección a 4 semanas, mapa de calor
por municipio y tabla con alertas (clic en una fila abre el detalle de variables).
Demás pestañas: casos y canal endémico, predicción individual, clima y detalle del modelo.

Ejecutar desde src_dashboard/:  streamlit run app.py
"""
import json
import math
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from plotly.subplots import make_subplots

from services import api_client as api
from services.api_client import ApiError

st.set_page_config(page_title="Alerta temprana de dengue – Bolívar", layout="wide")

GEO_DIR = Path(__file__).parent / "assets" / "geo"


# ---------- Datos (cacheados) ----------
@st.cache_data(ttl=600, show_spinner=False)
def cargar_municipios():
    return api.get_municipios()


@st.cache_data(ttl=600, show_spinner=False)
def cargar_rango():
    return api.get_rango_fechas()


@st.cache_data(ttl=600, show_spinner=False)
def cargar_serie(municipio_id: str) -> pd.DataFrame:
    df = pd.DataFrame(api.get_serie(municipio_id))
    df["semana"] = pd.to_datetime(df["semana"])
    return df


@st.cache_data(ttl=600, show_spinner=False)
def cargar_modelo():
    return api.get_modelo()


@st.cache_data(ttl=60, show_spinner=False)
def cargar_estado() -> dict:
    return api.get_estado_datos()


def serie_o_vacia(municipio_id: str) -> pd.DataFrame:
    """Serie del municipio, o un DataFrame vacío si no hay datos cargados."""
    try:
        return cargar_serie(municipio_id)
    except ApiError:
        return pd.DataFrame(columns=["semana", "casos", "lluvia_mm", "temp_media_c", "umbral_canal", "exceso"])


@st.cache_data(show_spinner=False)
def cargar_geo(municipio_ids: tuple[str, ...]) -> dict:
    """FeatureCollection con una feature por municipio; el id de la feature es el municipio_id."""
    features = []
    for mid in municipio_ids:
        geo = json.loads((GEO_DIR / f"{mid}_cabecera_urbana_mgn2021.geojson").read_text(encoding="utf-8"))
        feature = geo if geo.get("type") == "Feature" else geo["features"][0]
        feature["id"] = mid
        feature["properties"] = {"municipio_id": mid}
        features.append(feature)
    return {"type": "FeatureCollection", "features": features}


def semana_de_corte(df: pd.DataFrame, fecha) -> pd.Series | None:
    """Fila de la semana que contiene la fecha (igual criterio que la API)."""
    previas = df.loc[df["semana"] <= pd.Timestamp(fecha)]
    return None if previas.empty else previas.iloc[-1]


# ---------- Gráficas ----------
def tramos_exceso(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Rangos de semanas consecutivas en exceso (épocas de alto riesgo)."""
    tramos, inicio, previo = [], None, None
    for fila in df.itertuples():
        if fila.exceso and inicio is None:
            inicio = fila.semana
        if not fila.exceso and inicio is not None:
            tramos.append((inicio, previo))
            inicio = None
        previo = fila.semana
    if inicio is not None:
        tramos.append((inicio, previo))
    return tramos


def limites_geo(geo: dict, margen: float = 0.08) -> tuple[list[float], list[float]]:
    """Rangos (lon, lat) que encuadran todos los polígonos, con margen en grados."""
    lons, lats = [], []

    def recorrer(coords):
        if isinstance(coords[0], (int, float)):
            lons.append(coords[0])
            lats.append(coords[1])
        else:
            for c in coords:
                recorrer(c)

    for f in geo["features"]:
        recorrer(f["geometry"]["coordinates"])
    return [min(lons) - margen, max(lons) + margen], [min(lats) - margen, max(lats) + margen]


ESCALA_CALOR = [
    (0.0, (255, 255, 178)),
    (0.33, (254, 204, 92)),
    (0.66, (253, 141, 60)),
    (1.0, (189, 0, 38)),
]
COLOR_SIN_PROYECCION = [158, 158, 158, 170]
MAPA_BASE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"


def color_calor(prob: float) -> list[int]:
    p = min(max(prob, 0.0), 1.0)
    for (a, ca), (b, cb) in zip(ESCALA_CALOR[:-1], ESCALA_CALOR[1:]):
        if p <= b:
            t = 0 if b == a else (p - a) / (b - a)
            return [round(ca[i] + (cb[i] - ca[i]) * t) for i in range(3)] + [190]
    return [189, 0, 38, 190]


# Alerta: el color y el radio pulsan entre un mínimo y un máximo semitransparente.
COLOR_ALERTA_MIN = [255, 150, 150, 70]
COLOR_ALERTA_MAX = [220, 0, 20, 170]
RADIO_ALERTA_MIN, RADIO_ALERTA_MAX = 14, 24
PERIODO_PULSO_S = 2.0


def fase_pulso(t: float) -> float:
    """Valor en [0, 1] que sube y baja suavemente con periodo PERIODO_PULSO_S."""
    return (1 - math.cos(2 * math.pi * t / PERIODO_PULSO_S)) / 2


def color_alerta(fase: float) -> list[int]:
    return [round(a + (b - a) * fase) for a, b in zip(COLOR_ALERTA_MIN, COLOR_ALERTA_MAX)]


def centroide(geometry: dict) -> tuple[float, float]:
    pts: list[list[float]] = []

    def recorrer(coords):
        if isinstance(coords[0], (int, float)):
            pts.append(coords)
        else:
            for c in coords:
                recorrer(c)

    recorrer(geometry["coordinates"])
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def mapa_municipios(geo: dict, probabilidades: dict[str, float] | None,
                    nombres: dict[str, str], alertas: frozenset = frozenset(),
                    fase: float = 0.0) -> pdk.Deck:
    """Mapa base de Colombia (CARTO) con zoom en Bolívar.

    Por municipio: polígono (color por probabilidad) y un punto en su centroide, visible
    aunque el polígono sea pequeño. En alerta, polígono y punto pulsan entre rojo
    semitransparente y rosa claro según `fase` (0 a 1). Sin bordes. Colores fijos por capa.
    Requiere conexión a internet en el navegador para cargar el mapa base.
    """
    capas = []
    lon_rng, lat_rng = limites_geo(geo, margen=0.25)
    span = max(lon_rng[1] - lon_rng[0], lat_rng[1] - lat_rng[0])

    for f in geo["features"]:
        mid = f["id"]
        p = probabilidades.get(mid) if probabilidades else None
        en_alerta = mid in alertas
        texto = nombres[mid] if p is None else f"{nombres[mid]} · probabilidad {p:.0%}"
        if en_alerta:
            texto += " · ALERTA"
            color = color_alerta(fase)
            radio = round(RADIO_ALERTA_MIN + (RADIO_ALERTA_MAX - RADIO_ALERTA_MIN) * fase)
        else:
            color = COLOR_SIN_PROYECCION if p is None else color_calor(p)
            radio = 12

        capas.append(pdk.Layer(
            "GeoJsonLayer",
            data={"type": "FeatureCollection", "features": [{
                "type": "Feature", "geometry": f["geometry"], "properties": {"texto": texto},
            }]},
            filled=True,
            stroked=False,
            get_fill_color=color,
            pickable=True,
        ))

        lon, lat = centroide(f["geometry"])
        capas.append(pdk.Layer(
            "ScatterplotLayer",
            data=[{"position": [lon, lat], "texto": texto}],
            get_fill_color=color,
            radius_min_pixels=radio,
            radius_max_pixels=radio + 10,
            stroked=False,
            pickable=True,
        ))

    vista = pdk.ViewState(
        latitude=sum(lat_rng) / 2,
        longitude=sum(lon_rng) / 2,
        zoom=min(max(math.log2(360 / span), 4), 11),
        pitch=0,
    )
    return pdk.Deck(
        layers=capas,
        initial_view_state=vista,
        map_style=MAPA_BASE,
        tooltip={"text": "{texto}"},
    )


def grafica_canal_clima(df: pd.DataFrame) -> go.Figure:
    """Tres gráficas con eje X compartido: casos y canal, lluvia y temperatura.

    Con shared_xaxes, hacer zoom o desplazar una gráfica mueve las tres.
    """
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=("Casos y umbral del canal endémico", "Lluvia (mm)", "Temperatura media (°C)"),
        specs=[[{"secondary_y": True}], [{}], [{}]],
    )
    for inicio, fin in tramos_exceso(df):
        fig.add_vrect(
            x0=inicio, x1=fin, fillcolor="red", opacity=0.12, line_width=0,
            row="all", col=1,
        )
    fig.add_trace(go.Scatter(x=df["semana"], y=df["casos"], name="Casos semanales",
                             mode="lines", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["semana"], y=df["umbral_canal"], name="Umbral canal endémico",
                             mode="lines", line=dict(color="#d62728", dash="dash")), row=1, col=1)
    # Etiqueta actual (brote en la semana) en el eje derecho 0–1, como línea de escalón roja translúcida.
    if "etiqueta" in df.columns and df["etiqueta"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["semana"], y=df["etiqueta"].astype(float), name="Brote en la semana",
            mode="lines", line=dict(color="rgba(214, 39, 40, 0.55)", width=2, shape="hv"),
            fill="tozeroy", fillcolor="rgba(214, 39, 40, 0.15)",
        ), row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Brote en la semana (0 = no, 1 = sí)", range=[0, 1], tickvals=[0, 1],
                         showgrid=False, secondary_y=True, row=1, col=1)
    fig.add_trace(go.Scatter(x=df["semana"], y=df["lluvia_mm"], name="Lluvia",
                             mode="lines", line=dict(color="#2ca02c")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["semana"], y=df["temp_media_c"], name="Temperatura",
                             mode="lines", line=dict(color="#ff7f0e")), row=3, col=1)
    fig.update_yaxes(title_text="Casos", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="mm", row=2, col=1)
    fig.update_yaxes(title_text="°C", row=3, col=1)
    fig.update_xaxes(title_text="Semana epidemiológica (inicio)", row=3, col=1)
    fig.update_layout(height=760, legend=dict(orientation="h", y=-0.08), margin=dict(t=60, b=40))
    return fig




# ---------- Modal de detalle ----------
@st.dialog("Variables del municipio", width="large")
def modal_detalle(pred: dict, fila: dict):
    st.subheader(pred["municipio"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidad de exceso", f"{pred['probabilidad_exceso']:.0%}")
    c2.metric("Umbral de alerta", f"{pred['umbral_decision']:.0%}")
    c3.metric("Estado", "⚠️ ALERTA" if pred["alerta"] else "Sin alerta")
    st.write(
        f"Semana de corte **{pred['fecha_corte']}** → semana objetivo **{pred['semana_objetivo']}** "
        f"({pred['horizonte_semanas']} semanas de anticipación)."
    )

    st.markdown("**Valores observados en la semana de corte**")
    st.dataframe(pd.DataFrame({
        "Variable": ["Casos", "Lluvia (mm)", "Temperatura media (°C)", "Umbral del canal endémico (casos)"],
        "Valor": [str(fila["casos"]), str(fila["lluvia_mm"]), str(fila["temp_media_c"]),
                  str(round(fila["umbral_canal"], 1))],
    }), hide_index=True, use_container_width=True)

    st.markdown("**Variables que más influyen en la probabilidad**")
    st.dataframe(pd.DataFrame(pred["variables_influyentes"]), hide_index=True, use_container_width=True)

    if pred.get("exceso_observado") is not None:
        obs = "sí" if pred["exceso_observado"] else "no"
        st.caption(f"Comparación de prueba: en la semana objetivo hubo exceso registrado → **{obs}**.")


# ---------- Encabezado ----------
st.markdown(
    """<style>
    [data-testid="stTab"] [data-testid="stMarkdownContainer"],
    [data-testid="stTab"] [data-testid="stMarkdownContainer"] * {
        color: #ffffff !important;
        opacity: 1 !important;
    }
    [data-testid="stTab"][aria-selected="true"] {
        background-color: rgba(31, 119, 180, 0.45) !important;
        border-radius: 6px 6px 0 0;
    }
    </style>""",
    unsafe_allow_html=True,
)
st.title("Alerta temprana de excesos de dengue – Bolívar")
st.caption("Apoyo a la decisión. No reemplaza el criterio epidemiológico.")

try:
    municipios = cargar_municipios()
    modelo_info = cargar_modelo()
    estado = cargar_estado()
except ApiError as exc:
    st.error(str(exc))
    st.info("Inicie la API (docker compose up, o uvicorn en src_api/).")
    st.stop()

st.caption("Datos: SQLite (cargados desde CSV en la pestaña Configuración).")
try:
    rango = cargar_rango()
    error_rango = ""
except ApiError as exc:
    # Sin datos cargados: las pestañas de datos lo indican;
    # la pestaña de configuración sigue disponible.
    rango, error_rango = None, str(exc)

nombres = {m["id"]: m["nombre"] for m in municipios}
geo = cargar_geo(tuple(m["id"] for m in municipios))

opciones = {m["nombre"]: m["id"] for m in municipios}

if rango is not None:
    fecha_min = pd.Timestamp(rango["fecha_corte_min"]).date()
    fecha_max = pd.Timestamp(rango["fecha_corte_max"]).date()

tab_mapa, tab_casos, tab_modelo, tab_config, tab_acerca = st.tabs([
    "🗺️ 1. Mapa de alertas",
    "📈 2. Casos y canal endémico",
    "🧪 3. Detalle del modelo",
    "⚙️ 4. Configuración",
    "ℹ️ 5. Acerca de",
])

# ---------- 0. Mapa de alertas ----------
with tab_mapa:
    if rango is None:
        st.info(
            f"Aún no hay datos para el mapa ({error_rango}). "
            "Cargue los datos en la pestaña Configuración."
        )
    else:
        # Barra superior: calendario, proyección y umbral
        b1, b2, b3, b4 = st.columns([2, 2, 2, 2], vertical_alignment="bottom")
        with b1:
            fecha_sel = st.date_input("📅 Fecha de corte", value=fecha_max,
                                      min_value=fecha_min, max_value=fecha_max, key="fecha_mapa")
        with b2:
            st.write("")
            st.markdown(
                """<style>
                button[kind="primary"] {
                    background-color: #1f77b4 !important; border-color: #1f77b4 !important;
                    color: #ffffff !important;
                }
                button[kind="primary"]:hover {
                    background-color: #17629a !important; border-color: #17629a !important;
                }
                button[data-testid="stBaseButton-primary"] *,
                button[data-testid="stBaseButton-primary"] p {
                    color: #ffffff !important;
                }
                </style>""",
                unsafe_allow_html=True,
            )
            proyectar = st.button(
                f"Proyectar {modelo_info['horizonte_semanas']} semanas", type="primary",
                use_container_width=True, key="proyectar",
            )
        with b3:
            st.metric("Umbral de alerta", f"{modelo_info['umbral_decision']:.0%}",
                      help="Hay alerta cuando la probabilidad de exceso es igual o mayor a este valor.")
        with b4:
            fila_ref = semana_de_corte(serie_o_vacia(municipios[0]["id"]), fecha_sel)
            st.metric("Semana de corte", str(fila_ref["semana"].date()) if fila_ref is not None else "—")

        if proyectar:
            resultados, errores = {}, []
            for m in municipios:
                try:
                    resultados[m["id"]] = api.post_prediccion(m["id"], fecha_sel.isoformat())
                except ApiError as exc:
                    errores.append(f"{m['nombre']}: {exc}")
            st.session_state["proyeccion"] = {"fecha": fecha_sel.isoformat(), "resultados": resultados}
            st.session_state["modal_ultimo"] = None
            for e in errores:
                st.error(e)

        proy = st.session_state.get("proyeccion")
        proy_vigente = proy["resultados"] if proy and proy["fecha"] == fecha_sel.isoformat() else None
        if proy and not proy_vigente:
            st.info("La proyección mostrada corresponde a otra fecha. Pulse «Proyectar» para actualizarla.")
        elif proy_vigente:
            n_alertas = sum(1 for p in proy_vigente.values() if p["alerta"])
            st.success(f"Proyección para la semana objetivo "
                       f"{next(iter(proy_vigente.values()))['semana_objetivo']}: "
                       f"{n_alertas} municipio(s) con alerta.")
        else:
            st.info("Seleccione una fecha de corte y pulse «Proyectar» para ver la probabilidad por municipio.")

        alertas = frozenset(mid for mid, p in proy_vigente.items() if p["alerta"]) if proy_vigente else frozenset()
        probs = {mid: p["probabilidad_exceso"] for mid, p in proy_vigente.items()} if proy_vigente else None

        def bloque_mapa():
            # El fragmento se redibuja cada 0.25 s sólo si hay alertas: así el color pulsa suavemente.
            st.pydeck_chart(
                mapa_municipios(geo, probs, nombres, alertas, fase_pulso(time.time())),
                height=520, use_container_width=True,
            )

        col_mapa, col_tabla = st.columns([1.3, 1])
        with col_mapa:
            st.fragment(run_every=0.25 if alertas else None)(bloque_mapa)()
            st.caption(
                "Gris: sin proyección. Amarillo a rojo: probabilidad de exceso. "
                "Rojo parpadeante: municipio con alerta. El mapa base requiere internet."
            )

        with col_tabla:
            st.markdown("**Municipios**")
            st.markdown(
                """<style>
                div[data-testid="stVerticalBlockBorderWrapper"]:has(.fila-tabla):not(:has(div[data-testid="stVerticalBlockBorderWrapper"] .fila-tabla)) {
                    padding: 8px !important;
                }
                div[data-testid="stVerticalBlockBorderWrapper"]:has(.fila-tabla):not(:has(div[data-testid="stVerticalBlockBorderWrapper"] .fila-tabla)) > div {
                    padding: 0 !important;
                }
                div[data-testid="stVerticalBlockBorderWrapper"]:has(.fila-tabla):not(:has(div[data-testid="stVerticalBlockBorderWrapper"] .fila-tabla)) [data-testid="stVerticalBlock"] {
                    gap: 0.1rem !important; padding-top: 0 !important; padding-bottom: 0 !important;
                }
                div[data-testid="stVerticalBlockBorderWrapper"]:has(.fila-tabla) [data-testid="stElementContainer"]:has(.fila-tabla),
                div[data-testid="stVerticalBlockBorderWrapper"]:has(.fila-tabla) [data-testid="stElementContainer"]:has(.fila-alerta) {
                    display: none !important;
                }
                div[data-testid="stVerticalBlockBorderWrapper"]:has(.fila-alerta):not(:has(div[data-testid="stVerticalBlockBorderWrapper"] .fila-alerta)) {
                    background-color: #833535; border-color: #833535;
                }
                div[data-testid="stVerticalBlockBorderWrapper"]:has(.fila-alerta):not(:has(div[data-testid="stVerticalBlockBorderWrapper"] .fila-alerta)) * {
                    color: #ffffff !important;
                }
                div[data-testid="stVerticalBlockBorderWrapper"] button, div[data-testid="stVerticalBlockBorderWrapper"] button * {
                    color: #31333F !important;
                }
                div[data-testid="stVerticalBlockBorderWrapper"] button:not([data-testid="stBaseButton-primary"]) {
                    background-color: #000000 !important; border-color: #000000 !important;
                }
                div[data-testid="stVerticalBlockBorderWrapper"] button:not([data-testid="stBaseButton-primary"]):hover {
                    background-color: #000000 !important; border-color: #000000 !important;
                }
                div[data-testid="stVerticalBlockBorderWrapper"] button:not([data-testid="stBaseButton-primary"]), div[data-testid="stVerticalBlockBorderWrapper"] button:not([data-testid="stBaseButton-primary"]) * {
                    color: #ffffff !important;
                }
                </style>""",
                unsafe_allow_html=True,
            )
            ancho = [0.5, 2.2, 2.2, 1.2, 1.2, 1.5]
            encabezado = st.columns(ancho)
            for col, texto in zip(encabezado, ["", "Municipio", "Prob.", "Casos", "Umbral", ""]):
                col.markdown(f"**{texto}**")

            for m in municipios:
                mid = m["id"]
                pred = proy_vigente.get(mid) if proy_vigente else None
                fila_corte = semana_de_corte(serie_o_vacia(mid), fecha_sel)
                en_alerta = bool(pred and pred["alerta"])
                with st.container(border=True):
                    st.markdown('<span class="fila-tabla"></span>', unsafe_allow_html=True)
                    if en_alerta:
                        st.markdown('<span class="fila-alerta"></span>', unsafe_allow_html=True)
                    c = st.columns(ancho, vertical_alignment="center")
                    c[0].markdown("⚠️" if en_alerta else "")
                    c[1].markdown(f"**{m['nombre']}**" if en_alerta else m["nombre"])
                    if pred:
                        c[2].progress(pred["probabilidad_exceso"], text=f"{pred['probabilidad_exceso']:.0%}")
                    else:
                        c[2].markdown("—")
                    c[3].markdown(str(int(fila_corte["casos"])) if fila_corte is not None else "—")
                    c[4].markdown(f"{float(fila_corte['umbral_canal']):.0f}" if fila_corte is not None else "—")
                    if c[5].button("🔍 Ver", key=f"ver_{mid}", disabled=not pred,
                                   help="Ver las variables de este municipio"):
                        modal_detalle(pred, fila_corte.to_dict())

# ---------- 1. Casos y canal ----------
with tab_casos:
    st.subheader("Canal endémico, lluvia y temperatura")
    muni_canal_nombre = st.selectbox("Municipio", list(opciones.keys()), key="muni_canal")
    muni_canal_id = opciones[muni_canal_nombre]
    try:
        serie_canal = cargar_serie(muni_canal_id)
    except ApiError as exc:
        st.error(str(exc))
        serie_canal = None
    if serie_canal is not None:
        st.write(
            "Las zonas sombreadas marcan las épocas de **alto riesgo**: semanas en que los casos "
            "superan el umbral del canal endémico, seguidos. "
            "Al mover o hacer zoom en una gráfica, se mueven todas."
        )
        st.caption("Datos semanales cargados desde caracteristicas.csv (pestaña Configuración).")
        st.plotly_chart(grafica_canal_clima(serie_canal), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Semanas registradas", len(serie_canal))
        c2.metric("Semanas en exceso", int(serie_canal["exceso"].sum()))
        c3.metric("Épocas de alto riesgo", len(tramos_exceso(serie_canal)))

# ---------- 2. Modelo ----------
with tab_modelo:
    st.subheader("Detalle técnico del modelo")
    st.table(pd.DataFrame({
        "Concepto": ["Modelo", "Versión", "Horizonte (semanas)", "Umbral de decisión",
                     "Variables", "Métricas de evaluación"],
        "Valor": [modelo_info["nombre"], modelo_info["version"],
                  str(modelo_info["horizonte_semanas"]), f"{modelo_info['umbral_decision']:.0%}",
                  ", ".join(modelo_info["variables"]), modelo_info["metricas"]],
    }))
    st.info("")

# ---------- 4. Configuración ----------
with tab_config:
    st.subheader("Configuración de datos")
    st.markdown("**Cargar datos semanales de un municipio**")
    st.caption(
        "Archivo: `data/processed/municipios/<municipio>/caracteristicas.csv` del pipeline. "
        "Se usan las columnas `ANO`, `SEMANA`, `casos_confirmados_o_probables`, `precipitation_mm`, "
        "`air_temp_mean_c` y `umbral_brote`; las demás se ignoran. Las semanas que ya existen se "
        "**actualizan**, las nuevas se **agregan** y las que no vienen en el archivo se conservan."
    )
    col_muni, col_archivo = st.columns([1, 2], vertical_alignment="bottom")
    with col_muni:
        muni_cfg = st.selectbox("Municipio", list(opciones.keys()), key="muni_config")
    with col_archivo:
        archivo_csv = st.file_uploader("Archivo CSV", type=["csv"], key="csv_config")
    if st.button("Cargar en SQLite", type="primary", disabled=archivo_csv is None, key="btn_cargar_csv"):
        try:
            resp = api.post_csv(opciones[muni_cfg], archivo_csv.name, archivo_csv.getvalue())
            st.cache_data.clear()
            st.success(
                f"{muni_cfg}: {resp['nuevas']} semanas nuevas y {resp['actualizadas']} actualizadas. "
                f"Total guardado: {resp['semanas']} semanas ({resp['primer_periodo']} a {resp['ultimo_periodo']})."
            )
        except ApiError as exc:
            st.error(f"No se cargó el archivo: {exc}")

    st.markdown("**Estado de los datos en SQLite**")
    st.dataframe(pd.DataFrame([{
        "Municipio": m["nombre"],
        "Cargado": "Sí" if m["cargado"] else "No",
        "Semanas": m["semanas"],
        "Desde": m["primer_periodo"] or "—",
        "Hasta": m["ultimo_periodo"] or "—",
        "Última carga": (f"{m['ultimas_nuevas']} nuevas, {m['ultimas_actualizadas']} actualizadas"
                         if m["cargado"] else "—"),
        "Archivo": m["archivo"] or "—",
        "Cargado el": (m["cargado_en"] or "—").replace("T", " ")[:19],
    } for m in estado["municipios"]]), hide_index=True, use_container_width=True)


# ---------- 5. Acerca de ----------
with tab_acerca:
    st.subheader("Acerca de este proyecto")
    st.write(
        "**Sistema de alerta temprana para excesos de dengue en municipios de Bolívar "
        "mediante inteligencia artificial.**"
    )
    st.write(
        "Proyecto de grado de la Maestría en Inteligencia Artificial, Universidad de los Andes. "
        "Predice, con cuatro semanas de antelación, si los casos semanales de dengue de un "
        "municipio superarán el límite superior de su canal endémico."
    )
    st.markdown("**Equipo**")
    for nombre_integrante in [
        "Ana Milena Sánchez Villamil",
        "Daniel Santiago Vallejo Ortega",
        "Wilson Armando Bojacá González",
    ]:
        st.write(f"- {nombre_integrante}")
    st.markdown("Noviembre 2026")
    st.caption("Apoyo a la decisión. No reemplaza el criterio epidemiológico.")
