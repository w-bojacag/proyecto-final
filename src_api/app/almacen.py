"""Almacenamiento SQLite de las series semanales cargadas desde CSV.

Archivo de entrada: `data/processed/municipios/<municipio>/caracteristicas.csv` del pipeline.
Se usan estas columnas (las demás se ignoran):

    ANO                              -> anio
    SEMANA                           -> semana epidemiológica (1 a 53)
    casos_confirmados_o_probables    -> casos
    precipitation_mm                 -> lluvia_mm (acumulado semanal)
    air_temp_mean_c                  -> temp_media_c (promedio semanal)
    umbral_brote                     -> umbral_canal (límite superior del canal endémico de Bortman)

    es_brote_actual                  -> etiqueta (opcional: brote en la semana actual, 0 o 1)

exceso = casos > umbral_canal (igual que `es_brote_actual` del pipeline).
Si el archivo no trae `target_brote_t4`, la etiqueta se guarda vacía (NULL) y la gráfica no la muestra.

Clave de cada fila: (municipio, anio, semana). Al cargar un archivo:
- si la semana ya existe para ese municipio, se actualiza;
- si no existe, se agrega;
- las semanas que no vienen en el archivo se conservan.
"""
import io
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .config import settings

# columna del CSV (en minúsculas) -> nombre interno
MAPEO = {
    "ano": "anio",
    "semana": "semana",
    "casos_confirmados_o_probables": "casos",
    "precipitation_mm": "lluvia_mm",
    "air_temp_mean_c": "temp_media_c",
    "umbral_brote": "umbral_canal",
}
# columna opcional del CSV -> nombre interno (no obligatoria para cargar el archivo)
OPCIONAL = {"es_brote_actual": "etiqueta"}

ESQUEMA = """
CREATE TABLE IF NOT EXISTS serie_semanal (
    municipio_id TEXT NOT NULL,
    anio INTEGER NOT NULL,
    semana INTEGER NOT NULL,
    semana_inicio TEXT NOT NULL,
    casos INTEGER NOT NULL,
    lluvia_mm REAL NOT NULL,
    temp_media_c REAL NOT NULL,
    umbral_canal REAL NOT NULL,
    exceso INTEGER NOT NULL,
    etiqueta INTEGER,
    PRIMARY KEY (municipio_id, anio, semana)
);
CREATE TABLE IF NOT EXISTS cargas (
    municipio_id TEXT PRIMARY KEY,
    archivo TEXT NOT NULL,
    ultimas_nuevas INTEGER NOT NULL,
    ultimas_actualizadas INTEGER NOT NULL,
    cargado_en TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS configuracion (
    clave TEXT PRIMARY KEY,
    valor TEXT NOT NULL
);
"""


class CsvInvalido(ValueError):
    """El CSV no cumple el formato. El mensaje describe el problema para el usuario."""


def _conexion() -> sqlite3.Connection:
    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(settings.sqlite_path)
    con.row_factory = sqlite3.Row
    con.executescript(ESQUEMA)
    # Bases creadas antes de existir la etiqueta: se agrega la columna sin perder datos.
    columnas = {fila[1] for fila in con.execute("PRAGMA table_info(serie_semanal)")}
    if "etiqueta" not in columnas:
        con.execute("ALTER TABLE serie_semanal ADD COLUMN etiqueta INTEGER")
        con.commit()
    return con


def semana_inicio(anio: int, semana: int) -> pd.Timestamp:
    """Domingo que inicia la semana epidemiológica `semana` del año `anio`.

    Criterio INS/SIVIGILA (igual que src/cruce_ambiental_epidemiologico.py): la semana
    va de domingo a sábado y cada semana se ubica por su miércoles.
    """
    primer_dia = pd.Timestamp(year=anio, month=1, day=1)
    ventana_ini = primer_dia + pd.Timedelta(days=7 * (semana - 1))
    miercoles = ventana_ini + pd.Timedelta(days=(2 - ventana_ini.weekday()) % 7)
    return miercoles - pd.Timedelta(days=3)


def leer_csv(contenido: bytes) -> pd.DataFrame:
    """Valida el CSV de características y devuelve las filas a guardar (sin escribir en la base)."""
    try:
        texto = contenido.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = contenido.decode("latin-1")
    try:
        df = pd.read_csv(io.StringIO(texto), sep=None, engine="python")
    except Exception as exc:  # pandas no distingue bien los errores de formato
        raise CsvInvalido(f"No se pudo leer el archivo como CSV: {exc}") from exc

    df.columns = [str(c).strip().lower() for c in df.columns]
    faltan = [c for c in MAPEO if c not in df.columns]
    if faltan:
        raise CsvInvalido(
            "Faltan columnas: " + ", ".join(faltan) + ". "
            "Se requiere el archivo caracteristicas.csv del pipeline (columnas ANO, SEMANA, "
            "casos_confirmados_o_probables, precipitation_mm, air_temp_mean_c, umbral_brote)."
        )
    if df.empty:
        raise CsvInvalido("El archivo no tiene filas de datos.")

    tiene_etiqueta = "es_brote_actual" in df.columns
    df = df[list(MAPEO) + [c for c in OPCIONAL if c in df.columns]].rename(columns={**MAPEO, **OPCIONAL}).copy()
    for c in MAPEO.values():
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if tiene_etiqueta:
        df["etiqueta"] = pd.to_numeric(df["etiqueta"], errors="coerce")
        if not df["etiqueta"].dropna().isin([0, 1]).all():
            raise CsvInvalido("La columna es_brote_actual solo puede contener 0 o 1.")
    else:
        df["etiqueta"] = None

    malas = df[list(MAPEO.values())].isna().any(axis=1)
    malas = df[malas].index + 2  # +2: encabezado y base 1
    if len(malas) > 0:
        raise CsvInvalido(
            "Valores vacíos o no numéricos en las filas: "
            f"{', '.join(str(i) for i in malas[:10])}{'…' if len(malas) > 10 else ''}."
        )
    for c in ("anio", "semana", "casos"):
        if (df[c] % 1 != 0).any():
            raise CsvInvalido(f"La columna {c} debe contener enteros.")
    df[["anio", "semana", "casos"]] = df[["anio", "semana", "casos"]].astype(int)

    if not df["semana"].between(1, 53).all():
        raise CsvInvalido("La columna SEMANA debe estar entre 1 y 53.")
    if (df[["casos", "lluvia_mm", "umbral_canal"]] < 0).any().any():
        raise CsvInvalido("casos_confirmados_o_probables, precipitation_mm y umbral_brote no pueden ser negativos.")

    duplicados = df[df.duplicated(subset=["anio", "semana"], keep=False)]
    if not duplicados.empty:
        pares = sorted({f"{a}-S{s:02d}" for a, s in zip(duplicados["anio"], duplicados["semana"])})
        raise CsvInvalido(f"Hay semanas repetidas en el archivo: {', '.join(pares[:10])}.")

    df["exceso"] = (df["casos"] > df["umbral_canal"]).astype(int)
    df["semana_inicio"] = [semana_inicio(a, s).date().isoformat() for a, s in zip(df["anio"], df["semana"])]
    return df.sort_values(["anio", "semana"]).reset_index(drop=True)


def guardar_csv(municipio_id: str, nombre_archivo: str, df: pd.DataFrame) -> dict:
    """Inserta las semanas nuevas y actualiza las existentes. Las demás semanas no se tocan."""
    cargado_en = datetime.now(timezone.utc).isoformat(timespec="seconds")
    claves = [(int(a), int(s)) for a, s in zip(df["anio"], df["semana"])]
    filas = [
        (municipio_id, int(r.anio), int(r.semana), r.semana_inicio, int(r.casos),
         float(r.lluvia_mm), float(r.temp_media_c), float(r.umbral_canal), int(r.exceso),
         None if pd.isna(r.etiqueta) else int(r.etiqueta))
        for r in df.itertuples()
    ]
    with closing(_conexion()) as con, con:  # una sola transacción
        existentes = {
            (r["anio"], r["semana"])
            for r in con.execute(
                "SELECT anio, semana FROM serie_semanal WHERE municipio_id = ?", (municipio_id,)
            )
        }
        actualizadas = sum(1 for k in claves if k in existentes)
        nuevas = len(claves) - actualizadas
        con.executemany(
            "INSERT INTO serie_semanal (municipio_id, anio, semana, semana_inicio, casos, lluvia_mm, "
            "temp_media_c, umbral_canal, exceso, etiqueta) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(municipio_id, anio, semana) DO UPDATE SET "
            "semana_inicio=excluded.semana_inicio, casos=excluded.casos, lluvia_mm=excluded.lluvia_mm, "
            "temp_media_c=excluded.temp_media_c, umbral_canal=excluded.umbral_canal, exceso=excluded.exceso, "
            "etiqueta=excluded.etiqueta",
            filas,
        )
        con.execute(
            "INSERT INTO cargas (municipio_id, archivo, ultimas_nuevas, ultimas_actualizadas, cargado_en) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(municipio_id) DO UPDATE SET "
            "archivo=excluded.archivo, ultimas_nuevas=excluded.ultimas_nuevas, "
            "ultimas_actualizadas=excluded.ultimas_actualizadas, cargado_en=excluded.cargado_en",
            (municipio_id, nombre_archivo, nuevas, actualizadas, cargado_en),
        )
    return {"nuevas": nuevas, "actualizadas": actualizadas, **estado_municipio(municipio_id)}


def estado_municipio(municipio_id: str) -> dict:
    with closing(_conexion()) as con:
        carga = con.execute("SELECT * FROM cargas WHERE municipio_id = ?", (municipio_id,)).fetchone()
        total = con.execute(
            "SELECT COUNT(*) AS n, MIN(semana_inicio) AS ini, MAX(semana_inicio) AS fin "
            "FROM serie_semanal WHERE municipio_id = ?", (municipio_id,),
        ).fetchone()
        periodos = con.execute(
            "SELECT anio, semana FROM serie_semanal WHERE municipio_id = ? ORDER BY anio, semana "
            "LIMIT 1", (municipio_id,),
        ).fetchone()
        ultimo = con.execute(
            "SELECT anio, semana FROM serie_semanal WHERE municipio_id = ? ORDER BY anio DESC, semana DESC "
            "LIMIT 1", (municipio_id,),
        ).fetchone()
    if carga is None or total["n"] == 0:
        return {"cargado": False, "semanas": 0, "primer_periodo": None, "ultimo_periodo": None,
                "archivo": None, "cargado_en": None, "ultimas_nuevas": 0, "ultimas_actualizadas": 0}
    return {
        "cargado": True,
        "semanas": total["n"],
        "primer_periodo": f"{periodos['anio']}-S{periodos['semana']:02d}",
        "ultimo_periodo": f"{ultimo['anio']}-S{ultimo['semana']:02d}",
        "archivo": carga["archivo"],
        "cargado_en": carga["cargado_en"],
        "ultimas_nuevas": carga["ultimas_nuevas"],
        "ultimas_actualizadas": carga["ultimas_actualizadas"],
    }


def rango_datos(municipio_id: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Primera y última semana (inicio de semana) cargadas para el municipio."""
    with closing(_conexion()) as con:
        r = con.execute(
            "SELECT MIN(semana_inicio) AS ini, MAX(semana_inicio) AS fin FROM serie_semanal "
            "WHERE municipio_id = ?", (municipio_id,),
        ).fetchone()
    if r["ini"] is None:
        return None
    return pd.Timestamp(r["ini"]), pd.Timestamp(r["fin"])


def serie_semanal(municipio_id: str) -> pd.DataFrame | None:
    """Serie semanal del municipio, con las columnas que usa la API.

    `semana` es el domingo que inicia la semana epidemiológica.
    Devuelve None si el municipio no tiene datos cargados.
    """
    with closing(_conexion()) as con:
        filas = con.execute(
            "SELECT anio, semana AS semana_epi, semana_inicio, casos, lluvia_mm, temp_media_c, "
            "umbral_canal, exceso, etiqueta FROM serie_semanal WHERE municipio_id = ? ORDER BY anio, semana_epi",
            (municipio_id,),
        ).fetchall()
    if not filas:
        return None
    df = pd.DataFrame([dict(f) for f in filas])
    df["semana"] = pd.to_datetime(df["semana_inicio"])
    df["mes"] = df["semana"].dt.month
    df["exceso"] = df["exceso"].astype(bool)
    df["municipio_id"] = municipio_id
    df["etiqueta"] = df["etiqueta"].astype(float)
    return df[["municipio_id", "semana", "anio", "semana_epi", "mes", "casos", "lluvia_mm",
               "temp_media_c", "umbral_canal", "exceso", "etiqueta"]]


def leer_configuracion(valores_por_defecto: dict) -> dict:
    """Valores guardados en SQLite; los que faltan toman el valor por defecto (con su tipo)."""
    cfg = dict(valores_por_defecto)
    with closing(_conexion()) as con:
        for fila in con.execute("SELECT clave, valor FROM configuracion"):
            if fila["clave"] in cfg:
                cfg[fila["clave"]] = type(valores_por_defecto[fila["clave"]])(fila["valor"])
    return cfg


def guardar_configuracion(cambios: dict) -> None:
    """Inserta o actualiza los parámetros indicados."""
    with closing(_conexion()) as con, con:
        con.executemany(
            "INSERT INTO configuracion (clave, valor) VALUES (?, ?) "
            "ON CONFLICT(clave) DO UPDATE SET valor=excluded.valor",
            [(clave, str(valor)) for clave, valor in cambios.items()],
        )
