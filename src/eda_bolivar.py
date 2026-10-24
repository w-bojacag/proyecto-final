"""
EDA de casos de dengue - Departamento de Bolivar
Fuente: DatosUnificados/dengue_bolivar_unificado.csv (SIVIGILA, 2007-2025)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "processed" / "dengue_bolivar_unificado.csv"
FIG_DIR = BASE_DIR.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def _parse_fecha(serie):
    # Las fechas vienen como "26/04/2007 12:00:00 a. m." (hora siempre 12:00:00,
    # sin informacion util); solo se necesita la parte de fecha dd/mm/yyyy.
    solo_fecha = serie.astype(str).str.split(" ").str[0]
    return pd.to_datetime(solo_fecha, format="%d/%m/%Y", errors="coerce")


def cargar_datos():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["FEC_NOT"] = _parse_fecha(df["FEC_NOT"])
    df["INI_SIN"] = _parse_fecha(df["INI_SIN"])
    df["FEC_CON"] = _parse_fecha(df["FEC_CON"])
    return df


def reporte_calidad(df):
    print("=" * 70)
    print("1. CALIDAD DE DATOS")
    print("=" * 70)
    print(f"Filas: {df.shape[0]:,}  Columnas: {df.shape[1]}")

    dup_total = df.duplicated().sum()
    dup_consecutive = df.duplicated(subset=["CONSECUTIVE"]).sum()
    print(f"Duplicados (fila completa): {dup_total}")
    print(f"Duplicados (CONSECUTIVE repetido): {dup_consecutive}")

    nulos = df.isna().mean().sort_values(ascending=False) * 100
    nulos = nulos[nulos > 0]
    print("\n% de nulos por columna (solo columnas con nulos):")
    print(nulos.round(2).to_string())

    print("\nRango de SEMANA:", df["SEMANA"].min(), "-", df["SEMANA"].max())
    semanas_invalidas = df[(df["SEMANA"] < 1) | (df["SEMANA"] > 53)]
    print(f"Registros con SEMANA fuera de [1,53]: {len(semanas_invalidas)}")

    print("\nRango de EDAD:", df["EDAD"].min(), "-", df["EDAD"].max())
    print("Valores de UNI_MED (unidad de medida de edad):", df["UNI_MED"].unique())

    print("\nDistribucion TIP_CAS (tipo de caso: 1=sospechoso,2=probable,3=confirmado):")
    print(df["TIP_CAS"].value_counts())

    print("\nDistribucion Estado_final_de_caso:")
    print(df["nom_est_f_caso"].value_counts())

    incons_fecha = (df["FEC_NOT"] < df["INI_SIN"]).sum()
    print(f"\nRegistros donde FEC_NOT < INI_SIN (inconsistencia temporal): {incons_fecha}")

    print(f"\nRegistros con FEC_NOT nula: {df['FEC_NOT'].isna().sum()}")
    print(f"Registros con INI_SIN nula: {df['INI_SIN'].isna().sum()}")


def investigar_inconsistencia_fechas(df):
    """
    Caracteriza los registros donde FEC_NOT < INI_SIN (notificacion antes del
    inicio de sintomas, lo cual es logicamente imposible). Confirmado: el error
    ya viene asi en el archivo fuente Datos_2013_210.xlsx (no es un bug de la
    unificacion), y esta aislado al anio 2013.
    """
    print("\n" + "=" * 70)
    print("5. INVESTIGACION: FEC_NOT < INI_SIN")
    print("=" * 70)
    mask = df["FEC_NOT"] < df["INI_SIN"]
    incons = df[mask].copy()
    incons["gap_dias"] = (incons["INI_SIN"] - incons["FEC_NOT"]).dt.days

    print(f"Total de registros inconsistentes: {len(incons)}")
    print("\nDistribucion por ANO:")
    print(incons["ANO"].value_counts().sort_index().to_string())

    total_2013 = (df["ANO"] == 2013).sum()
    pct_2013 = mask[df["ANO"] == 2013].sum() / total_2013 * 100
    print(f"\nRegistros 2013 afectados: {mask[df['ANO'] == 2013].sum()} de {total_2013} ({pct_2013:.1f}%)")

    print("\nDistribucion del gap (dias) entre FEC_NOT e INI_SIN:")
    print(incons["gap_dias"].describe().round(1).to_string())

    print("\nPor TIP_CAS:")
    print(incons["TIP_CAS"].value_counts().to_string())

    print("\nPor municipio (top 10):")
    print(incons["Municipio_ocurrencia"].value_counts().head(10).to_string())

    print(
        "\nConclusion: el error esta aislado al 2013 (afecta ~64% de los "
        "registros de ese anio) y ya viene asi en el archivo fuente original "
        "(SIVIGILA), no es un bug del script de unificacion. SEMANA/ANO/FEC_NOT "
        "son consistentes entre si, por lo que la serie semanal de conteo de "
        "casos NO se ve afectada. El campo INI_SIN (y posiblemente FEC_CON) del "
        "2013 no debe usarse para analisis de retraso de notificacion sin "
        "excluir o corregir antes estos registros."
    )
    out = BASE_DIR / "registros_inconsistentes_fecha_2013.csv"
    incons.to_csv(out, index=False)
    print(f"\nRegistros inconsistentes exportados a: {out}")
    return incons


def cobertura_municipios(df, top_n=10):
    print("\n" + "=" * 70)
    print("2. COBERTURA POR MUNICIPIO (Municipio_ocurrencia)")
    print("=" * 70)
    conteo = df["Municipio_ocurrencia"].value_counts()
    print(conteo.head(top_n).to_string())

    anios_totales = df["ANO"].nunique()
    cobertura = (
        df.groupby("Municipio_ocurrencia")["ANO"]
        .nunique()
        .sort_values(ascending=False)
    )
    resumen = pd.DataFrame({"casos": conteo, "anios_con_datos": cobertura})
    resumen["pct_anios_cubiertos"] = (resumen["anios_con_datos"] / anios_totales * 100).round(1)
    resumen = resumen.sort_values("casos", ascending=False)
    print(f"\nTop {top_n} municipios por casos y cobertura de anios (de {anios_totales} anios en el dataset):")
    print(resumen.head(top_n).to_string())

    top5 = resumen.head(5).index.tolist()
    print(f"\n>> 5 municipios candidatos (mayor volumen y cobertura): {top5}")
    return top5


def serie_semanal(df, municipios, tip_cas=None, sufijo="", guardar_csv=True):
    """
    tip_cas: lista de valores de TIP_CAS a incluir (None = todos).
             2 = probable, 3 = confirmado por laboratorio o nexo, 5 = otro/ajustado.
    """
    print("\n" + "=" * 70)
    titulo = "3. SERIE TEMPORAL SEMANAL DE CASOS"
    if sufijo:
        titulo += f" ({sufijo})"
    print(titulo)
    print("=" * 70)
    sub = df[df["Municipio_ocurrencia"].isin(municipios)].copy()
    if tip_cas is not None:
        sub = sub[sub["TIP_CAS"].isin(tip_cas)]
    serie = (
        sub.groupby(["Municipio_ocurrencia", "ANO", "SEMANA"])
        .size()
        .reset_index(name="casos")
    )
    if guardar_csv:
        nombre = f"serie_semanal_top5_bolivar{sufijo}.csv"
        out = BASE_DIR / nombre
        serie.to_csv(out, index=False)
        print(f"Serie semanal guardada en: {out}")
    print(serie.groupby("Municipio_ocurrencia")["casos"].describe().round(1))
    return serie


def comparar_definiciones_caso(df, municipios):
    """
    Compara la serie usando solo 'confirmados' (TIP_CAS=3) vs 'confirmados+probables'
    (TIP_CAS in [2,3]), para llevar argumentos a la reunion con el experto sobre
    que definicion de caso usar en el sistema de alerta.
    """
    print("\n" + "=" * 70)
    print("4. COMPARACION: SOLO CONFIRMADOS  vs.  CONFIRMADOS + PROBABLES")
    print("=" * 70)

    serie_conf = serie_semanal(df, municipios, tip_cas=[3], sufijo="_solo_confirmados")
    serie_todo = serie_semanal(df, municipios, tip_cas=[2, 3], sufijo="_confirmados_y_probables")

    resumen = []
    for mun in municipios:
        c = serie_conf[serie_conf["Municipio_ocurrencia"] == mun]["casos"]
        t = serie_todo[serie_todo["Municipio_ocurrencia"] == mun]["casos"]
        resumen.append({
            "municipio": mun,
            "total_solo_confirmados": c.sum(),
            "total_confirmados_y_probables": t.sum(),
            "pct_aporte_probables": round((1 - c.sum() / t.sum()) * 100, 1) if t.sum() else np.nan,
            "max_semanal_solo_confirmados": c.max(),
            "max_semanal_confirmados_y_probables": t.max(),
        })
    tabla = pd.DataFrame(resumen)
    print("\nResumen comparativo por municipio:")
    print(tabla.to_string(index=False))

    fig, axes = plt.subplots(len(municipios), 1, figsize=(12, 3 * len(municipios)), sharex=False)
    for ax, mun in zip(axes, municipios):
        dc = serie_conf[serie_conf["Municipio_ocurrencia"] == mun].copy()
        dt = serie_todo[serie_todo["Municipio_ocurrencia"] == mun].copy()
        dc["t"] = dc["ANO"] + (dc["SEMANA"] - 1) / 52
        dt["t"] = dt["ANO"] + (dt["SEMANA"] - 1) / 52
        ax.plot(dt["t"], dt["casos"], label="Confirmados + Probables", linewidth=0.8, color="#e07b39")
        ax.plot(dc["t"], dc["casos"], label="Solo Confirmados", linewidth=0.8, color="#2b6cb0")
        ax.set_title(mun)
        ax.set_ylabel("casos/semana")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Año (fraccional por semana epidemiológica)")
    fig.tight_layout()
    out = FIG_DIR / "comparacion_definicion_caso.png"
    fig.savefig(out, dpi=130)
    print(f"\nFigura guardada en: {out}")

    return tabla


def graficar_series(serie, municipios):
    fig, axes = plt.subplots(len(municipios), 1, figsize=(12, 3 * len(municipios)), sharex=False)
    for ax, mun in zip(axes, municipios):
        d = serie[serie["Municipio_ocurrencia"] == mun].copy()
        d["t"] = d["ANO"] + (d["SEMANA"] - 1) / 52
        ax.plot(d["t"], d["casos"], linewidth=0.8)
        ax.set_title(mun)
        ax.set_ylabel("casos/semana")
    axes[-1].set_xlabel("Año (fraccional por semana epidemiológica)")
    fig.tight_layout()
    out = FIG_DIR / "series_semanales_top5.png"
    fig.savefig(out, dpi=130)
    print(f"\nFigura guardada en: {out}")


def graficar_casos_anuales(df):
    conteo = df["ANO"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(conteo.index.astype(str), conteo.values, color="#2b6cb0")
    ax.set_title("Casos de dengue por año - Bolívar (todos los municipios)")
    ax.set_ylabel("Casos notificados")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out = FIG_DIR / "casos_por_anio.png"
    fig.savefig(out, dpi=130)
    print(f"Figura guardada en: {out}")


def main():
    df = cargar_datos()
    reporte_calidad(df)
    investigar_inconsistencia_fechas(df)
    top5 = cobertura_municipios(df)
    serie = serie_semanal(df, top5, sufijo="_todos_tip_cas")
    graficar_series(serie, top5)
    graficar_casos_anuales(df)
    comparar_definiciones_caso(df, top5)


if __name__ == "__main__":
    main()
