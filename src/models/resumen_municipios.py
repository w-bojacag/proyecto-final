"""
Resumen comparativo de todos los municipios: reúne el eval_metrics.json de cada uno en una tabla
(reports/resumen_municipios.csv) y en un JSON de métricas para DVC.

Cada municipio tiene modelos independientes; esta tabla solo los pone lado a lado para comparar
en qué municipios funciona la alerta y con qué modelo.
"""

import json

import pandas as pd

from comun import ROOT_DIR, cargar_matriz, load_params
from rutas import rutas_municipio

COLUMNAS = [
    "roc_auc_val", "roc_auc_test", "ic95_roc_auc_test", "pr_auc_test", "prevalencia_test",
    "test_sensibilidad", "test_especificidad", "test_precision",
    "test_vp", "test_fp", "test_fn", "test_vn",
    "test_brotes", "test_brotes_avisados", "test_capacidad_deteccion", "test_razon_alertas",
    "test_desempeno_evento", "test_ventana_oportunidad_mediana",
]


def main():
    filas, resumen = [], {}
    for slug in load_params("municipios"):
        rutas = rutas_municipio(slug)
        with open(rutas.reportes / "eval_metrics.json", encoding="utf-8") as f:
            evaluacion = json.load(f)
        df = cargar_matriz(rutas)
        contexto = {
            "semanas": len(df),
            "casos_promedio_semana": round(float(df["casos_confirmados_o_probables"].mean()), 2),
            "semanas_con_umbral_cero": round(float((df["umbral_brote"] == 0).mean()), 3),
        }
        resumen[slug] = {"contexto": contexto, **{m: {k: r[k] for k in COLUMNAS} for m, r in evaluacion.items()}}
        for modelo, r in evaluacion.items():
            filas.append({"municipio": slug, "modelo": modelo, **contexto,
                          **{k: r[k] for k in COLUMNAS if k != "ic95_roc_auc_test"},
                          "ic95_auc_inf": r["ic95_roc_auc_test"][0], "ic95_auc_sup": r["ic95_roc_auc_test"][1]})

    tabla = pd.DataFrame(filas)
    tabla.to_csv(ROOT_DIR / "reports" / "resumen_municipios.csv", index=False)
    with open(ROOT_DIR / "reports" / "resumen_municipios.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=4)

    print("=" * 70)
    print("RESUMEN POR MUNICIPIO (prueba 2023-2025, walk-forward)")
    print("=" * 70)
    vista = tabla.pivot(index="municipio", columns="modelo", values="roc_auc_test").round(3)
    print("ROC-AUC de prueba:")
    print(vista.to_string())
    print("\nContexto de cada municipio:")
    print(tabla.drop_duplicates("municipio").set_index("municipio")[
        ["semanas", "casos_promedio_semana", "semanas_con_umbral_cero", "prevalencia_test"]].to_string())


if __name__ == "__main__":
    main()
