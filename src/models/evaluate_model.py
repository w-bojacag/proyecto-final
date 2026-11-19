"""
Evaluación clínica comparativa de los modelos de alerta temprana (brote a T+4) de un municipio.

Compara XGBoost, LSTM y regresión logística con cuatro líneas base:
- Persistencia: si hay brote en t, se anuncia brote en t+4.
- Regla casos/umbral: casos de la semana t divididos por el umbral de brote de la semana t+4.
- Alerta del canal endémico (Umaña et al., 2026): alerta cuando los casos pasan 2 o más semanas
  seguidas en la zona de alerta, entre la línea central del canal y su límite superior. Es lo que
  hoy lee el epidemiólogo en el canal. Se evalúa con la misma regla de brote avisado que los
  modelos, es decir, como aviso a 4 semanas.
- Probabilidad según el canal: frecuencia histórica de brote a 4 semanas según la zona del canal
  en las semanas t-1 y t, estimada solo con los años anteriores (walk-forward). Traduce el canal
  a una probabilidad, sin modelo.

Calidad de la probabilidad: Brier score de prueba y Brier skill score frente a la climatología
(proporción histórica de brote de los años de entrenamiento), más una curva de calibración.

Todas las predicciones son fuera de muestra (walk-forward). El umbral de decisión de cada año se
calibra con el índice de Youden sobre las predicciones del año anterior.

Incertidumbre: bootstrap por bloques de semanas consecutivas sobre el periodo de prueba
(las semanas vecinas están correlacionadas, así que no se remuestrean de forma independiente).
Para comparar modelos se usa la diferencia de AUC en las mismas muestras bootstrap.

Uso: python src/models/evaluate_model.py --municipio cartagena
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from comun import (  # noqa: E402
    TARGET, aplicar_umbral_dinamico, anos_walk_forward, cargar_matriz, load_params,
    mascara_entrenamiento, municipio_desde_argumentos, resumir_predicciones,
)

MODELOS = ["xgboost", "lstm", "logistica", "logistica_clima"]
REFERENCIA = "logistica"  # modelo contra el que se comparan los demás
CASOS = "casos_confirmados_o_probables"
# Reglas que ya emiten alertas binarias: no se les calibra umbral de Youden.
UMBRAL_FIJO = {"canal_alerta": 0.5}
# Puntajes que no son probabilidades: no se les calcula calibración.
SIN_PROBABILIDAD = {"base_casos_sobre_umbral"}
EN_CURVA_CALIBRACION = ["logistica", "xgboost", "lstm", "canal_probabilidad"]


def zona_canal(df):
    """0 = seguridad (hasta la línea central), 1 = alerta (entre la línea central y el límite
    superior), 2 = epidemia (sobre el límite superior)."""
    casos = df[CASOS]
    return pd.Series(np.select([casos > df["umbral_brote"], casos > df["centro_canal"]], [2, 1], 0),
                     index=df.index)


def alerta_canal(df):
    """Alerta del canal según Umaña et al. (2026): 2 o más semanas seguidas en zona de alerta."""
    en_alerta = zona_canal(df) == 1
    return (en_alerta & en_alerta.shift(1, fill_value=False)).astype(float)


def probabilidad_canal(df, particion, horizonte):
    """Frecuencia de brote a T+h según la zona del canal en t-1 y t, estimada en cada año del
    walk-forward con las semanas de entrenamiento. Se suaviza hacia la proporción de brote del
    entrenamiento (equivale a 2 semanas de prior) para los estados poco observados."""
    zona = zona_canal(df)
    estado = zona.shift(1, fill_value=0) * 3 + zona
    prob = pd.Series(np.nan, index=df.index)
    for ano in anos_walk_forward(df, particion):
        m = mascara_entrenamiento(df["ANO"], ano, horizonte)
        y = df.loc[m, TARGET]
        base = y.mean()
        conteo = y.groupby(estado[m]).agg(["sum", "count"])
        frecuencia = (conteo["sum"] + 2 * base) / (conteo["count"] + 2)
        del_ano = df["ANO"] == ano
        prob[del_ano] = estado[del_ano].map(frecuencia).fillna(base)
    return prob


def calibracion(predicciones, df, particion, horizonte):
    """Brier score de prueba y Brier skill score frente a la climatología: la proporción de
    semanas con brote en el entrenamiento de cada año del walk-forward."""
    climatologia = {
        ano: df.loc[mascara_entrenamiento(df["ANO"], ano, horizonte), TARGET].mean()
        for ano in anos_walk_forward(df, particion)
    }
    resultado = {}
    for nombre, p in predicciones.items():
        test = p[p["ANO"] > particion["val_end_year"]]
        if nombre in SIN_PROBABILIDAD:
            resultado[nombre] = {"brier_test": float("nan"), "brier_skill_test": float("nan")}
            continue
        brier = float(np.mean((test["prob"] - test["y"]) ** 2))
        referencia = float(np.mean((test["ANO"].map(climatologia) - test["y"]) ** 2))
        resultado[nombre] = {"brier_test": brier, "brier_skill_test": 1 - brier / referencia}
    return resultado


def curva_calibracion(predicciones, particion, rutas, n_grupos=5):
    """Probabilidad media frente a proporción observada de brote, en grupos de igual tamaño de las
    semanas de prueba. Pocos grupos porque hay unas 150 semanas de prueba."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Calibración perfecta")
    for nombre in EN_CURVA_CALIBRACION:
        test = predicciones[nombre]
        test = test[test["ANO"] > particion["val_end_year"]]
        grupos = pd.qcut(test["prob"].rank(method="first"), n_grupos, labels=False)
        puntos = test.groupby(grupos).agg(prob=("prob", "mean"), obs=("y", "mean"))
        ax.plot(puntos["prob"], puntos["obs"], marker="o", label=nombre)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Probabilidad predicha (media del grupo)")
    ax.set_ylabel("Proporción observada de semanas con brote")
    ax.set_title(f"Calibración en prueba 2023-2025 - {rutas.nombre}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(rutas.reportes / "calibracion.png", dpi=120)
    plt.close(fig)


def remuestras_por_bloques(n, tamano_bloque, repeticiones, semilla):
    """Índices bootstrap formados por bloques consecutivos (circulares) de semanas."""
    rng = np.random.default_rng(semilla)
    n_bloques = int(np.ceil(n / tamano_bloque))
    for _ in range(repeticiones):
        inicios = rng.integers(0, n, n_bloques)
        idx = (inicios[:, None] + np.arange(tamano_bloque)[None, :]).ravel() % n
        yield idx[:n]


def bootstrap(predicciones, particion, params):
    """Intervalos del 95 % para el AUC y la sensibilidad de prueba, y diferencias de AUC
    de cada modelo frente al de referencia."""
    prueba = {}
    for nombre, p in predicciones.items():
        p = aplicar_umbral_dinamico(p, UMBRAL_FIJO.get(nombre))
        prueba[nombre] = p[p["ANO"] > particion["val_end_year"]].reset_index(drop=True)
    y = prueba[REFERENCIA]["y"].to_numpy()

    aucs = {n: [] for n in prueba}
    sens = {n: [] for n in prueba}
    for idx in remuestras_por_bloques(len(y), params["tamano_bloque"], params["repeticiones"], params["semilla"]):
        if len(np.unique(y[idx])) < 2:
            continue
        for nombre, p in prueba.items():
            aucs[nombre].append(roc_auc_score(y[idx], p["prob"].to_numpy()[idx]))
            positivos = y[idx] == 1
            sens[nombre].append(p["alerta"].to_numpy()[idx][positivos].mean())

    def ic(valores):
        return [float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))]

    resultado = {}
    for nombre in prueba:
        diferencia = np.array(aucs[nombre]) - np.array(aucs[REFERENCIA])
        resultado[nombre] = {
            "ic95_roc_auc_test": ic(aucs[nombre]),
            "ic95_sensibilidad_test": ic(sens[nombre]),
            f"dif_auc_vs_{REFERENCIA}": float(np.mean(diferencia)),
            f"ic95_dif_auc_vs_{REFERENCIA}": ic(diferencia),
        }
    return resultado


def evaluar_modelos(rutas):
    particion = load_params("particion")
    params = load_params("evaluate")
    horizonte = load_params("target")["horizonte"]
    completo = cargar_matriz(rutas)
    canal_alerta = alerta_canal(completo)
    canal_prob = probabilidad_canal(completo, particion, horizonte)
    df = completo[completo["ANO"] >= particion["train_end_year"]]
    base = df[["ANO", "SEMANA"]].assign(y=df[TARGET].astype(int))

    predicciones = {
        nombre: pd.read_csv(rutas.reportes / f"predicciones_{nombre}.csv")
        .sort_values(["ANO", "SEMANA"]).reset_index(drop=True)
        for nombre in MODELOS
    }
    predicciones["base_persistencia"] = base.assign(prob=df["es_brote_actual"].astype(float))
    predicciones["base_casos_sobre_umbral"] = base.assign(prob=df["ratio_casos_umbral_objetivo"])
    predicciones["canal_alerta"] = base.assign(prob=canal_alerta[df.index])
    predicciones["canal_probabilidad"] = base.assign(prob=canal_prob[df.index])
    for nombre, p in predicciones.items():  # todos deben evaluar exactamente las mismas semanas
        if not p[["ANO", "SEMANA"]].reset_index(drop=True).equals(base[["ANO", "SEMANA"]].reset_index(drop=True)):
            raise ValueError(f"Las semanas de '{nombre}' no coinciden con las de la matriz")

    resultados = {
        nombre: resumir_predicciones(p, particion, UMBRAL_FIJO.get(nombre))
        for nombre, p in predicciones.items()
    }
    for extra in (bootstrap(predicciones, particion, params),
                  calibracion(predicciones, completo, particion, horizonte)):
        for nombre, valores in extra.items():
            resultados[nombre].update(valores)
    curva_calibracion(predicciones, particion, rutas)
    with open(rutas.reportes / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4)

    filas = []
    for nombre, r in resultados.items():
        filas.append({
            "modelo": nombre,
            "AUC val": f"{r['roc_auc_val']:.3f}",
            "AUC prueba [IC95]": f"{r['roc_auc_test']:.3f} [{r['ic95_roc_auc_test'][0]:.2f}-{r['ic95_roc_auc_test'][1]:.2f}]",
            f"dif. vs {REFERENCIA} [IC95]": (
                f"{r[f'dif_auc_vs_{REFERENCIA}']:+.3f} "
                f"[{r[f'ic95_dif_auc_vs_{REFERENCIA}'][0]:+.2f} a {r[f'ic95_dif_auc_vs_{REFERENCIA}'][1]:+.2f}]"
            ),
            "PR-AUC": f"{r['pr_auc_test']:.3f}",
            "Brier": f"{r['brier_test']:.3f}",
            "BSS": f"{r['brier_skill_test']:+.2f}",
            "sensib. [IC95]": f"{r['test_sensibilidad']:.0%} [{r['ic95_sensibilidad_test'][0]:.0%}-{r['ic95_sensibilidad_test'][1]:.0%}]",
            "especif.": f"{r['test_especificidad']:.0%}",
            "precisión": f"{r['test_precision']:.0%}",
            "VP/FP/FN/VN": f"{r['test_vp']}/{r['test_fp']}/{r['test_fn']}/{r['test_vn']}",
            "brotes avisados": f"{r['test_brotes_avisados']}/{r['test_brotes']}",
            "razón alertas": f"{r['test_razon_alertas']:.2f}",
            "desempeño evento": f"{r['test_desempeno_evento']:.2f}",
            "ventana (sem)": f"{r['test_ventana_oportunidad_mediana']:.0f}",
        })
    print("=" * 70)
    print(f"EVALUACIÓN COMPARATIVA - {rutas.nombre} (walk-forward; prueba 2023-2025)")
    print("=" * 70)
    print(f"Prevalencia de brote en prueba: {resultados[REFERENCIA]['prevalencia_test']:.1%}")
    print(pd.DataFrame(filas).to_string(index=False))
    print(f"\nMétricas guardadas en: {rutas.reportes / 'eval_metrics.json'}")


if __name__ == "__main__":
    evaluar_modelos(municipio_desde_argumentos("Evaluación comparativa de un municipio"))
