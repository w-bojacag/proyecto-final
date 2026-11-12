"""
Entrenamiento de una red LSTM para la alerta temprana de brote de dengue a T+4 en un municipio.

Usa la misma etiqueta, las mismas semanas y el mismo protocolo walk-forward que XGBoost:
1. Se fija el número de épocas con early stopping, entrenando hasta 2019 y validando en 2020-2022.
2. Para cada año Y se entrena con todos los años anteriores durante ese número de épocas y se
   predice Y. Cada predicción es el promedio de varias semillas (la red es inestable con pocas
   secuencias de entrenamiento).
El escalado se ajusta solo con los años de entrenamiento de cada iteración. Los modelos y
escaladores de cada año se guardan en models/<municipio>/walk_forward/.

Uso: python src/models/train_lstm.py --municipio cartagena
"""

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout

from comun import (
    TARGET, anos_walk_forward, cargar_matriz, carpeta_walk_forward, load_params,
    mascara_entrenamiento, municipio_desde_argumentos, resumir_predicciones,
)
from data_generator import crear_ventanas_temporales

CLIMA = [
    "precipitation_mm", "lst_day_c", "lst_night_c", "air_temp_max_c",
    "air_temp_mean_c", "air_temp_min_c", "dewpoint_c", "evi", "ndvi",
]
# Variables epidemiológicas en escala logarítmica: volumen de casos y su relación con el umbral
# (casos de la semana frente al umbral actual y al umbral de la semana objetivo).
EPI = ["casos_confirmados_o_probables", "exceso_actual", "ratio_casos_umbral_objetivo"]


def preparar_secuencias(df, window_size, incluir_clima):
    """Devuelve las variables por semana y el índice de fila (semana t) de cada ventana."""
    valores = df[(CLIMA if incluir_clima else []) + EPI].copy()
    valores[EPI] = np.log1p(valores[EPI])
    filas = np.arange(window_size - 1, len(df))
    return valores.to_numpy(dtype="float32"), filas


def construir_modelo(params, window_size, n_variables, semilla):
    tf.keras.utils.set_random_seed(semilla)
    modelo = Sequential([
        Input(shape=(window_size, n_variables)),
        LSTM(params["lstm_units"]),
        Dropout(params["dropout_rate"]),
        Dense(1, activation="sigmoid"),
    ])
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return modelo


def datos_hasta(df, valores, filas, mascara_train, window_size):
    """Ventanas escaladas con la media y desviación de las semanas de entrenamiento."""
    escalador = StandardScaler().fit(valores[np.where(mascara_train)[0]])
    X = crear_ventanas_temporales(escalador.transform(valores).astype("float32"), window_size)
    y = df[TARGET].to_numpy(dtype="float32")[filas]
    return X, y, escalador


def entrenar_modelo_lstm(rutas):
    print("=" * 70)
    print(f"LSTM - ALERTA TEMPRANA DE BROTE T+4 (WALK-FORWARD) - {rutas.nombre}")
    print("=" * 70)

    tf.config.experimental.enable_op_determinism()
    df = cargar_matriz(rutas)
    params = load_params("lstm")
    particion = load_params("particion")
    horizonte = load_params("target")["horizonte"]
    w = params["window_size"]

    valores, filas = preparar_secuencias(df, w, params["incluir_clima"])
    anos = df["ANO"].to_numpy()
    ano_fila = anos[filas]

    # 1. Número de épocas: early stopping con entrenamiento <= 2019 y validación 2020-2022
    m = mascara_entrenamiento(anos, particion["train_end_year"] + 1, horizonte)
    tr = m[filas]
    va = (ano_fila > particion["train_end_year"]) & (ano_fila <= particion["val_end_year"])
    X, y, _ = datos_hasta(df, valores, filas, m, w)
    epocas = []
    for semilla in params["semillas"]:
        modelo = construir_modelo(params, w, X.shape[2], semilla)
        parada = EarlyStopping(monitor="val_loss", patience=params["patience"], restore_best_weights=True)
        hist = modelo.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=params["max_epochs"],
                          batch_size=params["batch_size"], callbacks=[parada], verbose=0)
        epocas.append(int(np.argmin(hist.history["val_loss"])) + 1)
    n_epocas = max(1, int(round(np.median(epocas))))
    print(f"Épocas por semilla (early stopping): {epocas} -> se usan {n_epocas}")

    # 2. Walk-forward anual
    carpeta_wf = carpeta_walk_forward(rutas, "lstm")
    predicciones = []
    for ano in anos_walk_forward(df, particion):
        m = mascara_entrenamiento(anos, ano, horizonte)
        X, y, escalador = datos_hasta(df, valores, filas, m, w)
        joblib.dump(escalador, carpeta_wf / f"{ano}_escalador.joblib")
        tr, te = m[filas], ano_fila == ano
        probs = []
        for semilla in params["semillas"]:
            modelo = construir_modelo(params, w, X.shape[2], semilla)
            modelo.fit(X[tr], y[tr], epochs=n_epocas, batch_size=params["batch_size"], verbose=0)
            modelo.save(carpeta_wf / f"{ano}_semilla{semilla}.keras")
            probs.append(modelo.predict(X[te], verbose=0).ravel())
        predicciones.append(pd.DataFrame({
            "ANO": ano, "SEMANA": df["SEMANA"].to_numpy()[filas][te],
            "y": y[te].astype(int), "prob": np.mean(probs, axis=0),
        }))
        print(f"  {ano}: entrenado con {tr.sum()} secuencias (hasta {ano - 1})")

    pred = pd.concat(predicciones, ignore_index=True)
    pred.to_csv(rutas.reportes / "predicciones_lstm.csv", index=False)

    # Modelo final (primera semilla) entrenado con toda la historia, para uso operativo
    X, y, escalador = datos_hasta(df, valores, filas, np.ones(len(df), dtype=bool), w)
    modelo = construir_modelo(params, w, X.shape[2], params["semillas"][0])
    modelo.fit(X, y, epochs=n_epocas, batch_size=params["batch_size"], verbose=0)
    modelo.save(rutas.modelos / "lstm.keras")
    joblib.dump(escalador, rutas.modelos / "lstm_escalador.joblib")

    resumen = resumir_predicciones(pred, particion)
    metrics = {k: resumen[k] for k in ("roc_auc_val", "roc_auc_test", "pr_auc_test", "roc_auc_por_ano")}
    metrics["epocas"] = n_epocas
    with open(rutas.reportes / "metrics_lstm.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "-" * 50)
    print(f"Validación walk-forward (2020-2022) -> ROC-AUC: {metrics['roc_auc_val']:.4f}")
    print(f"Prueba walk-forward (2023-2025)     -> ROC-AUC: {metrics['roc_auc_test']:.4f}"
          f" | PR-AUC: {metrics['pr_auc_test']:.4f}")
    for ano, valor in metrics["roc_auc_por_ano"].items():
        print(f"    {ano}: {valor:.4f}")


if __name__ == "__main__":
    entrenar_modelo_lstm(municipio_desde_argumentos("Entrena la LSTM para un municipio"))
