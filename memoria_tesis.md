# Memoria Metodológica y Hallazgos del Proyecto (Para Informe Final)

Este documento es de uso estrictamente personal y no se subirá al repositorio de control de versiones. Su propósito es servir de insumo o borrador para redactar los hallazgos en el informe final de tesis (ej. en `informe_avance.docx`).

## 1. Arquitectura del Pipeline de Machine Learning
El proyecto utiliza **DVC (Data Version Control)** para garantizar la reproducibilidad de los experimentos epidemiológicos. Las etapas consolidadas son:
1. **`clima`**: Descarga de datos ambientales (GEE) para el polígono urbano.
2. **`cruce`**: Fusión de datos meteorológicos semanales con reportes de dengue (SIVIGILA).
3. **`features`**: Ingeniería de variables temporales (Rezagos temporales - *Lags*, Medias móviles espaciales y temporales).
4. **`train`**: Entrenamiento del modelo predictivo (XGBoost) manejando hiperparámetros estáticos desde `params.yaml`.
5. **`evaluate`**: Evaluación de clasificación clínica (Matriz de confusión) sobre el conjunto de prueba ciego (2023-2025).

## 2. Ajuste de Hiperparámetros (Evitar Sobreajuste)
* Inicialmente, variaciones en `build_features.py` indujeron a un fuerte *overfitting* (AUC > 0.96 en entrenamiento, pero cayendo drásticamente en validación/prueba).
* Se utilizaron técnicas de regularización en el XGBoost (`max_depth: 1`, regularización L1/L2 a través de `reg_alpha` y `reg_lambda`, e iteraciones controladas).

## 3. Matriz de Confusión y el Índice de Youden (Hallazgo Principal)
El modelo de clasificación binaria ("Brote" vs "Normal") requería un **umbral de decisión clínico**.

1. **Trade-off Sensibilidad vs. Falsos Positivos:**
   * Un umbral estático en **0.5**: 95.8% Sensibilidad / 31 Falsos Positivos. (Modelo "Paranoico", alto costo al sistema de salud).
   * Un umbral estricto en **0.7**: 0% Sensibilidad / 0 FP. (Modelo conservador e inútil, no detectó nada).

2. **Implementación del Índice de Youden ($J$):**
   * Metodológicamente, se implementó un script que **aprende el umbral ideal** exclusivamente basándose en la Curva ROC del conjunto de **Validación (2020-2022)**, evitando el sesgo o fuga de información.
   * La fórmula utilizada fue: $J = Sensibilidad (TPR) - Tasa de Falsos Positivos (FPR)$.
   * El umbral óptimo calculado fue **0.4455**.

3. **La Fase Híper-Endémica de Cartagena (2023-2025):**
   * Al aplicar ciegamente el umbral óptimo de 0.4455 al conjunto de **Prueba (2023-2025)**, el modelo clasificó 149 de 153 semanas (casi el 100%) como alertas de Brote (TN = 0).
   * **Conclusión Epidemiológica:** Los niveles base (normales) de casos de dengue en Cartagena experimentaron un alza estructural sin precedentes después de la pandemia de COVID-19. 
   * **Implicación Metodológica:** Al calcular el objetivo ("Umbral Percentil 75") tomando en cuenta **toda la historia histórica (2007-2019)**, los datos pos-pandémicos superan ese umbral constantemente.

## 4. Próximos Pasos (Hoja de Ruta)
1. **Canal Endémico Móvil/Dinámico:** En lugar de calcular el Percentil 75 sobre los 15 años de historia completa, se propondrá modificar el cálculo de la variable objetivo (`target_brote_t4`) usando una ventana móvil (ej. promedios semanales de los últimos 5 a 7 años), adaptando la sensibilidad del modelo al comportamiento endémico reciente.
2. **Despliegue a Bolívar:** Extender la estructura del proyecto DVC para modelar los municipios más afectados de Bolívar y observar el comportamiento geográfico de los brotes.

## 5. Implementación del Canal Endémico Móvil
* **Decisión de Diseño:** En `train_xgboost.py`, se modificó la función `crear_target_brote_t4`. El cálculo del "Percentil 75 global" fue reemplazado por un **Percentil 75 móvil de 5 años**. Es decir, para evaluar si la semana 10 de 2024 es un brote, el modelo busca el percentil 75 de las semanas 10 comprendidas entre 2019 y 2023 exclusivamente.
* **Impacto Clínico:** 
  * Los Casos Reales de Brote (Test) bajaron matemáticamente de 119 a **79**, pues el umbral exigido subió al nivel de la hiperendemia post-COVID.
  * El modelo logró una **Sensibilidad del 92.4%** (detectó 73 de 79 brotes a tiempo).
  * Los **Falsos Positivos subieron a 72**, lo cual revela que el XGBoost está perdiendo poder de discriminación (AUC Test cayó a 0.49). Esto sugiere que predecir a T+4 en escenarios altamente inestables requiere incluir más rezagos históricos de casos o repensar la importancia del clima.

## 6. Inyección de Variables de Velocidad y Estacionalidad
* **Diagnóstico:** El modelo estaba perdiendo poder predictivo bajo el Canal Endémico Móvil porque XGBoost descartaba inadvertidamente las variables de **tasa de cambio** (velocidad epidémica) en `train_xgboost.py` debido a una regla de exclusión de fugas demasiado estricta. Además, carecía de noción de estacionalidad (meses/trimestres).
* **Solución Técnica:**
  1. Se corrigió `train_xgboost.py` (y `evaluate_model.py`) para que incluyan las variables terminadas en `_tasa_cambio_1` (velocidad a 1 semana) y `_tasa_cambio_4` (velocidad a 1 mes).
  2. En `build_features.py` se añadieron las columnas de `mes_aproximado` y `trimestre_aproximado`.
* **Resultados Clínicos (Mejora del ROC-AUC):**
  * Las tasas de cambio irrumpieron inmediatamente en el **Top 10 de características más importantes** (posiciones 4 y 5), demostrando que XGBoost necesitaba "ver" la aceleración de los casos, no solo el volumen absoluto.
  * El **ROC-AUC en validación subió de 0.76 a 0.79**, y en **Test recuperó terreno de 0.49 a 0.57**.
  * A nivel de matriz de confusión (Youden: 0.4267), los Falsos Positivos se mantuvieron en 72, indicando que el modelo aún prioriza no dejar pasar ningún brote, pero su capacidad de ranqueo (AUC) va mejorando.

## 7. Límite de la Regularización y Desactivación del scale_pos_weight
* Para intentar reducir las alarmas falsas, se aplicó una regularización extrema (L1/L2 a 20.0, learning rate 0.005, min_child_weight 15).
* **Diagnóstico Metodológico:** La agresiva regularización estaba siendo contrarrestada por el parámetro `scale_pos_weight` calculado dinámicamente. Este multiplicador forzaba matemáticamente al modelo a temerle tanto a un Falso Negativo que prefería lanzar alertas indiscriminadas.
* **Solución y Hallazgo Final:** Se desactivó el sesgo artificial (`scale_pos = 1.0`). Como resultado, las probabilidades predichas se normalizaron y el **AUC en Validación alcanzó su máximo histórico (0.815)**.
* **Problema Persistente:** En el conjunto de Prueba (2023-2025), el AUC volvió a caer (0.53) y los Falsos Positivos se mantuvieron en 74 (True Negatives = 0). Al ver la importancia de variables, el `roll_mean_4` acapara el 99% del peso. 
* **Conclusión Definitiva sobre Cartagena:** El clima ha dejado de ser predictivo a T+4 en la actual fase de hiperendemia. El modelo XGBoost se convirtió en un "predictor ingenuo" (Naïve forecaster) que simplemente repite los casos del mes pasado, asumiendo que si la hiperendemia sigue, siempre habrá brote. Predecir a 4 semanas requerirá una nueva métrica epidemiológica que no se sature.

## 8. Transición Metodológica a Deep Learning (LSTM)
* **Justificación de la Transición:** Se ha demostrado empíricamente que XGBoost alcanzó su frontera metodológica. Al ser un algoritmo basado en árboles tabulares, no posee entendimiento nativo de secuencias temporales ni puede abstraer el "contexto" a largo plazo (ej. lluvias fuertes seguidas de calentamiento semanas después). La hiperendemia ahogó su capacidad de usar los rezagos climáticos artificiales.
* **Nueva Arquitectura:** Se decidió pivotar hacia **Redes Neuronales Recurrentes (LSTM)**. Las arquitecturas LSTM están diseñadas específicamente para retener memoria secuencial. 
* **Reto Principal:** A diferencia del formato tabular (2D), LSTM requiere transformar el dataset epidemiológico en **Tensores 3D** (Ventanas Temporales Deslizantes). Además, con un dataset limitado (~940 semanas), el control del sobreajuste (Regularización Dropout) será la principal prioridad técnica.
* **Primeros Resultados (Prueba de Concepto):** Se implementó una arquitectura base (1 capa LSTM de 32 unidades + Dropout 0.4 + Dense Sigmoid) con ventanas de 12 semanas (un trimestre).
  * El modelo sobreajustó rápidamente, activando el *Early Stopping* en la época 11.
  * Al restaurar los pesos de la época 1, el AUC de Validación fue `0.68`, pero en el conjunto hiperendémico de Prueba (Test) colapsó a `0.27`.
  * **Diagnóstico Inicial LSTM:** Las redes neuronales profundas sufren drásticamente ante "Data Shifts" (cambios estructurales en la distribución de los datos, como pasar de endemia a hiperendemia).
* **El Problema de la Normalización por Ventana:** Para combatir el Data Shift, se intentó aplicar una normalización Z-score aislada para cada ventana temporal de 12 semanas, junto con una reducción de la arquitectura (16 neuronas, 0.5 Dropout) para evitar el sobreajuste.
  * **Hallazgo Metodológico:** El AUC de entrenamiento cayó a `0.41` y validación a `0.29`. Al normalizar independientemente cada ventana, se destruyó la "magnitud absoluta" de la epidemia. Una ventana con [1, 2, 3] casos se normaliza exactamente igual que una ventana con [100, 200, 300] casos. La red LSTM pudo ver la "forma" de la curva, pero quedó completamente ciega al volumen de pacientes, imposibilitándola para predecir si se superaría el umbral de brote.
* **Estrategia de Normalización Híbrida:** Se implementó una solución donde las variables climáticas se normalizaron globalmente (Z-score), y a los casos epidemiológicos se les aplicó una transformación logarítmica (`np.log1p`) para comprimir la escala hiperendémica sin perder la magnitud absoluta.
  * **Resultados Clínicos:** El entrenamiento mejoró sustancialmente (Train AUC: 0.98, Val AUC: 0.76). La red logró converger de manera mucho más estable y aprender la dinámica general de los datos históricos. 
  * **Conclusión Final de Modelado (El Límite del Clima):** A pesar de la mejora técnica en la red LSTM, el AUC en el conjunto hiperendémico de Prueba (2023-2025) volvió a caer a `0.40`. Esto consolida de manera irrefutable nuestro hallazgo anterior: **Durante una fase hiperendémica prolongada, la señal climática y el historial de casos a mediano plazo (T+4) pierden su capacidad predictiva frente a umbrales móviles.** El ecosistema ya está tan saturado que el inicio de un nuevo pico no depende de las lluvias de hace 4 semanas, sino de dinámicas complejas que escapan a este conjunto de datos. 
  * **Siguientes Pasos (Expansión Espacial):** Concluido el análisis exhaustivo en Cartagena, la tesis debe avanzar hacia la validación de estos modelos en municipios con dinámicas diferentes (Magangué, El Carmen de Bolívar, etc.) para comprobar si la predicción climática es viable en entornos no hiperendémicos.

## 9. Validación Walk-Forward (Re-entrenamiento Continuo en Hiperendemia)
* **Planteamiento:** Se cuestionó si el colapso del modelo en la fase hiperendémica (2023-2025) se debía a la "Fuga de Datos Estática". Es decir, el modelo entrenado hasta 2022 quedó obsoleto. Para remediarlo, se implementó en `train_xgboost.py` una validación **Walk-Forward (Ventana Deslizante)**.
* **Mecánica:** El modelo se re-entrenó año tras año. Entrenó hasta 2022 para predecir 2023; luego ingirió los datos de 2023, se re-entrenó y predijo 2024; etc.
* **Resultados Clínicos:**
  * **2023 (ROC-AUC 0.63):** Al predecir el primer año de hiperendemia, el modelo mejoró su ordenamiento local frente a la partición estática.
  * **2024 (ROC-AUC 0.62):** Tras aprender de 2023, mantuvo un ordenamiento local aceptable.
  * **2025 (ROC-AUC 0.52):** La variabilidad climática extrema hizo colapsar de nuevo la precisión.
  * **El Hallazgo Paradójico (AUC Global: 0.40):** Aunque dentro de cada año el modelo puede ordenar moderadamente bien el riesgo, al juntar las predicciones de los 3 años, el AUC global se desploma a `0.40`. ¿Por qué? Porque al re-entrenarse cada año con más datos hiperendémicos, **la escala absoluta de las probabilidades se desfasa**. Un 70% de probabilidad en 2023 no significa lo mismo que un 70% en 2025. 
* **Corolario para el Sistema de Alerta:** Un sistema que se re-entrena continuamente (Walk-Forward) requiere **umbrales de decisión dinámicos**. Un umbral fijo (Índice de Youden calculado en 2022) dejará de funcionar en 2024 porque el modelo recalibró sus probabilidades.
