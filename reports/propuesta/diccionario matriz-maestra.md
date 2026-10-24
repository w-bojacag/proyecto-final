# Diccionario de Variables del Dataset Consolidado: Matriz Maestra Ambiental (Cartagena, 2007-2025)

Este documento describe la naturaleza, descripción detallada y fuente técnica de cada una de las variables que componen la matriz maestra ambiental (`matriz_maestra_ambiental_cartagena_2007-2025.csv`), diseñada para alimentar los modelos predictivos y de alerta temprana de dengue.

---

- **`date`**
  - **Descripción:** Fecha del registro diario de la serie temporal (formato AAAA-MM-DD). Actúa como la clave temporal principal para realizar el cruce (*merge*) posterior con los datos epidemiológicos de SIVIGILA.
  - **Fuente:** Calendario gregoriano.

- **`precipitation_mm`**
  - **Descripción:** Precipitación pluvial diaria acumulada promedio calculada para el área de estudio en milímetros ($\text{mm}$). La lluvia es el factor determinante para la creación y persistencia de criaderos de agua estancada donde se reproduce el vector *Aedes aegypti*.
  - **Fuente:** Satélite CHIRPS (`UCSB-CHG/CHIRPS/DAILY`).

- **`lst_day_c`**
  - **Descripción:** Temperatura de la superficie terrestre durante el día (*Land Surface Temperature - Day*) expresada en grados Celsius ($\text{°C}$). Refleja el calor directo absorbido por el suelo y los tejados urbanos, el cual acelera el desarrollo larvario del mosquito.
  - **Fuente:** Satélite MODIS (`MODIS/061/MOD11A1`).

- **`lst_night_c`**
  - **Descripción:** Temperatura de la superficie terrestre durante la noche (*Land Surface Temperature - Night*) en grados Celsius ($\text{°C}$). Permite evaluar la amplitud térmica diaria y las condiciones mínimas de calor nocturno en el hábitat del vector.
  - **Fuente:** Satélite MODIS (`MODIS/061/MOD11A1`).

- **`air_temp_max_c`**
  - **Descripción:** Temperatura máxima del aire registrada a 2 metros de altura sobre la superficie, en grados Celsius ($\text{°C}$). Las temperaturas extremas muy altas pueden afectar la longevidad del mosquito adulto, mientras que los rangos moderados altos optimizan la incubación del virus.
  - **Fuente:** Modelo ERA5-Land (`ECMWF/ERA5_LAND/DAILY_AGGR`).

- **`air_temp_mean_c`**
  - **Descripción:** Temperatura media diaria del aire a 2 metros de altura en grados Celsius ($\text{°C}$). Es la métrica estándar de calor ambiental utilizada comúnmente en epidemiología para estimar la tasa metabólica del vector.
  - **Fuente:** Modelo ERA5-Land (`ECMWF/ERA5_LAND/DAILY_AGGR`).

- **`air_temp_min_c`**
  - **Descripción:** Temperatura mínima diaria del aire a 2 metros de altura en grados Celsius ($\text{°C}$). Define las condiciones térmicas de las horas más frescas de la madrugada.
  - **Fuente:** Modelo ERA5-Land (`ECMWF/ERA5_LAND/DAILY_AGGR`).

- **`dewpoint_c`**
  - **Descripción:** Temperatura de punto de rocío a 2 metros de altura en grados Celsius ($\text{°C}$). Sirve como indicador directo del contenido de humedad y vapor de agua en la atmósfera; valores altos de punto de rocío se relacionan con una alta humedad relativa, la cual incrementa la supervivencia y actividad de búsqueda de hospederos del mosquito.
  - **Fuente:** Modelo ERA5-Land (`ECMWF/ERA5_LAND/DAILY_AGGR`).

- **`evi`**
  - **Descripción:** Índice de Vegetación Mejorado (*Enhanced Vegetation Index*), adimensional. Mide la densidad y verdor de la cobertura vegetal con mayor precisión en zonas de alta biomasa, corrigiendo la saturación atmosférica. Funciona como un proxy de la humedad retenida en el entorno urbano y periurbano.
  - **Fuente:** Satélite MODIS (`MODIS/061/MOD13Q1`).

- **`ndvi`**
  - **Descripción:** Índice de Vegetación de Diferencia Normalizada (*Normalized Difference Vegetation Index*), adimensional. Cuantifica la presencia, salud y distribución de la vegetación verde superficial, asociándose a microclimas favorables para el refugio de los vectores.
  - **Fuente:** Satélite MODIS (`MODIS/061/MOD13Q1`).
