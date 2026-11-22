# API – alerta temprana de dengue (Bolívar)

API FastAPI que predice, para cada municipio, la probabilidad de exceso de casos de dengue cuatro semanas después de una fecha de corte. El modelo es la regresión logística del paquete `dengue-modelo-logistica` (`.whl`). Las series se leen de una base SQLite, que se llena cargando el `caracteristicas.csv` de cada municipio.

Para que la API responda predicciones, hay que cargar antes los datos de cada municipio (ver [Carga de datos](#carga-de-datos)).

## Endpoints

| Método | Ruta | Uso |
|---|---|---|
| GET | `/api/v1/health` | Estado del servicio y del modelo |
| GET | `/api/v1/municipios` | Lista de municipios |
| GET | `/api/v1/rango-fechas` | Fechas de corte válidas (las que tienen datos) |
| GET | `/api/v1/municipios/{id}/serie` | Casos, lluvia, temperatura, umbral del canal, exceso y etiqueta |
| POST | `/api/v1/prediccion` | Probabilidad de exceso a 4 semanas para una fecha de corte |
| GET | `/api/v1/modelo` | Información del modelo y estado de métricas |
| GET | `/api/v1/configuracion` | Parámetros ajustables guardados en SQLite (umbral de decisión, año mínimo de gráficas, ventana de historial) |
| PUT | `/api/v1/configuracion` | Cambia uno o más de esos parámetros. Los campos omitidos no cambian |
| GET | `/api/v1/datos/estado` | Lo cargado en SQLite por municipio |
| POST | `/api/v1/datos/{municipio_id}/csv` | Carga `caracteristicas.csv` semanal en SQLite |

Documentación interactiva: `http://localhost:8000/docs`.

Ejemplo de predicción:

```json
POST /api/v1/prediccion
{"municipio_id": "cartagena", "fecha_corte": "2024-06-02"}
```

## Carga de datos

Los datos se cargan desde el dashboard, sin comandos: abre la pestaña **3. Configuración**, sección **Configuración de datos**, elige el municipio, selecciona su archivo `data/processed/municipios/<municipio>/caracteristicas.csv` (el que genera el pipeline) y cárgalo. Repite para los cinco municipios. El dashboard envía el archivo a la API, que lo guarda en SQLite.

Se carga el archivo tal cual lo genera el pipeline (una fila por año y semana epidemiológica). La API usa estas columnas y ignora el resto:

| Columna del CSV | Se guarda como | Significado |
|---|---|---|
| `ANO` | `anio` | Año epidemiológico |
| `SEMANA` | `semana` | Semana epidemiológica (1 a 53) |
| `casos_confirmados_o_probables` | `casos` | Casos de dengue de la semana (confirmados o probables) |
| `precipitation_mm` | `lluvia_mm` | Lluvia acumulada de la semana (mm) |
| `air_temp_mean_c` | `temp_media_c` | Temperatura media de la semana (°C) |
| `umbral_brote` | `umbral_canal` | Límite superior del canal endémico de Bortman |
| `es_brote_actual` (opcional) | `etiqueta` | Brote en la semana actual (0 o 1). Si no viene en el archivo, la gráfica no la muestra |

- `exceso` = `casos > umbral_canal`, igual que `es_brote_actual` en el pipeline.
- La fecha de inicio de cada semana (domingo) se calcula con el criterio INS/SIVIGILA.
- **Carga por actualización:** la clave es `(municipio, ANO, SEMANA)`. Si la semana ya existe se **actualiza**; si no existe se **agrega**; las semanas que no vienen en el archivo se **conservan**. La respuesta indica cuántas fueron nuevas y cuántas actualizadas.
- Si faltan columnas, hay valores vacíos o no numéricos, o hay semanas repetidas, la API responde `422` con el detalle.
- Si un municipio no tiene datos cargados, sus endpoints de serie y predicción responden `404`.
- Los archivos no pasan por la consola. El endpoint `POST /api/v1/datos/{municipio_id}/csv` lo usa el dashboard por dentro.

## Empaquetar el modelo (.whl)

El paquete del modelo está en `models/paquete_modelo/`. Contiene las variables, la carga y los artefactos entrenados de los cinco municipios (`modelo_dengue/artefactos/`). La API no lee `models/` directamente: instala el `.whl` que está en `src_api/model_pkg/`.

Flujo: empaquetar en `models/paquete_modelo/` → copiar el `.whl` a `src_api/model_pkg/` → subir al repositorio → construir la imagen de la API.

Windows (PowerShell), desde `proyecto-final/`:

```powershell
cd models\paquete_modelo
python -m pip wheel . --no-deps -w dist
cd ..\..
Copy-Item "models\paquete_modelo\dist\dengue_modelo_logistica-0.1.0-py3-none-any.whl" "src_api\model_pkg\" -Force
```

Para instalarlo en un entorno local: `pip install src_api/model_pkg/dengue_modelo_logistica-0.1.0-py3-none-any.whl`.

Para una versión nueva del modelo, cambiar `version` en `models/paquete_modelo/pyproject.toml` antes de empaquetar, y borrar el `.whl` anterior de `src_api/model_pkg/` para que la imagen no instale dos versiones. La imagen toma el `.whl` de `src_api/model_pkg/`, así que hay que copiarlo antes de construirla.

## Ejecución local (entorno virtual)

Windows (PowerShell):

```powershell
cd src_api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install model_pkg\dengue_modelo_logistica-0.1.0-py3-none-any.whl
uvicorn app.main:app --reload --port 8000
```

Verificación rápida: `http://localhost:8000/api/v1/health` debe responder `{"estado": "ok", ...}`.

## Docker

Requisitos: Docker Desktop en marcha y el `.whl` copiado en `src_api/model_pkg/` (ver arriba).

Desde la carpeta `proyecto-final/`:

```bash
# Construir las imágenes de la API y del dashboard
docker compose build

# Levantar los servicios en segundo plano
docker compose up -d

# Ver los registros de la API
docker compose logs -f api

# Detener los servicios
docker compose down
```

Los parámetros de análisis (`UMBRAL_DECISION`, `ANIO_MINIMO_GRAFICOS`, `VENTANA_HISTORIAL`) no hace falta pasarlos en los comandos de despliegue. La API arranca con sus valores por defecto y después se ajustan desde la pestaña Configuración del dashboard, que se guardan en la base de datos del volumen.

Solo la API, sin Compose:

```bash
docker build -t dengue-alerta-api:0.1 ./src_api
docker run -d --name dengue-api -p 8000:8000 \
  -e SQLITE_PATH=/app/data/dengue.sqlite \
  -v "$(pwd)/src_api/data:/app/data" \
  dengue-alerta-api:0.1
```

Solo el dashboard, con la API corriendo en tu equipo (Windows o Mac):

```bash
docker build -t dengue-alerta-dashboard:0.1 ./src_dashboard
docker run -d --name dengue-dashboard -p 8501:8501 \
  -e API_URL=http://host.docker.internal:8000/api/v1 \
  dengue-alerta-dashboard:0.1
```

En Linux, `host.docker.internal` no existe por defecto. Para ese caso usa la red compartida descrita en el despliegue en EC2.

Con Compose, el dashboard ya apunta a la API del mismo `docker-compose.yml`. Accesos: API en `http://localhost:8000/docs`, dashboard en `http://localhost:8501`.

Tras levantar los servicios, cargar los datos de cada municipio desde el dashboard (`http://localhost:8501`, pestaña Configuración; ver [Carga de datos](#carga-de-datos)).

Verificación: `curl http://localhost:8000/api/v1/health` debe responder con `"estado": "ok"` y el nombre del modelo.

## Despliegue en AWS EC2

La instancia clona el repositorio, construye las imágenes localmente y ejecuta los contenedores. No se usa un registro de imágenes (ECR).

Antes de empezar, en tu equipo:

- El `.whl` debe estar en el repositorio remoto, en `src_api/model_pkg/`. Si no está, la construcción de la imagen de la API falla. Verifica con `git status` y haz commit y push de esa carpeta.
- Los CSV de los municipios **no están en git** (se versionan con DVC). Se suben desde tu navegador en el paso 4.

### 1. Preparar la instancia

- Sistema operativo: Amazon Linux 2023.
- Grupo de seguridad: SSH (22) solo desde tu IP; dashboard (8501) desde donde lo vayan a usar. La API (8000) no necesita ser pública: el dashboard la llama desde dentro de la red de Docker.

```bash
ssh -i tu-clave.pem ec2-user@<IP_PUBLICA>

sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user      # cerrar la sesión SSH y volver a entrar
```

Si no quieres usar el grupo `docker`, antepón `sudo` a todos los comandos de `docker` de esta sección. Usa siempre la misma forma en todos los pasos.

### 2. Clonar y construir las imágenes

```bash
cd ~
git clone <URL_DEL_REPOSITORIO> proyecto-final
cd proyecto-final
ls src_api/model_pkg                  # debe mostrar el .whl

docker build -t dengue-alerta-api:0.1 ./src_api
docker build -t dengue-alerta-dashboard:0.1 ./src_dashboard
```

### 3. Red y contenedores

Los dos contenedores se conectan por la red `dengue`, así el dashboard llama a la API por el nombre `dengue-api`.

```bash
docker network create dengue
docker volume create dengue-data      # aquí queda la base SQLite, aunque se reinicien los contenedores

docker run -d --name dengue-api --network dengue --restart unless-stopped -p 8000:8000 \
  -e SQLITE_PATH=/app/data/dengue.sqlite \
  -v dengue-data:/app/data \
  dengue-alerta-api:0.1

docker run -d --name dengue-dashboard --network dengue --restart unless-stopped -p 8501:8501 \
  -e API_URL=http://dengue-api:8000/api/v1 \
  dengue-alerta-dashboard:0.1
```

No mezcles estos comandos con `docker compose` en la misma instancia: Compose crea su propia red y sus nombres de contenedor.

Los parámetros de análisis no van en estos comandos. Si necesitas un valor distinto al inicial, cámbialo desde la pestaña Configuración del dashboard. El volumen `dengue-data` debe conservarse entre actualizaciones, porque ahí se guardan esos valores.

### 4. Cargar los datos desde el dashboard

Abre `http://<IP_PUBLICA>:8501` en tu navegador, ve a la pestaña **3. Configuración** y, en **Configuración de datos**, carga el `caracteristicas.csv` de cada municipio (cartagena, magangue, el_carmen_de_bolivar, santa_rosa_del_sur, turbaco). Los archivos están en `data/processed/municipios/<municipio>/` de tu equipo, no en la instancia, así que no hace falta copiarlos con `scp`.

Al terminar, la sección de estado de datos del dashboard debe mostrar cada municipio como cargado.

### 5. Verificar

```bash
curl -s http://localhost:8000/api/v1/health
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8501
docker exec dengue-dashboard python -c "import urllib.request; print(urllib.request.urlopen('http://dengue-api:8000/api/v1/health').read())"
```

La salud de la API debe responder `"estado": "ok"`, el dashboard debe devolver `200` y el último comando debe imprimir el mismo JSON. Luego abre `http://<IP_PUBLICA>:8501` en el navegador.

### 6. Actualizar una versión

En la instancia, con el repositorio ya clonado:

```bash
cd ~/proyecto-final
git pull
docker build -t dengue-alerta-api:0.1 ./src_api
docker rm -f dengue-api
docker run -d --name dengue-api --network dengue --restart unless-stopped -p 8000:8000 \
  -e SQLITE_PATH=/app/data/dengue.sqlite -v dengue-data:/app/data \
  dengue-alerta-api:0.1
```

El dashboard se actualiza igual, con `docker build` y `docker rm -f dengue-dashboard` seguido del `docker run` del paso 3. El volumen `dengue-data` conserva los datos cargados entre versiones.

## Configuración de variables

Hay dos tipos de configuración.

### Parámetros del análisis (desde el dashboard)

Se cambian en la pestaña **3. Configuración**, sección **Parámetros del análisis**:

| Parámetro | Valor inicial | Qué controla |
|---|---|---|
| Umbral de decisión | `0.5` | Probabilidad a partir de la cual hay alerta. Pendiente de validar |
| Año mínimo de las gráficas | `2012` | Primer año de las gráficas de casos, canal y clima. La predicción usa toda la serie |
| Ventana de historial | `8` semanas | Historial mínimo para aceptar una fecha de corte |

Ajusta los valores, pulsa **Guardar parámetros** y las pestañas se actualizan. Los valores se guardan en SQLite (tabla `configuracion`), en el volumen `dengue-data`, así que sobreviven a reinicios y actualizaciones. Los cambios aplican a todos los usuarios del dashboard.

No hace falta pasar estos parámetros en los comandos de despliegue. La API también acepta `PUT /api/v1/configuracion` si necesitas cambiarlos desde otro sistema:

```bash
curl -X PUT http://localhost:8000/api/v1/configuracion \
  -H "Content-Type: application/json" \
  -d '{"umbral_decision": 0.3}'
```

Rangos válidos: umbral entre 0 y 1; año mínimo entre 2007 y 2030; ventana entre 1 y 52 semanas. Un valor fuera de rango responde `422`.

### Variables de entorno (valores iniciales)

Las variables de entorno solo se usan cuando todavía no hay un valor guardado en SQLite. Una vez guardado un parámetro, cambiar la variable no tiene efecto.

| Variable | Por defecto | Descripción |
|---|---|---|
| `SQLITE_PATH` | `src_api/data/dengue.sqlite` | Archivo de la base SQLite. En Docker se apunta al volumen (`/app/data/dengue.sqlite`) |
| `MODEL_VERSION` | `logistica-0.1` | Versión reportada en cada respuesta |
| `HORIZONTE_SEMANAS` | `4` | Semanas de anticipación. No cambiarla sin reentrenar el modelo |
| `UMBRAL_DECISION` | `0.5` | Valor inicial del umbral de decisión |
| `VENTANA_HISTORIAL` | `8` | Valor inicial de la ventana de historial |
| `ANIO_MINIMO_GRAFICOS` | `2012` | Valor inicial del año mínimo de las gráficas |

Para restablecer un parámetro al valor inicial, borra la fila correspondiente de la tabla `configuracion` o guárdalo con el valor deseado.

La API no tiene autenticación. Quien pueda llegar al puerto 8000 puede cambiar estos parámetros, así que en un despliegue público conviene no exponerlo.

## Agregar otro modelo

1. Implementar una clase que herede de `Predictor` en `app/modelo.py`.
2. Construir las variables exactamente como en entrenamiento, usando sólo el historial hasta `fecha_corte` y el umbral de la semana objetivo.
3. Devolver la clase desde `cargar_predictor()`.

## Métricas del modelo

`GET /api/v1/modelo` devuelve, por municipio, las métricas de la regresión logística en el conjunto de prueba 2023–2025: AUC-ROC (con su intervalo del 95 %), AUC-PR, recall, especificidad, precisión, F1 y el conteo de brotes avisados.

Las métricas vienen de `src_api/metricas/logistica_por_municipio.csv`, copia de las filas `logistica` de `reports/resumen_municipios.csv` que genera el pipeline. Son una copia fija: la API no recalcula nada.

**Actualizar las métricas** después de reentrenar, desde `proyecto-final/`:

```bash
head -1 reports/resumen_municipios.csv > src_api/metricas/logistica_por_municipio.csv
grep ",logistica," reports/resumen_municipios.csv >> src_api/metricas/logistica_por_municipio.csv
```

Luego reconstruir la imagen de la API. Las métricas calculadas con el umbral de Youden del entrenamiento no son las mismas que las que daría el umbral de decisión de la API (`UMBRAL_DECISION`).

## Pendientes

- **Umbral de decisión:** el valor inicial es `0.5` y se ajusta desde Configuración. Falta que el equipo lo valide. El umbral de Youden del entrenamiento no está guardado en el paquete del modelo, así que la API no puede usarlo todavía.
