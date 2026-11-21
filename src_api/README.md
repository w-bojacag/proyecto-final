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
| GET | `/api/v1/datos/estado` | Lo cargado en SQLite por municipio |
| POST | `/api/v1/datos/{municipio_id}/csv` | Carga `caracteristicas.csv` semanal en SQLite |

Documentación interactiva: `http://localhost:8000/docs`.

Ejemplo de predicción:

```json
POST /api/v1/prediccion
{"municipio_id": "cartagena", "fecha_corte": "2024-06-02"}
```

## Carga de datos

Se carga el archivo `data/processed/municipios/<municipio>/caracteristicas.csv` tal cual lo genera el pipeline (una fila por año y semana epidemiológica). La API usa estas columnas y ignora el resto:

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

Ejemplo:

```bash
curl -F "archivo=@data/processed/municipios/cartagena/caracteristicas.csv" http://localhost:8000/api/v1/datos/cartagena/csv
```

## Empaquetar el modelo (.whl)

El paquete del modelo está en `models/paquete_modelo/`. Contiene las variables, la carga y los artefactos entrenados de los cinco municipios (`modelo_dengue/artefactos/`). La API no lee `models/` directamente: instala el `.whl` que está en `src_api/model_pkg/`.

Flujo: empaquetar en `models/paquete_modelo/` → copiar el `.whl` a `src_api/model_pkg/` → construir la imagen de la API.

Windows (PowerShell), desde `proyecto-final/`:

```powershell
cd models\paquete_modelo
python -m pip wheel . --no-deps -w dist
cd ..\..
Copy-Item "models\paquete_modelo\dist\dengue_modelo_logistica-0.1.0-py3-none-any.whl" "src_api\model_pkg\" -Force
```

Linux / macOS / Git Bash, desde `proyecto-final/`:

```bash
cd models/paquete_modelo && python -m pip wheel . --no-deps -w dist && cd ../..
cp models/paquete_modelo/dist/dengue_modelo_logistica-0.1.0-py3-none-any.whl src_api/model_pkg/
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

Linux / macOS / Git Bash:

```bash
cd src_api
python -m venv .venv
source .venv/bin/activate        # en Git Bash en Windows: source .venv/Scripts/activate
pip install -r requirements.txt
pip install model_pkg/dengue_modelo_logistica-0.1.0-py3-none-any.whl
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

Solo la API, sin Compose:

```bash
docker build -t dengue-alerta-api:0.1 ./src_api
docker run -d --name dengue-api -p 8000:8000 \
  -e SQLITE_PATH=/app/data/dengue.sqlite \
  -v "$(pwd)/src_api/data:/app/data" \
  dengue-alerta-api:0.1
```

Solo el dashboard:

```bash
docker build -t dengue-alerta-dashboard:0.1 ./src_dashboard
docker run -d --name dengue-dashboard -p 8501:8501 \
  -e API_URL=http://host.docker.internal:8000/api/v1 \
  dengue-alerta-dashboard:0.1
```

Con Compose, el dashboard ya apunta a la API del mismo `docker-compose.yml`. Accesos: API en `http://localhost:8000/docs`, dashboard en `http://localhost:8501`.

Tras levantar la API, cargar los datos de cada municipio:

```bash
curl -F "archivo=@data/processed/municipios/cartagena/caracteristicas.csv" http://localhost:8000/api/v1/datos/cartagena/csv
```

Verificación: `curl http://localhost:8000/api/v1/health` debe responder con `"estado": "ok"` y el nombre del modelo.

## Despliegue en AWS EC2

Flujo: construir las imágenes en tu equipo → subirlas a Amazon ECR → en la instancia EC2, descargarlas y ejecutarlas. Los dos servicios se conectan por una red de Docker propia, así el dashboard llama a la API por el nombre `dengue-api`.

Los comandos usan variables que debes reemplazar: `AWS_REGION` (por ejemplo `us-east-1`) y `AWS_ACCOUNT_ID` (tu número de cuenta de 12 dígitos). No se incluyen credenciales: usa `aws configure` o un perfil de IAM con permiso para ECR.

### 1. En tu equipo: construir y subir las imágenes

Requisitos: AWS CLI, Docker, `.whl` copiado en `src_api/model_pkg/` (ver [Empaquetar el modelo](#empaquetar-el-modelo-whl)). Desde `proyecto-final/`, en Git Bash o Linux/macOS:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=123456789012
export REGISTRY=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Crear los repositorios (solo la primera vez)
aws ecr create-repository --repository-name dengue-alerta-api --region $AWS_REGION
aws ecr create-repository --repository-name dengue-alerta-dashboard --region $AWS_REGION

# Iniciar sesión en ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $REGISTRY

# Construir para la arquitectura de EC2 (x86_64) y subir
docker build --platform linux/amd64 -t $REGISTRY/dengue-alerta-api:0.1 ./src_api
docker push $REGISTRY/dengue-alerta-api:0.1

docker build --platform linux/amd64 -t $REGISTRY/dengue-alerta-dashboard:0.1 ./src_dashboard
docker push $REGISTRY/dengue-alerta-dashboard:0.1
```

Si la instancia usa procesadores ARM (Graviton, por ejemplo `t4g`), cambia `linux/amd64` por `linux/arm64`.

### 2. Preparar la instancia EC2

- Sistema operativo: Amazon Linux 2023 (incluye AWS CLI).
- Rol de IAM con la política `AmazonEC2ContainerRegistryReadOnly`, para que la instancia pueda descargar las imágenes.
- Grupo de seguridad: SSH (22) solo desde tu IP; dashboard (8501) desde donde lo vayan a usar; API (8000) solo si necesitas acceso externo, por ejemplo para cargar datos.

Conectarse y preparar Docker:

```bash
ssh -i tu-clave.pem ec2-user@<IP_PUBLICA>

sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user      # cerrar la sesión SSH y volver a entrar
```

### 3. En la instancia: descargar y ejecutar

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=123456789012
export REGISTRY=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $REGISTRY

docker network create dengue
docker volume create dengue-data      # aquí queda la base SQLite, aunque se reinicien los contenedores

docker pull $REGISTRY/dengue-alerta-api:0.1
docker pull $REGISTRY/dengue-alerta-dashboard:0.1

docker run -d --name dengue-api --network dengue --restart unless-stopped -p 8000:8000 \
  -e SQLITE_PATH=/app/data/dengue.sqlite \
  -v dengue-data:/app/data \
  $REGISTRY/dengue-alerta-api:0.1

docker run -d --name dengue-dashboard --network dengue --restart unless-stopped -p 8501:8501 \
  -e API_URL=http://dengue-api:8000/api/v1 \
  $REGISTRY/dengue-alerta-dashboard:0.1
```

Comprobar: `curl http://localhost:8000/api/v1/health` debe responder `"estado": "ok"`. El dashboard queda en `http://<IP_PUBLICA>:8501`.

### 4. Cargar los datos en la instancia

Copiar el CSV de cada municipio desde tu equipo, con un nombre distinto por municipio para que no se sobrescriban:

```bash
# En tu equipo (Git Bash), desde proyecto-final/
for m in cartagena magangue el_carmen_de_bolivar santa_rosa_del_sur turbaco; do
  scp -i tu-clave.pem data/processed/municipios/$m/caracteristicas.csv ec2-user@<IP_PUBLICA>:~/$m.csv
done
```

Luego, en la instancia, cargar cada archivo en su municipio:

```bash
for m in cartagena magangue el_carmen_de_bolivar santa_rosa_del_sur turbaco; do
  curl -F "archivo=@$HOME/$m.csv" http://localhost:8000/api/v1/datos/$m/csv
done
```

Verificar con `curl http://localhost:8000/api/v1/datos/estado`: cada municipio debe aparecer con `"cargado": true`.

### 5. Actualizar una versión

Subir la nueva imagen desde tu equipo (paso 1, con un tag nuevo como `0.2`), y en la instancia:

```bash
docker pull $REGISTRY/dengue-alerta-api:0.2
docker rm -f dengue-api
docker run -d --name dengue-api --network dengue --restart unless-stopped -p 8000:8000 \
  -e SQLITE_PATH=/app/data/dengue.sqlite -v dengue-data:/app/data \
  $REGISTRY/dengue-alerta-api:0.2
```

El volumen `dengue-data` conserva los datos cargados entre versiones.

## Variables de entorno

| Variable | Por defecto | Descripción |
|---|---|---|
| `SQLITE_PATH` | `src_api/data/dengue.sqlite` | Archivo de la base SQLite (se crea al cargar el primer CSV) |
| `MODEL_VERSION` | `logistica-0.1` | Versión reportada en cada respuesta |
| `HORIZONTE_SEMANAS` | `4` | Semanas de anticipación |
| `VENTANA_HISTORIAL` | `8` | Semanas de historial mínimo antes de una fecha de corte válida |
| `UMBRAL_DECISION` | `0.5` | Probabilidad a partir de la cual hay alerta (pendiente de validar) |
| `ANIO_MINIMO_GRAFICOS` | `2012` | Primer año que devuelve `municipios/{id}/serie` (gráficas del canal endémico y clima). Los años anteriores se omiten; la predicción no se ve afectada |

## Agregar otro modelo

1. Implementar una clase que herede de `Predictor` en `app/modelo.py`.
2. Construir las variables exactamente como en entrenamiento, usando sólo el historial hasta `fecha_corte` y el umbral de la semana objetivo.
3. Devolver la clase desde `cargar_predictor()`.

## Pendientes

- Métricas de evaluación por municipio (recall, especificidad, precisión, F1 y AUC-ROC).
- Umbral de decisión validado con el equipo.
