# Dashboard – alerta temprana de dengue (Streamlit)

Interfaz con cinco pestañas: mapa de alertas por fecha de corte, casos y canal endémico con épocas de alto riesgo (con lluvia y temperatura), detalle técnico del modelo, configuración de datos y acerca del proyecto. Consume la API de `../src_api`, que lee las series de SQLite y predice con la regresión logística del paquete `.whl`.

Para que haya datos y predicciones, la API debe estar levantada y con los CSV de los municipios cargados (ver [../src_api/README.md](../src_api/README.md)).

## Ejecución local (entorno virtual)

Primero debe estar activa la API en `localhost:8000`.

Windows (PowerShell):

```powershell
cd src_dashboard
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

Linux / macOS / Git Bash:

```bash
cd src_dashboard
python -m venv .venv
source .venv/bin/activate        # en Git Bash en Windows: source .venv/Scripts/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

Abrir `http://localhost:8501`.

Variables de entorno: `API_URL` (por defecto `http://localhost:8000/api/v1`) y `API_TIMEOUT_S` (10).

## Docker

Desde `proyecto-final/`, la imagen del dashboard se construye junto con la de la API:

```bash
docker compose build dashboard
docker compose up -d
```

Sin Compose:

```bash
docker build -t dengue-alerta-dashboard:0.1 ./src_dashboard
docker run -d --name dengue-dashboard -p 8501:8501 \
  -e API_URL=http://host.docker.internal:8000/api/v1 \
  dengue-alerta-dashboard:0.1
```

La API debe estar levantada antes. Comandos completos, incluido el empaquetado del modelo, en [../src_api/README.md](../src_api/README.md).

## Despliegue en AWS EC2

El dashboard se despliega junto con la API, en la misma instancia y la misma red de Docker. El procedimiento completo (crear repositorios en ECR, subir imágenes, preparar la instancia y cargar los datos) está en [../src_api/README.md](../src_api/README.md#despliegue-en-aws-ec2).

Solo para el dashboard, desde `proyecto-final/` en tu equipo:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=123456789012
export REGISTRY=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

aws ecr create-repository --repository-name dengue-alerta-dashboard --region $AWS_REGION   # solo la primera vez
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $REGISTRY
docker build --platform linux/amd64 -t $REGISTRY/dengue-alerta-dashboard:0.1 ./src_dashboard
docker push $REGISTRY/dengue-alerta-dashboard:0.1
```

En la instancia EC2 (con la API ya levantada en la red `dengue`):

```bash
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $REGISTRY
docker pull $REGISTRY/dengue-alerta-dashboard:0.1
docker run -d --name dengue-dashboard --network dengue --restart unless-stopped -p 8501:8501 \
  -e API_URL=http://dengue-api:8000/api/v1 \
  $REGISTRY/dengue-alerta-dashboard:0.1
```

Para actualizar: subir el tag nuevo, y en la instancia `docker pull`, `docker rm -f dengue-dashboard` y volver a ejecutar el `docker run` con el nuevo tag.

El dashboard queda accesible en `http://<IP_PUBLICA>:8501`. La API no necesita ser pública para el dashboard, porque el dashboard la llama desde dentro de la red de Docker.

## Pestañas

1. **Mapa de alertas**: selección de la fecha de corte (solo fechas con datos), proyección a 4 semanas, mapa de calor por municipio y tabla de alertas. Al hacer clic en una fila se abre el detalle de variables influyentes.
2. **Casos y canal endémico**: casos semanales, umbral del canal y tramos de exceso sombreados (épocas de alto riesgo). La línea de escalón en el eje derecho muestra si hubo brote en la semana. Debajo, las gráficas de lluvia y temperatura.
3. **Detalle del modelo**: nombre, versión, horizonte, umbral de decisión, variables y estado de las métricas (pendientes).
4. **Configuración**: permite elegir un municipio y cargar su `caracteristicas.csv` semanal en SQLite. Las semanas existentes se actualizan y las nuevas se agregan. Formato en [../src_api/README.md](../src_api/README.md).
5. **Acerca de**: descripción del proyecto y del equipo.
