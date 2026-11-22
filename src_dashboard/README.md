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
Abrir `http://localhost:8501`.

Variables de entorno: `API_URL` (por defecto `http://localhost:8000/api/v1`) y `API_TIMEOUT_S` (10).

## Docker

Desde `proyecto-final/`, la imagen del dashboard se construye junto con la de la API:

```bash
docker compose build dashboard
docker compose up -d
```

Sin Compose, con la API corriendo en tu equipo (Windows o Mac):

```bash
docker build -t dengue-alerta-dashboard:0.1 ./src_dashboard
docker run -d --name dengue-dashboard -p 8501:8501 \
  -e API_URL=http://host.docker.internal:8000/api/v1 \
  dengue-alerta-dashboard:0.1
```

La API debe estar levantada antes. Comandos completos, incluido el empaquetado del modelo, en [../src_api/README.md](../src_api/README.md).

## Despliegue en AWS EC2

El dashboard se despliega en la misma instancia que la API, clonando el repositorio y construyendo la imagen allí. El procedimiento completo (preparar la instancia, clonar, crear la red `dengue`, cargar los datos y verificar) está en [../src_api/README.md](../src_api/README.md#despliegue-en-aws-ec2). No se usa un registro de imágenes.

Solo los pasos del dashboard, dentro de la instancia, con el repositorio clonado en `~/proyecto-final` y la API ya corriendo en la red `dengue`:

```bash
cd ~/proyecto-final
docker build -t dengue-alerta-dashboard:0.1 ./src_dashboard

docker rm -f dengue-dashboard 2>/dev/null
docker run -d --name dengue-dashboard --network dengue --restart unless-stopped -p 8501:8501 \
  -e API_URL=http://dengue-api:8000/api/v1 \
  dengue-alerta-dashboard:0.1
```

Para actualizar, `git pull`, volver a construir la imagen y ejecutar de nuevo los dos comandos anteriores. El dashboard no guarda datos propios, así que no hay nada que conservar.

El dashboard queda accesible en `http://<IP_PUBLICA>:8501`. La API no necesita ser pública para el dashboard, porque el dashboard la llama desde dentro de la red de Docker.

## Pestañas

1. **Mapa de alertas**: selección de la fecha de corte (solo fechas con datos), proyección a 4 semanas, mapa de calor por municipio y tabla de alertas. Al hacer clic en una fila se abre el detalle de variables influyentes.
2. **Casos y canal endémico**: casos semanales, umbral del canal y tramos de exceso sombreados (épocas de alto riesgo). La línea de escalón en el eje derecho muestra si hubo brote en la semana. Debajo, las gráficas de lluvia y temperatura.
3. **Configuración**: dos secciones.
   - **Parámetros del análisis:** umbral de decisión, año mínimo de las gráficas y ventana de historial. Se guardan en SQLite y se aplican a todos los usuarios; al guardar, todas las pestañas se actualizan. Ver [../src_api/README.md](../src_api/README.md#variables-de-entorno).
   - **Carga de datos:** permite elegir un municipio y cargar su `caracteristicas.csv` semanal en SQLite. Las semanas existentes se actualizan y las nuevas se agregan. Formato en [../src_api/README.md](../src_api/README.md).
4. **Detalle del modelo**: nombre, versión, horizonte, umbral de decisión y variables. Abajo, una tabla con las métricas de evaluación de cada municipio en el conjunto de prueba 2023–2025 (AUC-ROC, recall, especificidad, precisión, F1 y brotes avisados). Las métricas vienen de la API; cómo actualizarlas está en [../src_api/README.md](../src_api/README.md#métricas-del-modelo).
5. **Acerca de**: descripción del proyecto y del equipo.
