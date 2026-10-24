import ee
import pandas as pd

# 1. Inicializar Earth Engine con tu proyecto personal
ee.Initialize(project='dengue-project-2026')

# Polígono de Cartagena
poligono_cartagena = ee.Geometry.Polygon([
    [
        [-75.5800, 10.3200],  # Suroeste
        [-75.4000, 10.3200],  # Sureste
        [-75.4000, 10.5000],  # Noreste
        [-75.5800, 10.5000],  # Noroeste
        [-75.5800, 10.3200]   # Cierre del polígono
    ]
])

def extraer_chirps(year):
    """1. Precipitación Diaria (CHIRPS)"""
    col = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterDate(f"{year}-01-01", f"{year}-12-31").select('precipitation')
    def map_func(img):
        val = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poligono_cartagena, scale=5000).get('precipitation')
        return ee.Feature(None, {'date': img.date().format('YYYY-MM-dd'), 'precipitation_mm': val})
    return pd.DataFrame([f['properties'] for f in col.map(map_func).getInfo()['features']])

def extraer_modis_lst(year):
    """2. Temperatura Superficial Día y Noche (MODIS LST - Diaria)"""
    col = ee.ImageCollection('MODIS/061/MOD11A1').filterDate(f"{year}-01-01", f"{year}-12-31").select(['LST_Day_1km', 'LST_Night_1km'])
    def map_func(img):
        day = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poligono_cartagena, scale=1000).get('LST_Day_1km')
        night = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poligono_cartagena, scale=1000).get('LST_Night_1km')
        
        day_c = ee.Algorithms.If(day, ee.Number(day).multiply(0.02).subtract(273.15), None)
        night_c = ee.Algorithms.If(night, ee.Number(night).multiply(0.02).subtract(273.15), None)
        return ee.Feature(None, {'date': img.date().format('YYYY-MM-dd'), 'lst_day_c': day_c, 'lst_night_c': night_c})
    return pd.DataFrame([f['properties'] for f in col.map(map_func).getInfo()['features']])

def extraer_era5_clima(year):
    """3. Temperatura de Aire (Media, Min, Max) y Punto de Rocío (ERA5-Land - Diaria)"""
    col = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterDate(f"{year}-01-01", f"{year}-12-31").select([
        'temperature_2m', 'temperature_2m_min', 'temperature_2m_max', 'dewpoint_temperature_2m'
    ])
    def map_func(img):
        t_mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poligono_cartagena, scale=11132).get('temperature_2m')
        t_min = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poligono_cartagena, scale=11132).get('temperature_2m_min')
        t_max = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poligono_cartagena, scale=11132).get('temperature_2m_max')
        dew = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poligono_cartagena, scale=11132).get('dewpoint_temperature_2m')
        
        return ee.Feature(None, {
            'date': img.date().format('YYYY-MM-dd'),
            'air_temp_mean_c': ee.Algorithms.If(t_mean, ee.Number(t_mean).subtract(273.15), None),
            'air_temp_min_c': ee.Algorithms.If(t_min, ee.Number(t_min).subtract(273.15), None),
            'air_temp_max_c': ee.Algorithms.If(t_max, ee.Number(t_max).subtract(273.15), None),
            'dewpoint_c': ee.Algorithms.If(dew, ee.Number(dew).subtract(273.15), None)
        })
    return pd.DataFrame([f['properties'] for f in col.map(map_func).getInfo()['features']])

def extraer_modis_vegetacion(year):
    """4. Índices de Vegetación NDVI y EVI (MODIS - 16 días)"""
    col = ee.ImageCollection('MODIS/061/MOD13Q1').filterDate(f"{year}-01-01", f"{year}-12-31").select(['NDVI', 'EVI'])
    def map_func(img):
        ndvi = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poligono_cartagena, scale=250).get('NDVI')
        evi = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poligono_cartagena, scale=250).get('EVI')
        
        # Factor de escala de MODIS Vegetation es 0.0001
        return ee.Feature(None, {
            'date': img.date().format('YYYY-MM-dd'),
            'ndvi': ee.Algorithms.If(ndvi, ee.Number(ndvi).multiply(0.0001), None),
            'evi': ee.Algorithms.If(evi, ee.Number(evi).multiply(0.0001), None)
        })
    return pd.DataFrame([f['properties'] for f in col.map(map_func).getInfo()['features']])

# --- EJECUCIÓN DEL PROCESO POR AÑOS (2007 - 2025) ---
print("Iniciando extracción masiva por bloques anuales...")

all_chirps, all_lst, all_era5, all_veg = [], [], [], []

for anio in range(2007, 2026):
    print(f"Procesando año {anio}...")
    all_chirps.append(extraer_chirps(anio))
    all_lst.append(extraer_modis_lst(anio))
    all_era5.append(extraer_era5_clima(anio))
    all_veg.append(extraer_modis_vegetacion(anio))

# Consolidar cada bloque en un DataFrame independiente
df_chirps = pd.concat(all_chirps, ignore_index=True)
df_lst = pd.concat(all_lst, ignore_index=True)
df_era5 = pd.concat(all_era5, ignore_index=True)
df_veg = pd.concat(all_veg, ignore_index=True)

# Limpiar y convertir fechas
for df in [df_chirps, df_lst, df_era5, df_veg]:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

# --- FUSIÓN MAESTRO EN PANDAS ---
# Unimos primero las fuentes diarias usando una base de fechas completa (2007-2005)
rango_fechas = pd.date_range(start='2007-01-01', end='2025-12-31', freq='D')
df_maestro = pd.DataFrame(index=rango_fechas)
df_maestro.index.name = 'date'

# Combinar información diaria
df_maestro = df_maestro.join(df_chirps).join(df_lst).join(df_era5)

# Combinar información de 16 días (NDVI/EVI) aplicando un reindex y relleno (ffill) para que el ML no tenga nulos diarios
df_veg = df_veg[~df_veg.index.duplicated(keep='first')] # Limpiar posibles duplicados de fecha
df_maestro = df_maestro.join(df_veg.reindex(df_maestro.index).ffill())

# Guardar la matriz final consolidada
ruta_salida = 'c:/Wilson/IA/Dengue/analisis/matriz_maestra_ambiental_cartagena_2007_2025.csv'
df_maestro.to_csv(ruta_salida)

print(f"\n¡Proceso finalizado con éxito!")
print(f"Matriz completa guardada en: {ruta_salida}")
print(df_maestro.head())