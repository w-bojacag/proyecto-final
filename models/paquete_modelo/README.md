## Empaquetar el modelo (.whl)

El modelo logístico se empaqueta como un paquete de Python (`paquete_modelo/`) que contiene las variables, la carga y los artefactos entrenados de los cinco municipios. La API lo instala desde el `.whl`.

Los artefactos se copian desde `models/` (salida del entrenamiento) antes de empaquetar. Desde `proyecto-final/`:

Windows (PowerShell):

```powershell
foreach ($m in "cartagena","magangue","el_carmen_de_bolivar","santa_rosa_del_sur","turbaco") {
    New-Item -ItemType Directory -Force -Path "models\paquete_modelo\modelo_dengue\artefactos\$m" | Out-Null
    Copy-Item "models\$m\logistica.joblib" "models\paquete_modelo\modelo_dengue\artefactos\$m\"
}

cd models\paquete_modelo
python -m pip wheel . --no-deps -w dist
```

Resultado: `models/paquete_modelo/dist/dengue_modelo_logistica-0.1.0-py3-none-any.whl`. Para instalarlo en un entorno local: `pip install dist/dengue_modelo_logistica-0.1.0-py3-none-any.whl`.

Para una versión nueva del modelo, cambiar `version` en `paquete_modelo/pyproject.toml` antes de empaquetar. La imagen de la API toma el `.whl` de `paquete_modelo/dist/`, así que hay que empaquetar antes de construirla.