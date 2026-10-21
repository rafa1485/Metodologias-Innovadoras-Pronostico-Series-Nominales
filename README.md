# Pronóstico de series temporales multivariadas

Versión Python modular de `ajuste_y_comparacion_modelos_ML-21-oct-26.ipynb`.
El notebook se conserva como referencia. El script mantiene sus algoritmos,
parámetros y formatos de Excel y elimina la dependencia de Google Colab/Drive.

## Instalación y ejecución

Se requiere Python 3.10 o posterior.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python ajuste_y_comparacion_modelos_ML.py
```

La ejecución completa crea muestras, compara SARIMA, SARIMAX, CNN y TCN con
validación temporal y genera los gráficos. Usa `Input_series.xlsx` del directorio
del proyecto y guarda los archivos en `resultados/`, sin sobrescribir los
resultados históricos de la raíz. El entrenamiento utiliza hasta 250 épocas por
red y fold y puede demorar. PyTorch selecciona CUDA si está disponible.

Las etapas se pueden ejecutar por separado, en este orden:

```bash
python ajuste_y_comparacion_modelos_ML.py muestras --input Input_series.xlsx --output-dir resultados
python ajuste_y_comparacion_modelos_ML.py modelos --output-dir resultados
python ajuste_y_comparacion_modelos_ML.py graficos --output-dir resultados --horizonte 6
```

`modelos` necesita los Excel de muestras y objetivos previamente generados en
`--output-dir`. `graficos` necesita allí `comparacion_modelos_temporal_loo.xlsx`.
Para graficar resultados históricos de la raíz se puede usar `--output-dir .`;
esto escribe los gráficos y el Excel de pronósticos en esa misma carpeta.
`--epochs N` permite cambiar el máximo de épocas de las redes.
`python ajuste_y_comparacion_modelos_ML.py --help` muestra las opciones.

## Organización por uso

| Módulo | Responsabilidad |
| --- | --- |
| `pronostico/config_muestras.py` | Entrada, ventana, faltantes, semilla e imputación |
| `pronostico/muestras.py` | Lectura, imputación, ventanas y exportación de muestras/objetivos |
| `pronostico/config_modelos.py` | Series seleccionadas, modelos, semillas y entrenamiento |
| `pronostico/datos.py` | Carga de muestras y ordenamiento temporal |
| `pronostico/modelos_estadisticos.py` | Pronósticos SARIMA y SARIMAX |
| `pronostico/redes.py` | Dataset PyTorch, CNN, TCN, escalado y entrenamiento |
| `pronostico/comparacion.py` | Folds temporales, métricas, complejidad y exportación |
| `pronostico/config_graficos.py` | Horizonte y nombres de archivos de gráficos |
| `pronostico/graficos.py` | Tabla de pronósticos y gráficos con backend sin interfaz gráfica |

Los tres archivos `config_*.py` conservan configuraciones independientes por
etapa: la generación usa semilla 20 y SARIMAX estacional `(1, 1, 1, 12)`;
la comparación usa semilla 42 y SARIMAX estacional `(0, 1, 1, 12)`.
Al cambiar `WINDOW_SIZE`, ajustar tanto `config_muestras.py` como
`config_modelos.py`. La variable objetivo debe estar entre `SELECTED_SERIES`.
Las rutas de entrada/salida se establecen desde la consola; los demás parámetros
se editan en los módulos de configuración. Importar los módulos no ejecuta el
proceso ni monta Google Drive.

Se conserva la metodología del notebook, incluyendo el criterio de parada basado
en pérdida de entrenamiento y los mecanismos de respaldo de SARIMA/SARIMAX. Esta
conversión no constituye una revisión metodológica. Como mejora de manejo de
errores, si no se generan muestras válidas se informa antes de exportar Excel.

## Verificación

```bash
python -m pip install pytest
python -m pytest -q
```

Las pruebas verifican la equivalencia del código de las funciones con el notebook,
la generación de ventanas y objetivos, y la exportación de gráficos. Las pruebas
que necesitan dependencias ausentes se omiten explícitamente.
