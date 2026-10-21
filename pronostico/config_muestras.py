"""Parámetros de la etapa de muestras."""

INPUT_EXCEL = "Input_series.xlsx"

OUTPUT_SAMPLES_EXCEL = "muestras_entrenamiento.xlsx"
OUTPUT_TARGETS_EXCEL = "outputs_entrenamiento.xlsx"
OUTPUT_SKIPPED_EXCEL = "muestras_descartadas.xlsx"

# Los nombres de columnas están en la fila 9 de Excel.
# pandas cuenta desde 0, entonces header=8.
EXCEL_HEADER_ROW = 8

WINDOW_SIZE = 24

RANDOM_STEPS = [2, 3]

RANDOM_SEED = 20

# Nombre de la variable que se desea predecir.
# Debe coincidir exactamente con el nombre de una columna del Excel.
TARGET_COLUMN = "vsup"

# Lista o diccionario con los periodos faltantes finales por variable.
#
# Opción 1: lista, alineada con las columnas de series, sin contar la columna periodo.
# Si el Excel tiene:
#   Periodo | U1 | U2 | U3 | U4
# entonces:
#   MISSING_PERIODS = [0, 1, 2, 3]
#
# Opción 2: diccionario por nombre de variable.
#   MISSING_PERIODS = {
#       "U1": 0,
#       "U2": 1,
#       "U3": 2,
#       "U4": 3,
#   }
#


MISSING_PERIODS = {
    "vsup": 0, #2,
    "vasmay": 0, #2,
    "vcc": 0, #2,
    "velec": 0, #2,
    "vmcame": 0,
    "argiva": 0, #1,
    "ica_arg": 0,
    "bc_arg": 0,
}

# Política frente a faltantes iniciales dentro de una ventana.
#
# "skip_sample":
#     Descarta la muestra si alguna columna conserva NaN luego de imputar
#     los faltantes finales con SARIMAX.
#
# "leave_blank":
#     Deja los NaN en el Excel. No recomendado si luego se entrena una red
#     directamente con esas muestras.
#
# "fill_first_valid":
#     Rellena los NaN iniciales con el primer valor observado de esa columna
#     dentro de la muestra. Es una imputación fuerte y debe usarse con cuidado.
#
INITIAL_MISSING_POLICY = "skip_sample"

# Configuración SARIMAX.
# Para series mensuales con estacionalidad anual:
SARIMAX_ORDER = (1, 1, 1)
SARIMAX_SEASONAL_ORDER = (1, 1, 1, 12)

# Si no se quiere estacionalidad:
# SARIMAX_SEASONAL_ORDER = (0, 0, 0, 0)

MIN_HISTORY_FOR_SARIMAX = 18

