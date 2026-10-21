"""Parámetros de la etapa de modelos."""

import torch

SAMPLES_EXCEL = "muestras_entrenamiento.xlsx"
TARGETS_EXCEL = "outputs_entrenamiento.xlsx"

OUTPUT_RESULTS_EXCEL = "comparacion_modelos_temporal_loo.xlsx"

WINDOW_SIZE = 24
TARGET_VALUE_COLUMN = "target_value"

ORDER_MODE = "auto"  # "auto", "target_period", "sample_number_desc"

MIN_TRAIN_SAMPLES = 12

RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# SELECCIÓN DE COLUMNAS
# ============================================================

# Lista de series que se usarán como entrada del modelo.
# Deben coincidir exactamente con los nombres de columnas de las hojas
# de muestras_entrenamiento.xlsx.
#
# Si SELECTED_SERIES = None, se usan todas las columnas de series disponibles.
#
# Ejemplo:
# SELECTED_SERIES = ["U1", "U2", "U3", "U5"]
#
SELECTED_SERIES = [
    "vsup", #
    #"vasmay",
    "vcc", # 6.6
    "velec", #
    "vmcame", #
    "argiva", #
    "ica_arg", #
    "bc_arg" #
    ] # None

# Si querés seleccionar manualmente:
# SELECTED_SERIES = [
#     "U1",
#     "U2",
#     "U3",
#     "U4",
# ]

# Variable objetivo. Si TARGET_VARIABLE_NAME = None, se intentará tomar
# desde la columna target_variable del archivo outputs_entrenamiento.xlsx.
TARGET_VARIABLE_NAME = None

# Ejemplo:
# TARGET_VARIABLE_NAME = "U1"


# ============================================================
# CONFIGURACIÓN SARIMA / SARIMAX
# ============================================================

SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (0, 1, 1, 12)

SARIMAX_ORDER = (1, 1, 1)
SARIMAX_SEASONAL_ORDER = (0, 1, 1, 12)

# Si querés desactivar estacionalidad:
# SARIMA_SEASONAL_ORDER = (0, 0, 0, 0)
# SARIMAX_SEASONAL_ORDER = (0, 0, 0, 0)


# ============================================================
# CONFIGURACIÓN REDES NEURONALES PEQUEÑAS
# ============================================================

BATCH_SIZE = 4
EPOCHS = 250
PATIENCE = 35
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3

CNN_CHANNELS = 8
CNN_KERNEL_SIZE = 3
CNN_DROPOUT = 0.30

TCN_CHANNELS = [8, 8]
TCN_KERNEL_SIZE = 2
TCN_DROPOUT = 0.30

