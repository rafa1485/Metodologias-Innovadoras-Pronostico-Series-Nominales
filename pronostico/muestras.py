"""1. Lectura, imputación y creación de muestras."""

import warnings
import random
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from . import config_muestras as cfg


def read_time_series_excel(path: str, header_row: int = 8) -> pd.DataFrame:
    """
    Lee el Excel usando como encabezado la fila 9.
    La primera columna es la etiqueta del periodo.
    Todas las demás columnas son series temporales.
    """

    df = pd.read_excel(path, header=header_row)

    df = df.dropna(how="all").reset_index(drop=True)

    if df.shape[1] < 2:
        raise ValueError(
            "El archivo debe tener al menos una columna de periodo y una columna de serie."
        )

    # Quitar columnas completamente vacías, por si el Excel tiene columnas residuales.
    df = df.dropna(axis=1, how="all")

    period_col = df.columns[0]
    series_cols = list(df.columns[1:])

    # Convertir las columnas de series a numéricas.
    for col in series_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("Archivo leído correctamente.")
    print(f"Columna de periodo: {period_col}")
    print(f"Columnas de series: {series_cols}")

    return df


def print_dataset_size(samples: list[dict], targets_df: pd.DataFrame):
    """
    Imprime el tamaño del dataset generado.
    """

    n_samples = len(samples)

    if n_samples == 0:
        print("No se generaron muestras.")
        return

    first_sample_df = samples[0]["sample_df"]

    window_size = first_sample_df.shape[0]
    n_columns_total = first_sample_df.shape[1]

    # Restamos 1 porque la primera columna es la etiqueta del periodo
    n_series = n_columns_total - 1

    print()
    print("Tamaño del dataset generado:")
    print(f"  X: ({n_samples}, {window_size}, {n_series})")
    print(f"  y: ({len(targets_df)},)")
    print(f"  Cantidad de muestras: {n_samples}")
    print(f"  Periodos por muestra: {window_size}")
    print(f"  Series por muestra: {n_series}")


def normalize_missing_periods(missing_periods, series_cols: list[str]) -> dict:
    """
    Convierte missing_periods a un diccionario:
        {nombre_columna: cantidad_faltantes_finales}
    """

    if isinstance(missing_periods, dict):
        result = {}

        for col in series_cols:
            result[col] = int(missing_periods.get(col, 0))

        unknown_cols = set(missing_periods.keys()) - set(series_cols)
        if unknown_cols:
            raise ValueError(
                f"Hay variables en MISSING_PERIODS que no existen en el Excel: {unknown_cols}"
            )

        return result

    if isinstance(missing_periods, list):
        if len(missing_periods) != len(series_cols):
            raise ValueError(
                f"MISSING_PERIODS debe tener {len(series_cols)} valores, "
                f"uno por cada serie: {series_cols}"
            )

        return {
            col: int(k)
            for col, k in zip(series_cols, missing_periods)
        }

    raise TypeError("MISSING_PERIODS debe ser una lista o un diccionario.")


def fit_sarimax_and_forecast(
    serie: pd.Series,
    steps: int,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    min_history: int = 18,
) -> np.ndarray:
    """
    Ajusta SARIMAX sobre la historia disponible y pronostica 'steps' valores.

    Si hay muy pocos datos o el modelo falla, usa como respaldo el último
    valor observado.
    """

    serie = pd.to_numeric(serie, errors="coerce")

    # Se eliminan faltantes iniciales y cualquier otro NaN histórico.
    serie = serie.dropna()

    if steps <= 0:
        return np.array([])

    if len(serie) == 0:
        return np.repeat(np.nan, steps)

    if len(serie) < min_history:
        last_value = serie.iloc[-1]
        return np.repeat(last_value, steps)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = SARIMAX(
                serie,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

            result = model.fit(disp=False)
            forecast = result.forecast(steps=steps)

        return forecast.to_numpy()

    except Exception:
        last_value = serie.iloc[-1]
        return np.repeat(last_value, steps)


def handle_initial_missing_values(
    sample: pd.DataFrame,
    series_cols: list[str],
    policy: str,
) -> tuple[pd.DataFrame, bool, str]:
    """
    Maneja faltantes que permanecen dentro de la ventana luego de imputar
    los faltantes finales.

    Devuelve:
        sample, is_valid, reason
    """

    remaining_nan_cols = [
        col for col in series_cols
        if sample[col].isna().any()
    ]

    if not remaining_nan_cols:
        return sample, True, ""

    if policy == "skip_sample":
        reason = (
            "La muestra conserva faltantes iniciales o internos en las columnas: "
            + ", ".join(remaining_nan_cols)
        )
        return sample, False, reason

    if policy == "leave_blank":
        return sample, True, ""

    if policy == "fill_first_valid":
        sample = sample.copy()

        for col in remaining_nan_cols:
            first_valid = sample[col].dropna()

            if len(first_valid) == 0:
                reason = f"La columna {col} no tiene ningún valor válido dentro de la muestra."
                return sample, False, reason

            sample[col] = sample[col].fillna(first_valid.iloc[0])

        return sample, True, ""

    raise ValueError(
        "INITIAL_MISSING_POLICY debe ser: "
        "'skip_sample', 'leave_blank' o 'fill_first_valid'."
    )


def build_sample_with_imputation(
    df: pd.DataFrame,
    end_idx: int,
    window_size: int,
    missing_periods_by_col: dict,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    initial_missing_policy: str = "skip_sample",
) -> tuple[pd.DataFrame, bool, str]:
    """
    Construye una muestra de longitud window_size terminando en end_idx.

    Para cada columna:
    - toma los últimos window_size periodos;
    - simula que los últimos k periodos no estaban disponibles;
    - imputa esos k periodos con SARIMAX;
    - controla faltantes iniciales o internos.
    """

    period_col = df.columns[0]
    series_cols = list(df.columns[1:])

    start_idx = end_idx - window_size + 1

    if start_idx < 0:
        return pd.DataFrame(), False, "No hay suficientes periodos para construir la ventana."

    sample = df.iloc[start_idx:end_idx + 1].copy().reset_index(drop=True)

    for col_name in series_cols:
        k_missing = int(missing_periods_by_col.get(col_name, 0))

        if k_missing <= 0:
            continue

        if k_missing >= window_size:
            return (
                sample,
                False,
                f"La cantidad de faltantes finales para {col_name} "
                f"no puede ser mayor o igual que WINDOW_SIZE."
            )

        available_until_idx = end_idx - k_missing

        if available_until_idx < 0:
            return (
                sample,
                False,
                f"No hay historia suficiente para imputar {col_name}."
            )

        history = df.loc[:available_until_idx, col_name]

        forecast_values = fit_sarimax_and_forecast(
            serie=history,
            steps=k_missing,
            order=order,
            seasonal_order=seasonal_order,
            min_history=cfg.MIN_HISTORY_FOR_SARIMAX,
        )

        positions_to_impute = range(window_size - k_missing, window_size)

        for pos, imputed_value in zip(positions_to_impute, forecast_values):
            sample.loc[pos, col_name] = imputed_value

    sample, is_valid, reason = handle_initial_missing_values(
        sample=sample,
        series_cols=series_cols,
        policy=initial_missing_policy,
    )

    return sample, is_valid, reason


def generate_training_samples(
    df: pd.DataFrame,
    target_column: str,
    window_size: int,
    missing_periods_by_col: dict,
    random_steps: list[int],
    seed: int = 42,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    initial_missing_policy: str = "skip_sample",
):
    """
    Genera muestras de entrenamiento.

    Para una ventana que termina en end_idx:
        X = ventana [end_idx - window_size + 1, ..., end_idx]
        y = target_column en end_idx + 1

    Las ventanas se recorren hacia atrás con saltos aleatorios de 2 o 3 periodos.
    """

    random.seed(seed)

    period_col = df.columns[0]
    series_cols = list(df.columns[1:])

    if target_column not in series_cols:
        raise ValueError(
            f"La variable objetivo '{target_column}' no existe entre las columnas de series: "
            f"{series_cols}"
        )

    samples = []
    targets = []
    skipped = []

    # Debe existir el valor objetivo en end_idx + 1.
    end_idx = len(df) - 2

    sample_number = 1
    candidate_number = 1

    while end_idx - window_size + 1 >= 0:

        target_idx = end_idx + 1
        target_value = df.loc[target_idx, target_column]
        target_period = df.loc[target_idx, period_col]

        if pd.isna(target_value):
            skipped.append(
                {
                    "candidate_number": candidate_number,
                    "end_idx": end_idx,
                    "target_idx": target_idx,
                    "reason": f"Objetivo {target_column} en el {target_period} está vacío.",
                }
            )

            #end_idx -= random.choice(random_steps)
            end_idx -= 1
            candidate_number += 1
            continue

        sample_df, is_valid, reason = build_sample_with_imputation(
            df=df,
            end_idx=end_idx,
            window_size=window_size,
            missing_periods_by_col=missing_periods_by_col,
            order=order,
            seasonal_order=seasonal_order,
            initial_missing_policy=initial_missing_policy,
        )

        if not is_valid:
            skipped.append(
                {
                    "candidate_number": candidate_number,
                    "end_idx": end_idx,
                    "target_idx": target_idx,
                    "reason": reason,
                }
            )

            #end_idx -= random.choice(random_steps)
            end_idx -= 1
            candidate_number += 1
            continue

        samples.append(
            {
                "sample_number": sample_number,
                "candidate_number": candidate_number,
                "end_idx": end_idx,
                "target_idx": target_idx,
                "sample_df": sample_df,
            }
        )

        targets.append(
            {
                "sample_number": sample_number,
                "candidate_number": candidate_number,
                "input_start_period": sample_df.iloc[0][period_col],
                "input_end_period": sample_df.iloc[-1][period_col],
                "target_period": target_period,
                "target_variable": target_column,
                "target_value": target_value,
            }
        )

        step = random.choice(random_steps)
        end_idx -= step
        sample_number += 1
        candidate_number += 1

    targets_df = pd.DataFrame(targets)
    skipped_df = pd.DataFrame(skipped)

    print(f"end_idx: {end_idx}")
    print(f"len(df): {len(df)}")
    print("Dataframe")
    print(targets_df)
    print()
    print("Skipped")
    print(skipped_df)
    print("=================================")
    return samples, targets_df, skipped_df


def save_samples_to_excel(samples: list[dict], output_path: str):
    """
    Guarda cada muestra en una hoja separada.
    """

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sample in samples:
            sample_number = sample["sample_number"]
            sample_df = sample["sample_df"]

            sheet_name = f"muestra_{sample_number:04d}"[:31]

            sample_df.to_excel(writer, sheet_name=sheet_name, index=False)


def save_targets_to_excel(targets_df: pd.DataFrame, output_path: str):
    """
    Guarda los outputs de entrenamiento.
    """

    targets_df.to_excel(output_path, index=False)


def save_skipped_to_excel(skipped_df: pd.DataFrame, output_path: str):
    """
    Guarda un registro de las muestras descartadas.
    """

    if len(skipped_df) > 0:
        skipped_df.to_excel(output_path, index=False)


def main_crear_muestras():
    input_path = Path(cfg.INPUT_EXCEL)

    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {cfg.INPUT_EXCEL}")

    df = read_time_series_excel(
        path=cfg.INPUT_EXCEL,
        header_row=cfg.EXCEL_HEADER_ROW,
    )

    period_col = df.columns[0]
    series_cols = list(df.columns[1:])

    missing_periods_by_col = normalize_missing_periods(
        missing_periods=cfg.MISSING_PERIODS,
        series_cols=series_cols,
    )

    print()
    print("Configuración de faltantes finales:")
    for col, k in missing_periods_by_col.items():
        print(f"  {col}: {k}")

    print()
    print(f"Variable objetivo: {cfg.TARGET_COLUMN}")
    print(f"Tamaño de ventana: {cfg.WINDOW_SIZE}")
    print(f"Política de faltantes iniciales: {cfg.INITIAL_MISSING_POLICY}")

    samples, targets_df, skipped_df = generate_training_samples(
        df=df,
        target_column=cfg.TARGET_COLUMN,
        window_size=cfg.WINDOW_SIZE,
        missing_periods_by_col=missing_periods_by_col,
        random_steps=cfg.RANDOM_STEPS,
        seed=cfg.RANDOM_SEED,
        order=cfg.SARIMAX_ORDER,
        seasonal_order=cfg.SARIMAX_SEASONAL_ORDER,
        initial_missing_policy=cfg.INITIAL_MISSING_POLICY,
    )

    if not samples:
        raise ValueError("No se generaron muestras válidas; revisá la ventana y los datos de entrada.")

    print_dataset_size(samples, targets_df)
    save_samples_to_excel(samples, cfg.OUTPUT_SAMPLES_EXCEL)
    save_targets_to_excel(targets_df, cfg.OUTPUT_TARGETS_EXCEL)
    save_skipped_to_excel(skipped_df, cfg.OUTPUT_SKIPPED_EXCEL)

    print()
    print("Proceso finalizado.")
    print(f"Muestras generadas: {len(samples)}")
    print(f"Muestras descartadas: {len(skipped_df)}")
    print(f"Archivo de muestras: {cfg.OUTPUT_SAMPLES_EXCEL}")
    print(f"Archivo de outputs: {cfg.OUTPUT_TARGETS_EXCEL}")

    if len(skipped_df) > 0:
        print(f"Archivo de muestras descartadas: {cfg.OUTPUT_SKIPPED_EXCEL}")
