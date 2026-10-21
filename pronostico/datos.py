"""2. Carga y ordenamiento temporal del dataset."""

from pathlib import Path
import numpy as np
import pandas as pd
from . import config_modelos as cfg


def sheet_name_from_sample_number(sample_number: int) -> str:
    return f"muestra_{int(sample_number):04d}"


def infer_target_variable_from_metadata(targets_df: pd.DataFrame):
    """
    Determina la variable objetivo.

    Prioridad:
    1) TARGET_VARIABLE_NAME si fue definido manualmente.
    2) Columna target_variable en outputs_entrenamiento.xlsx.
    """

    if cfg.TARGET_VARIABLE_NAME is not None:
        return cfg.TARGET_VARIABLE_NAME

    if "target_variable" in targets_df.columns:
        unique_targets = targets_df["target_variable"].dropna().unique()

        if len(unique_targets) == 1:
            return unique_targets[0]

        if len(unique_targets) > 1:
            raise ValueError(
                "El archivo outputs_entrenamiento.xlsx contiene más de una "
                "variable objetivo en la columna 'target_variable'. "
                "Definí TARGET_VARIABLE_NAME manualmente."
            )

    raise ValueError(
        "No se pudo determinar la variable objetivo. "
        "Definí TARGET_VARIABLE_NAME o asegurate de que exista la columna "
        "'target_variable' en outputs_entrenamiento.xlsx."
    )


def load_dataset_from_generated_excels(
    samples_excel: str,
    targets_excel: str,
    selected_series=None,
):
    """
    Lee los Excel generados previamente.

    Devuelve:
        X: np.ndarray con forma (n_muestras, WINDOW_SIZE, n_features)
        y: np.ndarray con forma (n_muestras,)
        metadata_df: metadatos del archivo outputs_entrenamiento.xlsx
        feature_names: nombres de las columnas utilizadas como entrada
        target_variable: nombre de la variable objetivo
    """

    samples_path = Path(samples_excel)
    targets_path = Path(targets_excel)

    if not samples_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {samples_excel}")

    if not targets_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {targets_excel}")

    targets_df = pd.read_excel(targets_excel)

    required_columns = ["sample_number", cfg.TARGET_VALUE_COLUMN]

    for col in required_columns:
        if col not in targets_df.columns:
            raise ValueError(f"No se encontró la columna '{col}' en {targets_excel}")

    target_variable = infer_target_variable_from_metadata(targets_df)

    xls = pd.ExcelFile(samples_excel)
    existing_sheets = set(xls.sheet_names)

    X_list = []
    y_list = []
    metadata_rows = []
    feature_names = None

    for _, row in targets_df.iterrows():
        sample_number = int(row["sample_number"])
        sheet_name = sheet_name_from_sample_number(sample_number)

        if sheet_name not in existing_sheets:
            raise ValueError(
                f"No se encontró la hoja '{sheet_name}' en {samples_excel}"
            )

        sample_df = pd.read_excel(samples_excel, sheet_name=sheet_name)

        if sample_df.shape[0] != cfg.WINDOW_SIZE:
            raise ValueError(
                f"La hoja {sheet_name} tiene {sample_df.shape[0]} filas, "
                f"pero se esperaban {cfg.WINDOW_SIZE}."
            )

        all_series_cols = list(sample_df.columns[1:])

        if selected_series is None:
            current_feature_names = all_series_cols
        else:
            missing_selected = [
                col for col in selected_series
                if col not in all_series_cols
            ]

            if missing_selected:
                raise ValueError(
                    f"En la hoja {sheet_name} faltan columnas seleccionadas: "
                    + ", ".join(missing_selected)
                )

            current_feature_names = list(selected_series)

        if target_variable not in current_feature_names:
            raise ValueError(
                f"La variable objetivo '{target_variable}' no está incluida "
                f"en SELECTED_SERIES. Para SARIMA/SARIMAX y redes, la serie "
                f"objetivo debe formar parte de la entrada."
            )

        if feature_names is None:
            feature_names = current_feature_names
        elif current_feature_names != feature_names:
            raise ValueError(
                f"La hoja {sheet_name} tiene columnas distintas a las anteriores."
            )

        values = sample_df[current_feature_names].to_numpy(dtype=float)

        if np.isnan(values).any():
            raise ValueError(
                f"La hoja {sheet_name} contiene NaN en las columnas seleccionadas. "
                f"Revisar imputación antes de entrenar."
            )

        target_value = float(row[cfg.TARGET_VALUE_COLUMN])

        if np.isnan(target_value):
            raise ValueError(
                f"El output de la muestra {sample_number} contiene NaN."
            )

        X_list.append(values)
        y_list.append(target_value)
        metadata_rows.append(row.to_dict())

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=float)
    metadata_df = pd.DataFrame(metadata_rows)

    return X, y, metadata_df, feature_names, target_variable


def order_dataset_temporally(X, y, metadata_df, order_mode="auto"):
    """
    Ordena las muestras de más antiguas a más recientes.

    Si puede interpretar target_period como fecha, ordena por fecha.
    Si no, usa sample_number descendente, asumiendo que muestra_0001
    es la más reciente.
    """

    metadata_df = metadata_df.copy()

    if order_mode in ["auto", "target_period"] and "target_period" in metadata_df.columns:
        parsed_dates = pd.to_datetime(metadata_df["target_period"], errors="coerce")

        if parsed_dates.notna().all():
            metadata_df["_sort_key"] = parsed_dates
            order_indices = metadata_df.sort_values("_sort_key").index.to_numpy()

            X_ord = X[order_indices]
            y_ord = y[order_indices]
            metadata_ord = (
                metadata_df
                .loc[order_indices]
                .drop(columns=["_sort_key"])
                .reset_index(drop=True)
            )

            return X_ord, y_ord, metadata_ord

        if order_mode == "target_period":
            raise ValueError(
                "No se pudo convertir target_period a fecha. "
                "Usar ORDER_MODE='sample_number_desc'."
            )

    if order_mode in ["auto", "sample_number_desc"]:
        order_indices = metadata_df.sort_values(
            "sample_number",
            ascending=False,
        ).index.to_numpy()

        X_ord = X[order_indices]
        y_ord = y[order_indices]
        metadata_ord = metadata_df.loc[order_indices].reset_index(drop=True)

        return X_ord, y_ord, metadata_ord

    raise ValueError(
        "ORDER_MODE debe ser 'auto', 'target_period' o 'sample_number_desc'."
    )
