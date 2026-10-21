"""5. Validación temporal y resumen de modelos."""

import math
import numpy as np
import pandas as pd
from . import config_modelos as cfg
from .datos import load_dataset_from_generated_excels, order_dataset_temporally
from .modelos_estadisticos import sarima_forecast_one_step, sarimax_forecast_one_step
from .redes import SmallCNN1D, SmallTCN, count_trainable_parameters, scale_fold, set_seed, train_neural_model_for_fold


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))

    mask = y_true != 0

    if np.any(mask):
        mape = np.mean(
            np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        ) * 100
    else:
        mape = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
    }


def temporal_leave_one_out_comparison(
    X,
    y,
    metadata_df,
    feature_names,
    target_variable,
):
    n_samples, window_size, n_features = X.shape

    if n_samples <= cfg.MIN_TRAIN_SAMPLES:
        raise ValueError(
            f"No hay suficientes muestras. "
            f"n_samples={n_samples}, MIN_TRAIN_SAMPLES={cfg.MIN_TRAIN_SAMPLES}"
        )

    if target_variable not in feature_names:
        raise ValueError(
            f"La variable objetivo '{target_variable}' no está en feature_names."
        )

    target_col_idx = feature_names.index(target_variable)

    results = []

    for val_idx in range(cfg.MIN_TRAIN_SAMPLES, n_samples):
        print(f"Fold temporal: entrenar [0:{val_idx}] -> validar muestra {val_idx}")

        X_train = X[:val_idx]
        y_train = y[:val_idx]

        X_val = X[val_idx:val_idx + 1]
        y_val = y[val_idx]

        meta = metadata_df.iloc[val_idx].to_dict()

        # -----------------------------
        # 1) SARIMA baseline univariado
        # -----------------------------
        target_history = X_val[0, :, target_col_idx]

        pred_sarima, n_params_sarima = sarima_forecast_one_step(
            target_history=target_history,
            order=cfg.SARIMA_ORDER,
            seasonal_order=cfg.SARIMA_SEASONAL_ORDER,
        )

        # -----------------------------
        # 2) SARIMAX baseline con exógenas
        # -----------------------------
        exog_indices = [
            idx for idx in range(n_features)
            if idx != target_col_idx
        ]

        if len(exog_indices) > 0:
            exog_window = X_val[0][:, exog_indices]

            exog_window = np.asarray(exog_window, dtype=float)

            if exog_window.ndim == 1:
                exog_window = exog_window.reshape(-1, 1)

            pred_sarimax, n_params_sarimax = sarimax_forecast_one_step(
                target_window=target_history,
                exog_window=exog_window,
                order=cfg.SARIMAX_ORDER,
                seasonal_order=cfg.SARIMAX_SEASONAL_ORDER,
            )
        else:
            pred_sarimax = pred_sarima
            n_params_sarimax = n_params_sarima

        # -----------------------------
        # Escalado para redes
        # -----------------------------
        X_train_scaled, X_val_scaled, y_train_scaled, scaler_y = scale_fold(
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
        )

        # -----------------------------
        # 3) 1D-CNN pequeña
        # -----------------------------
        set_seed(cfg.RANDOM_SEED + val_idx)

        cnn_model = SmallCNN1D(
            n_features=n_features,
            n_channels=cfg.CNN_CHANNELS,
            kernel_size=cfg.CNN_KERNEL_SIZE,
            dropout=cfg.CNN_DROPOUT,
        )

        n_params_cnn_1d = count_trainable_parameters(cnn_model)

        pred_cnn_scaled = train_neural_model_for_fold(
            model=cnn_model,
            X_train=X_train_scaled,
            y_train=y_train_scaled,
            X_val=X_val_scaled,
            device=cfg.DEVICE,
            batch_size=cfg.BATCH_SIZE,
            epochs=cfg.EPOCHS,
            patience=cfg.PATIENCE,
            learning_rate=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
        )

        pred_cnn = scaler_y.inverse_transform(
            np.array([[pred_cnn_scaled]])
        )[0, 0]

        # -----------------------------
        # 4) TCN pequeña
        # -----------------------------
        set_seed(cfg.RANDOM_SEED + 10_000 + val_idx)

        tcn_model = SmallTCN(
            n_features=n_features,
            channels=cfg.TCN_CHANNELS,
            kernel_size=cfg.TCN_KERNEL_SIZE,
            dropout=cfg.TCN_DROPOUT,
        )

        n_params_tcn = count_trainable_parameters(tcn_model)

        pred_tcn_scaled = train_neural_model_for_fold(
            model=tcn_model,
            X_train=X_train_scaled,
            y_train=y_train_scaled,
            X_val=X_val_scaled,
            device=cfg.DEVICE,
            batch_size=cfg.BATCH_SIZE,
            epochs=cfg.EPOCHS,
            patience=cfg.PATIENCE,
            learning_rate=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
        )

        pred_tcn = scaler_y.inverse_transform(
            np.array([[pred_tcn_scaled]])
        )[0, 0]

        row = {
            "fold": val_idx - cfg.MIN_TRAIN_SAMPLES + 1,
            "val_position_chronological": val_idx,
            "train_samples": val_idx,
            "y_real": y_val,

            "target_variable": target_variable,
            "selected_series": ", ".join(feature_names),
            "n_selected_series": len(feature_names),

            "pred_sarima": pred_sarima,
            "pred_sarimax": pred_sarimax,
            "pred_cnn_1d": pred_cnn,
            "pred_tcn": pred_tcn,

            "error_sarima": y_val - pred_sarima,
            "error_sarimax": y_val - pred_sarimax,
            "error_cnn_1d": y_val - pred_cnn,
            "error_tcn": y_val - pred_tcn,

            "abs_error_sarima": abs(y_val - pred_sarima),
            "abs_error_sarimax": abs(y_val - pred_sarimax),
            "abs_error_cnn_1d": abs(y_val - pred_cnn),
            "abs_error_tcn": abs(y_val - pred_tcn),

            "params_sarima": n_params_sarima,
            "params_sarimax": n_params_sarimax,
            "params_cnn_1d": n_params_cnn_1d,
            "params_tcn": n_params_tcn,
        }

        for key, value in meta.items():
            row[f"meta_{key}"] = value

        results.append(row)

    results_df = pd.DataFrame(results)

    return results_df


def build_metrics_summary(results_df):
    y_true = results_df["y_real"].to_numpy(dtype=float)

    model_columns = {
        "SARIMA": {
            "pred": "pred_sarima",
            "params": "params_sarima",
        },
        "SARIMAX": {
            "pred": "pred_sarimax",
            "params": "params_sarimax",
        },
        "1D-CNN pequeña": {
            "pred": "pred_cnn_1d",
            "params": "params_cnn_1d",
        },
        "TCN pequeña": {
            "pred": "pred_tcn",
            "params": "params_tcn",
        },
    }

    rows = []

    for model_name, cols in model_columns.items():
        y_pred = results_df[cols["pred"]].to_numpy(dtype=float)
        metrics = compute_metrics(y_true, y_pred)

        params_values = results_df[cols["params"]].to_numpy(dtype=float)

        rows.append(
            {
                "modelo": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE": metrics["MAPE"],
                "parametros_promedio": np.nanmean(params_values),
                "parametros_min": np.nanmin(params_values),
                "parametros_max": np.nanmax(params_values),
                "n_selected_series": results_df["n_selected_series"].iloc[0],
                "target_variable": results_df["target_variable"].iloc[0],
                "selected_series": results_df["selected_series"].iloc[0],
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("MAE").reset_index(drop=True)

    return summary_df


def build_model_complexity_summary(results_df):
    """
    Genera una tabla separada enfocada únicamente en complejidad.
    """

    rows = [
        {
            "modelo": "SARIMA",
            "parametros_promedio": results_df["params_sarima"].mean(),
            "parametros_min": results_df["params_sarima"].min(),
            "parametros_max": results_df["params_sarima"].max(),
        },
        {
            "modelo": "SARIMAX",
            "parametros_promedio": results_df["params_sarimax"].mean(),
            "parametros_min": results_df["params_sarimax"].min(),
            "parametros_max": results_df["params_sarimax"].max(),
        },
        {
            "modelo": "1D-CNN pequeña",
            "parametros_promedio": results_df["params_cnn_1d"].mean(),
            "parametros_min": results_df["params_cnn_1d"].min(),
            "parametros_max": results_df["params_cnn_1d"].max(),
        },
        {
            "modelo": "TCN pequeña",
            "parametros_promedio": results_df["params_tcn"].mean(),
            "parametros_min": results_df["params_tcn"].min(),
            "parametros_max": results_df["params_tcn"].max(),
        },
    ]

    return pd.DataFrame(rows)


def modelos():
    set_seed(cfg.RANDOM_SEED)

    print(f"Dispositivo usado: {cfg.DEVICE}")

    X, y, metadata_df, feature_names, target_variable = load_dataset_from_generated_excels(
        samples_excel=cfg.SAMPLES_EXCEL,
        targets_excel=cfg.TARGETS_EXCEL,
        selected_series=cfg.SELECTED_SERIES,
    )

    X, y, metadata_df = order_dataset_temporally(
        X=X,
        y=y,
        metadata_df=metadata_df,
        order_mode=cfg.ORDER_MODE,
    )

    print()
    print("Dataset cargado y ordenado temporalmente.")
    print(f"X.shape: {X.shape}")
    print(f"y.shape: {y.shape}")
    print(f"Variable objetivo: {target_variable}")
    print(f"Variables seleccionadas: {feature_names}")
    print(f"Cantidad de variables seleccionadas: {len(feature_names)}")
    print(f"Cantidad de folds temporales: {X.shape[0] - cfg.MIN_TRAIN_SAMPLES}")

    print()
    print("Iniciando comparación temporal leave-one-out...")

    results_df = temporal_leave_one_out_comparison(
        X=X,
        y=y,
        metadata_df=metadata_df,
        feature_names=feature_names,
        target_variable=target_variable,
    )

    summary_df = build_metrics_summary(results_df)
    complexity_df = build_model_complexity_summary(results_df)

    with pd.ExcelWriter(cfg.OUTPUT_RESULTS_EXCEL, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="resumen_metricas", index=False)
        complexity_df.to_excel(writer, sheet_name="complejidad_modelos", index=False)
        results_df.to_excel(writer, sheet_name="predicciones_por_fold", index=False)

    print()
    print("Comparación finalizada.")
    print()
    print("Resumen de métricas:")
    print(summary_df)

    print()
    print("Resumen de complejidad:")
    print(complexity_df)

    print()
    print(f"Archivo generado: {cfg.OUTPUT_RESULTS_EXCEL}")
