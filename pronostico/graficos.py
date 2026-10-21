"""6. Exportación y gráficos de los últimos pronósticos."""

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import config_graficos as cfg


def plot_forecasts_T_periods(forecast_df, output_png):
    """
    Grafica valores reales y pronósticos de:
    - SARIMA
    - SARIMAX
    - 1D-CNN
    - TCN
    """

    x = forecast_df["periodo"].astype(str)

    plt.figure(figsize=(13, 6))

    plt.plot(
        x,
        forecast_df["y_real"],
        marker="o",
        linewidth=2.5,
        label="Real",
    )

    plt.plot(
        x,
        forecast_df["pred_sarima"],
        marker="o",
        linestyle="--",
        label="SARIMA",
    )

    plt.plot(
        x,
        forecast_df["pred_sarimax"],
        marker="o",
        linestyle="--",
        label="SARIMAX",
    )

    plt.plot(
        x,
        forecast_df["pred_cnn_1d"],
        marker="o",
        linestyle="--",
        label="1D-CNN pequeña",
    )

    plt.plot(
        x,
        forecast_df["pred_tcn"],
        marker="o",
        linestyle="--",
        label="TCN pequeña",
    )

    plt.title(f"Comparación de pronósticos - últimos {len(forecast_df)} periodos")
    plt.xlabel("Periodo")
    plt.ylabel("Valor")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_png, dpi=300)
    plt.close()


def plot_absolute_errors_T_periods(forecast_df, output_png):
    """
    Grafica el error absoluto de cada modelo por periodo.
    Esto ayuda a ver qué modelo se comporta mejor en cada punto.
    """

    x = forecast_df["periodo"].astype(str)

    plt.figure(figsize=(13, 6))

    plt.plot(
        x,
        forecast_df["abs_error_sarima"],
        marker="o",
        linestyle="-",
        label="SARIMA",
    )

    plt.plot(
        x,
        forecast_df["abs_error_sarimax"],
        marker="o",
        linestyle="-",
        label="SARIMAX",
    )

    plt.plot(
        x,
        forecast_df["abs_error_cnn_1d"],
        marker="o",
        linestyle="-",
        label="1D-CNN pequeña",
    )

    plt.plot(
        x,
        forecast_df["abs_error_tcn"],
        marker="o",
        linestyle="-",
        label="TCN pequeña",
    )

    plt.title(f"Errores absolutos por periodo - últimos {len(forecast_df)} periodos")
    plt.xlabel("Periodo")
    plt.ylabel("Error absoluto")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_png, dpi=300)
    plt.close()


def comparacion_grafica():
    results_df = pd.read_excel(
        cfg.INPUT_RESULTS_EXCEL,
        sheet_name="predicciones_por_fold",
    )

    # Intentamos usar el periodo real desde la metadata.
    # En el script anterior suele quedar como meta_target_period.
    if "meta_target_period" in results_df.columns:
        period_col = "meta_target_period"
    else:
        period_col = "val_position_chronological"

    forecast_df = results_df.tail(cfg.FORECAST_HORIZON_T).copy()

    forecast_df = forecast_df.rename(
        columns={
            period_col: "periodo",
        }
    )

    required_columns = [
        "periodo",
        "y_real",
        "pred_sarima",
        "pred_sarimax",
        "pred_cnn_1d",
        "pred_tcn",
        "error_sarima",
        "error_sarimax",
        "error_cnn_1d",
        "error_tcn",
        "abs_error_sarima",
        "abs_error_sarimax",
        "abs_error_cnn_1d",
        "abs_error_tcn",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in forecast_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Faltan columnas en el archivo de resultados: "
            + ", ".join(missing_columns)
            + "\nVerificá que el script de comparación haya generado resultados "
              "para SARIMA, SARIMAX, 1D-CNN y TCN."
        )

    forecast_df = forecast_df[required_columns]

    forecast_df.to_excel(cfg.OUTPUT_FORECAST_EXCEL, index=False)

    plot_forecasts_T_periods(
        forecast_df=forecast_df,
        output_png=cfg.OUTPUT_FORECAST_PLOT,
    )

    plot_absolute_errors_T_periods(
        forecast_df=forecast_df,
        output_png=cfg.OUTPUT_ERRORS_PLOT,
    )

    print("Comparación gráfica generada.")
    print(f"Archivo Excel: {cfg.OUTPUT_FORECAST_EXCEL}")
    print(f"Gráfico pronósticos: {cfg.OUTPUT_FORECAST_PLOT}")
    print(f"Gráfico errores absolutos: {cfg.OUTPUT_ERRORS_PLOT}")
