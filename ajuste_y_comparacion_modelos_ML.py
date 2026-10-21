"""Entrada por consola del proceso de pronóstico de series temporales."""

import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("El valor debe ser mayor que cero.")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "etapa", nargs="?", default="todo",
        choices=("todo", "muestras", "modelos", "graficos"),
        help="Etapa a ejecutar (por defecto: todo).",
    )
    parser.add_argument("--input", type=Path, default=BASE_DIR / "Input_series.xlsx",
                        help="Excel de entrada para crear muestras.")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "resultados",
                        help="Carpeta compartida por las tres etapas.")
    parser.add_argument("--epochs", type=positive_int, default=None,
                        help="Épocas máximas de las redes; por defecto se usa config_modelos.py.")
    parser.add_argument("--horizonte", type=positive_int, default=None,
                        help="Cantidad de predicciones finales a graficar.")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Las importaciones por etapa permiten graficar sin cargar PyTorch.
    if args.etapa in ("todo", "muestras"):
        from pronostico import config_muestras as cfg
        from pronostico.muestras import main_crear_muestras

        cfg.INPUT_EXCEL = str(args.input.resolve())
        cfg.OUTPUT_SAMPLES_EXCEL = str(output / "muestras_entrenamiento.xlsx")
        cfg.OUTPUT_TARGETS_EXCEL = str(output / "outputs_entrenamiento.xlsx")
        cfg.OUTPUT_SKIPPED_EXCEL = str(output / "muestras_descartadas.xlsx")
        main_crear_muestras()

    if args.etapa in ("todo", "modelos"):
        from pronostico import config_modelos as cfg
        from pronostico.comparacion import modelos

        cfg.SAMPLES_EXCEL = str(output / "muestras_entrenamiento.xlsx")
        cfg.TARGETS_EXCEL = str(output / "outputs_entrenamiento.xlsx")
        cfg.OUTPUT_RESULTS_EXCEL = str(output / "comparacion_modelos_temporal_loo.xlsx")
        if args.epochs is not None:
            cfg.EPOCHS = args.epochs
        modelos()

    if args.etapa in ("todo", "graficos"):
        from pronostico import config_graficos as cfg
        from pronostico.graficos import comparacion_grafica

        cfg.INPUT_RESULTS_EXCEL = str(output / "comparacion_modelos_temporal_loo.xlsx")
        cfg.OUTPUT_FORECAST_EXCEL = str(output / "pronosticos_T_periodos.xlsx")
        cfg.OUTPUT_FORECAST_PLOT = str(output / "comparacion_pronosticos_T_periodos.png")
        cfg.OUTPUT_ERRORS_PLOT = str(output / "comparacion_errores_T_periodos.png")
        if args.horizonte is not None:
            cfg.FORECAST_HORIZON_T = args.horizonte
        comparacion_grafica()


if __name__ == "__main__":
    main()
