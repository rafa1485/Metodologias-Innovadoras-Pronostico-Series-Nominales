"""Comprobaciones de la conversión y de las etapas con datos pequeños."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "ajuste_y_comparacion_modelos_ML-21-oct-26.ipynb"


class RemoveConfigNamespace(ast.NodeTransformer):
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "cfg":
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node


def test_all_functions_preserve_notebook_logic():
    notebook = json.loads(NOTEBOOK.read_text())
    original = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    original[node.name] = node
    converted = {}
    for path in (ROOT / "pronostico").glob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                node = RemoveConfigNamespace().visit(node)
                if node.name == "main_crear_muestras":
                    # Única mejora deliberada: impedir exportación de cero muestras.
                    node.body = [x for x in node.body if not (
                        isinstance(x, ast.If) and isinstance(x.test, ast.UnaryOp)
                        and isinstance(x.test.op, ast.Not)
                        and isinstance(x.test.operand, ast.Name)
                        and x.test.operand.id == "samples"
                    )]
                converted[node.name] = node
    assert converted.keys() == original.keys()
    for name in original:
        assert ast.dump(converted[name]) == ast.dump(original[name]), name


def test_samples_match_notebook_and_use_last_target(tmp_path):
    pytest.importorskip("statsmodels")
    import numpy as np
    import pandas as pd
    from pronostico import muestras

    notebook = json.loads(NOTEBOOK.read_text())
    original = {}
    exec("".join(notebook["cells"][1]["source"]), original)
    df = pd.DataFrame({"Periodo": pd.date_range("2020-01-01", periods=40, freq="MS"),
                       "vsup": np.arange(40, dtype=float), "vcc": np.arange(40) * 2.0})
    kwargs = dict(df=df, target_column="vsup", window_size=24,
                  missing_periods_by_col={"vsup": 0, "vcc": 0},
                  random_steps=[2, 3], seed=20)
    expected, expected_y, expected_skipped = original["generate_training_samples"](**kwargs)
    actual, actual_y, actual_skipped = muestras.generate_training_samples(**kwargs)
    pd.testing.assert_frame_equal(actual_y, expected_y)
    pd.testing.assert_frame_equal(actual_skipped, expected_skipped)
    assert actual_y.iloc[0]["target_value"] == 39
    for a, b in zip(actual, expected):
        pd.testing.assert_frame_equal(a["sample_df"], b["sample_df"])
    muestras.save_samples_to_excel(actual, tmp_path / "muestras.xlsx")
    muestras.save_targets_to_excel(actual_y, tmp_path / "objetivos.xlsx")
    assert len(pd.ExcelFile(tmp_path / "muestras.xlsx").sheet_names) == len(actual)
    pd.testing.assert_frame_equal(
        pd.read_excel(tmp_path / "objetivos.xlsx"), actual_y, check_dtype=False)


def test_graphics_cli_exports_last_predictions(tmp_path):
    import shutil
    import pandas as pd

    pytest.importorskip("matplotlib")
    filename = "comparacion_modelos_temporal_loo.xlsx"
    shutil.copyfile(ROOT / filename, tmp_path / filename)
    subprocess.run([sys.executable, str(ROOT / "ajuste_y_comparacion_modelos_ML.py"),
                    "graficos", "--output-dir", str(tmp_path), "--horizonte", "3"],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    original = pd.read_excel(tmp_path / filename, sheet_name="predicciones_por_fold").tail(3)
    exported = pd.read_excel(tmp_path / "pronosticos_T_periodos.xlsx")
    assert len(exported) == 3
    for col in ("y_real", "pred_sarima", "pred_sarimax", "pred_cnn_1d", "pred_tcn"):
        assert exported[col].tolist() == original[col].tolist()
    for name in ("comparacion_pronosticos_T_periodos.png", "comparacion_errores_T_periodos.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_temporal_comparison_one_fold(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("statsmodels")
    pytest.importorskip("sklearn")
    import numpy as np
    import pandas as pd
    import torch
    from pronostico import comparacion, config_modelos as cfg

    for name, value in dict(MIN_TRAIN_SAMPLES=2, EPOCHS=1, PATIENCE=1,
                            DEVICE=torch.device("cpu"), SARIMA_ORDER=(0, 0, 0),
                            SARIMA_SEASONAL_ORDER=(0, 0, 0, 0),
                            SARIMAX_ORDER=(0, 0, 0),
                            SARIMAX_SEASONAL_ORDER=(0, 0, 0, 0)).items():
        monkeypatch.setattr(cfg, name, value)
    X = np.random.default_rng(42).normal(size=(3, 24, 2))
    result = comparacion.temporal_leave_one_out_comparison(
        X, np.array([1., 2., 3.]), pd.DataFrame({"sample_number": [3, 2, 1]}),
        ["vsup", "vcc"], "vsup")
    assert result["train_samples"].tolist() == [2]
    for model in ("sarima", "sarimax", "cnn_1d", "tcn"):
        assert np.isfinite(result[f"pred_{model}"]).all()
        assert result[f"abs_error_{model}"].iloc[0] == pytest.approx(
            abs(3 - result[f"pred_{model}"].iloc[0]))


def test_complete_pipeline_exports_workbooks(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("statsmodels")
    pytest.importorskip("sklearn")
    import numpy as np
    import pandas as pd
    import torch
    from ajuste_y_comparacion_modelos_ML import main
    from pronostico import config_modelos as cfg

    monkeypatch.setattr(cfg, "MIN_TRAIN_SAMPLES", 6)
    monkeypatch.setattr(cfg, "DEVICE", torch.device("cpu"))
    for name in ("SARIMA_ORDER", "SARIMAX_ORDER"):
        monkeypatch.setattr(cfg, name, (0, 0, 0))
    for name in ("SARIMA_SEASONAL_ORDER", "SARIMAX_SEASONAL_ORDER"):
        monkeypatch.setattr(cfg, name, (0, 0, 0, 0))
    df = pd.DataFrame({"Periodo": pd.date_range("2020-01-01", periods=40, freq="MS")})
    for i, col in enumerate(["vsup", "vasmay", "vcc", "velec", "vmcame", "argiva", "ica_arg", "bc_arg"]):
        df[col] = np.arange(40, dtype=float) + i + 1
    input_path = tmp_path / "entrada.xlsx"
    df.to_excel(input_path, startrow=8, index=False)
    output = tmp_path / "salida"
    main(["todo", "--input", str(input_path), "--output-dir", str(output), "--epochs", "1"])
    result = pd.read_excel(output / "comparacion_modelos_temporal_loo.xlsx", sheet_name=None)
    assert set(result) == {"resumen_metricas", "complejidad_modelos", "predicciones_por_fold"}
    assert len(result["predicciones_por_fold"]) == 1
    assert len(pd.read_excel(output / "pronosticos_T_periodos.xlsx")) == 1
