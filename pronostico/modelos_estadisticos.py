"""3. Pronósticos base SARIMA y SARIMAX."""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def sarima_forecast_one_step(
    target_history,
    order=(1, 1, 1),
    seasonal_order=(0, 1, 1, 12),
):
    """
    SARIMA univariado.

    Devuelve:
        predicción, número de parámetros ajustados
    """

    target_history = pd.Series(target_history).dropna()

    if len(target_history) == 0:
        return np.nan, 0

    if len(target_history) < 8:
        return float(target_history.iloc[-1]), 0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = SARIMAX(
                target_history,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

            result = model.fit(disp=False)
            pred = result.forecast(steps=1)

        n_params = len(result.params)

        return float(pred.iloc[0]), n_params

    except Exception:
        return float(target_history.iloc[-1]), 0


def sarimax_forecast_one_step(
    target_window,
    exog_window,
    order=(1, 1, 1),
    seasonal_order=(0, 1, 1, 12),
):
    """
    SARIMAX con variables exógenas.

    Usa una alineación rezagada para evitar requerir exógenas futuras:

        exog(t-23) -> y(t-22)
        exog(t-22) -> y(t-21)
        ...
        exog(t-1)  -> y(t)
        exog(t)    -> predice y(t+1)

    Devuelve:
        predicción, número de parámetros ajustados
    """

    target_window = np.asarray(target_window, dtype=float).reshape(-1)
    exog_window = np.asarray(exog_window, dtype=float)

    if exog_window.ndim == 1:
        exog_window = exog_window.reshape(-1, 1)

    # Si llega transpuesta: (n_exog, window_size) -> (window_size, n_exog)
    if (
        exog_window.shape[0] != len(target_window)
        and exog_window.shape[1] == len(target_window)
    ):
        exog_window = exog_window.T

    # Recorte defensivo si hubiera alguna diferencia menor.
    if exog_window.shape[0] != len(target_window):
        min_len = min(len(target_window), exog_window.shape[0])
        target_window = target_window[-min_len:]
        exog_window = exog_window[-min_len:, :]

    if len(target_window) < 8:
        return float(target_window[-1]), 0

    if exog_window.shape[1] == 0:
        return float(target_window[-1]), 0

    endog_history = pd.Series(target_window[1:]).dropna()
    exog_history = exog_window[:-1, :]
    exog_next = exog_window[-1:, :]

    if len(endog_history) != exog_history.shape[0]:
        min_len = min(len(endog_history), exog_history.shape[0])
        endog_history = endog_history.iloc[-min_len:]
        exog_history = exog_history[-min_len:, :]

    if len(endog_history) < 8:
        return float(target_window[-1]), 0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = SARIMAX(
                endog_history,
                exog=exog_history,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

            result = model.fit(disp=False)

            pred = result.forecast(
                steps=1,
                exog=exog_next,
            )

        n_params = len(result.params)

        return float(pred.iloc[0]), n_params

    except Exception:
        return float(target_window[-1]), 0
