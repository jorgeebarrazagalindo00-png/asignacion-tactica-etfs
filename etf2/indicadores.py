"""Indicadores por ETF.

Regla de oro: el valor de un indicador en la fecha t usa SOLO datos hasta t.
Nada mira hacia adelante (hay un test que lo comprueba).
"""

import numpy as np
import pandas as pd

from .config import DIAS_ANUALES, RETORNOS, VENTANA_VOL

COLUMNAS_IND = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_60d", "momentum", "drawdown"]


def calcular_indicadores(px: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """px: precios ajustados (fechas x tickers). Devuelve un DataFrame ancho por indicador."""
    ind = {}
    for nombre, dias in RETORNOS.items():
        ind[nombre] = px / px.shift(dias) - 1.0

    log_ret = np.log(px).diff()
    ind["vol_60d"] = log_ret.rolling(VENTANA_VOL).std() * np.sqrt(DIAS_ANUALES)

    # Momentum: promedio simple de los retornos a 3, 6 y 12 meses.
    ind["momentum"] = (ind["ret_3m"] + ind["ret_6m"] + ind["ret_12m"]) / 3.0

    # Distancia al máximo de los últimos 12 meses (0 = en máximos; -0,2 = 20 % abajo).
    ind["drawdown"] = px / px.rolling(DIAS_ANUALES, min_periods=1).max() - 1.0
    return ind


def a_largo(ind: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une los indicadores en una tabla larga (fecha, ticker, ret_1m, ...), sin filas vacías."""
    largo = None
    for nombre in COLUMNAS_IND:
        t = (ind[nombre].rename_axis(index="fecha", columns="ticker")
             .reset_index()
             .melt(id_vars="fecha", var_name="ticker", value_name=nombre))
        largo = t if largo is None else largo.merge(t, on=["fecha", "ticker"])
    largo = largo.replace([np.inf, -np.inf], np.nan)
    largo = largo.dropna(subset=COLUMNAS_IND, how="all")
    return largo.sort_values(["ticker", "fecha"]).reset_index(drop=True)
