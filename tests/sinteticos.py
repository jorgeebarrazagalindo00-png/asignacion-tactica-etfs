"""Precios inventados pero con forma realista, para pruebas y capturas de pantalla.

No son datos de mercado: son caminatas aleatorias con semilla fija, correlacionadas
entre sí de forma parecida a los activos reales (acciones juntas, bonos al revés).
"""

import numpy as np
import pandas as pd

from etf2.config import TICKERS

_INICIAL = dict(zip(TICKERS, [205.0, 103.0, 119.0, 114.0, 126.0]))
_DERIVA = dict(zip(TICKERS, [0.00050, 0.00070, 0.00030, 0.00030, 0.00000]))
_SIGMA = dict(zip(TICKERS, [0.0105, 0.0135, 0.0145, 0.0090, 0.0092]))
_CORR = np.array([
    [1.00, 0.90, 0.85, 0.05, -0.25],
    [0.90, 1.00, 0.75, 0.05, -0.20],
    [0.85, 0.75, 1.00, 0.08, -0.25],
    [0.05, 0.05, 0.08, 1.00, 0.20],
    [-0.25, -0.20, -0.25, 0.20, 1.00],
])


def generar(n: int = 1600, fin: str = "2025-02-18", semilla: int = 42) -> pd.DataFrame:
    """Tabla larga con ticker, fecha, apertura, maximo, minimo, cierre, cierre_ajustado y volumen."""
    rng = np.random.default_rng(semilla)
    fechas = pd.bdate_range(end=fin, periods=n)
    ruido = rng.standard_normal((n, len(TICKERS))) @ np.linalg.cholesky(_CORR).T
    filas = []
    for j, t in enumerate(TICKERS):
        ret = _DERIVA[t] + _SIGMA[t] * ruido[:, j]
        cierre = _INICIAL[t] * np.cumprod(1 + ret)
        previo = np.concatenate([[_INICIAL[t]], cierre[:-1]])
        apertura = previo * (1 + rng.normal(0, _SIGMA[t] * 0.35, n))
        alto = np.maximum(apertura, cierre) * (1 + np.abs(rng.normal(0, _SIGMA[t] * 0.45, n)))
        bajo = np.minimum(apertura, cierre) * (1 - np.abs(rng.normal(0, _SIGMA[t] * 0.45, n)))
        volumen = (4e7 * np.exp(rng.normal(0, 0.35, n)) * (1 + 18 * np.abs(ret))).round()
        filas.append(pd.DataFrame({
            "ticker": t, "fecha": fechas, "apertura": apertura, "maximo": alto, "minimo": bajo,
            "cierre": cierre, "cierre_ajustado": cierre, "volumen": volumen}))
    return pd.concat(filas, ignore_index=True)


def a_csv(df: pd.DataFrame, ruta) -> str:
    """Guarda en el formato de CSV que acepta el pipeline (Date,Open,High,Low,Close,Volume,Ticker)."""
    c = df.rename(columns={"fecha": "Date", "apertura": "Open", "maximo": "High", "minimo": "Low",
                           "cierre": "Close", "volumen": "Volume", "ticker": "Ticker"})
    c[["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]].to_csv(ruta, index=False)
    return str(ruta)
