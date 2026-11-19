"""Indicadores técnicos clásicos sobre series de precios: RSI, MACD y ATR.

Igual que en indicadores.py, el valor de la fecha t usa solo datos hasta t.
"""

import numpy as np
import pandas as pd


def rsi(cierre: pd.Series, n: int = 14) -> pd.Series:
    """Índice de fuerza relativa (Wilder), de 0 a 100. Más de 70: sobrecompra; menos de 30: sobreventa."""
    d = cierre.diff()
    sube = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    baja = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    fuerza = sube / baja.replace(0.0, np.nan)
    valor = 100.0 - 100.0 / (1.0 + fuerza)
    valor = valor.where(~((baja == 0) & (sube > 0)), 100.0)     # solo subidas en la ventana
    return valor.where(~((baja == 0) & (sube == 0)), 50.0)      # precio sin cambios


def macd(cierre: pd.Series, rapida: int = 12, lenta: int = 26, senal: int = 9) -> pd.DataFrame:
    """MACD: diferencia entre dos medias exponenciales, su señal y el histograma (MACD - señal)."""
    linea = cierre.ewm(span=rapida, adjust=False).mean() - cierre.ewm(span=lenta, adjust=False).mean()
    sen = linea.ewm(span=senal, adjust=False).mean()
    return pd.DataFrame({"macd": linea, "senal": sen, "hist": linea - sen})


def atr(maximo: pd.Series, minimo: pd.Series, cierre: pd.Series, n: int = 14) -> pd.Series:
    """Rango verdadero promedio (Wilder): cuánto se mueve el precio en un día, en dólares."""
    previo = cierre.shift(1)
    rango = pd.concat([maximo - minimo, (maximo - previo).abs(), (minimo - previo).abs()], axis=1).max(axis=1)
    return rango.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
