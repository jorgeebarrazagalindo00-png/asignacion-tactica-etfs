"""Extracción de precios: Yahoo Finance (yfinance) o un CSV de respaldo.

Salida común de ambas fuentes: una tabla "larga" con las columnas
ticker, fecha, apertura, maximo, minimo, cierre, cierre_ajustado, volumen.

Se usa el CIERRE AJUSTADO para todos los cálculos porque incluye dividendos y
splits; sin eso, TLT (bonos) y SPY quedarían subestimados frente a GLD, que no paga.
"""

import pandas as pd

COLUMNAS = ["ticker", "fecha", "apertura", "maximo", "minimo",
            "cierre", "cierre_ajustado", "volumen"]


class ErrorDatos(RuntimeError):
    """Los datos descargados no sirven (vacíos, columnas faltantes, valores raros)."""


def descargar_yfinance(tickers, inicio, fin=None) -> pd.DataFrame:
    """Baja el histórico diario de cada ticker desde Yahoo Finance."""
    import yfinance as yf

    trozos = []
    for t in tickers:
        h = yf.Ticker(t).history(start=inicio, end=fin, auto_adjust=False, actions=False)
        if h is None or h.empty:
            raise ErrorDatos(f"Yahoo Finance no devolvió datos para {t} desde {inicio}.")
        if "Adj Close" not in h.columns:
            raise ErrorDatos(f"La respuesta de Yahoo para {t} no trae 'Adj Close'.")
        idx = pd.to_datetime(h.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        t_df = pd.DataFrame({
            "ticker": t,
            "fecha": idx.normalize(),
            "apertura": h["Open"].to_numpy(),
            "maximo": h["High"].to_numpy(),
            "minimo": h["Low"].to_numpy(),
            "cierre": h["Close"].to_numpy(),
            "cierre_ajustado": h["Adj Close"].to_numpy(),
            "volumen": h["Volume"].to_numpy(),
        })
        trozos.append(t_df)
    return validar_precios(pd.concat(trozos, ignore_index=True))


def cargar_csv(ruta) -> pd.DataFrame:
    """Lee precios desde un CSV (Date,Open,High,Low,Close,Volume,Ticker).

    Sirve para la carga inicial desde los CSV del proyecto anterior o para
    probar el pipeline sin internet. Ese formato no trae cierre ajustado, así
    que se usa el cierre normal en su lugar.
    """
    c = pd.read_csv(ruta)
    faltan = {"Date", "Open", "High", "Low", "Close", "Volume", "Ticker"} - set(c.columns)
    if faltan:
        raise ErrorDatos(f"Al CSV le faltan columnas: {sorted(faltan)}")
    df = pd.DataFrame({
        "ticker": c["Ticker"],
        "fecha": pd.to_datetime(c["Date"]),
        "apertura": c["Open"], "maximo": c["High"], "minimo": c["Low"],
        "cierre": c["Close"],
        "cierre_ajustado": c["Close"],
        "volumen": c["Volume"],
    })
    return validar_precios(df)


def validar_precios(df: pd.DataFrame) -> pd.DataFrame:
    """Chequeos mínimos de calidad antes de guardar nada."""
    faltan = set(COLUMNAS) - set(df.columns)
    if faltan:
        raise ErrorDatos(f"Faltan columnas: {sorted(faltan)}")
    df = df[COLUMNAS].copy()
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
    for col in COLUMNAS[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["cierre", "cierre_ajustado"])
    df = df[(df["cierre"] > 0) & (df["cierre_ajustado"] > 0)]
    df = df.drop_duplicates(subset=["ticker", "fecha"], keep="last")
    df["volumen"] = df["volumen"].round().astype("Int64")     # entero (o vacío): así lo espera la base
    if df.empty:
        raise ErrorDatos("No quedaron filas válidas después de validar los precios.")
    return df.sort_values(["ticker", "fecha"]).reset_index(drop=True)


def matriz_precios(df_largo: pd.DataFrame) -> pd.DataFrame:
    """Pasa a formato ancho (fechas x tickers) con el cierre ajustado.

    Solo se conservan las fechas en que TODOS los ETFs tienen precio, para que
    los cálculos cruzados (z-score, correlación) comparen lo mismo con lo mismo.
    """
    px = df_largo.pivot(index="fecha", columns="ticker", values="cierre_ajustado")
    return px.sort_index().dropna(how="any")
