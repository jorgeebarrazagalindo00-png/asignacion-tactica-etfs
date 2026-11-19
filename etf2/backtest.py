"""Backtest honesto de la regla de pesos contra dos referencias simples.

Supuestos (todos conservadores y declarados):

* Se decide al cierre del último día hábil de cada mes con datos hasta ese día.
* La operación se ejecuta al cierre del SIGUIENTE día hábil (un día de rezago:
  no se puede operar al precio de un cierre que recién se conoce).
* Cada unidad de capital negociada paga FRICCION (por defecto 0,1 %), incluida
  la compra inicial.
* Entre rebalanceos los pesos se mueven libremente con los precios.
* Retornos de precios ajustados (incluyen dividendos). Tasa libre de riesgo = 0
  al calcular el Sharpe.
* Los parámetros de la regla NO se ajustaron mirando este backtest.

Referencias: (1) pesos iguales con el mismo calendario y costos, (2) solo SPY.
"""

import numpy as np
import pandas as pd

from .config import (DIAS_ANUALES, FRICCION, NOMBRE_ESTRATEGIA, NOMBRE_IGUALES, NOMBRE_SPY)


def fechas_de_decision(fechas: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Último día con datos de cada mes."""
    s = pd.Series(fechas, index=fechas)
    return list(s.groupby([fechas.year, fechas.month]).last().sort_values())


def simular(ret: pd.DataFrame, decisiones: dict, friccion: float = FRICCION):
    """Simula una cartera.

    ret: retornos diarios (fechas x tickers).
    decisiones: {fecha_de_decisión: vector de pesos objetivo}. Cada decisión se
    ejecuta al cierre del siguiente día hábil.
    Devuelve (curva de capital empezando en 1, turnover acumulado, n rebalanceos).
    """
    fechas = ret.index
    pos = {f: i for i, f in enumerate(fechas)}
    ejecuciones = {}
    for f, w in decisiones.items():
        i = pos[f] + 1
        if i < len(fechas):
            ejecuciones[i] = np.asarray(w, dtype=float)
    if not ejecuciones:
        raise ValueError("No hay ninguna decisión ejecutable en el rango de datos.")

    R = ret.to_numpy()
    i0 = min(ejecuciones)
    w0 = ejecuciones[i0]
    valor = 1.0 - friccion * w0.sum()          # compra inicial
    v = w0 * valor
    curva = {fechas[i0]: valor}
    turnover, n_reb = 0.0, 0

    for i in range(i0 + 1, len(fechas)):
        v = v * (1.0 + R[i])
        valor = v.sum()
        if i in ejecuciones:
            w = ejecuciones[i]
            turn = np.abs(w * valor - v).sum() / valor
            valor -= friccion * turn * valor
            v = w * valor
            turnover += turn
            n_reb += 1
        curva[fechas[i]] = valor
    return pd.Series(curva, name="capital"), turnover, n_reb


def metricas(curva: pd.Series, turnover: float, n_reb: int) -> dict:
    r = curva.pct_change().dropna()
    anios = (curva.index[-1] - curva.index[0]).days / 365.25
    vol = r.std() * np.sqrt(DIAS_ANUALES)
    return {
        "fecha_inicio": curva.index[0],
        "fecha_fin": curva.index[-1],
        "retorno_total": float(curva.iloc[-1] / curva.iloc[0] - 1.0),
        "cagr": float((curva.iloc[-1] / curva.iloc[0]) ** (1.0 / anios) - 1.0) if anios > 0 else np.nan,
        "volatilidad": float(vol),
        "sharpe": float(r.mean() * DIAS_ANUALES / vol) if vol > 0 else np.nan,
        "max_drawdown": float((curva / curva.cummax() - 1.0).min()),
        "turnover_anual": float(turnover / anios) if anios > 0 else np.nan,
        "n_rebalanceos": int(n_reb),
    }


def correr_backtest(px: pd.DataFrame, pesos_ancho: pd.DataFrame, friccion: float = FRICCION):
    """Compara la regla del proyecto con pesos iguales y con solo SPY.

    Devuelve (resumen: DataFrame de métricas por estrategia, curvas: DataFrame
    fechas x estrategias, todas normalizadas a 1 en su primer día común).
    """
    return comparar_estrategias(px, {NOMBRE_ESTRATEGIA: pesos_ancho}, friccion)


def comparar_estrategias(px: pd.DataFrame, estrategias: dict[str, pd.DataFrame],
                         friccion: float = FRICCION):
    """Simula varias estrategias de pesos con el mismo calendario y los mismos costos.

    estrategias: {nombre: pesos (fechas x tickers)}. Todas deben traer las mismas fechas:
    se decide el último día de cada mes de ese índice. Se añaden siempre las dos
    referencias (pesos iguales y solo SPY) para que la comparación sea justa.
    """
    tickers = list(px.columns)
    ret = px.pct_change().iloc[1:]
    pesos = {n: p.reindex(columns=tickers).dropna(how="any") for n, p in estrategias.items()}
    primera = next(iter(pesos.values()))

    decision_dias = [d for d in fechas_de_decision(primera.index) if d in ret.index]
    solo_spy = np.array([1.0 if t == "SPY" else 0.0 for t in tickers])

    corridas = {n: simular(ret, {d: p.loc[d].to_numpy() for d in decision_dias}, friccion)
                for n, p in pesos.items()}
    corridas[NOMBRE_IGUALES] = simular(
        ret, {d: np.full(len(tickers), 1.0 / len(tickers)) for d in decision_dias}, friccion)
    corridas[NOMBRE_SPY] = simular(ret, {decision_dias[0]: solo_spy}, friccion)
    curvas = pd.concat({k: v[0] for k, v in corridas.items()}, axis=1).dropna(how="any")
    curvas = curvas / curvas.iloc[0]

    filas = []
    for nombre, (curva, turnover, n_reb) in corridas.items():
        m = metricas(curva.loc[curvas.index], turnover, n_reb)
        m["estrategia"] = nombre
        m["friccion"] = friccion
        filas.append(m)
    resumen = pd.DataFrame(filas)
    cols = ["estrategia"] + [c for c in resumen.columns if c != "estrategia"]
    return resumen[cols], curvas
