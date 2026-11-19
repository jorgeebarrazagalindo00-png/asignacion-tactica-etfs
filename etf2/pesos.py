"""Regla de pesos: de los indicadores a un reparto sugerido entre los ETFs.

La regla es deliberadamente simple y explicable (nada de caja negra):

1. Peso base por riesgo ("inverse volatility", una versión sencilla de paridad de
   riesgo): cada ETF recibe un peso proporcional a 1 / volatilidad. Los ETFs más
   tranquilos pesan más, para que ninguno domine el riesgo del portafolio.
2. Inclinación por momentum: se compara el momentum de cada ETF contra los demás
   (z-score entre los 5). Los que van mejor suben su peso hasta x1,5; los que van
   peor lo bajan hasta x0,5.
3. Filtro de momentum absoluto: si el retorno a 6 meses de un ETF es <= 0, su
   peso se reduce a la mitad (no insistir con lo que está cayendo).
4. Límites: ningún ETF por debajo de PESO_MIN ni por encima de PESO_MAX. Los
   pesos se escalan hasta que, ya recortados, suman exactamente 1.
"""

import numpy as np
import pandas as pd

from . import formato as f
from .config import (FACTOR_FILTRO_6M, INCLINACION_ML, INCLINACION_MOMENTUM, PESO_MAX, PESO_MIN)


def aplicar_limites(w, wmin: float = PESO_MIN, wmax: float = PESO_MAX, total: float = 1.0):
    """Lleva los pesos al rango [wmin, wmax] de modo que sigan sumando `total` (1 por defecto).

    Método: se busca un factor de escala s tal que los pesos, ya recortados,
    sumen exactamente el total:   sum( clip(s * w, wmin, wmax) ) = total.
    La suma crece de forma continua con s, así que s se encuentra por bisección.
    Los pesos que no tocan ningún límite conservan sus proporciones relativas.

    (Primero se probó el enfoque "recortar y repartir el sobrante entre los
    libres", pero puede dejar todos los pesos congelados con suma < 1; la
    escala global no tiene ese problema mientras n*wmin <= total <= n*wmax.)

    `total` distinto de 1 sirve para repartir solo una parte de la cartera (p. ej. el 60 %
    que un perfil destina a renta variable) entre un grupo de ETFs.

    Devuelve (pesos, estado): -1 = quedó en el piso, +1 = en el tope, 0 = libre.
    """
    w = np.asarray(w, dtype=float)
    n = len(w)
    if not (n * wmin - 1e-9 <= total <= n * wmax + 1e-9):
        raise ValueError("Límites imposibles de cumplir para ese número de ETFs.")
    w = w / w.sum()

    bajo, alto = 0.0, 2.0 * wmax / max(w.min(), 1e-12)     # con `alto` todos quedan en el tope
    for _ in range(200):
        s = 0.5 * (bajo + alto)
        if np.clip(s * w, wmin, wmax).sum() < total:
            bajo = s
        else:
            alto = s
    final = np.clip(0.5 * (bajo + alto) * w, wmin, wmax)
    estado = np.where(final <= wmin + 1e-12, -1, np.where(final >= wmax - 1e-12, 1, 0))
    return final, estado


def _z_entre_etfs(valores: np.ndarray) -> np.ndarray:
    """Puntaje z de cada ETF frente a los demás, recortado a ±2 (0 si todos son iguales)."""
    sd = valores.std()
    z = np.zeros_like(valores) if sd < 1e-12 else (valores - valores.mean()) / sd
    return np.clip(z, -2.0, 2.0)


def pesos_para_fecha(vol, mom, ret6, prob=None):
    """Calcula los pesos de UN día a partir de vectores (uno por ETF).

    prob (opcional): probabilidad de subida que da el modelo de ML a cada ETF. Si se pasa,
    se aplica la misma idea que con el momentum: quien está por encima de los demás sube
    su peso hasta ×(1 + INCLINACION_ML) y quien está por debajo lo baja hasta ×(1 - INCLINACION_ML).
    """
    vol = np.maximum(np.asarray(vol, dtype=float), 1e-6)
    mom = np.asarray(mom, dtype=float)
    ret6 = np.asarray(ret6, dtype=float)

    base = 1.0 / vol
    base = base / base.sum()

    z = _z_entre_etfs(mom)
    factor = 1.0 + INCLINACION_MOMENTUM * (z / 2.0)
    filtro = ret6 <= 0
    factor = np.where(filtro, factor * FACTOR_FILTRO_6M, factor)

    factor_ml = np.ones_like(base)
    if prob is not None:
        prob = np.asarray(prob, dtype=float)
        factor_ml = 1.0 + INCLINACION_ML * (_z_entre_etfs(prob) / 2.0)

    bruto = base * factor * factor_ml
    bruto = bruto / bruto.sum()
    final, estado = aplicar_limites(bruto)
    return {"peso": final, "base": base, "factor": factor, "z": z, "factor_ml": factor_ml,
            "filtro": filtro, "estado": estado}


def _motivo(vol, base, z, filtro, estado, peso, prob=None) -> str:
    partes = [f"Volatilidad {f.pct(vol)} → peso por riesgo {f.pct(base)}",
              f"momentum {f.num_signo(z, 1)} desviaciones"]
    if prob is not None:
        partes.append(f"el modelo ve {f.pct(prob, 0)} de probabilidad de subida")
    if filtro:
        partes.append("retorno a 6 meses negativo (peso a la mitad)")
    if estado == 1:
        partes.append("limitado al tope")
    elif estado == -1:
        partes.append("sostenido en el mínimo")
    return "; ".join(partes) + f" → peso final {f.pct(peso)}"


def pesos_historicos(ind: dict[str, pd.DataFrame], prob: pd.DataFrame | None = None) -> pd.DataFrame:
    """Calcula los pesos sugeridos para cada fecha en que hay indicadores completos.

    Devuelve tabla larga: fecha, ticker, peso, peso_base, factor_momentum, motivo.
    Con `prob` (fechas x tickers: probabilidad de subida del modelo) se calcula solo para
    las fechas que tienen predicción y se agregan las columnas prob_sube y factor_ml.
    """
    vol, mom, r6 = ind["vol_60d"], ind["momentum"], ind["ret_6m"]
    tickers = list(vol.columns)
    ok = vol.notna().all(axis=1) & mom.notna().all(axis=1) & r6.notna().all(axis=1)
    if prob is not None:
        prob = prob.reindex(columns=tickers)
        ok &= vol.index.isin(prob.dropna(how="any").index)
    fechas = vol.index[ok.to_numpy()]
    V, M, R = (vol.loc[fechas].to_numpy(), mom.loc[fechas].to_numpy(), r6.loc[fechas].to_numpy())
    P = prob.loc[fechas].to_numpy() if prob is not None else None

    filas = []
    for i, fecha in enumerate(fechas):
        r = pesos_para_fecha(V[i], M[i], R[i], None if P is None else P[i])
        for j, t in enumerate(tickers):
            fila = {
                "fecha": fecha, "ticker": t,
                "peso": float(r["peso"][j]),
                "peso_base": float(r["base"][j]),
                "factor_momentum": float(r["factor"][j]),
                "motivo": _motivo(V[i][j], r["base"][j], r["z"][j], bool(r["filtro"][j]),
                                  int(r["estado"][j]), r["peso"][j], None if P is None else P[i][j]),
            }
            if P is not None:
                fila["prob_sube"] = float(P[i][j])
                fila["factor_ml"] = float(r["factor_ml"][j])
            filas.append(fila)
    columnas = ["fecha", "ticker", "peso", "peso_base", "factor_momentum", "motivo"]
    return pd.DataFrame(filas, columns=columnas + (["prob_sube", "factor_ml"] if prob is not None else []))


def pesos_anchos(pesos_largo: pd.DataFrame) -> pd.DataFrame:
    """Tabla larga -> ancha (fechas x tickers), lista para el backtest."""
    return pesos_largo.pivot(index="fecha", columns="ticker", values="peso").sort_index()
