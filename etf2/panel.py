"""Preparación y análisis de datos para el panel (pandas puro, sin Streamlit)."""

import json

import numpy as np
import pandas as pd

from . import config, formato as f, tecnico
from .prescriptivo import ACCIONES_ANTERIORES, nivel_de_alerta
from .config import DIAS_ANUALES, GRUPO, PESO_MAX, PESO_MIN, TICKERS

PERIODOS = {"1 A": 365, "3 A": 3 * 365, "5 A": 5 * 365, "Máx.": None}
RANGOS_VELAS = {"3 M": 92, "6 M": 183, "1 A": 365, "3 A": 3 * 365, "Máx.": None}
COLS_IND = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_60d", "momentum", "drawdown"]


# --- De la API a tablas -----------------------------------------------------------
def a_frame(filas: list[dict], fechas=(), numericas=()) -> pd.DataFrame:
    df = pd.DataFrame(filas)
    for c in fechas:
        if c in df:
            df[c] = pd.to_datetime(df[c])
    for c in numericas:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _en_orden(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas en el orden de config.TICKERS: colores y leyendas siempre iguales."""
    return df[[t for t in TICKERS if t in df.columns] + [t for t in df.columns if t not in TICKERS]]


def matriz_precios(precios: pd.DataFrame) -> pd.DataFrame:
    px = precios.pivot(index="fecha", columns="ticker", values="cierre_ajustado").sort_index()
    return _en_orden(px.dropna(how="any"))


def pesos_ancho(pesos: pd.DataFrame) -> pd.DataFrame:
    return _en_orden(pesos.pivot(index="fecha", columns="ticker", values="peso").sort_index())


def fin_de_mes(ancho: pd.DataFrame) -> pd.DataFrame:
    """Último dato de cada mes (cuando la regla decide rebalancear)."""
    return ancho.groupby([ancho.index.year, ancho.index.month]).tail(1)


def recortar(df, dias):
    """Últimos `dias` días de calendario (None = todo)."""
    if dias is None or len(df) == 0:
        return df
    return df[df.index >= df.index[-1] - pd.Timedelta(days=dias)]


def base100(px: pd.DataFrame) -> pd.DataFrame:
    return px / px.iloc[0] * 100.0


def correlaciones(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change().dropna().corr()


def cambio_de_peso(ancho: pd.DataFrame, dias: int = 30) -> pd.Series | None:
    """Peso de hoy menos el de hace ~`dias` días (None si falta historia)."""
    ultima = ancho.index[-1]
    previo = ancho.loc[: ultima - pd.Timedelta(days=dias)]
    if previo.empty:
        return None
    return ancho.iloc[-1] - previo.iloc[-1]


def curva_a_base100(curva: pd.DataFrame) -> pd.DataFrame:
    return curva.set_index("fecha").sort_index() * 100.0


# --- Velas -------------------------------------------------------------------------
def ohlc(precios: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Apertura, máximo, mínimo, cierre y volumen de un ETF, por fecha."""
    d = precios[precios["ticker"] == ticker].set_index("fecha").sort_index()
    d = d[["apertura", "maximo", "minimo", "cierre", "volumen"]].dropna(subset=["cierre"]).copy()
    for c in ("apertura", "maximo", "minimo"):
        d[c] = d[c].fillna(d["cierre"])
    d["volumen"] = d["volumen"].fillna(0)
    return d


def intervalo_velas(dias) -> tuple[str, str | None]:
    """Elige el tamaño de vela según el rango pedido, para que nunca haya demasiadas."""
    if dias is not None and dias <= 190:
        return "diarias", None
    if dias is not None and dias <= 3 * 366:
        return "semanales", "W-FRI"
    return "mensuales", "M"


def velas(o: pd.DataFrame, regla: str | None = None, medias=(20, 50)) -> pd.DataFrame:
    """Agrupa el OHLC diario en velas semanales o mensuales y añade medias móviles.

    Cada vela queda fechada en su último día con datos. Las medias se calculan sobre
    las velas resultantes (20 y 50 velas), como en cualquier graficador.
    """
    if regla:
        per = o.index.to_period(regla)
        v = o.groupby(per).agg(apertura=("apertura", "first"), maximo=("maximo", "max"),
                               minimo=("minimo", "min"), cierre=("cierre", "last"),
                               volumen=("volumen", "sum"))
        v.index = pd.DatetimeIndex(o.index.to_series().groupby(per).last().to_numpy())
    else:
        v = o.copy()
    for n in medias:
        v[f"mm{n}"] = v["cierre"].rolling(n).mean()
    v["variacion"] = v["cierre"].pct_change()
    return v


def resumen_universo(precios: pd.DataFrame, ind_hoy: pd.DataFrame) -> pd.DataFrame:
    """Una fila por ETF: último precio, variación del día, rango de 52 semanas e indicadores."""
    filas = []
    for tk in [t for t in TICKERS if t in set(precios["ticker"])]:
        d = precios[precios["ticker"] == tk].sort_values("fecha")
        aj, anio = d["cierre_ajustado"], d.tail(DIAS_ANUALES)
        fila = {
            "ticker": tk,
            "cierre": float(d["cierre"].iloc[-1]),
            "var_dia": float(aj.iloc[-1] / aj.iloc[-2] - 1) if len(aj) > 1 else np.nan,
            "min_52": float(anio["minimo"].fillna(anio["cierre"]).min()),
            "max_52": float(anio["maximo"].fillna(anio["cierre"]).max()),
            "volumen": float(d["volumen"].iloc[-1]) if pd.notna(d["volumen"].iloc[-1]) else np.nan,
            "tendencia": aj.tail(90).tolist(),
        }
        for c in COLS_IND:
            fila[c] = float(ind_hoy.at[tk, c]) if (tk in ind_hoy.index and c in ind_hoy
                                                    and pd.notna(ind_hoy.at[tk, c])) else np.nan
        filas.append(fila)
    return pd.DataFrame(filas).set_index("ticker")


# --- Indicadores técnicos y distribución de retornos -------------------------------------
def tecnicos(o: pd.DataFrame) -> pd.DataFrame:
    """RSI, MACD y ATR diarios de un ETF. o: salida de ohlc()."""
    t = tecnico.macd(o["cierre"])
    t["rsi"] = tecnico.rsi(o["cierre"])
    t["atr"] = tecnico.atr(o["maximo"], o["minimo"], o["cierre"])
    t["atr_pct"] = t["atr"] / o["cierre"]
    return t


def lectura_tecnica(ultimo: pd.Series) -> dict:
    """Cómo leer los tres indicadores de un día, en pocas palabras."""
    rsi, hist = ultimo["rsi"], ultimo["hist"]
    if rsi != rsi:
        zona = f.VACIO
    else:
        zona = "Sobrecompra (más de 70)" if rsi >= 70 else "Sobreventa (menos de 30)" if rsi <= 30 else "Zona neutral (30 a 70)"
    impulso = f.VACIO if hist != hist else "MACD sobre su señal: impulso al alza" if hist > 0 \
        else "MACD bajo su señal: impulso a la baja"
    return {"rsi": zona, "macd": impulso}


def estadisticas_retornos(px: pd.DataFrame) -> pd.DataFrame:
    """Una fila por ETF con la forma de sus retornos diarios en el rango."""
    r = px.pct_change().dropna(how="all")
    return pd.DataFrame({"media": r.mean(), "desviacion": r.std(), "asimetria": r.skew(), "curtosis": r.kurt(),
                         "mejor": r.max(), "peor": r.min(), "positivos": (r > 0).mean(), "n": r.count()})


def histograma(r: pd.Series, tramos: int = 31) -> pd.DataFrame:
    """Cuántos días cayó el retorno en cada tramo. Tramos iguales y simétricos alrededor de cero;
    los días extremos (el 0,5 % de cada cola) se cuentan en el tramo del borde."""
    r = r.dropna()
    limite = float(max(abs(r.quantile(0.005)), abs(r.quantile(0.995)), 1e-6))
    bordes = np.linspace(-limite, limite, tramos + 1)
    dias, _ = np.histogram(r.clip(-limite, limite), bins=bordes)
    return pd.DataFrame({"desde": bordes[:-1], "hasta": bordes[1:], "centro": (bordes[:-1] + bordes[1:]) / 2,
                         "dias": dias})


# --- Modelo predictivo ---------------------------------------------------------------------
def resueltas(pred: pd.DataFrame, modelo: str) -> pd.DataFrame:
    """Predicciones de un modelo cuyo resultado ya se conoce."""
    return pred[(pred["modelo"] == modelo) & pred["subio"].notna()]


def prob_de_hoy(pred: pd.DataFrame) -> tuple[pd.Timestamp, pd.DataFrame]:
    """La predicción más reciente: fecha y probabilidad por ETF (filas) y modelo (columnas)."""
    fecha = pred["fecha"].max()
    hoy = pred[pred["fecha"] == fecha].pivot(index="ticker", columns="modelo", values="prob_sube")
    return fecha, hoy.reindex([t for t in TICKERS if t in hoy.index])


def calibracion(pred: pd.DataFrame, modelo: str, tramos: int = 5) -> pd.DataFrame:
    """¿Cuando el modelo dice 60 %, sube el 60 % de las veces?

    Agrupa las predicciones resueltas en tramos con igual número de casos y compara la
    probabilidad media anunciada con la frecuencia con que el ETF realmente subió.
    """
    d = resueltas(pred, modelo)
    if d["prob_sube"].nunique() < 2:
        return pd.DataFrame(columns=["prob_media", "frecuencia", "n"])
    grupo = pd.qcut(d["prob_sube"], min(tramos, d["prob_sube"].nunique()), duplicates="drop")
    t = d.groupby(grupo, observed=True).agg(prob_media=("prob_sube", "mean"), frecuencia=("subio", "mean"),
                                           n=("subio", "size"))
    return t.reset_index(drop=True)


def matriz_confusion(pred: pd.DataFrame, modelo: str) -> dict:
    """Aciertos y errores con umbral 0,5: vp y vn son aciertos; fp y fn, errores."""
    d = resueltas(pred, modelo)
    dice, paso = d["prob_sube"] >= 0.5, d["subio"] == 1
    return {"vp": int((dice & paso).sum()), "fp": int((dice & ~paso).sum()),
            "fn": int((~dice & paso).sum()), "vn": int((~dice & ~paso).sum())}


def veredicto_del_modelo(metricas: pd.DataFrame, ml_resumen: pd.DataFrame) -> list[str]:
    """Frases que dicen sin adornos si el modelo aporta o no. Usa ** ** para resaltar."""
    frases = []
    m = metricas.set_index("modelo")
    principal = config.MODELO_PRINCIPAL
    if principal in m.index and "frecuencia" in m.index:
        mio, ref = m.at[principal, "exactitud"], m.at["frecuencia", "exactitud"]
        if mio > ref + 0.005:
            frases.append(f"El modelo acierta el **{f.pct(mio)}** de las veces; decir siempre lo más frecuente "
                          f"acierta el {f.pct(ref)}. La ventaja es de {f.pp(mio - ref)}.")
        else:
            frases.append(f"El modelo acierta el **{f.pct(mio)}** de las veces y la referencia más simple, el "
                          f"{f.pct(ref)}: **hoy no la supera en exactitud**.")
        auc = m.at[principal, "auc"]
        if auc == auc:
            lectura = "ordena mejor que el azar" if auc > 0.52 else "ordena casi como el azar (0,50)"
            frases.append(f"Su AUC es **{f.num(auc, 2)}**: {lectura}.")
    if not ml_resumen.empty:
        r = ml_resumen.set_index("estrategia")
        if config.NOMBRE_ESTRATEGIA in r.index and config.NOMBRE_ESTRATEGIA_ML in r.index:
            sin, con = r.at[config.NOMBRE_ESTRATEGIA, "sharpe"], r.at[config.NOMBRE_ESTRATEGIA_ML, "sharpe"]
            if con > sin:
                frases.append(f"Al inclinar los pesos con el modelo, el Sharpe de la regla pasa de {f.num(sin, 2)} a "
                              f"**{f.num(con, 2)}** en el mismo periodo: por eso la recomendación sí lo usa.")
            else:
                frases.append(f"Al inclinar los pesos con el modelo, el Sharpe de la regla pasa de {f.num(sin, 2)} a "
                              f"**{f.num(con, 2)}**: no mejora, así que **la recomendación no lo usa**.")
    return frases


# --- Recomendación y uso del LLM -------------------------------------------------------------
# Cómo arrancó cada corrida, en palabras. «github_actions» y «manual» son de las corridas anteriores al 18-nov.
ORIGENES = {"programada": "automático", "panel": "botón del panel", "github_manual": "manual en GitHub",
            "pc": "desde una PC", "github_actions": "automático", "manual": "desde una PC"}


def origen_legible(origen) -> str:
    return ORIGENES.get(origen, str(origen) if origen else "—")


def recomendacion_de(recs: pd.DataFrame, perfil: str) -> dict | None:
    """La recomendación más reciente de un perfil, o None si no hay."""
    if recs.empty:
        return None
    d = recs[(recs["perfil"] == perfil) & (recs["fecha"] == recs["fecha"].max())]
    if d.empty:
        return None
    rec = d.iloc[0].to_dict()
    if isinstance(rec["detalle"], str):          # según el cliente, el JSON puede llegar como texto
        rec["detalle"] = json.loads(rec["detalle"])
    for e in rec["detalle"].get("etfs", []):     # recomendaciones guardadas con los nombres anteriores
        if e.get("accion") in ACCIONES_ANTERIORES:
            e["accion"] = ACCIONES_ANTERIORES[e["accion"]]
    return rec


def decision_del_mes(rec: dict, universo: pd.DataFrame) -> pd.DataFrame:
    """Una fila por ETF: acción del perfil, peso, cambio frente al mes anterior y nivel de alerta.

    universo: salida de resumen_universo (último cierre y volatilidad anual de 60 días).
    """
    d = pd.DataFrame(rec["detalle"]["etfs"]).set_index("ticker")
    d = d.reindex([t for t in TICKERS if t in d.index])
    d["cambio"] = d["peso"] - d["peso_previo"].astype(float)
    d["cierre"] = universo["cierre"].reindex(d.index)
    d["alerta"] = nivel_de_alerta(d["cierre"], universo["vol_60d"].reindex(d.index))
    d["margen"] = d["alerta"] / d["cierre"] - 1
    return d


def uso_del_llm(log: pd.DataFrame, hoy=None) -> dict:
    """Resumen de la bitácora de llamadas al modelo de lenguaje."""
    hoy = pd.Timestamp(hoy or pd.Timestamp.now(tz="UTC").date())
    llamadas = log[log["estado"].isin(["ok", "rechazada", "error"])] if not log.empty else log
    ok = llamadas[llamadas["estado"] == "ok"] if not llamadas.empty else llamadas
    modelos = ok["modelo"].dropna() if not ok.empty else pd.Series(dtype=object)
    return {
        "llamadas": int(len(llamadas)), "aceptadas": int(len(ok)),
        "hoy": int((llamadas["dia"] == hoy).sum()) if not llamadas.empty else 0,
        "tokens_entrada": float(ok["tokens_entrada"].mean()) if len(ok) else np.nan,
        "tokens_salida": float(ok["tokens_salida"].mean()) if len(ok) else np.nan,
        "modelo": modelos.iloc[0] if len(modelos) else None,
    }


def costo_por_cien(uso: dict) -> float | None:
    """Costo estimado en USD de 100 recomendaciones con la tarifa de pago del modelo (None si no se conoce)."""
    precio = config.LLM_PRECIOS_USD_M.get(uso.get("modelo"))
    if not precio or uso["tokens_entrada"] != uso["tokens_entrada"]:
        return None
    return 100 * (uso["tokens_entrada"] * precio[0] + uso["tokens_salida"] * precio[1]) / 1e6


# --- Backtest ------------------------------------------------------------------------
def drawdowns(curvas: pd.DataFrame) -> pd.DataFrame:
    """Caída desde el máximo anterior (0 = en máximos)."""
    return curvas / curvas.cummax() - 1.0


def retornos_anuales(curvas: pd.DataFrame) -> tuple[pd.DataFrame, set[int]]:
    """Retorno por año calendario y qué años están incompletos (el primero y/o el último)."""
    fin = curvas.groupby(curvas.index.year).last()
    previo = fin.shift(1)
    previo.iloc[0] = curvas.iloc[0]
    parciales = set()
    if curvas.index[0].month > 1 or curvas.index[0].day > 7:
        parciales.add(int(curvas.index[0].year))
    if curvas.index[-1].month < 12 or curvas.index[-1].day < 24:
        parciales.add(int(curvas.index[-1].year))
    return fin / previo - 1.0, parciales


def retornos_mensuales(serie: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Retorno de cada mes (serie) y la misma información como tabla año x mes."""
    per = serie.index.to_period("M")
    fin = serie.groupby(per).last()
    previo = fin.shift(1)
    previo.iloc[0] = serie.iloc[0]
    r = fin / previo - 1.0
    tabla = (pd.DataFrame({"anio": r.index.year, "mes": r.index.month, "r": r.to_numpy()})
             .pivot(index="anio", columns="mes", values="r").reindex(columns=range(1, 13)))
    return r, tabla


def resumen_mensual(r: pd.Series) -> dict:
    r = r.dropna()
    return {"mejor": float(r.max()), "mejor_mes": r.idxmax(), "peor": float(r.min()),
            "peor_mes": r.idxmin(), "positivos": float((r > 0).mean()), "n": int(len(r))}


# --- Riesgo --------------------------------------------------------------------------
def contribucion_riesgo(pesos: pd.Series, retornos: pd.DataFrame) -> pd.Series:
    """Parte de la volatilidad de la cartera que aporta cada ETF (suma 1).

    Con w = pesos y C = covarianza de los retornos: aporte_i = w_i (C w)_i / (w' C w).
    """
    cols = [t for t in pesos.index if t in retornos.columns]
    w = pesos.reindex(cols).to_numpy(dtype=float)
    marginal = retornos[cols].cov().to_numpy() @ w
    total = float(w @ marginal)
    if not np.isfinite(total) or total <= 0:
        return pd.Series(np.nan, index=cols)
    return pd.Series(w * marginal / total, index=cols)


# --- Órdenes de rebalanceo ---------------------------------------------------------------
def ordenes(pesos: pd.Series, precios: pd.Series, actuales: pd.Series,
            tolerancia: float = 0.01) -> tuple[pd.DataFrame, dict]:
    """Qué comprar o vender para llevar una cartera a los pesos sugeridos.

    actuales: monto por ticker; puede traer además una fila 'Efectivo'. No se opera
    cuando la diferencia es menor que `tolerancia` x capital (banda de no operación).
    """
    capital = float(actuales.sum())
    tk = list(pesos.index)
    actual = actuales.reindex(tk).fillna(0.0).astype(float)
    objetivo = pesos.astype(float) * capital
    dif = objetivo - actual
    opera = dif.abs() >= max(tolerancia * capital, 0.005)
    monto = dif.where(opera, 0.0)
    tabla = pd.DataFrame({
        "peso": pesos.astype(float), "actual": actual, "objetivo": objetivo,
        "operacion": np.where(~opera, "Mantener", np.where(dif > 0, "Comprar", "Vender")),
        "monto": monto, "precio": precios.reindex(tk).astype(float),
    })
    tabla["acciones"] = tabla["monto"] / tabla["precio"]
    tabla["final"] = tabla["actual"] + tabla["monto"]
    negociado = float(monto.abs().sum())
    return tabla, {"capital": capital, "negociado": negociado,
                   "efectivo_final": capital - float(tabla["final"].sum()),
                   "compras": int((tabla["operacion"] == "Comprar").sum()),
                   "ventas": int((tabla["operacion"] == "Vender").sum())}


# --- Calendario y estado de los datos -------------------------------------------------
def proximo_rebalanceo(fecha) -> pd.Timestamp:
    """Último día hábil del mes de `fecha` (si ya es ese día, el mismo)."""
    return (pd.Timestamp(fecha).normalize() + pd.offsets.BMonthEnd(0)).normalize()


def dias_de_retraso(fecha_datos, hoy=None) -> int:
    """Días hábiles entre el último dato y hoy (1 es normal: el cierre de ayer)."""
    if hoy is None:
        hoy = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    return int(np.busday_count(pd.Timestamp(fecha_datos).date(), pd.Timestamp(hoy).date()))


# --- Lectura del reparto en palabras ---------------------------------------------------
def _lista(nombres: list[str]) -> str:
    return nombres[0] if len(nombres) == 1 else ", ".join(nombres[:-1]) + " y " + nombres[-1]


def lectura(hoy: pd.DataFrame, ind_hoy: pd.DataFrame, cambio: pd.Series | None) -> list[str]:
    """Frases que explican el reparto del día, armadas solo con los datos (sin IA).

    Usa ** ** para marcar lo que debe ir resaltado.
    """
    frases = []
    pesos = hoy["peso"]

    por_grupo = pesos.groupby(pd.Series(GRUPO)).sum().sort_values(ascending=False)
    partes = [f"**{f.pct(v)}** en {g.lower()}" for g, v in por_grupo.items()]
    frases.append("La cartera queda con " + _lista(partes) + ".")

    mayor = pesos.idxmax()
    motivo = ""
    if "vol_60d" in ind_hoy and ind_hoy["vol_60d"].notna().any() and ind_hoy["vol_60d"].idxmin() == mayor:
        motivo = f": es el activo menos volátil del grupo ({f.pct(ind_hoy.at[mayor, 'vol_60d'])} anual)"
    elif hoy.at[mayor, "factor_momentum"] >= 1.1:
        motivo = f", impulsado por su momentum (×{f.num(hoy.at[mayor, 'factor_momentum'], 2)})"
    tope = " y toca el tope permitido" if pesos[mayor] >= PESO_MAX - 1e-4 else ""
    frases.append(f"**{mayor}** es la mayor posición con **{f.pct(pesos[mayor])}**{motivo}{tope}.")

    lider = hoy["factor_momentum"].idxmax()
    if lider != mayor and hoy.at[lider, "factor_momentum"] >= 1.1:
        frases.append(f"**{lider}** recibe el mayor impulso por momentum: su peso se multiplica "
                      f"por **{f.num(hoy.at[lider, 'factor_momentum'], 2)}** frente al que tendría solo por riesgo.")

    if "ret_6m" in ind_hoy:
        negativos = [t for t in hoy.index if t in ind_hoy.index and ind_hoy.at[t, "ret_6m"] <= 0]
        if negativos:
            detalle = _lista([f"**{t}** ({f.pct_signo(ind_hoy.at[t, 'ret_6m'])})" for t in negativos])
            verbo = "tiene" if len(negativos) == 1 else "tienen"
            frases.append(f"{detalle} {verbo} retorno negativo a 6 meses: el filtro reduce su peso a la mitad.")
        else:
            frases.append("Ningún ETF tiene retorno negativo a 6 meses: el filtro no recorta a nadie.")

    en_piso = [t for t in hoy.index if pesos[t] <= PESO_MIN + 1e-4]
    if en_piso:
        frases.append(f"{_lista([f'**{t}**' for t in en_piso])} queda en el mínimo de {f.pct(PESO_MIN, 0)}: "
                      "la regla nunca saca del todo a un activo.")

    if cambio is not None:
        sube, baja = cambio.idxmax(), cambio.idxmin()
        if max(abs(cambio[sube]), abs(cambio[baja])) < 0.005:
            frases.append("El reparto casi no cambió frente a hace un mes.")
        else:
            frases.append(f"Frente a hace un mes, lo que más sube es **{sube}** ({f.pp(cambio[sube])}) "
                          f"y lo que más baja es **{baja}** ({f.pp(cambio[baja])}).")
    return frases
