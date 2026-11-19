"""Modelo predictivo: ¿cada ETF terminará el próximo mes en positivo?

Es un problema de clasificación binaria. Para cada ETF y cada día se calculan diez variables
con datos hasta ese día y se intenta anticipar si el precio ajustado, HORIZONTE_ML días hábiles
después, estará por encima del de hoy.

Cómo se evita mirar el futuro (lo que más invalida a un modelo de mercado):

* Validación «walk-forward»: el modelo se vuelve a entrenar cada fin de mes, solo con
  información que ya existía ese día, y predice el mes siguiente. Nunca se evalúa con datos
  que usó para aprender.
* Embargo: el resultado de una fila de entrenamiento (si el ETF subió en los 21 días
  siguientes) solo se conoce 21 días después. Por eso se entrena únicamente con filas cuyo
  resultado ya estaba cerrado el día de la decisión.
* Todo el preprocesamiento (el escalado) se ajusta dentro de cada entrenamiento.

Se comparan dos modelos con dos referencias que no aprenden nada. Si el modelo no le gana
a la referencia más simple, el panel lo dice.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config, tecnico
from .backtest import comparar_estrategias, fechas_de_decision
from .indicadores import calcular_indicadores
from .pesos import pesos_anchos, pesos_historicos

VARIABLES = list(config.ETIQUETAS_VARIABLES)
ETIQUETAS_VARIABLES = config.ETIQUETAS_VARIABLES
NOMBRES_MODELOS = config.NOMBRES_MODELOS
ORDEN_MODELOS = ["logit", "gb", "momentum", "frecuencia"]
SIN_ORDEN = {"frecuencia"}      # da la misma probabilidad a todos: no ordena, no tiene AUC ni IC
COLUMNAS_PRED = ["fecha", "ticker", "modelo", "prob_sube", "subio", "ret_futuro"]


def nuevos_modelos() -> dict:
    """Los dos modelos que se entrenan. Hiperparámetros fijos, definidos antes de ver resultados."""
    return {
        "logit": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
        "gb": GradientBoostingClassifier(n_estimators=config.ARBOLES_GB, max_depth=2, learning_rate=0.05,
                                         subsample=0.8, random_state=config.SEMILLA),
    }


# --- Variables y objetivo --------------------------------------------------------------
def variables_por_dia(px: pd.DataFrame, ind: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    """Una tabla (fechas x ETFs) por variable. Cada valor usa solo precios hasta su fecha."""
    ind = ind or calcular_indicadores(px)
    v = {k: ind[k] for k in ("ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_60d", "drawdown")}
    v["rsi_14"] = px.apply(tecnico.rsi)
    v["macd_rel"] = px.apply(lambda s: tecnico.macd(s)["hist"]) / px
    v["dist_ma50"] = px / px.rolling(50).mean() - 1.0
    log_ret = np.log(px).diff()
    v["vol_corta_larga"] = log_ret.rolling(20).std() / log_ret.rolling(config.VENTANA_VOL).std()
    return v


def armar_tabla(px: pd.DataFrame, horizonte: int = config.HORIZONTE_ML) -> pd.DataFrame:
    """Tabla larga: una fila por ETF y día con las variables, el retorno futuro y el objetivo.

    objetivo = 1 si el retorno de los próximos `horizonte` días es positivo; queda vacío
    en los últimos días, cuyo futuro todavía no ocurre. `pos` es la posición del día en el
    calendario de bolsa y sirve para aplicar el embargo.
    """
    v = variables_por_dia(px)
    futuro = px.shift(-horizonte) / px - 1.0
    partes = []
    for t in px.columns:
        d = pd.DataFrame({k: v[k][t] for k in VARIABLES})
        d["ret_futuro"] = futuro[t]
        d["ticker"] = t
        partes.append(d.rename_axis("fecha").reset_index())
    tabla = pd.concat(partes, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    tabla = tabla.dropna(subset=VARIABLES)
    tabla["objetivo"] = (tabla["ret_futuro"] > 0).astype(float).where(tabla["ret_futuro"].notna())
    tabla["pos"] = tabla["fecha"].map({f: i for i, f in enumerate(px.index)})
    return tabla.sort_values(["fecha", "ticker"]).reset_index(drop=True)


# --- Walk-forward --------------------------------------------------------------------
def _entrenamiento(tabla: pd.DataFrame, pos_decision: int, horizonte: int) -> pd.DataFrame:
    """Filas cuyo resultado ya se conocía el día de la decisión (embargo de `horizonte` días)."""
    return tabla[(tabla["pos"] + horizonte <= pos_decision) & tabla["objetivo"].notna()]


def _predecir_en(tabla: pd.DataFrame, fecha, horizonte: int, minimo: int):
    """Entrena con lo conocido hasta `fecha` y predice ese día. None si falta historia."""
    dia = tabla[tabla["fecha"] == fecha]
    if dia.empty:
        return None
    ent = _entrenamiento(tabla, int(dia["pos"].iloc[0]), horizonte)
    if ent["fecha"].nunique() < minimo or ent["objetivo"].nunique() < 2:
        return None

    prob, ajustados = {}, {}
    for nombre, modelo in nuevos_modelos().items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ajustados[nombre] = modelo.fit(ent[VARIABLES], ent["objetivo"])
        prob[nombre] = ajustados[nombre].predict_proba(dia[VARIABLES])[:, 1]
    # Referencias que no aprenden: «sube si viene subiendo» y «sube tanto como subió siempre».
    prob["momentum"] = (dia["ret_6m"].to_numpy() > 0).astype(float)
    prob["frecuencia"] = np.full(len(dia), float(ent["objetivo"].mean()))

    base = dia[["fecha", "ticker", "ret_futuro"]].assign(subio=dia["objetivo"])
    pred = pd.concat([base.assign(modelo=n, prob_sube=p) for n, p in prob.items()], ignore_index=True)
    return pred, ajustados, ent


def caminar(tabla: pd.DataFrame, fechas: list, horizonte: int = config.HORIZONTE_ML,
            minimo: int = config.MIN_ENTRENAMIENTO):
    """Walk-forward: predice cada fecha con un modelo entrenado solo con su pasado.

    Devuelve (predicciones, modelos del último ajuste, datos con que se entrenaron).
    """
    partes, ajustados, ent = [], None, None
    for fecha in fechas:
        r = _predecir_en(tabla, fecha, horizonte, minimo)
        if r is not None:
            partes.append(r[0])
            ajustados, ent = r[1], r[2]
    if not partes:
        return pd.DataFrame(columns=COLUMNAS_PRED), None, None
    return pd.concat(partes, ignore_index=True)[COLUMNAS_PRED], ajustados, ent


# --- Métricas ------------------------------------------------------------------------
def calcular_metricas(pred: pd.DataFrame) -> pd.DataFrame:
    """Métricas fuera de muestra por modelo, con umbral 0,5. Solo cuentan los casos ya resueltos."""
    filas = []
    resueltas = pred[pred["subio"].notna()]
    for modelo in [m for m in ORDEN_MODELOS if m in set(resueltas["modelo"])]:
        g = resueltas[resueltas["modelo"] == modelo]
        y = g["subio"].astype(int).to_numpy()
        p = g["prob_sube"].to_numpy(dtype=float)
        dice_sube = (p >= 0.5).astype(int)
        ordena = modelo not in SIN_ORDEN and len(set(y)) == 2 and len(set(p)) > 1
        filas.append({
            "modelo": modelo, "n": int(len(g)), "n_fechas": int(g["fecha"].nunique()),
            "fecha_desde": g["fecha"].min(), "fecha_hasta": g["fecha"].max(),
            "exactitud": float(accuracy_score(y, dice_sube)),
            "precision_pos": float(precision_score(y, dice_sube, zero_division=0)),
            "sensibilidad": float(recall_score(y, dice_sube, zero_division=0)),
            "f1": float(f1_score(y, dice_sube, zero_division=0)),
            "auc": float(roc_auc_score(y, p)) if ordena else np.nan,
            "brier": float(brier_score_loss(y, p)),
            "ic_spearman": float(g["prob_sube"].corr(g["ret_futuro"], method="spearman")) if ordena else np.nan,
            "tasa_positivos": float(y.mean()),
        })
    return pd.DataFrame(filas)


def importancias(ajustados: dict, ent: pd.DataFrame) -> pd.DataFrame:
    """Qué variables pesan más en el modelo ajustado con todo el pasado disponible."""
    logit = ajustados["logit"].named_steps["logisticregression"].coef_[0]
    gb = ajustados["gb"].feature_importances_
    filas = [{"modelo": "logit", "variable": v, "importancia": float(c), "signo": int(np.sign(c))}
             for v, c in zip(VARIABLES, logit)]
    filas += [{"modelo": "gb", "variable": v, "importancia": float(c), "signo": 0}
              for v, c in zip(VARIABLES, gb)]
    return pd.DataFrame(filas)


# --- Todo junto ----------------------------------------------------------------------
def entrenar_y_evaluar(px: pd.DataFrame, minimo: int = config.MIN_ENTRENAMIENTO,
                       horizonte: int = config.HORIZONTE_ML) -> dict | None:
    """Walk-forward mensual + predicción de hoy. None si todavía no hay historia suficiente."""
    tabla = armar_tabla(px, horizonte)
    conocidas = set(tabla["fecha"])
    fechas = [d for d in fechas_de_decision(px.index) if d in conocidas]
    ultima = px.index[-1]
    if ultima in conocidas and ultima not in fechas:
        fechas.append(ultima)                  # la predicción de hoy, aún sin resultado
    pred, ajustados, ent = caminar(tabla, fechas, horizonte, minimo)
    if pred.empty:
        return None
    return {
        "predicciones": pred, "metricas": calcular_metricas(pred),
        "importancias": importancias(ajustados, ent), "fecha_prediccion": pred["fecha"].max(),
        "n_entrenamiento": int(len(ent)),
    }


def probabilidades(pred: pd.DataFrame, modelo: str = config.MODELO_PRINCIPAL) -> pd.DataFrame:
    """Probabilidad de subida del modelo, en tabla ancha (fechas x ETFs)."""
    d = pred[pred["modelo"] == modelo]
    return d.pivot(index="fecha", columns="ticker", values="prob_sube").sort_index()


def efecto_del_modelo(px: pd.DataFrame, ind: dict, pesos_largo: pd.DataFrame, pred: pd.DataFrame):
    """Prueba de ablación: la misma regla con y sin la inclinación del modelo, en el mismo periodo.

    Devuelve (pesos con ML en tabla larga, resumen del backtest, curvas). Ambas estrategias
    y las dos referencias usan las mismas fechas de decisión y los mismos costos.
    """
    prob = probabilidades(pred)
    pesos_ml = pesos_historicos(ind, prob=prob)
    fechas = pd.DatetimeIndex(sorted(pesos_ml["fecha"].unique()))
    solo_regla = pesos_largo[pesos_largo["fecha"].isin(fechas)]
    resumen, curvas = comparar_estrategias(px, {
        config.NOMBRE_ESTRATEGIA: pesos_anchos(solo_regla),
        config.NOMBRE_ESTRATEGIA_ML: pesos_anchos(pesos_ml),
    })
    return pesos_ml, resumen, curvas
