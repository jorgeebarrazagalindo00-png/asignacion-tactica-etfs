"""Pruebas del modelo predictivo: sobre todo, que nunca use información del futuro."""

import numpy as np
import pandas as pd
import pytest

from etf2 import config, pesos, predictivo as pr, tecnico
from etf2.backtest import fechas_de_decision
from etf2.datos import matriz_precios
from etf2.indicadores import calcular_indicadores
from etf2.pesos import pesos_historicos

from sinteticos import generar

H = config.HORIZONTE_ML
MINIMO = 300


@pytest.fixture(scope="module")
def px():
    return matriz_precios(generar(1000))


@pytest.fixture(scope="module")
def resultado(px):
    return pr.entrenar_y_evaluar(px, minimo=MINIMO)


# --- Indicadores técnicos ---------------------------------------------------------------
def test_rsi_en_los_casos_extremos():
    sube = pd.Series(np.arange(1.0, 60.0))
    assert tecnico.rsi(sube).iloc[-1] == 100.0
    assert tecnico.rsi(sube[::-1].reset_index(drop=True)).iloc[-1] == pytest.approx(0.0)
    assert tecnico.rsi(pd.Series(np.full(60, 10.0))).iloc[-1] == 50.0
    azar = pd.Series(100 + np.random.default_rng(1).normal(0, 1, 400).cumsum())
    r = tecnico.rsi(azar).dropna()
    assert r.between(0, 100).all() and len(r) == 400 - 14


def test_macd_y_atr(px):
    m = tecnico.macd(px["SPY"])
    assert np.allclose(m["hist"], m["macd"] - m["senal"])
    plano = tecnico.macd(pd.Series(np.full(80, 50.0)))
    assert np.allclose(plano, 0.0)
    fijo = pd.Series(np.full(60, 100.0))
    assert np.allclose(tecnico.atr(fijo + 1, fijo - 1, fijo).dropna(), 2.0)
    salto = fijo.copy()
    salto.iloc[30:] = 110.0                                  # un hueco de 10 entre dos cierres
    a = tecnico.atr(salto + 1, salto - 1, salto)
    assert a.iloc[30] > a.iloc[29] and a.iloc[13:].notna().all() and a.iloc[:13].isna().all()


def test_los_indicadores_tecnicos_no_miran_adelante(px):
    corte = 600
    recortado = px["QQQ"].iloc[:corte]
    assert tecnico.rsi(px["QQQ"]).iloc[corte - 1] == pytest.approx(tecnico.rsi(recortado).iloc[-1])
    assert tecnico.macd(px["QQQ"])["hist"].iloc[corte - 1] == pytest.approx(tecnico.macd(recortado)["hist"].iloc[-1])


# --- Tabla de entrenamiento ---------------------------------------------------------------
def test_la_tabla_deja_sin_objetivo_los_ultimos_dias(px):
    tabla = pr.armar_tabla(px)
    assert not tabla[pr.VARIABLES].isna().any().any()
    ultimos = tabla[tabla["pos"] > len(px) - 1 - H]
    assert ultimos["objetivo"].isna().all() and len(ultimos) == H * 5
    resto = tabla[tabla["pos"] <= len(px) - 1 - H]
    assert resto["objetivo"].isin([0.0, 1.0]).all()
    fila = resto[resto["ticker"] == "GLD"].iloc[100]
    i = int(fila["pos"])
    assert fila["ret_futuro"] == pytest.approx(px["GLD"].iloc[i + H] / px["GLD"].iloc[i] - 1)
    assert fila["objetivo"] == float(fila["ret_futuro"] > 0)


def test_las_variables_de_un_dia_no_cambian_si_cambia_el_futuro(px):
    corte = 700
    otro = px.copy()
    otro.iloc[corte + 1:] *= 1.5
    a, b = pr.armar_tabla(px), pr.armar_tabla(otro)
    antes_a, antes_b = a[a["pos"] <= corte], b[b["pos"] <= corte]
    assert np.allclose(antes_a[pr.VARIABLES], antes_b[pr.VARIABLES])


def test_el_entrenamiento_respeta_el_embargo(px):
    tabla = pr.armar_tabla(px)
    pos = 800
    ent = pr._entrenamiento(tabla, pos, H)
    assert (ent["pos"] + H <= pos).all()
    assert ent["pos"].max() == pos - H                       # usa todo lo permitido, ni un día más
    assert ent["objetivo"].notna().all()


def test_la_prediccion_de_un_dia_no_depende_de_lo_que_pasa_despues(px):
    d = fechas_de_decision(px.index)[-8]
    corte = px.index.get_loc(d)
    rng = np.random.default_rng(3)
    otro = px.copy()
    otro.iloc[corte + 1:] = otro.iloc[corte + 1:] * rng.uniform(0.5, 1.5, (len(px) - corte - 1, 5))

    a, *_ = pr.caminar(pr.armar_tabla(px), [d], minimo=MINIMO)
    b, *_ = pr.caminar(pr.armar_tabla(otro), [d], minimo=MINIMO)
    assert len(a) == 5 * 4
    assert np.allclose(a["prob_sube"], b["prob_sube"])
    assert not np.allclose(a["ret_futuro"], b["ret_futuro"])   # el futuro sí era distinto


# --- Walk-forward ---------------------------------------------------------------------------
def test_sin_historia_suficiente_no_hay_modelo():
    assert pr.entrenar_y_evaluar(matriz_precios(generar(500))) is None


def test_predice_cada_fin_de_mes_y_hoy(px, resultado):
    pred = resultado["predicciones"]
    assert set(pred["modelo"]) == set(pr.ORDEN_MODELOS)
    assert pred["prob_sube"].between(0, 1).all()
    assert not pred.duplicated(["fecha", "ticker", "modelo"]).any()

    fines = set(fechas_de_decision(px.index))
    fechas = set(pred["fecha"])
    assert fechas - fines <= {px.index[-1]}
    assert resultado["fecha_prediccion"] == px.index[-1]
    hoy = pred[pred["fecha"] == px.index[-1]]
    assert len(hoy) == 5 * 4 and hoy["subio"].isna().all()    # lo de hoy todavía no se puede evaluar

    primera = pred["fecha"].min()
    assert px.index.get_loc(primera) >= 252 + MINIMO + H - 1  # historia mínima + embargo antes de predecir


def test_es_reproducible(px, resultado):
    otra = pr.entrenar_y_evaluar(px, minimo=MINIMO)
    assert np.allclose(resultado["predicciones"]["prob_sube"], otra["predicciones"]["prob_sube"])


def test_metricas_por_modelo(resultado):
    m = resultado["metricas"].set_index("modelo")
    assert list(m.index) == pr.ORDEN_MODELOS
    pred = resultado["predicciones"]
    resueltas = pred[pred["subio"].notna() & (pred["modelo"] == "logit")]
    assert m.at["logit", "n"] == len(resueltas) and m["n"].nunique() == 1
    for col in ("exactitud", "precision_pos", "sensibilidad", "f1", "brier", "tasa_positivos"):
        assert m[col].between(0, 1).all(), col
    assert m.loc[["logit", "gb", "momentum"], "auc"].between(0, 1).all()
    assert np.isnan(m.at["frecuencia", "auc"]) and np.isnan(m.at["frecuencia", "ic_spearman"])
    acierto = ((resueltas["prob_sube"] >= 0.5) == (resueltas["subio"] == 1)).mean()
    assert m.at["logit", "exactitud"] == pytest.approx(acierto)


def test_metricas_de_un_modelo_perfecto_y_de_uno_invertido():
    fechas = pd.bdate_range("2024-01-01", periods=40)
    subio = np.tile([1.0, 0.0], 20)
    base = pd.DataFrame({"fecha": fechas, "ticker": "SPY", "subio": subio, "ret_futuro": np.where(subio == 1, 0.02, -0.02)})
    pred = pd.concat([base.assign(modelo="logit", prob_sube=np.where(subio == 1, 0.9, 0.1)),
                      base.assign(modelo="gb", prob_sube=np.where(subio == 1, 0.1, 0.9))])
    m = pr.calcular_metricas(pred).set_index("modelo")
    assert m.at["logit", "exactitud"] == 1.0 and m.at["logit", "auc"] == 1.0 and m.at["logit", "f1"] == 1.0
    assert m.at["gb", "exactitud"] == 0.0 and m.at["gb", "auc"] == 0.0
    assert m.at["logit", "ic_spearman"] > 0.99 and m.at["gb", "ic_spearman"] < -0.99


def test_importancias(resultado):
    imp = resultado["importancias"]
    assert len(imp) == 2 * len(pr.VARIABLES)
    gb = imp[imp["modelo"] == "gb"]
    assert gb["importancia"].sum() == pytest.approx(1.0) and (gb["importancia"] >= 0).all()
    logit = imp[imp["modelo"] == "logit"]
    assert set(logit["signo"]) <= {-1, 0, 1} and set(logit["variable"]) == set(pr.VARIABLES)
    assert set(pr.VARIABLES) == set(pr.ETIQUETAS_VARIABLES)


# --- Efecto del modelo sobre los pesos -------------------------------------------------------
def test_pesos_con_modelo_cumplen_los_limites(px, resultado):
    ind = calcular_indicadores(px)
    prob = pr.probabilidades(resultado["predicciones"])
    con_ml = pesos_historicos(ind, prob=prob)
    assert set(con_ml["fecha"]) == set(prob.index)             # solo donde hay predicción
    suma = con_ml.groupby("fecha")["peso"].sum()
    assert np.allclose(suma, 1.0)
    assert con_ml["peso"].between(config.PESO_MIN - 1e-9, config.PESO_MAX + 1e-9).all()
    assert con_ml["factor_ml"].between(1 - config.INCLINACION_ML - 1e-9, 1 + config.INCLINACION_ML + 1e-9).all()
    assert "probabilidad de subida" in con_ml["motivo"].iloc[0]


def test_mas_probabilidad_no_baja_el_peso():
    vol, mom, r6 = [0.15] * 5, [0.1] * 5, [0.1] * 5
    sin = pesos.pesos_para_fecha(vol, mom, r6)["peso"]
    con = pesos.pesos_para_fecha(vol, mom, r6, prob=[0.8, 0.5, 0.5, 0.5, 0.5])["peso"]
    assert np.allclose(sin, 0.2)
    assert con[0] > 0.2 and np.all(con[1:] < 0.2) and con.sum() == pytest.approx(1.0)
    iguales = pesos.pesos_para_fecha(vol, mom, r6, prob=[0.6] * 5)["peso"]
    assert np.allclose(iguales, sin)                           # si el modelo no distingue, no mueve nada


def test_ablacion_compara_en_el_mismo_periodo(px, resultado):
    ind = calcular_indicadores(px)
    regla = pesos_historicos(ind)
    pesos_ml, resumen, curvas = pr.efecto_del_modelo(px, ind, regla, resultado["predicciones"])
    assert list(resumen["estrategia"]) == [config.NOMBRE_ESTRATEGIA, config.NOMBRE_ESTRATEGIA_ML,
                                           config.NOMBRE_IGUALES, config.NOMBRE_SPY]
    assert resumen["fecha_inicio"].nunique() == 1 and resumen["fecha_fin"].nunique() == 1
    assert np.allclose(curvas.iloc[0], 1.0) and list(curvas.columns) == list(resumen["estrategia"])
    r = resumen.set_index("estrategia")
    assert r.at[config.NOMBRE_ESTRATEGIA, "n_rebalanceos"] == r.at[config.NOMBRE_ESTRATEGIA_ML, "n_rebalanceos"]
    assert curvas.index[0] > pesos_ml["fecha"].min()            # se opera al día siguiente de decidir


def test_sin_inclinacion_el_modelo_no_cambia_nada(px, resultado, monkeypatch):
    monkeypatch.setattr(pesos, "INCLINACION_ML", 0.0)
    ind = calcular_indicadores(px)
    _, resumen, curvas = pr.efecto_del_modelo(px, ind, pesos_historicos(ind), resultado["predicciones"])
    assert np.allclose(curvas[config.NOMBRE_ESTRATEGIA], curvas[config.NOMBRE_ESTRATEGIA_ML])
