"""Pruebas del motor: indicadores, pesos y backtest."""

import numpy as np
import pandas as pd
import pytest

from etf2.backtest import correr_backtest, simular
from etf2.config import PESO_MAX, PESO_MIN, TICKERS
from etf2.datos import matriz_precios
from etf2.indicadores import a_largo, calcular_indicadores
from etf2.pesos import aplicar_limites, pesos_anchos, pesos_historicos, pesos_para_fecha


# --- Indicadores ---------------------------------------------------------------
def test_indicadores_no_miran_al_futuro(precios_sinteticos):
    """El indicador de la fecha t debe ser igual con o sin los datos posteriores a t."""
    px = matriz_precios(precios_sinteticos)
    completo = calcular_indicadores(px)
    corte = 900
    parcial = calcular_indicadores(px.iloc[: corte + 1])
    for nombre in completo:
        a = completo[nombre].iloc[corte]
        b = parcial[nombre].iloc[corte]
        pd.testing.assert_series_equal(a, b, check_names=False)


def test_a_largo_no_deja_filas_vacias(precios_sinteticos):
    largo = a_largo(calcular_indicadores(matriz_precios(precios_sinteticos)))
    assert set(largo["ticker"]) == set(TICKERS)
    assert not largo.drop(columns=["fecha", "ticker"]).isna().all(axis=1).any()


# --- Pesos ---------------------------------------------------------------------
def test_limites_suman_uno_y_se_respetan_en_casos_aleatorios():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        w, _ = aplicar_limites(rng.dirichlet(np.ones(5) * rng.uniform(0.1, 3)))
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert w.min() >= PESO_MIN - 1e-9 and w.max() <= PESO_MAX + 1e-9


def test_pesos_para_fecha_aleatorios():
    rng = np.random.default_rng(1)
    for _ in range(500):
        r = pesos_para_fecha(rng.uniform(0.05, 0.4, 5), rng.normal(0, 0.2, 5), rng.normal(0, 0.15, 5))
        assert r["peso"].sum() == pytest.approx(1.0, abs=1e-9)
        assert r["peso"].min() >= PESO_MIN - 1e-9 and r["peso"].max() <= PESO_MAX + 1e-9


def test_menos_volatilidad_mas_peso():
    r = pesos_para_fecha([0.10, 0.20, 0.20, 0.20, 0.20], [0.1] * 5, [0.1] * 5)
    assert r["peso"][0] > r["peso"][1]


def test_mas_momentum_mas_peso():
    r = pesos_para_fecha([0.15] * 5, [0.30, 0.0, 0.0, 0.0, 0.0], [0.1] * 5)
    assert r["peso"][0] > r["peso"][1]


def test_filtro_de_retorno_6m_negativo_baja_el_peso():
    r = pesos_para_fecha([0.15] * 5, [0.1] * 5, [-0.05, 0.1, 0.1, 0.1, 0.1])
    assert r["filtro"][0] and r["peso"][0] < r["peso"][1]


def test_pesos_historicos_suman_uno_cada_dia(precios_sinteticos):
    ind = calcular_indicadores(matriz_precios(precios_sinteticos))
    largo = pesos_historicos(ind)
    assert not largo.empty
    assert (largo.groupby("fecha")["peso"].sum() - 1).abs().max() < 1e-9
    assert largo["motivo"].str.len().min() > 10


# --- Backtest ------------------------------------------------------------------
def _ret_simple():
    fechas = pd.bdate_range("2024-01-01", periods=10)
    return pd.DataFrame({"A": 0.01, "B": 0.0}, index=fechas)


def test_ejecucion_con_un_dia_de_rezago_sin_costo():
    ret = _ret_simple()
    d0 = ret.index[2]
    curva, _, _ = simular(ret, {d0: [1.0, 0.0]}, friccion=0.0)
    assert curva.index[0] == ret.index[3]                  # entra al cierre del día siguiente
    assert curva.iloc[0] == pytest.approx(1.0)
    assert curva.iloc[1] == pytest.approx(1.01)            # el primer retorno ganado es el de t+2


def test_costo_inicial_se_descuenta():
    ret = _ret_simple()
    curva, _, _ = simular(ret, {ret.index[2]: [0.5, 0.5]}, friccion=0.001)
    assert curva.iloc[0] == pytest.approx(1 - 0.001)


def test_rebalanceo_cobra_solo_lo_negociado():
    ret = _ret_simple()
    decisiones = {ret.index[1]: [0.5, 0.5], ret.index[4]: [0.5, 0.5]}
    _, turnover, n = simular(ret, decisiones, friccion=0.001)
    # Entre la compra (cierre 2) y el rebalanceo (cierre 5) A rinde 1 % por día durante 3 días:
    # el peso de A deriva de 0,5 a a/(a+1). Solo se negocia esa deriva, no toda la cartera.
    a = 1.01 ** 3
    assert n == 1 and turnover == pytest.approx(2 * (a / (a + 1) - 0.5))


def test_aplicar_limites_caso_extremo_que_rompia_el_metodo_anterior():
    """Dos pesos enormes y tres diminutos: antes quedaban todos congelados sumando 0,95."""
    w, estado = aplicar_limites([0.50, 0.45, 0.02, 0.02, 0.01])
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert w[0] == pytest.approx(PESO_MAX) and w[1] == pytest.approx(PESO_MAX)
    assert w[2:].min() >= PESO_MIN - 1e-12


def test_backtest_completo_con_datos_sinteticos(precios_sinteticos):
    px = matriz_precios(precios_sinteticos)
    pesos = pesos_anchos(pesos_historicos(calcular_indicadores(px)))
    resumen, curvas = correr_backtest(px, pesos)
    assert len(resumen) == 3 and curvas.shape[1] == 3
    assert (curvas.iloc[0] == 1.0).all()
    assert resumen["max_drawdown"].le(0).all()
    assert resumen["n_rebalanceos"].iloc[0] > 20
