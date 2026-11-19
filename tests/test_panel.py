"""Pruebas de formato y de los cálculos que alimentan el panel."""

import base64
import json

import numpy as np
import pandas as pd
import pytest

from etf2 import formato as f
from etf2 import panel
from etf2.backtest import correr_backtest
from etf2.config import NOMBRE_ESTRATEGIA, PESO_MAX, PESO_MIN, TICKERS
from etf2.datos import matriz_precios
from etf2.indicadores import a_largo, calcular_indicadores
from etf2.pesos import pesos_anchos, pesos_historicos


@pytest.fixture(scope="module")
def motor(precios_sinteticos):
    px = matriz_precios(precios_sinteticos)
    ind = calcular_indicadores(px)
    pesos = pesos_historicos(ind)
    resumen, curvas = correr_backtest(px, pesos_anchos(pesos))
    return {"px": px, "ind": ind, "pesos": pesos, "ancho": pesos_anchos(pesos), "resumen": resumen, "curvas": curvas}


@pytest.fixture(scope="module")
def hoy(motor):
    p = motor["pesos"]
    return p[p["fecha"] == p["fecha"].max()].set_index("ticker").reindex(TICKERS)


@pytest.fixture(scope="module")
def ind_hoy(motor):
    largo = a_largo(motor["ind"])
    return largo[largo["fecha"] == largo["fecha"].max()].set_index("ticker")


# --- formato ----------------------------------------------------------------------
def test_numeros_en_espanol_con_signo_menos_tipografico():
    assert f.num(1234.5, 1) == "1.234,5"
    assert f.num(-14.8) == "−14,8" and "-" not in f.pct(-0.148)
    assert f.pct(0.2405) == "24,1\u00a0%"
    assert f.pct_signo(0.0241) == "+2,4\u00a0%" and f.pct_signo(-0.013) == "−1,3\u00a0%"
    assert f.pct_signo(0.0001) == "0,0\u00a0%"                 # sin signo cuando redondea a cero
    assert f.pp(0.032) == "+3,2\u00a0pp" and f.pp(-0.022) == "−2,2\u00a0pp"
    assert f.usd(12345.6) == "USD\u00a012.346" and f.usd(-12.5, 2) == "−USD\u00a012,50"
    assert f.num(float("nan")) == f.VACIO and f.pct(None) == f.VACIO and f.usd(None) == f.VACIO


def test_cantidades_compactas():
    assert f.compacto(39_985_300) == "40,0\u00a0M"
    assert f.compacto(1_260) == "1,3\u00a0k" and f.compacto(950) == "950"
    assert f.compacto(2_300_000_000) == "2,30\u00a0mil\u00a0M"


def test_fechas_y_tiempo_transcurrido():
    assert f.fecha(pd.Timestamp("2026-11-14")) == "14 nov 2026"
    assert f.mes_anio("2026-03-09") == "mar 2026" and f.dia_mes("2026-03-09") == "9 mar"
    assert f.fecha_hora("2026-11-14T23:00:00+00:00") == "14 nov 2026 18:00"       # hora de Perú (UTC-5)
    ahora = pd.Timestamp("2026-11-14T12:00:00Z")
    assert f.hace("2026-11-14T11:59:30Z", ahora) == "hace un momento"
    assert f.hace("2026-11-14T11:15:00Z", ahora) == "hace 45 min"
    assert f.hace("2026-11-14T07:00:00Z", ahora) == "hace 5 h"
    assert f.hace("2026-11-10T12:00:00Z", ahora) == "hace 4 días"


def test_direccion():
    assert f.direccion(0.01) == "sube" and f.direccion(-0.01) == "baja"
    assert f.direccion(0.0) == "plano" and f.direccion(float("nan")) == "plano"
    assert f.direccion(0.0003, umbral=0.0005) == "plano"


def _jwt(rol: str) -> str:
    cuerpo = base64.urlsafe_b64encode(json.dumps({"role": rol}).encode()).decode().rstrip("=")
    return f"aaa.{cuerpo}.firma"


def test_guarda_contra_claves_secretas():
    assert f.es_clave_secreta("sb_secret_abc") and f.es_clave_secreta(_jwt("service_role"))
    assert not f.es_clave_secreta("sb_publishable_abc") and not f.es_clave_secreta(_jwt("anon"))
    assert not f.es_clave_secreta("") and not f.es_clave_secreta("cualquier-cosa")


def test_etiquetas_no_se_pisan_y_conservan_el_orden():
    pos = f.separar_etiquetas({"A": 100.0, "B": 100.5, "C": 101.0, "D": 150.0}, minimo=5.0)
    orden = sorted(pos, key=pos.get)
    assert orden == ["A", "B", "C", "D"]
    valores = [pos[k] for k in orden]
    assert all(b - a >= 5.0 - 1e-9 for a, b in zip(valores, valores[1:]))
    assert pos["D"] == pytest.approx(150.0)


# --- tablas base ------------------------------------------------------------------
def test_columnas_siempre_en_el_orden_de_config(motor, precios_sinteticos):
    barajado = precios_sinteticos.sample(frac=1, random_state=1)
    assert list(panel.matriz_precios(barajado).columns) == TICKERS
    assert list(panel.pesos_ancho(motor["pesos"].sample(frac=1, random_state=2)).columns) == TICKERS


def test_fin_de_mes_recortar_y_base100(motor):
    m = panel.fin_de_mes(motor["ancho"])
    assert m.index.to_series().groupby([m.index.year, m.index.month]).size().max() == 1
    ult = panel.recortar(motor["px"], 365)
    assert (ult.index[-1] - ult.index[0]).days <= 365 and panel.recortar(motor["px"], None).equals(motor["px"])
    assert (panel.base100(ult).iloc[0] == 100).all()


def test_cambio_de_peso_necesita_historia(motor):
    c = panel.cambio_de_peso(motor["ancho"], 30)
    assert c is not None and abs(c.sum()) < 1e-9
    assert panel.cambio_de_peso(motor["ancho"].iloc[:5], 30) is None


# --- velas ------------------------------------------------------------------------
def test_intervalo_segun_el_rango():
    assert panel.intervalo_velas(92) == ("diarias", None)
    assert panel.intervalo_velas(365) == ("semanales", "W-FRI")
    assert panel.intervalo_velas(None) == ("mensuales", "M")
    assert set(panel.RANGOS_VELAS.values()) >= {92, 183, 365, None}


def test_velas_semanales_agrupan_bien(precios_sinteticos):
    o = panel.ohlc(precios_sinteticos, "QQQ")
    v = panel.velas(o, "W-FRI")
    ultima = v.index[-2]                                        # una semana completa
    dias = o[(o.index <= ultima) & (o.index > v.index[-3])]
    assert set(v.index) <= set(o.index)                         # cada vela se fecha en un día que existe
    assert v.loc[ultima, "apertura"] == pytest.approx(dias["apertura"].iloc[0])
    assert v.loc[ultima, "cierre"] == pytest.approx(dias["cierre"].iloc[-1])
    assert v.loc[ultima, "maximo"] == pytest.approx(dias["maximo"].max())
    assert v.loc[ultima, "minimo"] == pytest.approx(dias["minimo"].min())
    assert v.loc[ultima, "volumen"] == pytest.approx(dias["volumen"].sum())
    assert v["volumen"].sum() == pytest.approx(o["volumen"].sum())
    assert (v["maximo"] >= v[["apertura", "cierre"]].max(axis=1) - 1e-9).all()
    assert (v["minimo"] <= v[["apertura", "cierre"]].min(axis=1) + 1e-9).all()


def test_velas_diarias_y_medias(precios_sinteticos):
    o = panel.ohlc(precios_sinteticos, "SPY")
    v = panel.velas(o)
    assert len(v) == len(o)
    assert v["mm20"].iloc[-1] == pytest.approx(o["cierre"].tail(20).mean())
    assert v["mm50"].iloc[:49].isna().all() and v["mm50"].iloc[49:].notna().all()
    assert v["variacion"].iloc[-1] == pytest.approx(o["cierre"].iloc[-1] / o["cierre"].iloc[-2] - 1)


def test_ohlc_rellena_huecos_con_el_cierre(precios_sinteticos):
    d = precios_sinteticos.copy()
    d.loc[d.index[:3], ["apertura", "maximo", "minimo", "volumen"]] = np.nan
    o = panel.ohlc(d, "SPY")
    assert not o.isna().any().any()
    assert (o.iloc[:3]["apertura"] == o.iloc[:3]["cierre"]).all() and (o.iloc[:3]["volumen"] == 0).all()


def test_resumen_universo(precios_sinteticos, ind_hoy):
    u = panel.resumen_universo(precios_sinteticos, ind_hoy)
    assert list(u.index) == TICKERS
    spy = precios_sinteticos[precios_sinteticos["ticker"] == "SPY"]
    assert u.at["SPY", "cierre"] == pytest.approx(spy["cierre"].iloc[-1])
    assert u.at["SPY", "var_dia"] == pytest.approx(spy["cierre"].iloc[-1] / spy["cierre"].iloc[-2] - 1)
    assert (u["min_52"] <= u["cierre"]).all() and (u["cierre"] <= u["max_52"]).all()
    assert u.at["TLT", "ret_6m"] == pytest.approx(ind_hoy.at["TLT", "ret_6m"])
    assert len(u.at["SPY", "tendencia"]) == 90
    sin_ind = panel.resumen_universo(precios_sinteticos, pd.DataFrame(columns=panel.COLS_IND))
    assert sin_ind["vol_60d"].isna().all()                      # sin indicadores no inventa nada


# --- backtest ---------------------------------------------------------------------
def test_drawdowns_y_retornos_componen_bien(motor):
    c = motor["curvas"]
    dd = panel.drawdowns(c)
    assert (dd <= 1e-12).all().all() and dd.iloc[0].abs().max() < 1e-12
    anual, parciales = panel.retornos_anuales(c)
    assert ((1 + anual).prod() - 1).to_numpy() == pytest.approx((c.iloc[-1] / c.iloc[0] - 1).to_numpy())
    assert int(c.index[-1].year) in parciales                   # los datos terminan en febrero
    r, tabla = panel.retornos_mensuales(c[NOMBRE_ESTRATEGIA])
    assert (1 + r).prod() - 1 == pytest.approx(c[NOMBRE_ESTRATEGIA].iloc[-1] / c[NOMBRE_ESTRATEGIA].iloc[0] - 1)
    assert list(tabla.columns) == list(range(1, 13)) and tabla.notna().sum().sum() == len(r)
    m = panel.resumen_mensual(r)
    assert m["mejor"] == r.max() and m["peor"] == r.min() and 0 <= m["positivos"] <= 1 and m["n"] == len(r)


# --- riesgo -----------------------------------------------------------------------
def test_contribucion_riesgo():
    rng = np.random.default_rng(3)
    indep = pd.DataFrame(rng.normal(0, 0.01, (4000, 3)), columns=["A", "B", "C"])
    iguales = pd.Series(1 / 3, index=["A", "B", "C"])
    rc = panel.contribucion_riesgo(iguales, indep)
    assert rc.sum() == pytest.approx(1.0) and rc.to_numpy() == pytest.approx([1 / 3] * 3, abs=0.03)
    doble = indep.assign(A=indep["A"] * 2)                       # A con el doble de volatilidad
    assert panel.contribucion_riesgo(iguales, doble)["A"] > 0.6
    cobertura = indep.assign(C=-indep["A"] + rng.normal(0, 0.001, 4000))
    assert panel.contribucion_riesgo(pd.Series({"A": 0.6, "B": 0.3, "C": 0.1}), cobertura)["C"] < 0


# --- órdenes ----------------------------------------------------------------------
PESOS = pd.Series({"SPY": 0.25, "QQQ": 0.25, "IWM": 0.10, "GLD": 0.15, "TLT": 0.25})
PRECIOS = pd.Series({"SPY": 500.0, "QQQ": 400.0, "IWM": 200.0, "GLD": 250.0, "TLT": 100.0})


def test_ordenes_desde_efectivo():
    t, tot = panel.ordenes(PESOS, PRECIOS, pd.Series({"Efectivo": 10_000.0}), tolerancia=0.01)
    assert (t["operacion"] == "Comprar").all() and tot["compras"] == 5 and tot["ventas"] == 0
    assert t["monto"].sum() == pytest.approx(10_000) and tot["efectivo_final"] == pytest.approx(0)
    assert t.at["TLT", "acciones"] == pytest.approx(25.0) and t.at["SPY", "acciones"] == pytest.approx(5.0)


def test_ordenes_desde_una_cartera_y_banda_de_no_operacion():
    actual = pd.Series({"SPY": 4000.0, "QQQ": 2500.0, "IWM": 1000.0, "GLD": 1450.0, "TLT": 1050.0, "Efectivo": 0.0})
    t, tot = panel.ordenes(PESOS, PRECIOS, actual, tolerancia=0.0)
    assert t["monto"].sum() == pytest.approx(0) and t["final"].to_numpy() == pytest.approx((PESOS * 10_000).to_numpy())
    assert t.at["SPY", "operacion"] == "Vender" and t.at["TLT", "operacion"] == "Comprar"
    assert t.at["QQQ", "operacion"] == "Mantener" and t.at["IWM", "operacion"] == "Mantener"
    t2, tot2 = panel.ordenes(PESOS, PRECIOS, actual, tolerancia=0.01)     # GLD se desvía 50 de 10.000: no se toca
    assert t2.at["GLD", "operacion"] == "Mantener" and t2.at["GLD", "monto"] == 0
    assert tot2["negociado"] < tot["negociado"]
    assert tot2["efectivo_final"] == pytest.approx(10_000 - t2["final"].sum())


# --- calendario -------------------------------------------------------------------
def test_proximo_rebalanceo_y_retraso():
    assert panel.proximo_rebalanceo("2026-11-13") == pd.Timestamp("2026-11-30")
    assert panel.proximo_rebalanceo("2026-11-30") == pd.Timestamp("2026-11-30")
    assert panel.proximo_rebalanceo("2026-10-12") == pd.Timestamp("2026-10-30")   # el 31 cae sábado
    assert panel.dias_de_retraso("2026-11-13", "2026-11-14") == 1                # viernes -> sábado
    assert panel.dias_de_retraso("2026-11-13", "2026-11-16") == 1                # lunes antes del cierre
    assert panel.dias_de_retraso("2026-11-13", "2026-11-18") == 3                # tres ruedas sin datos
    assert panel.dias_de_retraso("2026-11-13", "2026-11-13") == 0


# --- lectura ----------------------------------------------------------------------
def test_lectura_sale_de_los_datos(hoy, ind_hoy, motor):
    frases = panel.lectura(hoy, ind_hoy, panel.cambio_de_peso(motor["ancho"], 30))
    texto = " ".join(frases)
    mayor = hoy["peso"].idxmax()
    assert f"**{mayor}** es la mayor posición" in texto and f.pct(hoy["peso"].max()) in texto
    assert "renta variable" in frases[0] and "oro" in frases[0] and "bonos" in frases[0]
    for t in TICKERS:
        negativo = ind_hoy.at[t, "ret_6m"] <= 0
        assert (f"**{t}** ({f.pct_signo(ind_hoy.at[t, 'ret_6m'])})" in texto) == negativo
    assert "hace un mes" in texto
    assert not any("mes" in x and "hace un mes" in x for x in panel.lectura(hoy, ind_hoy, None))


def test_lectura_menciona_limites():
    hoy = pd.DataFrame({"peso": [PESO_MAX, 0.30, 0.20, PESO_MIN, PESO_MIN],
                        "peso_base": [0.2] * 5, "factor_momentum": [1.5, 1.2, 1.0, 0.3, 0.3]}, index=TICKERS)
    ind = pd.DataFrame({"vol_60d": [0.2, 0.1, 0.15, 0.12, 0.11], "ret_6m": [0.1, 0.1, 0.1, -0.1, -0.2]}, index=TICKERS)
    texto = " ".join(panel.lectura(hoy, ind, None))
    assert "toca el tope permitido" in texto and "queda en el mínimo" in texto
    assert "**GLD** (−10,0\u00a0%) y **TLT** (−20,0\u00a0%) tienen retorno negativo" in texto
