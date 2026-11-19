"""Cálculos, gráficos y piezas de las pestañas Descriptivo (técnicos), Predictivo y Prescriptivo."""

import json

import numpy as np
import pandas as pd
import pytest

from etf2 import componentes as c
from etf2 import config, graficos, panel, prescriptivo as ps, tema
from etf2.config import NOMBRE_ESTRATEGIA as E, NOMBRE_ESTRATEGIA_ML as EML, TICKERS

T = tema.T


@pytest.fixture(scope="module")
def ohlc(precios_sinteticos):
    return panel.ohlc(precios_sinteticos, "SPY")


def predicciones(n_fechas: int = 30, semilla: int = 5) -> pd.DataFrame:
    """Predicciones inventadas: el modelo «logit» acierta algo más que el azar; «frecuencia» dice siempre 0,6."""
    rng = np.random.default_rng(semilla)
    fechas = pd.bdate_range("2023-01-31", periods=n_fechas, freq="BME")
    filas = []
    for i, fecha in enumerate(fechas):
        for t in TICKERS:
            subio = float(rng.random() < 0.6) if i < n_fechas - 1 else np.nan       # la última aún no se resuelve
            ret = (0.03 if subio == 1 else -0.03) + rng.normal(0, 0.01) if subio == subio else np.nan
            senal = 0.6 + (0.15 if subio == 1 else -0.15 if subio == 0 else 0.0) + rng.normal(0, 0.12)
            for modelo, p in (("logit", senal), ("gb", senal + rng.normal(0, 0.05)), ("momentum", float(rng.random() < 0.5)),
                              ("frecuencia", 0.6)):
                filas.append({"fecha": fecha, "ticker": t, "modelo": modelo, "prob_sube": float(np.clip(p, 0.02, 0.98)),
                              "subio": subio, "ret_futuro": ret})
    return pd.DataFrame(filas)


# --- Indicadores técnicos y distribución ---------------------------------------------------
def test_tecnicos_y_su_lectura(ohlc):
    t = panel.tecnicos(ohlc)
    assert list(t.columns) == ["macd", "senal", "hist", "rsi", "atr", "atr_pct"] and len(t) == len(ohlc)
    ult = t.iloc[-1]
    assert 0 <= ult["rsi"] <= 100 and ult["atr"] > 0 and ult["atr_pct"] == pytest.approx(ult["atr"] / ohlc["cierre"].iloc[-1])
    assert panel.lectura_tecnica(pd.Series({"rsi": 75.0, "hist": 0.4})) == {
        "rsi": "Sobrecompra (más de 70)", "macd": "MACD sobre su señal: impulso al alza"}
    baja = panel.lectura_tecnica(pd.Series({"rsi": 22.0, "hist": -0.4}))
    assert baja["rsi"].startswith("Sobreventa") and baja["macd"].endswith("a la baja")
    assert panel.lectura_tecnica(pd.Series({"rsi": 50.0, "hist": 0.1}))["rsi"].startswith("Zona neutral")
    assert panel.lectura_tecnica(pd.Series({"rsi": np.nan, "hist": np.nan})) == {"rsi": "—", "macd": "—"}

    fig = graficos.tecnicos(t.dropna().tail(200), "RSI y MACD", "sub")
    assert [tr.name for tr in fig.data] == ["RSI (14 días)", "Histograma (MACD − señal)", "MACD (12, 26)", "Señal (9)"]
    assert list(fig.layout.yaxis.range) == [0, 100]
    colores = set(fig.data[1].marker.color)
    assert colores <= {tema.rgba(T["sube"], 0.55), tema.rgba(T["baja"], 0.55)}


def test_estadisticas_e_histograma_de_retornos(precios_sinteticos):
    px = panel.matriz_precios(precios_sinteticos).tail(400)
    est = panel.estadisticas_retornos(px)
    r = px["QQQ"].pct_change().dropna()
    assert est.at["QQQ", "media"] == pytest.approx(r.mean()) and est.at["QQQ", "peor"] == pytest.approx(r.min())
    assert est.at["QQQ", "positivos"] == pytest.approx((r > 0).mean()) and est.at["QQQ", "n"] == len(r)
    h = panel.histograma(r)
    assert h["dias"].sum() == len(r) and len(h) == 31                     # ningún día se pierde
    assert h["desde"].iloc[0] == pytest.approx(-h["hasta"].iloc[-1])      # simétrico alrededor de cero
    assert np.allclose(h["hasta"].iloc[:-1], h["desde"].iloc[1:])
    fig = graficos.histograma(h, float(r.mean()), tema.color_etf("QQQ"), "t", "s")
    assert fig.data[0].marker.color == tema.color_etf("QQQ") and sum(fig.data[0].y) == len(r)


# --- Predictivo ----------------------------------------------------------------------------
def test_prob_de_hoy_y_resueltas():
    pred = predicciones()
    fecha, hoy = panel.prob_de_hoy(pred)
    assert fecha == pred["fecha"].max() and list(hoy.index) == TICKERS
    assert set(hoy.columns) == {"logit", "gb", "momentum", "frecuencia"}
    res = panel.resueltas(pred, "logit")
    assert res["subio"].notna().all() and len(res) == 29 * 5 and set(res["modelo"]) == {"logit"}


def test_calibracion_y_matriz_de_confusion():
    pred = predicciones()
    cal = panel.calibracion(pred, "logit")
    assert len(cal) == 5 and cal["n"].sum() == 29 * 5
    assert cal["prob_media"].is_monotonic_increasing
    assert cal["frecuencia"].iloc[-1] > cal["frecuencia"].iloc[0]         # a más probabilidad, más subidas
    assert panel.calibracion(pred, "frecuencia").empty                    # una probabilidad constante no se calibra

    m = panel.matriz_confusion(pred, "logit")
    res = panel.resueltas(pred, "logit")
    assert sum(m.values()) == len(res)
    assert m["vp"] + m["fn"] == int(res["subio"].sum())
    assert m["vp"] == int(((res["prob_sube"] >= 0.5) & (res["subio"] == 1)).sum())
    html = c.confusion(m)
    assert f"<b>{m['vp']}</b>aciertos" in html and "falsas alarmas" in html and "subidas no vistas" in html

    fig = graficos.calibracion(cal, "t", "s")
    assert [tr.name for tr in fig.data] == ["Calibración perfecta", "Modelo"]
    assert list(fig.layout.xaxis.range) == list(fig.layout.yaxis.range)    # ejes iguales: la diagonal es la diagonal


def test_grafico_de_probabilidades_e_importancias():
    _, hoy = panel.prob_de_hoy(predicciones())
    fig = graficos.probabilidades(hoy, "logit", "gb", config.NOMBRES_MODELOS, "t", "s")
    orden = list(fig.data[0].y)
    assert sorted(orden) == sorted(TICKERS) and list(fig.data[0].x) == sorted(hoy["logit"], reverse=True)
    assert list(fig.data[0].marker.color) == [tema.color_etf(t) for t in orden]       # el color sigue al ETF
    assert fig.data[1].marker.symbol == "diamond" and fig.layout.showlegend is False
    solo = graficos.probabilidades(hoy, "logit", None, config.NOMBRES_MODELOS, "t", "s")
    assert len(solo.data) == 1

    imp = pd.DataFrame({"variable": ["ret_1m", "vol_60d", "rsi_14"], "importancia": [0.2, -0.5, 0.05]})
    con_signo = graficos.importancia(imp, True, "t", "s")
    assert list(con_signo.data[0].y) == ["RSI (14 d)", "Retorno 1 mes", "Volatilidad 60 d"]     # de menor a mayor peso
    assert list(con_signo.data[0].marker.color) == [T["polo_frio"], T["polo_frio"], T["polo_calido"]]
    sin_signo = graficos.importancia(imp.assign(importancia=[0.3, 0.6, 0.1]), False, "t", "s")
    assert sin_signo.data[0].marker.color == T["acero"] and sin_signo.layout.xaxis.range[0] == 0


def test_veredicto_dice_si_el_modelo_aporta_o_no():
    def metricas(exactitud, auc):
        return pd.DataFrame({"modelo": ["logit", "frecuencia"], "exactitud": [exactitud, 0.56], "auc": [auc, np.nan]})

    def resumen(sin, con):
        return pd.DataFrame({"estrategia": [E, EML], "sharpe": [sin, con]})

    bueno = " ".join(panel.veredicto_del_modelo(metricas(0.61, 0.58), resumen(0.80, 0.91)))
    assert "La ventaja es de" in bueno and "ordena mejor que el azar" in bueno and "sí lo usa" in bueno
    malo = " ".join(panel.veredicto_del_modelo(metricas(0.50, 0.48), resumen(0.87, 0.85)))
    assert "hoy no la supera en exactitud" in malo and "casi como el azar" in malo and "la recomendación no lo usa" in malo
    assert len(panel.veredicto_del_modelo(metricas(0.50, np.nan), pd.DataFrame())) == 1


def test_el_veredicto_del_panel_coincide_con_la_regla_del_pipeline():
    """El panel explica la misma decisión que toma prescriptivo.conviene_usar_el_modelo."""
    for sin, con in ((0.8, 0.9), (0.8, 0.8), (0.9, 0.7)):
        r = pd.DataFrame({"estrategia": [E, EML], "sharpe": [sin, con]})
        dice_que_si = "sí lo usa" in " ".join(panel.veredicto_del_modelo(pd.DataFrame({"modelo": []}), r))
        assert dice_que_si == ps.conviene_usar_el_modelo(r)


def test_la_estrategia_con_modelo_tiene_estilo_propio():
    estilo = graficos.ESTILO
    assert EML in estilo and estilo[EML]["color"] == T["modelo"]
    assert len({v["color"] for v in estilo.values()}) == len(estilo)
    assert T["modelo"] not in {T["sube"], T["baja"], T["acento"], *T["etf"].values()}
    fechas = pd.bdate_range("2024-01-01", periods=300)
    curvas = pd.DataFrame({n: 100 + np.arange(300) * k for n, k in zip(estilo, (0.1, 0.12, 0.15, 0.16))}, index=fechas)
    fig = graficos.capital(curvas, "t", "s")
    assert {tr.name for tr in fig.data} == set(estilo)
    assert {tr.name: tr.line.color for tr in fig.data}[EML] == T["modelo"]


# --- Prescriptivo ----------------------------------------------------------------------------
def test_recomendacion_de_un_perfil():
    detalle = {"etfs": [{"ticker": "SPY", "peso": 0.3}]}
    recs = pd.DataFrame({
        "fecha": pd.to_datetime(["2025-02-18", "2025-02-18", "2025-02-17"]),
        "perfil": ["Moderado", "Agresivo", "Moderado"], "resumen": ["hoy", "hoy agresivo", "ayer"],
        "detalle": [json.dumps(detalle), detalle, detalle]})
    r = panel.recomendacion_de(recs, "Moderado")
    assert r["resumen"] == "hoy" and r["detalle"] == detalle                  # la más reciente; el JSON en texto se lee
    assert panel.recomendacion_de(recs, "Agresivo")["detalle"] == detalle
    assert panel.recomendacion_de(recs, "Conservador") is None
    assert panel.recomendacion_de(pd.DataFrame(), "Moderado") is None


def test_recomendaciones_guardadas_con_los_nombres_anteriores():
    detalle = {"etfs": [{"ticker": "SPY", "accion": "Aumentar"}, {"ticker": "TLT", "accion": "Reducir"},
                        {"ticker": "GLD", "accion": "Mantener"}]}
    recs = pd.DataFrame({"fecha": pd.to_datetime(["2026-11-13"]), "perfil": ["Moderado"], "resumen": ["x"],
                         "detalle": [json.dumps(detalle)]})
    acciones = [e["accion"] for e in panel.recomendacion_de(recs, "Moderado")["detalle"]["etfs"]]
    assert acciones == ["Comprar", "Vender", "Mantener"]


def test_decision_del_mes():
    detalle = {"etfs": [
        {"ticker": "TLT", "peso": 0.25, "peso_previo": 0.30, "accion": "Vender", "prob_sube": 0.5, "explicacion": "b"},
        {"ticker": "SPY", "peso": 0.40, "peso_previo": None, "accion": "Mantener", "prob_sube": None, "explicacion": "a"},
    ]}
    universo = pd.DataFrame({"cierre": [600.0, 90.0], "vol_60d": [0.18, np.nan]}, index=["SPY", "TLT"])
    d = panel.decision_del_mes({"detalle": detalle}, universo)
    assert list(d.index) == ["SPY", "TLT"]                                   # en el orden de config.TICKERS
    assert d.at["TLT", "cambio"] == pytest.approx(-0.05) and pd.isna(d.at["SPY", "cambio"])
    assert d.at["SPY", "alerta"] == pytest.approx(600 * (1 - 2 * 0.18 / np.sqrt(12)))
    assert d.at["SPY", "margen"] == pytest.approx(-2 * 0.18 / np.sqrt(12))
    assert pd.isna(d.at["TLT", "alerta"]) and pd.isna(d.at["TLT", "margen"])


def test_uso_y_costo_del_llm():
    hoy = pd.Timestamp("2026-11-14")
    log = pd.DataFrame({
        "id": [6, 5, 4, 3, 2, 1],
        "dia": [hoy, hoy, hoy, hoy, hoy - pd.Timedelta(days=1), hoy - pd.Timedelta(days=1)],
        "estado": ["ok", "rechazada", "limite", "sin_clave", "ok", "error"],
        "modelo": ["gemini-3.5-flash-lite", "gemini-3.5-flash-lite", None, None, "gemini-2.5-flash-lite", None],
        "tokens_entrada": [1000.0, 1000.0, np.nan, np.nan, 800.0, np.nan],
        "tokens_salida": [400.0, 380.0, np.nan, np.nan, 200.0, np.nan]})
    uso = panel.uso_del_llm(log, hoy)
    assert uso["llamadas"] == 4 and uso["aceptadas"] == 2 and uso["hoy"] == 2     # límite y sin clave no son llamadas
    assert uso["tokens_entrada"] == 900 and uso["tokens_salida"] == 300 and uso["modelo"] == "gemini-3.5-flash-lite"
    entrada, salida = config.LLM_PRECIOS_USD_M["gemini-3.5-flash-lite"]
    assert panel.costo_por_cien(uso) == pytest.approx(100 * (900 * entrada + 300 * salida) / 1e6)
    assert panel.costo_por_cien({**uso, "modelo": "modelo-desconocido"}) is None

    vacio = panel.uso_del_llm(log[log["estado"] == "sin_clave"], hoy)
    assert vacio["llamadas"] == 0 and vacio["modelo"] is None and panel.costo_por_cien(vacio) is None


def test_piezas_del_prescriptivo_escapan_el_texto():
    """El texto del LLM llega de fuera: nunca debe poder inyectar HTML en la página."""
    malo = '<script>alert(1)</script> & "comillas"'
    tarjeta = c.tarjeta_recomendacion("Título", "sub", ['<span class="mt-etq">ok</span>'], malo, "<div>barra</div>")
    assert "<script>" not in tarjeta and "&lt;script&gt;" in tarjeta
    assert '<span class="mt-etq">ok</span>' in tarjeta and "<div>barra</div>" in tarjeta
    assert "<script>" not in c.lista([malo]) and "<script>" not in c.aviso(malo)
    assert 'class="mt-etq compra"' in c.etiqueta_accion("Comprar")
    assert 'class="mt-etq venta"' in c.etiqueta_accion("Vender")
    assert 'class="mt-etq "' in c.etiqueta_accion("Mantener") and "<b>" not in c.etiqueta_accion("<b>x</b>")
    tres = c.reportes([("Descriptivo", "¿Qué pasó?", "texto", "lidera **SPY**")] * 3)
    assert tres.count('class="mt-paso"') == 3 and "<b>SPY</b>" in tres and "mt-flujo c3" in tres
    assert "mt-flujo c5" in c.flujo([("a", "b", "c")] * 5) and 'class="mt-flujo"' in c.flujo([("a", "b", "c")] * 4)


def test_titular_de_la_decision():
    t = c.titular_decision("Moderado", {"SPY": "Comprar", "QQQ": "Mantener", "IWM": "Mantener", "GLD": "Vender",
                                        "TLT": "Mantener"}, "13 nov 2026")
    assert "Este mes, perfil moderado · datos al 13 nov 2026" in t
    assert '<span class="compra">Comprar</span> <b>SPY</b>' in t
    assert '<span class="venta">Vender una parte de</span> <b>GLD</b>' in t
    assert "Mantener</span> <b>QQQ</b>, <b>IWM</b> y <b>TLT</b>" in t
    quieto = c.titular_decision("Agresivo", dict.fromkeys(TICKERS, "Mantener"), "x")
    assert "Mantener los cinco ETFs" in quieto and "compra" not in quieto
    assert "<script>" not in c.titular_decision("<script>", {"<b>": "Comprar"}, "<i>")


def test_tabla_con_ayudas_en_los_encabezados():
    t = c.tabla(["ETF", "AUC"], [["SPY", "0,48"]], ayudas={"AUC": 'Ordena "bien" <si> pasa 0,50'})
    assert '<th class="ayuda" title="Ordena &quot;bien&quot; &lt;si&gt; pasa 0,50">AUC</th>' in t
    assert '<th class="izq">ETF</th>' in t
    assert 'class="ayuda izq"' in c.tabla(["ETF"], [["SPY"]], ayudas={"ETF": "x"})

