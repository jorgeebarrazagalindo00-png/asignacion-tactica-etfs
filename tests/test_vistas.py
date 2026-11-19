"""Pruebas de las piezas visuales: colores, HTML y gráficos."""

import re

import pandas as pd
import pytest

from etf2 import componentes as c
from etf2 import formato as f
from etf2 import graficos, panel, tema
from etf2.backtest import correr_backtest
from etf2.config import NOMBRE_ESTRATEGIA, NOMBRE_SPY, PESO_MAX, TICKERS
from etf2.datos import matriz_precios
from etf2.indicadores import a_largo, calcular_indicadores
from etf2.pesos import pesos_anchos, pesos_historicos

T = tema.T


@pytest.fixture(scope="module")
def m(precios_sinteticos):
    px = matriz_precios(precios_sinteticos)
    ind = calcular_indicadores(px)
    pesos = pesos_historicos(ind)
    ancho = pesos_anchos(pesos)
    resumen, curvas = correr_backtest(px, ancho)
    largo = a_largo(ind)
    ind_hoy = largo[largo["fecha"] == largo["fecha"].max()].set_index("ticker")
    hoy = pesos[pesos["fecha"] == pesos["fecha"].max()].set_index("ticker").reindex(TICKERS)
    return {"precios": precios_sinteticos, "px": px, "ind": ind, "ancho": ancho, "curvas": curvas * 100,
            "hoy": hoy, "ind_hoy": ind_hoy, "universo": panel.resumen_universo(precios_sinteticos, ind_hoy)}


# --- colores ----------------------------------------------------------------------
def _contraste(a: str, b: str) -> float:
    la, lb = sorted((graficos._luminancia(a), graficos._luminancia(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_cada_etf_tiene_un_color_propio_y_fijo():
    colores = [tema.color_etf(t) for t in TICKERS]
    assert len(set(colores)) == len(TICKERS) and all(re.fullmatch(r"#[0-9a-f]{6}", x) for x in colores)
    assert tema.color_etf("XXX") == T["tenue"]                     # un ticker desconocido no inventa color
    reservados = {T["sube"], T["baja"], T["acento"], T["tinta"]}
    assert not reservados & set(colores)                           # verde, rojo y dorado no son series


def test_contrastes_minimos_sobre_la_superficie():
    for color in [*T["etf"].values(), T["sube"], T["baja"], T["acento"], T["acero"]]:
        assert _contraste(color, T["superficie"]) >= 3.0, color    # marcas
    for texto in (T["tinta"], T["tinta2"], T["tenue"], T["sube"], T["baja"], T["acento"]):
        assert _contraste(texto, T["superficie"]) >= 4.5, texto    # texto
    assert _contraste(T["tenue"], T["pagina"]) >= 4.5


def test_tinta_sobre_elige_el_texto_mas_legible():
    assert graficos.tinta_sobre(T["neutro"]) == "#ffffff" and graficos.tinta_sobre(T["sube"]) == "#000000"
    # Con blanco y negro puros, cualquier fondo de las escalas deja el texto por encima de 4,5:1.
    for polo in (T["sube"], T["baja"], T["polo_frio"], T["polo_calido"]):
        for paso in range(0, 21):
            fondo = graficos._mezcla(T["neutro"], polo, paso / 20)
            assert _contraste(graficos.tinta_sobre(fondo), fondo) >= 4.5, fondo


def test_la_hoja_de_estilos_define_las_variables():
    css = tema.css()
    assert css.startswith("<style>") and css.rstrip().endswith("</style>")
    assert css.index("@import") < css.index(".stApp")              # @import debe ir primero
    for var in ("--pagina", "--sup", "--tinta", "--acento", "--sube", "--baja"):
        assert f"{var}:" in css and f"var({var})" in css
    assert tema.rgba("#ff8000", 0.5) == "rgba(255,128,0,0.5)"


# --- HTML -------------------------------------------------------------------------
MALICIOSO = '<script>alert(1)</script> "x" & y'


def test_todo_lo_que_viene_de_la_base_se_escapa(m):
    hoy = m["hoy"]
    piezas = [
        c.tarjeta_asignacion(hoy, {t: MALICIOSO for t in TICKERS}, {t: MALICIOSO for t in TICKERS}, None, MALICIOSO),
        c.tarjeta_lectura([MALICIOSO, "**fuerte** y normal"], MALICIOSO, 3),
        c.ficha_etf("SPY", MALICIOSO, MALICIOSO, m["universo"].loc["SPY"]),
        c.cabecera(MALICIOSO, MALICIOSO, MALICIOSO, [c.chip(MALICIOSO, MALICIOSO, "ok")]),
        c.kpis([{"etiqueta": MALICIOSO, "valor": MALICIOSO, "detalle": c.e(MALICIOSO)}], pie=MALICIOSO),
        c.seccion(MALICIOSO, MALICIOSO, MALICIOSO), c.nota(MALICIOSO), c.lista([MALICIOSO]),
        c.flujo([(MALICIOSO, MALICIOSO, MALICIOSO)]), c.tabla([MALICIOSO], [[c.e(MALICIOSO)]]),
        c.barra_segmentada([(MALICIOSO, 0.5, "#fff"), ("b", 0.5, "#000")]),
    ]
    for html in piezas:
        assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<b>fuerte</b> y normal" in piezas[1]


def test_delta_colorea_por_favorable_no_por_signo():
    assert 'class="up' in c.delta(0.02, "+2 %") and "▲" in c.delta(0.02, "+2 %")
    assert 'class="down' in c.delta(-0.02, "−2 %") and "▼" in c.delta(-0.02, "−2 %")
    assert 'class="up' in c.delta(-0.02, "−2 pp", bueno_si_sube=False)      # bajar la volatilidad es bueno
    assert 'class="plano' in c.delta(0.0, "0 %") and 'class="plano' in c.delta(0.02, "+2 pp", neutro=True)
    assert "▲" not in c.delta(0.02, "+2 %", flecha=False)


def test_sparkline():
    svg = c.sparkline([1, 3, 2, 5, float("nan"), 4], "#abcdef", 80, 24)
    assert svg.startswith("<svg") and svg.count("<polyline") == 1 and svg.count("<circle") == 1
    puntos = re.search(r'points="([^"]+)"', svg).group(1).split()
    assert len(puntos) == 5                                             # el NaN se descarta
    xs, ys = zip(*[map(float, p.split(",")) for p in puntos])
    assert 0 <= min(xs) and max(xs) <= 80 and 0 <= min(ys) and max(ys) <= 24
    assert c.sparkline([5], "#fff") == "" and c.sparkline([], "#fff") == ""
    assert c.sparkline([2, 2, 2], "#fff").startswith("<svg")            # serie plana: no divide entre cero


def test_barra_segmentada_solo_rotula_lo_que_cabe():
    html = c.barra_segmentada([("A", 0.56, "#111111"), ("B", 0.40, "#222222"), ("C", 0.04, "#333333"), ("D", 0.0, "#444")])
    bloques = re.findall(r'<div(?: class="angosto")? style="flex:([\d.]+) 1 0;background:(#\w+)" title="([^"]+)">([^<]*)</div>', html)
    assert [b[1] for b in bloques] == ["#111111", "#222222", "#333333"]   # el de valor 0 no se dibuja
    assert bloques[0][3] == f.pct(0.56) and bloques[2][3] == ""           # 4 % no cabe: va sin texto...
    assert bloques[2][2] == f"C: {f.pct(0.04)}" and "C 4,0" in html       # ...pero queda en el tooltip y el alt
    assert html.count('class="angosto"') == 1                             # solo el bloque chico se oculta en móviles


def test_tarjeta_de_asignacion(m):
    html = c.tarjeta_asignacion(m["hoy"], {}, {t: f"clase {t}" for t in TICKERS}, None, "hoy")
    orden = re.findall(r'</i>([A-Z]{3})</span>', html)
    assert orden == list(m["hoy"]["peso"].sort_values(ascending=False).index)   # de mayor a menor peso
    for t in TICKERS:
        assert f.pct(m["hoy"].at[t, "peso"]) in html and f"clase {t}" in html
    anchos = [float(x) for x in re.findall(r'<i style="width:([\d.]+)%', html)]
    assert max(anchos) == pytest.approx(m["hoy"]["peso"].max() / PESO_MAX * 100, abs=0.06)
    assert "sin cambio" in c.tarjeta_asignacion(m["hoy"], {}, {}, pd.Series(0.0, index=TICKERS), "hoy")


def test_tabla_y_universo(m):
    html = c.tabla(["A", "B", "C"], [["1", "2", "3"]], izquierda=(0,), texto=(2,), tenue=(1,), alto_max=200)
    assert len(re.findall(r"<th[ >]", html)) == 3 and len(re.findall(r"<td[ >]", html)) == 3
    assert "max-height:200px" in html
    assert '<td class="izq">1</td>' in html and '<td class="tenue">2</td>' in html and '<td class="izq txt">3</td>' in html
    uni = c.tabla_universo(m["universo"])
    assert uni.count("<tr>") == len(TICKERS) + 1 and uni.count("<svg") == len(TICKERS)


def test_html_compacto_no_deja_lineas_que_markdown_tome_por_codigo(m):
    for html in (c.cabecera("t", "k", "l", []), c.tarjeta_lectura(["a"], "p"), c.ficha_etf("SPY", "n", "c", m["universo"].loc["SPY"])):
        assert "\n" not in html and "    " not in html


# --- gráficos ---------------------------------------------------------------------
def test_ticks_de_fecha_en_espanol_y_sobre_dias_que_existen(m):
    idx = m["px"].index
    for dias, patron in ((92, r"^\d{1,2} [a-z]{3}$"), (365, r"^[a-z]{3} \d{4}$"), (None, r"^\d{4}$")):
        tramo = panel.recortar(m["px"], dias).index
        vals, textos = graficos.ticks_fecha(tramo, 7)
        assert 2 <= len(vals) <= 7 and len(vals) == len(textos)
        assert all(v in idx for v in vals) and vals == sorted(vals)
        assert all(re.match(patron, t) for t in textos), textos
    vals, textos = graficos.ticks_fecha(idx[-3:], 7)                    # rango mínimo: no falla
    assert len(vals) >= 1


def _todas_las_figuras(m) -> dict:
    hoy, px, curvas = m["hoy"], m["px"], m["curvas"]
    v = panel.recortar(panel.velas(panel.ohlc(m["precios"], "SPY")), 183)
    riesgo = panel.contribucion_riesgo(hoy["peso"], px.pct_change().tail(60))
    anual, parciales = panel.retornos_anuales(curvas)
    _, mensual = panel.retornos_mensuales(curvas[NOMBRE_ESTRATEGIA])
    return {
        "velas": graficos.velas(v, "diarias", ("Media 20", "Media 50", "Volumen"), "t", "s"),
        "rendimiento": graficos.rendimiento(panel.base100(panel.recortar(px, 365)), "t", "s"),
        "volatilidad": graficos.volatilidad(m["ind"]["vol_60d"].dropna(), "t", "s"),
        "correlacion": graficos.correlacion(panel.correlaciones(px), "t", "s"),
        "base_vs_final": graficos.base_vs_final(hoy, "t", "s"),
        "peso_vs_riesgo": graficos.peso_vs_riesgo(hoy["peso"], riesgo, "t", "s"),
        "evolucion": graficos.evolucion_pesos(panel.fin_de_mes(m["ancho"]), "t", "s"),
        "capital": graficos.capital(curvas, "t", "s"),
        "caidas": graficos.caidas(panel.drawdowns(curvas), "t", "s"),
        "anuales": graficos.anuales(anual, parciales, "t", "s"),
        "calor": graficos.calor_mensual(mensual, "t", "s"),
    }


def test_todas_las_figuras_se_construyen_con_el_estilo_del_panel(m):
    for nombre, fig in _todas_las_figuras(m).items():
        d = fig.to_dict()
        assert d["data"], nombre
        assert fig.layout.paper_bgcolor == T["superficie"] and fig.layout.plot_bgcolor == T["superficie"], nombre
        assert fig.layout.separators == ",.", nombre                      # coma decimal
        assert "<b>t</b>" in fig.layout.title.text and "s</span>" in fig.layout.title.text, nombre
        # nunca dos escalas distintas sobre la misma zona del gráfico
        assert not any(k.startswith("yaxis") and v.get("overlaying") for k, v in d["layout"].items()), nombre


def test_velas(m):
    v = panel.recortar(panel.velas(panel.ohlc(m["precios"], "QQQ")), 183)
    fig = graficos.velas(v, "diarias", ("Media 20", "Media 50", "Volumen"), "t", "s")
    tipos = [t.type for t in fig.data]
    assert tipos == ["candlestick", "scatter", "scatter", "bar"]
    vela = fig.data[0]
    assert vela.increasing.fillcolor == T["sube"] and vela.decreasing.fillcolor == T["baja"]
    assert list(vela.close) == pytest.approx(list(v["cierre"])) and len(vela.x) == len(set(vela.x)) == len(v)
    assert vela.x[-1] == f.fecha(v.index[-1]) and "Cierre" in vela.text[-1] and "Vol." in vela.text[-1]
    assert fig.layout.xaxis.type == "category" and fig.layout.xaxis.rangeslider.visible is False
    assert set(fig.layout.xaxis.tickvals) <= set(vela.x)                  # las marcas existen en el eje
    assert fig.data[3].yaxis == "y2" and fig.layout.yaxis2.domain[1] < fig.layout.yaxis.domain[0]   # volumen aparte
    sube = v["cierre"] >= v["apertura"]
    assert [("52,211,153" in col) for col in fig.data[3].marker.color] == list(sube)
    assert any(a.text == f.num(v["cierre"].iloc[-1], 2) for a in fig.layout.annotations)       # etiqueta del último precio

    sin_capas = graficos.velas(v, "diarias", (), "t", "s")
    assert [t.type for t in sin_capas.data] == ["candlestick"] and sin_capas.layout.showlegend is False
    mensual = panel.velas(panel.ohlc(m["precios"], "QQQ"), "M")
    assert graficos.velas(mensual, "mensuales", ("Volumen",), "t", "s").data[0].x[-1] == f.mes_anio(mensual.index[-1])


def test_lineas_usan_el_color_del_etf_y_llevan_etiqueta_final(m):
    idx = panel.base100(panel.recortar(m["px"], 365))
    fig = graficos.rendimiento(idx, "t", "s")
    for traza in fig.data:
        assert traza.line.color == tema.color_etf(traza.name) and traza.line.width == 2
    textos = [a.text for a in fig.layout.annotations]
    assert all(any(t.startswith(tk) for t in textos) for tk in TICKERS)
    assert list(fig.layout.xaxis.range) == [idx.index[0], idx.index[-1]]   # las etiquetas quedan en el margen


def test_backtest_resalta_la_estrategia_y_spy_conserva_su_azul(m):
    fig = graficos.capital(m["curvas"], "t", "s")
    por_nombre = {t.name: t for t in fig.data}
    assert por_nombre[NOMBRE_ESTRATEGIA].line.width > max(t.line.width for n, t in por_nombre.items() if n != NOMBRE_ESTRATEGIA)
    assert por_nombre[NOMBRE_ESTRATEGIA].line.color == T["tinta"]
    assert por_nombre[NOMBRE_SPY].line.color == tema.color_etf("SPY")
    assert fig.data[-1].name == NOMBRE_ESTRATEGIA                         # se dibuja encima


def test_mapas_de_calor_rotulan_cada_celda_por_posicion(m):
    _, mensual = panel.retornos_mensuales(m["curvas"][NOMBRE_ESTRATEGIA])
    fig = graficos.calor_mensual(mensual, "t", "s")
    notas = fig.layout.annotations
    assert len(notas) == int(mensual.notna().sum().sum())
    assert all(isinstance(a.x, int) and isinstance(a.y, int) for a in notas)   # un año "2024" se leería como índice
    assert fig.layout.yaxis.type == "category" and fig.data[0].zmin == -fig.data[0].zmax
    corr = panel.correlaciones(m["px"])
    assert len(graficos.correlacion(corr, "t", "s").layout.annotations) == corr.size


def test_aporte_al_riesgo_negativo_cabe_en_el_grafico(m):
    pesos = pd.Series({"SPY": 0.5, "TLT": 0.5})
    fig = graficos.peso_vs_riesgo(pesos, pd.Series({"SPY": 1.08, "TLT": -0.08}), "t", "s")
    assert fig.layout.xaxis.range[0] < -0.08 and fig.layout.xaxis.range[1] > 1.08


def test_marcas_del_eje_de_volumen():
    assert graficos._marcas_volumen(1.05e8) == [5e7, 1e8] and graficos._marcas_volumen(0) == []
    assert [graficos._volumen_corto(x) for x in (5e7, 1e9, 2.5e6, 300)] == ["50 M", "1 mil M", "2,5 M", "300"]
    for maximo in (950, 4.9e6, 7.2e7, 2.6e8, 1.3e9):
        marcas = graficos._marcas_volumen(maximo)
        assert len(marcas) == 2 and marcas[0] < marcas[1] <= maximo * 1.02
