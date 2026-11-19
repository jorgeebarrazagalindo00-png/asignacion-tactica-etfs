"""Gráficos del panel (Plotly).

Criterios comunes: un solo eje por gráfico, rejilla tenue, líneas finas, leyenda
cuando hay dos o más series y etiqueta directa al final de cada línea. El texto
va siempre en color de texto; el color de la serie lo lleva la marca.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import formato as f
from .config import (ETIQUETAS_VARIABLES, NOMBRE_ESTRATEGIA, NOMBRE_ESTRATEGIA_ML, NOMBRE_IGUALES, NOMBRE_SPY,
                     PESO_MAX, PESO_MIN)
from .tema import FUENTE, T, color_etf, rgba

CONFIG = {"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "showTips": False}
MESES_CORTOS = [m.capitalize() for m in f.MESES]

# Estrategias del backtest: la regla del proyecto va en blanco (es el foco), "Solo SPY"
# usa el azul de SPY, "Pesos iguales" un gris acero y la regla con modelo, verde lima.
ESTILO = {
    NOMBRE_SPY: {"color": color_etf("SPY"), "ancho": 1.5},
    NOMBRE_IGUALES: {"color": T["acero"], "ancho": 1.5},
    NOMBRE_ESTRATEGIA: {"color": T["tinta"], "ancho": 2.4},
    NOMBRE_ESTRATEGIA_ML: {"color": T["modelo"], "ancho": 2.0},
}


# --- Piezas comunes ---------------------------------------------------------------
def _lienzo(alto: int, titulo: str = "", sub: str = "", leyenda: bool = True, **margen) -> dict:
    arriba = 16 + (24 if titulo else 0) + (18 if sub else 0) + (30 if leyenda else 8)
    m = {"l": 58, "r": 26, "t": arriba, "b": 42}
    m.update(margen)
    L = dict(
        height=alto, margin=m, separators=",.",
        paper_bgcolor=T["superficie"], plot_bgcolor=T["superficie"],
        font=dict(family=FUENTE, size=12, color=T["tinta2"]),
        hovermode="x unified", showlegend=leyenda,
        hoverlabel=dict(bgcolor=T["superficie2"], bordercolor=T["base"], namelength=-1,
                        font=dict(family=FUENTE, color=T["tinta"], size=12)),
        legend=dict(orientation="h", x=0, xanchor="left", y=1.0, yanchor="bottom", bgcolor="rgba(0,0,0,0)",
                    font=dict(color=T["tinta2"], size=12), itemsizing="constant"),
        xaxis=_eje_x(), yaxis=_eje_y(),
    )
    if titulo:
        texto = f"<b>{titulo}</b>"
        if sub:
            texto += f"<br><span style='font-size:12px;color:{T['tinta2']}'>{sub}</span>"
        L["title"] = dict(text=texto, x=0, xref="container", xanchor="left", y=1, yref="container",
                          yanchor="top", pad=dict(l=20, t=18), font=dict(size=14.5, color=T["tinta"]))
    return L


def _eje_x(**extra) -> dict:
    eje = dict(showgrid=False, linecolor=T["base"], tickcolor=T["base"], ticks="outside", ticklen=4,
               tickfont=dict(color=T["tenue"], size=11.5), fixedrange=True, hoverformat="%d/%m/%Y",
               showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1,
               spikecolor=T["base"], spikedash="solid")
    eje.update(extra)
    return eje


def _eje_y(**extra) -> dict:
    eje = dict(gridcolor=T["rejilla"], gridwidth=1, zeroline=False, showline=False, fixedrange=True,
               tickfont=dict(color=T["tenue"], size=11.5))
    eje.update(extra)
    return eje


def _mezcla(a: str, b: str, t: float) -> str:
    """Color intermedio entre a (t=0) y b (t=1), como lo interpola la escala del mapa de calor."""
    ca, cb = (tuple(int(h.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) for h in (a, b))
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb))


def _luminancia(hex_: str) -> float:
    canales = [int(hex_.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in canales)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def tinta_sobre(fondo: str) -> str:
    """Blanco o negro para rotular dentro de una celda: el que más contraste dé sobre ese fondo.

    Con blanco y negro puros el contraste nunca baja de 4,5:1, sea cual sea el fondo.
    """
    lf = _luminancia(fondo)
    return "#ffffff" if 1.05 / (lf + 0.05) >= (lf + 0.05) / 0.05 else "#000000"


def _color_celda(valor: float, limite: float, negativo: str, positivo: str) -> str:
    t = max(-1.0, min(1.0, valor / limite))
    return _mezcla(T["neutro"], positivo if t >= 0 else negativo, abs(t))


def ticks_fecha(indice, max_ticks: int = 7) -> tuple[list, list[str]]:
    """Marcas del eje de fechas en español, ajustadas a días que existen en los datos.

    Elige el paso (semanas, meses o años) más fino que no pase de `max_ticks` marcas.
    """
    idx = pd.DatetimeIndex(indice)
    ini, fin = idx[0], idx[-1]
    marcas, formato = [], f.dia_mes
    for unidad, paso in (("S", 1), ("S", 2), ("M", 1), ("M", 2), ("M", 3), ("M", 6), ("A", 1), ("A", 2), ("A", 5)):
        if unidad == "S":
            candidatas, fmt = list(pd.date_range(ini, fin, freq="W-MON")[::paso]), f.dia_mes
        elif unidad == "M":
            candidatas = [m for m in pd.date_range(ini, fin, freq="MS") if (m.month - 1) % paso == 0]
            fmt = f.mes_anio
        else:
            candidatas = [m for m in pd.date_range(ini, fin, freq="YS") if m.year % paso == 0]
            fmt = lambda d: str(d.year)                                         # noqa: E731
        if 2 <= len(candidatas) <= max_ticks:
            marcas, formato = candidatas, fmt
            break
    if not marcas:
        marcas, formato = [ini, idx[len(idx) // 2], fin], f.fecha
    valores, textos, usados = [], [], set()
    for m in marcas:
        p = int(idx.searchsorted(m))
        if p < len(idx) and p not in usados:
            usados.add(p)
            valores.append(idx[p])
            textos.append(formato(m))
    return valores, textos


def _etiquetas_finales(fig, finales: dict, textos: dict, colores: dict, x_final, separacion: float) -> None:
    """Etiqueta cada línea en su extremo derecho. Si dos quedan muy juntas se separan
    y un trazo corto del color de la serie las une con su línea."""
    pos = f.separar_etiquetas(finales, separacion)
    for nombre, y in finales.items():
        fig.add_annotation(x=x_final, y=y, xref="x", yref="y", ax=20, axref="pixel", ay=pos[nombre], ayref="y",
                           text=textos[nombre], showarrow=True, arrowhead=0, arrowwidth=1.6,
                           arrowcolor=colores[nombre], xanchor="left", standoff=3,
                           font=dict(color=T["tinta2"], size=11.5, family=FUENTE))


def _eje_fechas(fig, indice, max_ticks: int = 7, **extra) -> None:
    """Marcas en español y rango exacto de los datos (lo anotado al final queda en el margen)."""
    vals, textos = ticks_fecha(indice, max_ticks)
    fig.update_xaxes(tickmode="array", tickvals=vals, ticktext=textos, range=[indice[0], indice[-1]], **extra)


def _lineas_fecha(fig, datos: pd.DataFrame, colores: dict, anchos: dict | None = None, plantilla="%{y:,.1f}") -> None:
    for col in datos.columns:
        fig.add_trace(go.Scatter(
            x=datos.index, y=datos[col], name=col, mode="lines",
            line=dict(width=(anchos or {}).get(col, 2), color=colores[col]),
            hovertemplate=plantilla + "<extra>" + col + "</extra>"))


# --- Mercado -------------------------------------------------------------------------
def _marcas_volumen(maximo: float) -> list[float]:
    """Dos marcas redondas para el eje de volumen (por ejemplo 50 M y 100 M)."""
    if maximo <= 0:
        return []
    paso = 10 ** int(len(str(int(maximo))) - 1)
    for mult in (5, 2, 1):
        if 2 * mult * paso <= maximo * 1.02:
            return [mult * paso, 2 * mult * paso]
    return [paso / 2, paso]


def _volumen_corto(x: float) -> str:
    for divisor, sufijo in ((1e9, " mil M"), (1e6, " M"), (1e3, " k")):
        if x >= divisor:
            return f"{x / divisor:g}".replace(".", ",") + sufijo
    return f"{x:g}"


def velas(v: pd.DataFrame, intervalo: str, capas=("Media 20", "Media 50", "Volumen"),
          titulo: str = "", sub: str = "") -> go.Figure:
    """Velas japonesas con medias móviles y volumen. v: salida de panel.velas()."""
    con_vol = "Volumen" in capas and float(v["volumen"].sum()) > 0
    medias = [(n, c) for n, c in ((20, T["acento"]), (50, T["acero"])) if f"Media {n}" in capas]
    etq = {d: (f.mes_anio(d) if intervalo == "mensuales" else f.fecha(d)) for d in v.index}
    x = list(etq.values())
    sube = (v["cierre"] >= v["apertura"]).to_numpy()

    fig = make_subplots(rows=2 if con_vol else 1, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.77, 0.23] if con_vol else [1.0])
    texto = []
    for (_, r), arriba in zip(v.iterrows(), sube):
        var = f"  {'▲' if r['variacion'] >= 0 else '▼'} {f.pct_signo(r['variacion'], 2)}" if r["variacion"] == r["variacion"] else ""
        vol = f" · Vol. {f.compacto(r['volumen'])}" if r["volumen"] > 0 else ""
        texto.append(f"Apertura {f.num(r['apertura'], 2)} · Máx. {f.num(r['maximo'], 2)} · Mín. {f.num(r['minimo'], 2)}"
                     f" · <b>Cierre {f.num(r['cierre'], 2)}</b>{var}{vol}")
    fig.add_trace(go.Candlestick(
        x=x, open=v["apertura"], high=v["maximo"], low=v["minimo"], close=v["cierre"], name="Precio",
        increasing=dict(line=dict(color=T["sube"], width=1), fillcolor=T["sube"]),
        decreasing=dict(line=dict(color=T["baja"], width=1), fillcolor=T["baja"]),
        whiskerwidth=0, text=texto, hoverinfo="text", showlegend=False), row=1, col=1)
    for n, color in medias:
        fig.add_trace(go.Scatter(x=x, y=v[f"mm{n}"], name=f"Media de {n} velas", mode="lines",
                                 line=dict(color=color, width=1.5),
                                 hovertemplate="%{y:,.2f}<extra>Media " + str(n) + "</extra>"), row=1, col=1)
    if con_vol:
        fig.add_trace(go.Bar(
            x=x, y=v["volumen"], name="Volumen", showlegend=False,
            marker=dict(color=[rgba(T["sube"] if s else T["baja"], 0.55) for s in sube], line=dict(width=0)),
            customdata=[f.compacto(z) for z in v["volumen"]],
            hovertemplate="%{customdata}<extra>Volumen</extra>"), row=2, col=1)

    L = _lienzo(540 if con_vol else 450, titulo, sub, leyenda=bool(medias), l=20, r=74)
    del L["xaxis"], L["yaxis"]
    fig.update_layout(**L, bargap=0.25, xaxis_rangeslider_visible=False)
    vals, textos = ticks_fecha(v.index, 8)
    fig.update_xaxes(**_eje_x(type="category", tickmode="array", tickvals=[etq[d] for d in vals], ticktext=textos))
    fig.update_xaxes(showticklabels=False, ticks="", showline=not con_vol, row=1, col=1)
    fig.update_xaxes(showticklabels=True, ticks="outside", showline=True, row=2 if con_vol else 1, col=1)
    fig.update_yaxes(**_eje_y(side="right", tickformat=",.2f"), row=1, col=1)
    if con_vol:
        marcas = _marcas_volumen(float(v["volumen"].max()))
        fig.update_yaxes(**_eje_y(side="right", showgrid=False, tickmode="array", tickvals=marcas,
                                  ticktext=[_volumen_corto(m) for m in marcas]), row=2, col=1)

    ultimo = float(v["cierre"].iloc[-1])
    fig.add_hline(y=ultimo, line=dict(color=rgba(T["acento"], 0.75), width=1, dash="dot"), row=1, col=1)
    fig.add_annotation(xref="paper", x=1, xanchor="left", xshift=5, yref="y", y=ultimo, showarrow=False,
                       text=f.num(ultimo, 2), bgcolor=T["acento"], borderpad=3,
                       font=dict(color=T["sobre_acento"], size=11.5, family=FUENTE))
    return fig


def rendimiento(idx: pd.DataFrame, titulo: str, sub: str) -> go.Figure:
    """Precios en base 100: todos los ETFs en un mismo eje y comparables."""
    fig = go.Figure()
    colores = {tk: color_etf(tk) for tk in idx.columns}
    _lineas_fecha(fig, idx, colores)
    fig.add_hline(y=100, line=dict(color=T["base"], width=1))
    fig.update_layout(**_lienzo(390, titulo, sub, r=96))
    _eje_fechas(fig, idx.index, 6)
    finales = {tk: float(idx[tk].iloc[-1]) for tk in idx.columns}
    rango = (float(idx.max().max()) - float(idx.min().min())) or 1.0
    _etiquetas_finales(fig, finales, {tk: f"{tk}  {f.num(v, 0)}" for tk, v in finales.items()}, colores,
                       idx.index[-1], rango * 0.062)
    return fig


def volatilidad(vol: pd.DataFrame, titulo: str, sub: str) -> go.Figure:
    fig = go.Figure()
    colores = {tk: color_etf(tk) for tk in vol.columns}
    _lineas_fecha(fig, vol, colores, plantilla="%{y:.1%}")
    fig.update_layout(**_lienzo(360, titulo, sub, r=100))
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    _eje_fechas(fig, vol.index, 7)
    finales = {tk: float(vol[tk].iloc[-1]) for tk in vol.columns}
    _etiquetas_finales(fig, finales, {tk: f"{tk}  {f.pct(v, 0)}" for tk, v in finales.items()}, colores,
                       vol.index[-1], float(vol.max().max()) * 0.07)
    return fig


def correlacion(corr: pd.DataFrame, titulo: str, sub: str) -> go.Figure:
    """Mapa de calor: azul = se mueven juntos, ámbar = se mueven al revés, neutro = sin relación."""
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1, xgap=2, ygap=2,
        colorscale=[[0.0, T["polo_calido"]], [0.5, T["neutro"]], [1.0, T["polo_frio"]]],
        colorbar=dict(thickness=8, len=0.86, outlinewidth=0, tickvals=[-1, 0, 1],
                      ticktext=["−1", "0", "+1"], tickfont=dict(color=T["tenue"], size=11)),
        hovertemplate="%{y} y %{x}: %{z:.2f}<extra></extra>"))
    L = _lienzo(390, titulo, sub, leyenda=False, l=52, r=16, b=22)
    L["hovermode"] = "closest"
    fig.update_layout(**L)
    fig.update_xaxes(**_eje_x(showspikes=False, ticks="", showline=False, side="bottom", type="category",
                              tickfont=dict(color=T["tinta2"], size=12)))
    fig.update_yaxes(**_eje_y(showgrid=False, autorange="reversed", type="category", ticks="",
                              tickfont=dict(color=T["tinta2"], size=12)))
    for i, fila in enumerate(corr.index):
        for j, col in enumerate(corr.columns):
            v = float(corr.iloc[i, j])
            fondo = _color_celda(v, 1.0, T["polo_calido"], T["polo_frio"])
            fig.add_annotation(x=j, y=i, text=f.num(v, 2), showarrow=False,
                               font=dict(color=tinta_sobre(fondo), size=12, family=FUENTE))
    return fig


def tecnicos(t: pd.DataFrame, titulo: str, sub: str) -> go.Figure:
    """RSI arriba y MACD abajo, con el mismo eje de fechas. t: salida de panel.tecnicos()."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09, row_heights=[0.45, 0.55])
    fig.add_trace(go.Scatter(x=t.index, y=t["rsi"], name="RSI (14 días)", mode="lines",
                             line=dict(color=T["tinta"], width=1.6),
                             hovertemplate="%{y:.1f}<extra>RSI</extra>"), row=1, col=1)
    for nivel in (30, 70):
        fig.add_hline(y=nivel, line=dict(color=T["base"], width=1, dash="dot"), row=1, col=1)
        fig.add_annotation(xref="paper", x=1, xanchor="left", xshift=6, yref="y", y=nivel, showarrow=False,
                           text=str(nivel), font=dict(color=T["tenue"], size=11, family=FUENTE))
    sube = (t["hist"] >= 0).to_numpy()
    fig.add_trace(go.Bar(x=t.index, y=t["hist"], name="Histograma (MACD − señal)", showlegend=False,
                         marker=dict(color=[rgba(T["sube"] if s else T["baja"], 0.55) for s in sube], line=dict(width=0)),
                         hovertemplate="%{y:+.3f}<extra>Histograma</extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t.index, y=t["macd"], name="MACD (12, 26)", mode="lines",
                             line=dict(color=T["polo_frio"], width=1.6),
                             hovertemplate="%{y:+.3f}<extra>MACD</extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t.index, y=t["senal"], name="Señal (9)", mode="lines",
                             line=dict(color=T["acento"], width=1.4),
                             hovertemplate="%{y:+.3f}<extra>Señal</extra>"), row=2, col=1)
    L = _lienzo(430, titulo, sub, r=44)
    del L["xaxis"], L["yaxis"]
    fig.update_layout(**L, bargap=0.15)
    vals, textos = ticks_fecha(t.index, 7)
    fig.update_xaxes(**_eje_x(tickmode="array", tickvals=vals, ticktext=textos, range=[t.index[0], t.index[-1]]))
    fig.update_xaxes(showticklabels=False, ticks="", row=1, col=1)
    fig.update_yaxes(**_eje_y(range=[0, 100], tickvals=[0, 50, 100], title=dict(text="RSI", font=dict(size=11, color=T["tenue"]))),
                     row=1, col=1)
    fig.update_yaxes(**_eje_y(zeroline=True, zerolinecolor=T["base"], zerolinewidth=1,
                              title=dict(text="MACD (USD)", font=dict(size=11, color=T["tenue"]))), row=2, col=1)
    return fig


def histograma(h: pd.DataFrame, media: float, color: str, titulo: str, sub: str) -> go.Figure:
    """Distribución de los retornos diarios. h: salida de panel.histograma()."""
    fig = go.Figure(go.Bar(
        x=h["centro"], y=h["dias"], width=(h["hasta"] - h["desde"]) * 0.92,
        marker=dict(color=color, line=dict(width=0)),
        customdata=[[f.pct_signo(a, 2), f.pct_signo(b, 2)] for a, b in zip(h["desde"], h["hasta"])],
        hovertemplate="De %{customdata[0]} a %{customdata[1]}: %{y} días<extra></extra>"))
    L = _lienzo(360, titulo, sub, leyenda=False)
    L["hovermode"] = "closest"
    fig.update_layout(**L)
    fig.update_xaxes(**_eje_x(showspikes=False, tickformat="+.1%", zeroline=True, zerolinecolor=T["base"], zerolinewidth=1))
    fig.update_yaxes(**_eje_y(rangemode="tozero"))
    fig.add_vline(x=media, line=dict(color=T["tinta"], width=1.4, dash="dot"))
    fig.add_annotation(x=media, y=1, yref="paper", yanchor="bottom", showarrow=False,
                       text=f"media {f.pct_signo(media, 2)}", font=dict(color=T["tinta2"], size=11, family=FUENTE))
    return fig


# --- Predictivo ----------------------------------------------------------------------
def probabilidades(hoy: pd.DataFrame, principal: str, otro: str | None, nombres: dict, titulo: str, sub: str) -> go.Figure:
    """Probabilidad de subida por ETF: barra para el modelo principal y rombo para el segundo modelo."""
    d = hoy.sort_values(principal, ascending=False)
    tks = list(d.index)
    fig = go.Figure(go.Bar(
        y=tks, x=d[principal], orientation="h", name=nombres[principal],
        marker=dict(color=[color_etf(t) for t in tks], cornerradius=4, line=dict(width=0)),
        text=[f.pct(v, 0) for v in d[principal]], textposition="outside", cliponaxis=False,
        textfont=dict(color=T["tinta"], size=12, family=FUENTE),
        hovertemplate="%{y}: %{x:.1%}<extra>" + nombres[principal] + "</extra>"))
    if otro and otro in d:
        fig.add_trace(go.Scatter(
            y=tks, x=d[otro], mode="markers", name=nombres[otro],
            marker=dict(symbol="diamond", size=11, color=T["superficie"], line=dict(color=T["tinta"], width=1.8)),
            hovertemplate="%{y}: %{x:.1%}<extra>" + nombres[otro] + "</extra>"))
    L = _lienzo(340, titulo, sub, leyenda=False, l=54, r=40)      # el subtítulo ya dice qué es la barra y qué el rombo
    L["margin"]["t"] += 16
    L["hovermode"] = "closest"
    fig.update_layout(**L, bargap=0.42)
    fig.update_xaxes(**_eje_x(showspikes=False, showgrid=True, gridcolor=T["rejilla"], tickformat=".0%",
                              range=[0, 1.04], dtick=0.25))
    fig.update_yaxes(**_eje_y(showgrid=False, autorange="reversed", ticksuffix="   ",
                              tickfont=dict(color=T["tinta"], size=12.5)))
    fig.add_vline(x=0.5, line=dict(color=T["borde_fuerte"], width=1, dash="dot"))
    fig.add_annotation(x=0.5, y=1, yref="paper", yanchor="bottom", text="50 %: no se inclina", showarrow=False,
                       font=dict(color=T["tenue"], size=11, family=FUENTE))
    return fig


def calibracion(tabla: pd.DataFrame, titulo: str, sub: str) -> go.Figure:
    """Probabilidad anunciada frente a frecuencia observada; sobre la diagonal, el modelo está bien calibrado."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibración perfecta", hoverinfo="skip",
                             line=dict(color=T["acero"], width=1.2, dash="dot")))
    fig.add_trace(go.Scatter(
        x=tabla["prob_media"], y=tabla["frecuencia"], mode="lines+markers", name="Modelo",
        line=dict(color=T["tinta"], width=2), marker=dict(size=9, color=T["tinta"], line=dict(color=T["superficie"], width=2)),
        customdata=tabla["n"],
        hovertemplate="Dijo %{x:.0%} · subió el %{y:.0%} de las veces (%{customdata} casos)<extra></extra>"))
    L = _lienzo(340, titulo, sub, l=78, b=58)
    L["hovermode"] = "closest"
    fig.update_layout(**L)
    lo = max(0.0, float(min(tabla["prob_media"].min(), tabla["frecuencia"].min())) - 0.1)
    hi = min(1.0, float(max(tabla["prob_media"].max(), tabla["frecuencia"].max())) + 0.1)
    fig.update_xaxes(**_eje_x(showspikes=False, showgrid=True, gridcolor=T["rejilla"], tickformat=".0%", range=[lo, hi],
                              title=dict(text="Probabilidad que dio el modelo", standoff=10,
                                         font=dict(size=11.5, color=T["tenue"]))))
    fig.update_yaxes(**_eje_y(tickformat=".0%", range=[lo, hi],
                              title=dict(text="Veces que subió", standoff=12, font=dict(size=11.5, color=T["tenue"]))))
    return fig


def importancia(imp: pd.DataFrame, con_signo: bool, titulo: str, sub: str) -> go.Figure:
    """Barras por variable. Con signo (regresión logística): azul empuja hacia «sube», ámbar hacia «no sube»."""
    d = imp.assign(orden=imp["importancia"].abs()).sort_values("orden")
    etiquetas = [ETIQUETAS_VARIABLES.get(v, v) for v in d["variable"]]
    if con_signo:
        colores = [T["polo_frio"] if v >= 0 else T["polo_calido"] for v in d["importancia"]]
        textos = [f.num_signo(v, 2) for v in d["importancia"]]
    else:
        colores, textos = T["acero"], [f.pct(v, 0) for v in d["importancia"]]
    fig = go.Figure(go.Bar(
        y=etiquetas, x=d["importancia"], orientation="h", marker=dict(color=colores, cornerradius=3, line=dict(width=0)),
        text=textos, textposition="outside", cliponaxis=False, textfont=dict(color=T["tinta2"], size=11.5, family=FUENTE),
        hovertemplate="%{y}: %{text}<extra></extra>"))
    L = _lienzo(380, titulo, sub, leyenda=False, l=190, r=48)
    L["hovermode"] = "closest"
    fig.update_layout(**L, bargap=0.35)
    tope = float(d["importancia"].abs().max()) * 1.4 or 1.0
    fig.update_xaxes(**_eje_x(showspikes=False, showgrid=True, gridcolor=T["rejilla"], showticklabels=False, ticks="",
                              range=[-tope if con_signo else 0, tope], zeroline=con_signo, zerolinecolor=T["base"],
                              zerolinewidth=1))
    fig.update_yaxes(**_eje_y(showgrid=False, ticksuffix="   ", tickfont=dict(color=T["tinta2"], size=12)))
    return fig


# --- Asignación ----------------------------------------------------------------------
def base_vs_final(hoy: pd.DataFrame, titulo: str, sub: str) -> go.Figure:
    """Del peso por riesgo (círculo hueco) al peso final (círculo lleno), por ETF."""
    d = hoy.sort_values("peso", ascending=False)
    tks = list(d.index)
    fig = go.Figure()
    for tk, r in d.iterrows():
        fig.add_trace(go.Scatter(x=[r["peso_base"], r["peso"]], y=[tk, tk], mode="lines", hoverinfo="skip",
                                 showlegend=False, line=dict(color=rgba(color_etf(tk), 0.55), width=3)))
    fig.add_trace(go.Scatter(
        x=d["peso_base"], y=tks, mode="markers", name="Solo por riesgo",
        marker=dict(size=11, color=T["superficie"], line=dict(color=T["acero"], width=2)),
        hovertemplate="%{y}: %{x:.1%}<extra>Solo por riesgo</extra>"))
    fig.add_trace(go.Scatter(
        x=d["peso"], y=tks, mode="markers+text", showlegend=False,
        marker=dict(size=14, color=[color_etf(t) for t in tks], line=dict(color=T["superficie"], width=2)),
        text=[f.pct(v) for v in d["peso"]],
        textposition=["middle right" if r["peso"] >= r["peso_base"] else "middle left" for _, r in d.iterrows()],
        textfont=dict(color=T["tinta"], size=12, family=FUENTE), cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%}<extra>Peso final</extra>"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Peso final",
                             marker=dict(size=12, color=T["tinta"])))
    L = _lienzo(360, titulo, sub, l=54, r=30, b=74)
    L["margin"]["t"] -= 6
    L["hovermode"] = "closest"
    L["legend"].update(y=-0.17, yanchor="top")
    fig.update_layout(**L)
    tope = max(PESO_MAX + 0.06, float(d[["peso", "peso_base"]].max().max()) + 0.06)
    fig.update_xaxes(**_eje_x(showspikes=False, showgrid=True, gridcolor=T["rejilla"], tickformat=".0%",
                              range=[0, tope], dtick=0.1))
    fig.update_yaxes(**_eje_y(showgrid=False, autorange="reversed", ticksuffix="   ",
                              tickfont=dict(color=T["tinta"], size=12.5)))
    for x, texto in ((PESO_MIN, f"mínimo {f.pct(PESO_MIN, 0)}"), (PESO_MAX, f"tope {f.pct(PESO_MAX, 0)}")):
        fig.add_vline(x=x, line=dict(color=T["base"], width=1))
        fig.add_annotation(x=x, y=1, yref="paper", yanchor="bottom", text=texto, showarrow=False,
                           font=dict(color=T["tenue"], size=11, family=FUENTE))
    return fig


def peso_vs_riesgo(pesos: pd.Series, riesgo: pd.Series, titulo: str, sub: str) -> go.Figure:
    """Dos barras por ETF en el mismo eje (%): cuánto pesa y cuánto riesgo aporta."""
    orden = list(pesos.sort_values(ascending=False).index)
    fig = go.Figure()
    for nombre, serie, color in (("Peso en la cartera", pesos, T["acero"]),
                                 ("Aporte al riesgo", riesgo, T["acento"])):
        s = serie.reindex(orden)
        fig.add_trace(go.Bar(
            y=orden, x=s, orientation="h", name=nombre, marker=dict(color=color, cornerradius=4, line=dict(width=0)),
            text=[f.pct(v) for v in s], textposition="outside", cliponaxis=False,
            textfont=dict(color=T["tinta2"], size=11.5, family=FUENTE),
            hovertemplate="%{y}: %{x:.1%}<extra>" + nombre + "</extra>"))
    L = _lienzo(360, titulo, sub, l=54, r=30)
    L["hovermode"] = "closest"
    fig.update_layout(**L, barmode="group", bargap=0.34, bargroupgap=0.12)
    tope = max(float(pesos.max()), float(riesgo.max())) + 0.09
    piso = min(0.0, float(riesgo.min()) - 0.07)
    fig.update_xaxes(**_eje_x(showspikes=False, showgrid=True, gridcolor=T["rejilla"], tickformat=".0%",
                              range=[piso, tope], zeroline=piso < 0, zerolinecolor=T["base"], zerolinewidth=1))
    fig.update_yaxes(**_eje_y(showgrid=False, autorange="reversed", ticksuffix="   ",
                              tickfont=dict(color=T["tinta"], size=12.5)))
    return fig


def evolucion_pesos(mensual: pd.DataFrame, titulo: str, sub: str) -> go.Figure:
    """Área apilada al 100 %: el reparto sugerido al cierre de cada mes."""
    fig = go.Figure()
    for tk in mensual.columns:
        fig.add_trace(go.Scatter(
            x=mensual.index, y=mensual[tk], name=tk, mode="lines", stackgroup="uno",
            line=dict(width=2, color=T["superficie"]), fillcolor=color_etf(tk),
            hovertemplate="%{y:.1%}<extra>" + tk + "</extra>"))
    L = _lienzo(380, titulo, sub, r=58)
    L["legend"]["traceorder"] = "normal"
    fig.update_layout(**L)
    fig.update_yaxes(tickformat=".0%", range=[0, 1], dtick=0.25)
    _eje_fechas(fig, mensual.index, 7, hoverformat="%m/%Y")
    acumulado, ultimo = 0.0, mensual.iloc[-1]
    for tk in mensual.columns:
        if float(ultimo[tk]) >= 0.045:                        # la etiqueta solo si la franja tiene alto para ella
            fig.add_annotation(xref="paper", x=1, xanchor="left", xshift=8, yref="y",
                               y=acumulado + float(ultimo[tk]) / 2, text=tk, showarrow=False,
                               font=dict(color=T["tinta2"], size=11.5, family=FUENTE))
        acumulado += float(ultimo[tk])
    return fig


# --- Backtest ------------------------------------------------------------------------
def _orden_estrategias(columnas) -> list[str]:
    return [n for n in ESTILO if n in columnas]


def capital(curvas: pd.DataFrame, titulo: str, sub: str, alto: int = 420) -> go.Figure:
    """Crecimiento de 100 invertidos en cada estrategia."""
    nombres = _orden_estrategias(curvas.columns)
    fig = go.Figure()
    _lineas_fecha(fig, curvas[nombres], {n: ESTILO[n]["color"] for n in nombres},
                  {n: ESTILO[n]["ancho"] for n in nombres})
    fig.add_hline(y=100, line=dict(color=T["base"], width=1))
    fig.update_layout(**_lienzo(alto, titulo, sub, r=172))
    _eje_fechas(fig, curvas.index, 7)
    finales = {n: float(curvas[n].iloc[-1]) for n in nombres}
    rango = (float(curvas[nombres].max().max()) - float(curvas[nombres].min().min())) or 1.0
    _etiquetas_finales(fig, finales, {n: f"{n}  {f.num(v, 0)}" for n, v in finales.items()},
                       {n: ESTILO[n]["color"] for n in nombres}, curvas.index[-1], rango * 0.07 * 420 / alto)
    return fig


def caidas(dd: pd.DataFrame, titulo: str, sub: str) -> go.Figure:
    """Caída desde el máximo anterior; la estrategia va sombreada."""
    nombres = _orden_estrategias(dd.columns)
    fig = go.Figure()
    for n in nombres:
        principal = n == NOMBRE_ESTRATEGIA
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd[n], name=n, mode="lines",
            line=dict(color=ESTILO[n]["color"], width=1.8 if principal else 1.3),
            fill="tozeroy" if principal else None, fillcolor=rgba(T["tinta"], 0.10),
            hovertemplate="%{y:.1%}<extra>" + n + "</extra>"))
    fig.update_layout(**_lienzo(330, titulo, sub))
    fig.update_yaxes(tickformat=".0%")
    _eje_fechas(fig, dd.index, 6)
    return fig


def anuales(tabla: pd.DataFrame, parciales: set, titulo: str, sub: str) -> go.Figure:
    """Retorno por año calendario, una barra por estrategia. Los años incompletos llevan *."""
    nombres = [n for n in (NOMBRE_ESTRATEGIA, NOMBRE_ESTRATEGIA_ML, NOMBRE_IGUALES, NOMBRE_SPY) if n in tabla.columns]
    x = [f"{a}*" if a in parciales else str(a) for a in tabla.index]
    fig = go.Figure()
    for n in nombres:
        fig.add_trace(go.Bar(x=x, y=tabla[n], name=n,
                             marker=dict(color=ESTILO[n]["color"], cornerradius=3, line=dict(width=0)),
                             hovertemplate="%{y:.1%}<extra>" + n + "</extra>"))
    fig.update_layout(**_lienzo(360, titulo, sub), barmode="group", bargap=0.3, bargroupgap=0.12)
    fig.update_xaxes(**_eje_x(type="category", showspikes=False))
    fig.update_yaxes(tickformat=".0%", zeroline=True, zerolinecolor=T["base"], zerolinewidth=1)
    return fig


def calor_mensual(tabla: pd.DataFrame, titulo: str, sub: str) -> go.Figure:
    """Retorno de cada mes: verde si ganó, rojo si perdió, neutro cerca de cero."""
    z = tabla.to_numpy(dtype=float) * 100
    validos = pd.Series(z.ravel()).dropna().abs()
    limite = max(float(validos.quantile(0.95)) if len(validos) else 1.0, 1.0)
    anios = [str(a) for a in tabla.index]
    fig = go.Figure(go.Heatmap(
        z=z, x=MESES_CORTOS, y=anios, zmin=-limite, zmax=limite, xgap=2, ygap=2, hoverongaps=False,
        colorscale=[[0.0, T["baja"]], [0.5, T["neutro"]], [1.0, T["sube"]]],
        colorbar=dict(thickness=8, len=0.86, outlinewidth=0, ticksuffix=" %", tickfont=dict(color=T["tenue"], size=11)),
        hovertemplate="%{x} %{y}: %{z:+.2f} %<extra></extra>"))
    L = _lienzo(96 + 30 * len(anios), titulo, sub, leyenda=False, l=52, r=16, b=30)
    L["hovermode"] = "closest"
    fig.update_layout(**L)
    fig.update_xaxes(**_eje_x(showspikes=False, ticks="", showline=False, side="bottom", type="category",
                              tickfont=dict(color=T["tinta2"], size=11.5)))
    fig.update_yaxes(**_eje_y(showgrid=False, autorange="reversed", type="category", ticks="",
                              tickfont=dict(color=T["tinta2"], size=11.5)))
    for i, anio in enumerate(anios):
        for j, mes in enumerate(MESES_CORTOS):
            v = z[i, j]
            if v == v:
                fondo = _color_celda(v, limite, T["baja"], T["sube"])
                fig.add_annotation(x=j, y=i, text=f.num_signo(v, 1), showarrow=False,
                                   font=dict(color=tinta_sobre(fondo), size=11, family=FUENTE))
    return fig
