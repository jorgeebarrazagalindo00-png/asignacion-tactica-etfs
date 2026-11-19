"""Piezas de HTML del panel (cabecera, indicadores, tablas, fichas).

Todas devuelven texto HTML y escapan cualquier dato que venga de la base, así que
se pueden probar sin Streamlit.
"""

import html
import re

from . import formato as f
from .config import PESO_MAX
from .tema import T, color_etf, rgba

FLECHA = {"sube": "▲", "baja": "▼", "plano": "■"}
CLASE = {"sube": "up", "baja": "down", "plano": "plano"}


def e(x) -> str:
    return html.escape(str(x), quote=True)


def compacto(s: str) -> str:
    """Una sola línea sin sangrías: así Markdown no confunde el HTML con un bloque de código."""
    return " ".join(linea.strip() for linea in s.splitlines() if linea.strip())


def negritas(s: str) -> str:
    """Escapa el texto y convierte **así** en <b>así</b>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", e(s))


def delta(valor, texto: str, bueno_si_sube: bool = True, flecha: bool = True, neutro: bool = False) -> str:
    """Variación con flecha y color.

    El color dice si el cambio es favorable (no solo si sube): una volatilidad que baja
    va en verde. Con neutro=True no se colorea (un cambio de peso no es bueno ni malo).
    """
    d = f.direccion(valor)
    clase = "plano" if (d == "plano" or neutro) else ("up" if (d == "sube") == bueno_si_sube else "down")
    marca = f"{FLECHA[d]} " if flecha else ""
    return f'<span class="{clase} mt-num">{marca}{e(texto)}</span>'


def sparkline(valores, color: str, ancho: int = 92, alto: int = 28) -> str:
    """Mini gráfico de línea en SVG con un punto en el último valor."""
    v = [float(x) for x in valores if x == x]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    rango = (hi - lo) or 1.0
    m = 4
    xs = [m + i * (ancho - 2 * m) / (len(v) - 1) for i in range(len(v))]
    ys = [alto - m - (y - lo) / rango * (alto - 2 * m) for y in v]
    puntos = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return (f'<svg width="{ancho}" height="{alto}" viewBox="0 0 {ancho} {alto}" role="img" aria-hidden="true">'
            f'<polyline points="{puntos}" fill="none" stroke="{color}" stroke-width="1.6" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
            f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="3" fill="{color}" stroke="{T["superficie"]}" stroke-width="1.5"/></svg>')


def etiqueta_etf(ticker: str) -> str:
    return f'<span class="mt-tk"><i class="mt-sw" style="background:{color_etf(ticker)}"></i>{e(ticker)}</span>'


# --- Cabecera ------------------------------------------------------------------------
def _logo() -> str:
    return ('<svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">'
            f'<path d="M12 3a9 9 0 0 1 9 9" stroke="{T["acento"]}" stroke-width="2.6" stroke-linecap="round"/>'
            f'<path d="M20.5 15.6A9 9 0 0 1 7.6 19.9" stroke="{T["tinta"]}" stroke-width="2.6" stroke-linecap="round"/>'
            f'<path d="M4.4 16.6A9 9 0 0 1 8.4 3.7" stroke="{T["acero"]}" stroke-width="2.6" stroke-linecap="round"/></svg>')


def cabecera(titulo: str, kicker: str, lema: str, chips: list[str], linea: bool = True) -> str:
    """chips: fragmentos HTML ya armados (ver `chip`). linea=False quita la línea inferior, cuando la cabecera
    comparte fila con otros elementos y la línea se dibuja aparte."""
    return compacto(f"""
        <div class="mt-top{'' if linea else ' sin-linea'}">
          <div class="mt-marca">
            <div class="mt-logo">{_logo()}</div>
            <div>
              <div class="mt-kicker">{e(kicker)}</div>
              <div class="mt-titulo">{e(titulo)}</div>
              <div class="mt-lema">{e(lema)}</div>
            </div>
          </div>
          <div class="mt-estado">{''.join(chips)}</div>
        </div>""")


def chip(texto: str, valor: str = "", estado: str | None = None) -> str:
    """Píldora de estado. estado: 'ok' o 'aviso' dibuja un punto de color junto al texto."""
    punto = f'<i class="mt-punto{" aviso" if estado == "aviso" else ""}"></i>' if estado else ""
    fuerte = f" <b>{e(valor)}</b>" if valor else ""
    return f'<span class="mt-chip">{punto}{e(texto)}{fuerte}</span>'


def kpis(items: list[dict], pie: str = "", columnas: int | None = None) -> str:
    """Fila de indicadores. items: {'etiqueta', 'valor', 'detalle' (HTML), 'grafico' (SVG opcional)}."""
    celdas = ""
    for it in items:
        grafico = it.get("grafico") or ""
        celdas += (f'<div class="mt-kpi"><div class="l">{e(it["etiqueta"])}</div>'
                   f'<div class="fila"><div class="v">{e(it["valor"])}</div>{grafico}</div>'
                   f'<div class="d">{it.get("detalle", "")}</div></div>')
    estilo = f' style="grid-template-columns:repeat({columnas}, minmax(0, 1fr))"' if columnas else ""
    cierre = f'<div class="mt-pie">{e(pie)}</div>' if pie else ""
    return f'<div class="mt-kpis"{estilo}>{celdas}</div>{cierre}'


def seccion(titulo: str, texto: str = "", kicker: str = "") -> str:
    k = f'<div class="k">{e(kicker)}</div>' if kicker else ""
    p = f"<p>{e(texto)}</p>" if texto else ""
    return f'<div class="mt-sec">{k}<h2>{e(titulo)}</h2>{p}</div>'


def nota(texto: str) -> str:
    return f'<div class="mt-nota">{negritas(texto)}</div>'


# --- Tablas --------------------------------------------------------------------------
def tabla(columnas: list[str], filas: list[list[str]], izquierda=(0,), texto=(), tenue=(),
          alto_max: int | None = None, ayudas: dict | None = None) -> str:
    """Tabla con números alineados. Las celdas llegan ya escapadas (o con HTML propio).

    izquierda / texto / tenue: índices de columna alineados a la izquierda, con texto
    largo (pueden partir línea) o en color secundario. ayudas: {columna: texto} que aparece
    al pasar el mouse sobre ese encabezado.
    """
    ayudas = ayudas or {}
    def clase(i: int, es_td: bool) -> str:
        c = []
        if i in izquierda or i in texto:
            c.append("izq")
        if es_td and i in texto:
            c.append("txt")
        if es_td and i in tenue:
            c.append("tenue")
        return f' class="{" ".join(c)}"' if c else ""

    def th(i: int, c: str) -> str:
        if c not in ayudas:
            return f"<th{clase(i, False)}>{e(c)}</th>"
        cl = clase(i, False)
        cl = cl.replace('class="', 'class="ayuda ') if cl else ' class="ayuda"'
        return f'<th{cl} title="{e(ayudas[c])}">{e(c)}</th>'

    cab = "".join(th(i, c) for i, c in enumerate(columnas))
    cuerpo = "".join("<tr>" + "".join(f"<td{clase(i, True)}>{v}</td>" for i, v in enumerate(fila)) + "</tr>"
                     for fila in filas)
    estilo = f' style="max-height:{alto_max}px"' if alto_max else ""
    return (f'<div class="mt-tabla-caja"{estilo}><table class="mt-tabla"><thead><tr>{cab}</tr></thead>'
            f"<tbody>{cuerpo}</tbody></table></div>")


# --- Resumen: asignación y lectura --------------------------------------------------------
def barra_segmentada(partes: list[tuple[str, float, str]]) -> str:
    """Un todo partido en bloques. partes: (nombre, proporción 0-1, color).

    El porcentaje solo se imprime dentro del bloque si cabe; siempre queda en la tabla
    de abajo y en el texto alternativo de la barra.
    """
    bloques = ""
    for nombre, valor, color in partes:
        if valor <= 0:
            continue
        texto = f.pct(valor) if valor >= 0.075 else ""
        angosto = ' class="angosto"' if valor < 0.16 else ""      # en pantallas chicas estos no llevan texto
        bloques += (f'<div{angosto} style="flex:{valor:.6f} 1 0;background:{color}" '
                    f'title="{e(nombre)}: {e(f.pct(valor))}">{e(texto)}</div>')
    alt = ", ".join(f"{n} {f.pct(v)}" for n, v, _ in partes)
    return f'<div class="mt-seg" role="img" aria-label="{e(alt)}">{bloques}</div>'


def tarjeta_asignacion(hoy, nombres: dict, clases: dict, cambio, fecha_texto: str) -> str:
    """hoy: DataFrame por ticker con peso, peso_base y factor_momentum."""
    orden = hoy.sort_values("peso", ascending=False)
    barra = barra_segmentada([(tk, float(r["peso"]), color_etf(tk)) for tk, r in orden.iterrows()])
    filas = []
    for tk, r in orden.iterrows():
        c = color_etf(tk)
        ancho = min(float(r["peso"]) / PESO_MAX, 1.0) * 100
        medidor = (f'<div class="mt-medidor" style="background:{rgba(c, 0.20)}">'
                   f'<i style="width:{ancho:.1f}%;background:{c}"></i></div>')
        if cambio is None:
            var = '<span class="plano">—</span>'
        elif f.direccion(cambio.get(tk), 0.0005) == "plano":
            var = '<span class="plano mt-num">sin cambio</span>'
        else:
            var = delta(cambio.get(tk), f.pp(cambio.get(tk)), neutro=True)
        filas.append([etiqueta_etf(tk), e(clases.get(tk, nombres.get(tk, ""))), medidor,
                      f'<b>{e(f.pct(r["peso"]))}</b>', var])
    cuerpo = tabla(["ETF", "Clase de activo", f"Uso del tope ({f.pct(PESO_MAX, 0)})", "Peso", "Frente a hace 1 mes"],
                   filas, izquierda=(0, 1, 2), tenue=(1,))
    return compacto(f"""
        <div class="mt-card">
          <h3>Reparto de la regla</h3>
          <div class="sub">Antes de ajustar al perfil de riesgo · datos al {e(fecha_texto)}</div>
          {barra}
          <div style="height:.9rem"></div>
          {cuerpo}
        </div>""")


def tarjeta_lectura(frases: list[str], proximo: str, dias_faltan: int | None = None) -> str:
    items = "".join(f'<li><span class="n">{i}</span><span>{negritas(t)}</span></li>'
                    for i, t in enumerate(frases, 1))
    falta = ""
    if dias_faltan is not None:
        falta = " (hoy)" if dias_faltan <= 0 else f" (en {dias_faltan} día{'s' if dias_faltan != 1 else ''})"
    return compacto(f"""
        <div class="mt-card">
          <h3>Lectura del día</h3>
          <div class="sub">Por qué queda así el reparto, explicado con los datos</div>
          <ul class="mt-lect">{items}</ul>
          <div class="mt-prox"><span>Próxima decisión de rebalanceo</span><b>{e(proximo)}{e(falta)}</b></div>
        </div>""")


# --- Mercado: ficha del ETF ----------------------------------------------------------------
def _dato(etiqueta: str, valor_html: str) -> str:
    return f'<div class="dato"><span>{e(etiqueta)}</span><b>{valor_html}</b></div>'


def ficha_etf(ticker: str, nombre: str, clase: str, r) -> str:
    """r: fila de panel.resumen_universo para ese ticker."""
    hoy = delta(r["var_dia"], f"{f.pct_signo(r['var_dia'], 2)} en el día")
    ancho = max(r["max_52"] - r["min_52"], 1e-9)
    pos = min(max((r["cierre"] - r["min_52"]) / ancho, 0.0), 1.0) * 100

    def var(x):
        return delta(x, f.pct_signo(x), flecha=False)

    return compacto(f"""
        <div class="mt-etf">
          <div class="id">
            <i class="barra" style="background:{color_etf(ticker)}"></i>
            <div><div class="tk">{e(ticker)}</div><div class="nm">{e(nombre)} · {e(clase)}</div></div>
          </div>
          <div class="px"><div class="v mt-num">{e(f.num(r['cierre'], 2))}</div><div class="d">{hoy}</div></div>
          {_dato("1 mes", var(r['ret_1m']))}
          {_dato("3 meses", var(r['ret_3m']))}
          {_dato("12 meses", var(r['ret_12m']))}
          {_dato("Volatilidad 60 d", e(f.pct(r['vol_60d'])))}
          <div class="mt-r52">
            <div class="cab"><span>Mín. 52 sem. <b>{e(f.num(r['min_52'], 2))}</b></span><span>Máx. <b>{e(f.num(r['max_52'], 2))}</b></span></div>
            <div class="pista" role="img" aria-label="Precio actual en el {pos:.0f} % de su rango de 52 semanas"><i style="left:{pos:.1f}%"></i></div>
          </div>
        </div>""")


def tabla_universo(resumen) -> str:
    def var(x, dec=1):
        return delta(x, f.pct_signo(x, dec), flecha=False)

    filas = []
    for tk, r in resumen.iterrows():
        d = f.direccion(r["tendencia"][-1] - r["tendencia"][0]) if len(r["tendencia"]) > 1 else "plano"
        color = {"sube": T["sube"], "baja": T["baja"], "plano": T["acero"]}[d]
        filas.append([etiqueta_etf(tk), e(f.num(r["cierre"], 2)), var(r["var_dia"], 2), var(r["ret_1m"]),
                      var(r["ret_3m"]), var(r["ret_6m"]), var(r["ret_12m"]), e(f.pct(r["vol_60d"])),
                      e(f.pct(r["drawdown"])), sparkline(r["tendencia"], color)])
    return tabla(["ETF", "Último cierre", "Día", "1 mes", "3 meses", "6 meses", "12 meses", "Volatilidad 60 d",
                  "Desde máx. 12 m", "Últimos 90 días"], filas)


# --- Datos y método ----------------------------------------------------------------------
def flujo(pasos: list[tuple[str, str, str]]) -> str:
    """pasos: (título, descripción, tecnología)."""
    celdas = "".join(f'<div class="mt-paso"><div class="n">Paso {i}</div><h4>{e(t)}</h4><p>{e(d)}</p>'
                     f'<div class="tec">{e(tec)}</div></div>' for i, (t, d, tec) in enumerate(pasos, 1))
    return f'<div class="mt-flujo{" c5" if len(pasos) == 5 else ""}">{celdas}</div>'


def reportes(items: list[tuple[str, str, str, str]]) -> str:
    """Tres tarjetas, una por reporte. items: (rótulo, pregunta, explicación, dato del día con ** **)."""
    celdas = "".join(f'<div class="mt-paso"><div class="n">{e(k)}</div><h4>{e(t)}</h4><p>{e(d)}</p>'
                     f'<div class="dato">{negritas(dato)}</div></div>' for k, t, d, dato in items)
    return f'<div class="mt-flujo c3">{celdas}</div>'


# --- Predictivo y prescriptivo ---------------------------------------------------------------
def confusion(m: dict) -> str:
    """Matriz de confusión 2 x 2. m: salida de panel.matriz_confusion()."""
    def celda(clase: str, n: int, texto: str) -> str:
        return f'<div class="{clase}"><b>{e(f.num(n, 0))}</b>{e(texto)}</div>'

    return compacto(f"""
        <div class="mt-conf" role="table" aria-label="Matriz de confusión">
          <div class="cab"></div><div class="cab">Después subió</div><div class="cab">Después no subió</div>
          <div class="cab">El modelo dijo «sube»</div>
          {celda("ok", m["vp"], "aciertos")}{celda("mal", m["fp"], "falsas alarmas")}
          <div class="cab">El modelo dijo «no sube»</div>
          {celda("mal", m["fn"], "subidas no vistas")}{celda("ok", m["vn"], "aciertos")}
        </div>""")


def titular_decision(perfil: str, acciones: dict, fecha_texto: str) -> str:
    """La decisión del mes en una línea. acciones: {ticker: Comprar | Vender | Mantener}."""
    grupos = {a: [t for t, x in acciones.items() if x == a] for a in ("Comprar", "Vender", "Mantener")}

    def lista(tks: list[str]) -> str:
        nombres = [f"<b>{e(t)}</b>" for t in tks]
        return nombres[0] if len(nombres) == 1 else ", ".join(nombres[:-1]) + " y " + nombres[-1]

    if not grupos["Comprar"] and not grupos["Vender"]:
        frase = 'Mantener los cinco ETFs: ningún peso cambia lo suficiente para operar.'
    else:
        partes = [f'<span class="{c}">{v}</span> {lista(grupos[a])}'
                  for a, v, c in (("Comprar", "Comprar", "compra"), ("Vender", "Vender una parte de", "venta"),
                                  ("Mantener", "Mantener", "")) if grupos[a]]
        frase = '<span class="sep">·</span>'.join(partes)
    return (f'<div class="mt-titular"><div class="l">Este mes, perfil {e(perfil.lower())} · datos al {e(fecha_texto)}</div>'
            f'<div class="v">{frase}</div></div>')


def etiqueta_accion(accion: str) -> str:
    clase = {"Comprar": "compra", "Vender": "venta"}.get(accion, "")
    return f'<span class="mt-etq {clase}">{e(accion)}</span>'


def tarjeta_recomendacion(titulo: str, sub: str, etiquetas: list[str], texto: str, barra_html: str) -> str:
    """Cabecera de la recomendación: título, etiquetas (HTML ya armado), el resumen y la barra de pesos."""
    return compacto(f"""
        <div class="mt-rec">
          <div class="cab">
            <div><h3>{e(titulo)}</h3><div class="sub">{e(sub)}</div></div>
            <div class="etqs">{''.join(etiquetas)}</div>
          </div>
          <div class="texto">{e(texto)}</div>
          {barra_html}
        </div>""")


def aviso(texto: str) -> str:
    return f'<div class="mt-aviso">{e(texto)}</div>'


def lista(items: list[str]) -> str:
    return '<ul class="mt-lista">' + "".join(f"<li>{negritas(i)}</li>" for i in items) + "</ul>"


def tarjeta(titulo: str, cuerpo_html: str, sub: str = "") -> str:
    s = f'<div class="sub">{e(sub)}</div>' if sub else '<div style="height:.6rem"></div>'
    return f'<div class="mt-card"><h3>{e(titulo)}</h3>{s}{cuerpo_html}</div>'
