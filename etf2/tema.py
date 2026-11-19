"""Identidad visual del panel: colores, tipografía y hoja de estilos.

El panel tiene un solo tema (oscuro). Los cinco colores de los ETFs se eligieron con
el validador de paletas de la guía de visualización: pasan la banda de luminosidad,
el contraste contra la superficie (>= 3:1) y la separación entre TODOS los pares
también para personas con daltonismo (peor par: 9,7; el mínimo exigido es 8).
Verde/rojo de subida y bajada también se comprobaron entre sí (12,1).

Reglas de uso:
* el color sigue al ETF, no a su posición: SPY siempre es azul, GLD siempre ámbar;
* verde y rojo quedan reservados para sube/baja y siempre van con signo o flecha;
* el dorado es el color de la casa (detalles de la interfaz), no una serie de datos;
* el verde lima es de la estrategia con modelo predictivo, y de nadie más.
"""

FUENTE = "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"

T = {
    "pagina": "#0a0e16", "superficie": "#111723", "superficie2": "#172032",
    "borde": "rgba(148,163,184,0.14)", "borde_fuerte": "rgba(148,163,184,0.30)",
    "tinta": "#e7ebf3", "tinta2": "#a9b3c6", "tenue": "#7d899f",
    "rejilla": "#1c2536", "base": "#2c3852",
    "acento": "#d4b06a", "acero": "#8b98b3",
    "sube": "#34d399", "baja": "#ef5350", "aviso": "#f5b83d", "modelo": "#b8d94a",
    "neutro": "#1e2738", "polo_frio": "#3b8eea", "polo_calido": "#d9912a",
    "sobre_relleno": "#ffffff", "sobre_acento": "#14100a",
    "etf": {"SPY": "#0267c7", "QQQ": "#c32e85", "IWM": "#1faaa4", "GLD": "#c8800d", "TLT": "#a473ff"},
}


def color_etf(ticker: str) -> str:
    return T["etf"].get(ticker, T["tenue"])


def rgba(hex_: str, alfa: float) -> str:
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alfa})"


_VARIABLES = f"""
:root, .stApp {{
  --pagina: {T['pagina']}; --sup: {T['superficie']}; --sup2: {T['superficie2']};
  --borde: {T['borde']}; --borde-f: {T['borde_fuerte']};
  --tinta: {T['tinta']}; --tinta2: {T['tinta2']}; --tenue: {T['tenue']};
  --rejilla: {T['rejilla']}; --base: {T['base']};
  --acento: {T['acento']}; --acento-suave: {rgba(T['acento'], 0.13)}; --acento-borde: {rgba(T['acento'], 0.50)};
  --sobre-acento: {T['sobre_acento']};
  --sube: {T['sube']}; --baja: {T['baja']}; --aviso: {T['aviso']};
  --fuente: {FUENTE};
}}
"""

_ESTILOS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ---------- base ---------- */
.stApp { background: var(--pagina); color: var(--tinta); font-family: var(--fuente); }
.stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp summary, .stApp table,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] div { font-family: var(--fuente); }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 1340px; padding: 1.7rem 2.2rem 4rem; }
.up { color: var(--sube); } .down { color: var(--baja); } .plano { color: var(--tinta2); }
.mt-num { font-variant-numeric: tabular-nums; }

/* ---------- cabecera ---------- */
.mt-top { display: flex; justify-content: space-between; align-items: flex-end; gap: 1.2rem 2rem; flex-wrap: wrap;
          padding-bottom: 1.15rem; border-bottom: 1px solid var(--borde); }
.mt-marca { display: flex; align-items: center; gap: .95rem; }
.mt-logo { width: 46px; height: 46px; border-radius: 12px; display: grid; place-items: center; flex: none;
           background: var(--sup2); border: 1px solid var(--borde-f); }
.mt-kicker { font-size: .68rem; letter-spacing: .15em; text-transform: uppercase; color: var(--tenue); font-weight: 600; }
.mt-titulo { font-size: 1.56rem; font-weight: 650; letter-spacing: -.02em; line-height: 1.15; margin: .12rem 0 0; color: var(--tinta); }
.mt-lema { color: var(--tinta2); font-size: .86rem; margin-top: .2rem; }
.mt-estado { display: flex; gap: .5rem; flex-wrap: wrap; align-items: center; justify-content: flex-end; }
.mt-chip { display: inline-flex; align-items: center; gap: .45rem; border: 1px solid var(--borde); background: var(--sup);
           color: var(--tinta2); border-radius: 999px; padding: .3rem .75rem; font-size: .76rem; white-space: nowrap; }
.mt-chip b { color: var(--tinta); font-weight: 600; }
.mt-top.sin-linea { border-bottom: 0; padding-bottom: 0; }
.mt-linea { border-bottom: 1px solid var(--borde); margin: -.2rem 0 0; }
.mt-estado.barra { justify-content: flex-start; }
.mt-pop { display: grid; gap: .3rem; margin-bottom: .2rem; max-width: 22rem; }
.mt-pop b { color: var(--tinta); font-weight: 620; font-size: .92rem; }
.mt-pop span { color: var(--tinta2); font-size: .8rem; line-height: 1.5; }
button[data-testid="stPopoverButton"] { border: 1px solid var(--acento-borde); color: var(--acento); background: transparent; }
button[data-testid="stPopoverButton"]:hover { color: var(--acento); }
button[data-testid="stPopoverButton"] p { font-size: .8rem; font-weight: 600; }
div[data-testid="stPopoverBody"] div[data-testid="stForm"] { background: transparent; border: 0; padding: 0; }
.mt-punto { position: relative; width: 7px; height: 7px; border-radius: 50%; background: var(--sube); box-shadow: 0 0 0 3px rgba(52,211,153,.16); }
.mt-punto.aviso { background: var(--aviso); box-shadow: 0 0 0 3px rgba(245,184,61,.16); }

/* ---------- indicadores clave ---------- */
.mt-kpis { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .8rem; margin-top: 1.15rem; }
.mt-kpi { background: var(--sup); border: 1px solid var(--borde); border-radius: 12px; padding: .9rem 1rem .95rem; min-width: 0; }
.mt-kpi .l { font-size: .78rem; color: var(--tenue); font-weight: 500; }
.mt-kpi .v { font-size: 1.72rem; font-weight: 650; letter-spacing: -.025em; color: var(--tinta); margin-top: .28rem; line-height: 1.1; }
.mt-kpi .d { font-size: .76rem; color: var(--tinta2); margin-top: .4rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.mt-kpi .fila { display: flex; align-items: flex-end; justify-content: space-between; gap: .5rem; }
.mt-pie { color: var(--tenue); font-size: .76rem; margin: .55rem 0 0; }

/* ---------- secciones y tarjetas ---------- */
.mt-sec { margin: 2rem 0 .1rem; }
.mt-sec .k { font-size: .66rem; letter-spacing: .15em; text-transform: uppercase; color: var(--tenue); font-weight: 600; }
.mt-sec h2 { font-size: 1.12rem; font-weight: 620; color: var(--tinta); margin: .15rem 0 0; padding: 0; letter-spacing: -.01em; }
.mt-sec p { color: var(--tinta2); font-size: .86rem; margin: .25rem 0 0; max-width: 84ch; line-height: 1.5; }
.mt-duo { display: grid; grid-template-columns: minmax(0, 7fr) minmax(0, 5fr); gap: .9rem; }
.mt-duo.par { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
.mt-card { background: var(--sup); border: 1px solid var(--borde); border-radius: 12px; padding: 1.05rem 1.15rem 1.1rem; min-width: 0;
           display: flex; flex-direction: column; }
.mt-card h3 { font-size: .92rem; font-weight: 600; margin: 0; padding: 0; color: var(--tinta); }
.mt-card .sub { color: var(--tinta2); font-size: .8rem; margin: .15rem 0 .95rem; }
.mt-nota { color: var(--tenue); font-size: .78rem; margin: .1rem 0 .5rem; max-width: 118ch; line-height: 1.55; }
.mt-nota b { color: var(--tinta2); font-weight: 600; }

/* ---------- barra segmentada ---------- */
.mt-seg { display: flex; gap: 2px; height: 28px; width: 100%; }
.mt-seg > div { border-radius: 4px; display: flex; align-items: center; padding-left: .5rem; color: #fff;
                font-size: .76rem; font-weight: 600; min-width: 0; font-variant-numeric: tabular-nums; }
.mt-sw { width: 9px; height: 9px; border-radius: 3px; display: inline-block; flex: none; }
.mt-tk { display: inline-flex; align-items: center; gap: .5rem; font-weight: 600; color: var(--tinta); }

/* ---------- tablas ---------- */
.mt-tabla-caja { background: var(--sup); border: 1px solid var(--borde); border-radius: 12px; overflow: auto; }
.mt-card .mt-tabla-caja { border: none; border-radius: 0; background: transparent; }
table.mt-tabla { width: 100%; border-collapse: collapse; font-size: .84rem; margin: 0; }
.mt-tabla th { position: sticky; top: 0; z-index: 1; background: var(--sup2); color: var(--tenue); font-weight: 600; font-size: .72rem;
               text-align: right; padding: .55rem .9rem; border: none; border-bottom: 1px solid var(--borde); white-space: nowrap; }
.mt-tabla td { text-align: right; padding: .62rem .9rem; border: none; border-bottom: 1px solid var(--rejilla); color: var(--tinta);
               font-variant-numeric: tabular-nums; white-space: nowrap; vertical-align: middle; }
.mt-tabla tr:last-child td { border-bottom: none; }
.mt-tabla tbody tr:hover td { background: rgba(148,163,184,.07); }
.mt-tabla th.ayuda { cursor: help; text-decoration: underline dotted rgba(148,163,184,.55); text-underline-offset: 3px; }
.mt-tabla a { color: var(--tinta2); text-underline-offset: 3px; transition: color .15s; }
.mt-tabla a:hover { color: var(--tinta); }
.mt-tabla .izq { text-align: left; }
.mt-tabla td.txt { text-align: left; color: var(--tinta2); white-space: normal; min-width: 22ch; line-height: 1.45; }
.mt-tabla td.tenue { color: var(--tinta2); }
.mt-card table.mt-tabla th { background: transparent; padding-left: 0; }
.mt-card table.mt-tabla td { padding-left: 0; }
.mt-card table.mt-tabla th:last-child, .mt-card table.mt-tabla td:last-child { padding-right: 0; }
.mt-medidor { height: 6px; border-radius: 999px; min-width: 90px; }
.mt-medidor > i { display: block; height: 100%; border-radius: 999px; }
.mt-etq { display: inline-block; border-radius: 6px; padding: .12rem .5rem; font-size: .74rem; font-weight: 600;
          border: 1px solid var(--borde); color: var(--tinta2); background: rgba(148,163,184,.06); }
.mt-etq.compra { color: var(--sube); border-color: rgba(52,211,153,.35); background: rgba(52,211,153,.08); }
.mt-etq.venta { color: var(--baja); border-color: rgba(239,83,80,.38); background: rgba(239,83,80,.08); }
.mt-etq.acento { color: var(--tinta); border-color: var(--borde-f); background: rgba(148,163,184,.10); }

/* ---------- titular de la decisión del mes ---------- */
.mt-titular { display: flex; flex-wrap: wrap; align-items: baseline; gap: .35rem 1rem; background: var(--sup);
              border: 1px solid var(--borde); border-radius: 12px; padding: .95rem 1.15rem; margin-bottom: .2rem; }
.mt-titular .l { font-size: .74rem; color: var(--tenue); font-weight: 500; flex-basis: 100%; }
.mt-titular .v { font-size: 1.28rem; font-weight: 620; letter-spacing: -.015em; line-height: 1.35; color: var(--tinta2); }
.mt-titular .v b { color: var(--tinta); font-weight: 650; }
.mt-titular .compra { color: var(--sube); } .mt-titular .venta { color: var(--baja); }
.mt-titular .sep { color: var(--base); margin: 0 .45rem; }

/* ---------- recomendación ---------- */
.mt-rec { background: var(--sup); border: 1px solid var(--borde); border-radius: 12px; padding: 1.05rem 1.15rem 1.1rem; }
.mt-rec .cab { display: flex; justify-content: space-between; align-items: flex-start; gap: .6rem 1.2rem; flex-wrap: wrap; }
.mt-rec h3 { font-size: 1rem; font-weight: 620; margin: 0; padding: 0; color: var(--tinta); }
.mt-rec .sub { color: var(--tinta2); font-size: .8rem; margin-top: .15rem; }
.mt-rec .etqs { display: flex; gap: .4rem; flex-wrap: wrap; }
.mt-rec .texto { color: var(--tinta); font-size: .92rem; line-height: 1.6; margin: .9rem 0 1rem; max-width: 110ch; }
.mt-aviso { border: 1px solid var(--borde); border-left: 3px solid var(--borde-f); border-radius: 8px; background: var(--sup);
            color: var(--tinta2); font-size: .8rem; line-height: 1.5; padding: .65rem .9rem; margin: .8rem 0 .2rem; }
.mt-conf { display: grid; grid-template-columns: auto repeat(2, minmax(0, 1fr)); gap: 2px; font-size: .82rem; max-width: 30rem; }
.mt-conf > div { padding: .6rem .7rem; border-radius: 6px; background: var(--sup2); color: var(--tinta2); }
.mt-conf .cab { background: transparent; color: var(--tenue); font-size: .72rem; font-weight: 600; }
.mt-conf .ok { background: rgba(52,211,153,.10); color: var(--tinta); }
.mt-conf .mal { background: rgba(239,83,80,.09); color: var(--tinta); }
.mt-conf b { display: block; font-size: 1.25rem; font-weight: 650; font-variant-numeric: tabular-nums; }

/* ---------- lectura ---------- */
.mt-lect { list-style: none; padding: 0; margin: 0 0 1rem; display: grid; gap: .7rem; }
.mt-lect li { display: grid; grid-template-columns: auto 1fr; gap: .65rem; font-size: .86rem; color: var(--tinta2); line-height: 1.5; margin: 0; padding: 0; }
.mt-lect li b { color: var(--tinta); font-weight: 600; }
.mt-lect .n { width: 20px; height: 20px; border-radius: 6px; background: var(--sup2); color: var(--tinta2);
              display: grid; place-items: center; font-size: .68rem; font-weight: 700; margin-top: .12rem; }
.mt-prox { margin-top: auto; padding-top: .9rem; border-top: 1px solid var(--borde); display: flex; justify-content: space-between;
           gap: 1rem; font-size: .8rem; color: var(--tinta2); }
.mt-prox b { color: var(--tinta); font-weight: 600; }

/* ---------- ficha del ETF ---------- */
.mt-etf { display: flex; align-items: center; gap: 1.2rem 2rem; flex-wrap: wrap; background: var(--sup); border: 1px solid var(--borde);
          border-radius: 12px; padding: .9rem 1.15rem; }
.mt-etf .id { display: flex; align-items: center; gap: .75rem; min-width: 15rem; flex: 1 1 15rem; }
.mt-etf .barra { width: 4px; height: 38px; border-radius: 4px; flex: none; }
.mt-etf .tk { font-size: 1.15rem; font-weight: 650; letter-spacing: -.01em; color: var(--tinta); line-height: 1.2; }
.mt-etf .nm { font-size: .78rem; color: var(--tinta2); }
.mt-etf .px .v { font-size: 1.5rem; font-weight: 650; letter-spacing: -.02em; line-height: 1.15; color: var(--tinta); }
.mt-etf .px .d { font-size: .8rem; font-weight: 600; }
.mt-etf .dato { display: grid; gap: .1rem; }
.mt-etf .dato > span { font-size: .72rem; color: var(--tenue); }
.mt-etf .dato b { font-size: .9rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.mt-r52 { min-width: 13rem; flex: 1 1 13rem; max-width: 19rem; }
.mt-r52 .cab { display: flex; justify-content: space-between; font-size: .72rem; color: var(--tenue); }
.mt-r52 .cab b { color: var(--tinta2); font-weight: 600; font-variant-numeric: tabular-nums; }
.mt-r52 .pista { position: relative; height: 4px; border-radius: 999px; background: var(--base); margin: .5rem 0 .1rem; }
.mt-r52 .pista i { position: absolute; top: 50%; width: 10px; height: 10px; border-radius: 50%; background: var(--tinta);
                   box-shadow: 0 0 0 2px var(--sup); transform: translate(-50%, -50%); }

/* ---------- flujo del sistema ---------- */
.mt-flujo { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .8rem; }
.mt-flujo.c3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.mt-flujo.c5 { grid-template-columns: repeat(5, minmax(0, 1fr)); }
.mt-paso { background: var(--sup); border: 1px solid var(--borde); border-radius: 12px; padding: .9rem 1rem; position: relative; }
.mt-paso .n { font-size: .68rem; letter-spacing: .12em; color: var(--tenue); font-weight: 600; text-transform: uppercase; }
.mt-paso h4 { font-size: .92rem; font-weight: 600; margin: .2rem 0 .25rem; padding: 0; color: var(--tinta); }
.mt-paso p { font-size: .8rem; color: var(--tinta2); margin: 0; line-height: 1.45; }
.mt-paso .tec { margin-top: .55rem; font-size: .72rem; color: var(--tenue); }
.mt-paso .dato { margin-top: .6rem; padding-top: .55rem; border-top: 1px solid var(--borde); font-size: .8rem; color: var(--tinta); }
.mt-lista { margin: 0; padding-left: 1.1rem; color: var(--tinta2); }
[data-testid="stMarkdownContainer"] ul.mt-lista li { font-size: .84rem; line-height: 1.55; margin: 0 0 .4rem; }
.mt-lista b { color: var(--tinta); font-weight: 600; }

/* ---------- controles de Streamlit ---------- */
label[data-testid="stWidgetLabel"] p { font-size: .76rem; color: var(--tenue); font-weight: 500; }
div[data-testid="stButtonGroup"] button { background: var(--sup); border: 1px solid var(--borde); color: var(--tinta2);
                                         font-weight: 550; min-height: 2.05rem; }
div[data-testid="stButtonGroup"] button p { font-size: .8rem; }
div[data-testid="stButtonGroup"] button:hover { border-color: var(--borde-f); color: var(--tinta); }
div[data-testid="stButtonGroup"] button[aria-checked="true"], div[data-testid="stButtonGroup"] button[aria-pressed="true"] {
    background: var(--acento-suave); border-color: var(--acento-borde); color: var(--tinta); }
div[data-testid="stNumberInputContainer"] { background: var(--sup); border: 1px solid var(--borde); border-radius: 9px; }
div[data-testid="stNumberInputContainer"] input { background: transparent; color: var(--tinta); font-variant-numeric: tabular-nums; }
div[data-testid="stNumberInputContainer"] button { background: transparent; color: var(--tinta2); }
div[data-testid="stExpander"] details { background: var(--sup); border: 1px solid var(--borde); border-radius: 12px; }
div[data-testid="stExpander"] summary { color: var(--tinta2); }
div[data-testid="stExpander"] summary p { font-size: .8rem; }
div[data-testid="stPlotlyChart"] { background: var(--sup); border: 1px solid var(--borde); border-radius: 12px; overflow: hidden; }
div[data-testid="stDownloadButton"] button, div[data-testid="stButton"] button {
    background: transparent; border: 1px solid var(--borde-f); color: var(--tinta2); }
div[data-testid="stDownloadButton"] button:hover, div[data-testid="stButton"] button:hover {
    border-color: var(--acento-borde); color: var(--tinta); }
div[data-testid="stDownloadButton"] button p, div[data-testid="stButton"] button p { font-size: .8rem; }
div[data-testid="stForm"] { background: var(--sup); border: 1px solid var(--borde); border-radius: 12px; padding: .9rem 1rem; }
div[data-testid="stFormSubmitButton"] button { background: var(--acento); border: 1px solid var(--acento);
    color: var(--sobre-acento); }
div[data-testid="stFormSubmitButton"] button p { font-size: .82rem; font-weight: 600; }
div[data-testid="stFormSubmitButton"] button:hover, div[data-testid="stFormSubmitButton"] button:focus,
div[data-testid="stFormSubmitButton"] button:active, div[data-testid="stFormSubmitButton"] button:focus:not(:active) {
    background: var(--acento); border-color: var(--acento); color: var(--sobre-acento); }
div[data-testid="stFormSubmitButton"] button:hover { filter: brightness(1.08); }
div[data-testid="stTabs"] div[role="tablist"] { gap: .35rem; }
div[data-testid="stTab"] { padding-left: .15rem; padding-right: .95rem; color: var(--tinta2); }
div[data-testid="stTab"] p { font-size: .92rem; font-weight: 550; }
div[data-testid="stTab"][aria-selected="true"] { color: var(--tinta); }

/* ---------- pasar el mouse y hacer clic ----------
   Solo lo que se puede tocar reacciona: se aclara, toma el borde dorado y crece un poco;
   al hacer clic se hunde. Lo elegido queda en dorado. Sin animación si el sistema la reduce. */
div[data-testid="stButtonGroup"] button, button[data-testid^="stBaseButton"], button[data-testid="stPopoverButton"],
div[data-testid="stTab"],
div[data-testid="stExpander"] summary, div[data-testid="stRadioGroup"] label, div[data-testid="stNumberInputContainer"],
div[data-testid="stTextInputRootElement"] {
    transition: transform .16s ease, background-color .16s ease, border-color .16s ease, color .16s ease, box-shadow .16s ease; }
div[data-testid="stButtonGroup"] button:hover, button[data-testid^="stBaseButton-secondary"]:hover,
button[data-testid="stPopoverButton"]:hover {
    background: var(--sup2); border-color: var(--acento-borde); color: var(--tinta); box-shadow: 0 0 0 3px var(--acento-suave);
    position: relative; z-index: 1; }
div[data-testid="stButtonGroup"] button[data-selected="true"]:hover { background: var(--acento-suave); }
div[data-testid="stTab"]:hover { color: var(--tinta); }
div[data-testid="stTab"] p { transition: transform .16s ease; transform-origin: center bottom; }
div[data-testid="stExpander"] summary:hover { background: var(--sup2); color: var(--tinta); }
div[data-testid="stRadioGroup"] label:hover p { color: var(--tinta); }
div[data-testid="stNumberInputContainer"]:hover, div[data-testid="stTextInputRootElement"]:hover { border-color: var(--borde-f); }
div[data-testid="stNumberInputContainer"]:focus-within, div[data-testid="stTextInputRootElement"]:focus-within {
    border-color: var(--acento-borde); box-shadow: 0 0 0 3px var(--acento-suave); }
.stApp button:focus-visible, div[data-testid="stTab"]:focus-visible, div[data-testid="stExpander"] summary:focus-visible {
    outline: 2px solid var(--acento); outline-offset: 2px; }
@media (prefers-reduced-motion: no-preference) {
  div[data-testid="stButtonGroup"] button:hover, button[data-testid^="stBaseButton"]:hover,
  button[data-testid="stPopoverButton"]:hover { transform: translateY(-1px) scale(1.035); }
  div[data-testid="stTab"]:hover p { transform: scale(1.05); }
  div[data-testid="stRadioGroup"] label:hover { transform: translateX(1px) scale(1.03); }
  div[data-testid="stButtonGroup"] button:active, button[data-testid^="stBaseButton"]:active,
  button[data-testid="stPopoverButton"]:active { transform: scale(.97); transition-duration: .06s; }
  div[data-testid="stTab"]:active p { transform: scale(.97); }
}

/* ---------- movimiento ----------
   Poco, corto y con motivo: el contenido entra al cambiar de pestaña, los indicadores clave
   aparecen en orden, las barras de reparto se llenan de izquierda a derecha y el punto de
   «datos al día» late. Los gráficos se vuelven a crear al cambiar un filtro, así que entran con
   un fundido en vez de aparecer de golpe. Nada se mueve si el sistema pide movimiento reducido. */
@media (prefers-reduced-motion: no-preference) {
  @keyframes mt-entrar { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
  @keyframes mt-aparecer { from { opacity: 0; } to { opacity: 1; } }
  @keyframes mt-llenar { from { clip-path: inset(0 100% 0 0); } to { clip-path: inset(0 0 0 0); } }
  @keyframes mt-latido { 0% { transform: scale(1); opacity: .6; } 70%, 100% { transform: scale(2.8); opacity: 0; } }
  div[data-testid="stTabs"] [role="tabpanel"] > div { animation: mt-entrar .28s cubic-bezier(.2, .7, .2, 1) both; }
  div[data-testid="stPlotlyChart"] { animation: mt-aparecer .35s ease-out both; }
  .mt-kpi { animation: mt-entrar .42s cubic-bezier(.2, .7, .2, 1) both; }
  .mt-kpi:nth-child(2) { animation-delay: .05s; } .mt-kpi:nth-child(3) { animation-delay: .1s; }
  .mt-kpi:nth-child(4) { animation-delay: .15s; } .mt-kpi:nth-child(5) { animation-delay: .2s; }
  .mt-seg { animation: mt-llenar .75s cubic-bezier(.3, .7, .2, 1) .1s both; }
  .mt-medidor > i { animation: mt-llenar .6s cubic-bezier(.3, .7, .2, 1) .15s both; }
  .mt-tabla tbody tr:nth-child(2) .mt-medidor > i { animation-delay: .22s; }
  .mt-tabla tbody tr:nth-child(3) .mt-medidor > i { animation-delay: .29s; }
  .mt-tabla tbody tr:nth-child(4) .mt-medidor > i { animation-delay: .36s; }
  .mt-tabla tbody tr:nth-child(5) .mt-medidor > i { animation-delay: .43s; }
  .mt-punto:not(.aviso)::after { content: ""; position: absolute; inset: 0; border-radius: 50%; background: var(--sube);
                                 animation: mt-latido 2.4s ease-out infinite; }
}

/* ---------- pantallas angostas ---------- */
@media (max-width: 1100px) {
  .mt-kpis { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .mt-duo { grid-template-columns: minmax(0, 1fr); }
  .mt-flujo, .mt-flujo.c5 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding: 1.2rem 1rem 3rem; }
  .mt-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
  .mt-kpi .d { white-space: normal; }
  .mt-duo.par { grid-template-columns: minmax(0, 1fr); }
  .mt-tabla td.txt { min-width: 44ch; }
  .mt-seg > div.angosto { font-size: 0; padding: 0; }
  .mt-flujo, .mt-flujo.c3, .mt-flujo.c5 { grid-template-columns: minmax(0, 1fr); }
  .mt-estado { justify-content: flex-start; }
}
"""


def css() -> str:
    return f"<style>{_ESTILOS}{_VARIABLES}</style>"
