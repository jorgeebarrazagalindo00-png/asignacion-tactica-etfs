"""Panel de asignación táctica. Ejecutar con:  streamlit run dashboard.py

Solo LEE de Supabase, con la clave publicable (permiso de solo lectura). El botón
«Actualizar ahora» tampoco escribe: le pide a GitHub Actions que corra el workflow.
"""

import hmac
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from etf2 import componentes as c
from etf2 import actualizacion, config, formato as f, graficos, panel, prescriptivo, tema
from etf2.api_db import ClienteSupabase, ErrorAPI
from etf2.indicadores import calcular_indicadores

ICONO = Path(__file__).parent / "assets" / "icono.png"
st.set_page_config(page_title=config.NOMBRE_PANEL, page_icon=str(ICONO) if ICONO.exists() else None, layout="wide")

E, EML, IG, SPY = config.NOMBRE_ESTRATEGIA, config.NOMBRE_ESTRATEGIA_ML, config.NOMBRE_IGUALES, config.NOMBRE_SPY
PRINCIPAL = config.MODELO_PRINCIPAL
AYUDAS_METRICAS = {
    "Exactitud": "Aciertos sobre el total de predicciones.",
    "Precisión": "De las veces que dijo «sube», cuántas subió.",
    "Sensibilidad": "De las veces que subió, cuántas anticipó.",
    "F1": "Promedio armónico de precisión y sensibilidad.",
    "AUC": "Capacidad de ordenar los meses que suben antes que los que bajan. 0,50 es azar.",
    "Brier": "Error de la probabilidad: menos es mejor.",
    "IC de Spearman": "Correlación entre la probabilidad y el retorno real. Cerca de 0, sin relación.",
}
METRICAS_BT = ["retorno_total", "cagr", "volatilidad", "sharpe", "max_drawdown", "turnover_anual", "n_rebalanceos",
               "friccion"]


def html(texto: str) -> None:
    st.markdown(texto, unsafe_allow_html=True)


def grafico(fig) -> None:
    # theme=None: se respeta el estilo definido en etf2/graficos.py (Streamlit no lo retoca).
    st.plotly_chart(fig, width="stretch", theme=None, config=graficos.CONFIG)


def elegir(etiqueta: str, opciones: list[str], defecto: str, clave: str) -> str:
    """Selector de una opción; si el usuario la desmarca, vuelve a la opción por defecto."""
    return st.segmented_control(etiqueta, opciones, default=defecto, key=clave) or defecto


def datos_del_grafico(titulo: str, columnas: list[str], filas: list[list[str]], **kw) -> None:
    """Tabla equivalente a un gráfico, plegada: los mismos datos sin depender del color ni del cursor."""
    with st.expander(titulo):
        html(c.tabla(columnas, filas, alto_max=320, **kw))


def texto(x, vacio: str = "") -> str:
    """Texto de una celda; `vacio` si el dato falta en la base."""
    return vacio if pd.isna(x) else str(x)


def con_mayuscula(t: str) -> str:
    """Primera letra en mayúscula sin tocar el resto («desde una PC» → «Desde una PC»)."""
    return t[:1].upper() + t[1:]


def dos_tarjetas(izquierda: str, derecha: str) -> None:
    html(f'<div class="mt-duo par">{izquierda}{derecha}</div>')


# ---------------------------------------------------------------------------
# Credenciales y carga de datos
# ---------------------------------------------------------------------------
def secreto(nombre: str) -> str | None:
    """Primero el entorno (el .env de tu PC); si no está, los secretos de Streamlit Cloud."""
    valor = os.getenv(nombre)
    if valor:
        return valor
    try:
        return st.secrets.get(nombre) or None
    except Exception:
        return None


def credenciales() -> tuple[str | None, str | None]:
    config.cargar_env()
    return secreto("SUPABASE_URL"), secreto("SUPABASE_PUBLISHABLE_KEY")


def leer_opcional(cli: ClienteSupabase, tabla: str, params: dict, **kw) -> list[dict]:
    """Tablas de los reportes predictivo y prescriptivo: si la base aún no las tiene, no hay datos."""
    try:
        return cli.leer(tabla, params, **kw)
    except ErrorAPI as err:
        if err.codigo == 404:
            return []
        raise


@st.cache_data(ttl=600, show_spinner="Leyendo la base de datos…")
def cargar(url: str, clave: str) -> dict:
    cli = ClienteSupabase(url, clave)
    hoy = panel.a_frame(cli.leer("v_pesos_actuales", {"select": "*"}),
                        fechas=["fecha"], numericas=["peso", "peso_base", "factor_momentum"])
    ind_hoy = pd.DataFrame()
    if not hoy.empty:
        ind_hoy = panel.a_frame(
            cli.leer("indicadores", {"select": "*", "fecha": f"eq.{hoy['fecha'].max():%Y-%m-%d}"}),
            numericas=panel.COLS_IND)
    return {
        "hoy": hoy, "ind_hoy": ind_hoy,
        "etfs": panel.a_frame(cli.leer("etfs", {"select": "*"})),
        "precios": panel.a_frame(
            cli.leer("precios_diarios", {"select": "ticker,fecha,apertura,maximo,minimo,cierre,cierre_ajustado,volumen",
                                         "order": "ticker.asc,fecha.asc"}, hilos=6),
            fechas=["fecha"], numericas=["apertura", "maximo", "minimo", "cierre", "cierre_ajustado", "volumen"]),
        "pesos": panel.a_frame(
            cli.leer("pesos_sugeridos", {"select": "fecha,ticker,peso", "order": "fecha.asc,ticker.asc"}, hilos=6),
            fechas=["fecha"], numericas=["peso"]),
        "resumen": panel.a_frame(
            cli.leer("backtest_resumen", {"select": "*"}), fechas=["fecha_inicio", "fecha_fin"], numericas=METRICAS_BT),
        "curva": panel.a_frame(cli.leer("backtest_curva", {"select": "*", "order": "fecha.asc"}, hilos=4),
                               fechas=["fecha"], numericas=["estrategia", "pesos_iguales", "spy"]),
        "corridas": panel.a_frame(cli.leer("corridas", {"select": "*", "order": "id.desc"}, maximo=12),
                                  fechas=["inicio", "fin"],
                                  numericas=["id", "filas_precios", "filas_indicadores", "filas_pesos"]),
        "pred": panel.a_frame(
            leer_opcional(cli, "modelo_predicciones", {"select": "*", "order": "fecha.asc,ticker.asc,modelo.asc"},
                          hilos=4),
            fechas=["fecha"], numericas=["prob_sube", "subio", "ret_futuro"]),
        "metricas": panel.a_frame(
            leer_opcional(cli, "modelo_metricas", {"select": "*"}), fechas=["fecha_desde", "fecha_hasta"],
            numericas=["n", "n_fechas", "exactitud", "precision_pos", "sensibilidad", "f1", "auc", "brier",
                       "ic_spearman", "tasa_positivos"]),
        "importancias": panel.a_frame(leer_opcional(cli, "modelo_importancias", {"select": "*"}),
                                      numericas=["importancia", "signo"]),
        "ml_resumen": panel.a_frame(leer_opcional(cli, "backtest_ml_resumen", {"select": "*"}),
                                    fechas=["fecha_inicio", "fecha_fin"], numericas=METRICAS_BT),
        "ml_curva": panel.a_frame(
            leer_opcional(cli, "backtest_ml_curva", {"select": "*", "order": "fecha.asc"}, hilos=4),
            fechas=["fecha"], numericas=["estrategia", "estrategia_ml", "pesos_iguales", "spy"]),
        "recs": panel.a_frame(leer_opcional(cli, "recomendaciones", {"select": "*", "order": "fecha.desc"}, maximo=12),
                              fechas=["fecha"]),
        "llm_log": panel.a_frame(leer_opcional(cli, "llm_log", {"select": "*", "order": "id.desc"}, maximo=200),
                                 fechas=["dia"], numericas=["id", "tokens_entrada", "tokens_salida", "latencia_ms"]),
    }


# ---------------------------------------------------------------------------
# Botón «Actualizar ahora»: GitHub Actions corre el proceso, el panel solo lo pide y lo sigue
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def ejecuciones_github(token: str, repo: str, tic: int) -> list[dict]:
    """Últimas ejecuciones del workflow. Mientras se sigue una, `tic` cambia y obliga a consultar de nuevo."""
    return actualizacion.ClienteGitHub(token, repo).ejecuciones()


def _segundos_utc(ts) -> float:
    t = pd.Timestamp(ts)
    return t.timestamp() if t is not pd.NaT else 0.0


def fila_ejecucion(x: dict) -> list[str]:
    clase, etiqueta = actualizacion.estado(x)
    creada, fin = pd.Timestamp(x.get("creada")), pd.Timestamp(x.get("actualizada"))
    url = x.get("url") or ""
    enlace = f'<a href="{c.e(url)}" target="_blank" rel="noopener">Ver en GitHub</a>' \
        if url.startswith("https://github.com/") else ""
    return [c.e(f"#{x.get('numero')}"), f'<span class="{clase}">{c.e(etiqueta)}</span>',
            c.e(actualizacion.DISPAROS.get(x.get("evento"), x.get("evento") or "")), c.e(f.fecha_hora(creada)),
            c.e("en curso" if actualizacion.activa(x) else f.duracion(creada, fin)), enlace]


def credenciales_boton() -> tuple[str | None, str | None, str | None]:
    """Token de GitHub, repositorio y contraseña del botón. Sin los tres, el botón no aparece."""
    return tuple(secreto(v) for v in (config.VARIABLE_TOKEN_GITHUB, config.VARIABLE_REPO_GITHUB,
                                      config.VARIABLE_CLAVE_PANEL))


def seguimiento(token: str, repo: str, completo: bool) -> None:
    """Estado de las ejecuciones en GitHub. Mientras una corre, se consulta sola cada pocos segundos.

    completo=False: una línea con la última ejecución, en la ventana del botón. Es la que nota cuándo
    terminó y hace que el panel vuelva a leer la base. completo=True: la tabla de Datos y método.
    """
    lanzada = st.session_state.get("gh_lanzada", 0.0)
    vigilar = st.session_state.get("gh_activa", False) or time.time() - lanzada < 600

    @st.fragment(run_every=config.SEGUNDOS_SEGUIMIENTO if vigilar else None)
    def vista() -> None:
        tic = int(time.time() // config.SEGUNDOS_SEGUIMIENTO) if vigilar else 0
        try:
            ejs = ejecuciones_github(token, repo, tic)
        except actualizacion.ErrorGitHub as err:
            st.warning(str(err))
            return
        hay = actualizacion.hay_activa(ejs)
        if not completo:
            vistas = st.session_state.setdefault("gh_vistas", set())
            # Terminó una que se vio correr, o una creada después de pulsar el botón.
            terminadas = [x for x in ejs if not actualizacion.activa(x)
                          and (x["id"] in vistas or (lanzada and _segundos_utc(x.get("creada")) >= lanzada - 30))]
            vistas.update(x["id"] for x in ejs if actualizacion.activa(x))
            if terminadas:
                vistas.difference_update(x["id"] for x in terminadas)
                st.session_state.update(gh_lanzada=0.0, gh_activa=hay, gh_aviso="ok" if all(
                    x.get("resultado") == "success" for x in terminadas) else "error")
                cargar.clear()
                ejecuciones_github.clear()
                st.rerun()
            st.session_state["gh_activa"] = hay
            if hay and not vigilar:
                st.rerun()                      # hay una en curso: activar el seguimiento automático
        if lanzada and not hay and vigilar:
            html(c.nota("Esperando que GitHub inicie la ejecución…"))
        elif not ejs:
            html(c.nota("GitHub aún no registra ejecuciones de este workflow."))
        elif not completo:
            x = ejs[0]
            clase, etiqueta = actualizacion.estado(x)
            html(f'<div class="mt-nota">Última ejecución #{c.e(x.get("numero"))} · '
                 f'<span class="{clase}">{c.e(etiqueta)}</span> · '
                 f'{c.e(f.fecha_hora(pd.Timestamp(x.get("creada"))))}</div>')
        if completo and ejs:
            html(c.tabla(["Ejecución", "Estado", "Tipo", "Inicio (hora de Perú)", "Duración", "Registro"],
                         [fila_ejecucion(x) for x in ejs], izquierda=(0, 1, 2, 5)))

    vista()


def boton_en_cabecera(token: str, repo: str, contrasena: str) -> None:
    """«Actualizar ahora» junto a «Actualizado hace …»: abre una ventana con el tipo de carga y la contraseña."""
    with st.popover("Actualizar ahora", icon=":material/refresh:", key="gh_ventana"):
        html('<div class="mt-pop"><b>Traer datos nuevos ahora</b><span>GitHub Actions corre el proceso completo: '
             'precios, indicadores, modelo y recomendaciones. Tarda unos minutos y, al terminar, el panel se '
             'actualiza solo.</span></div>')
        with st.form("gh_form", clear_on_submit=True, border=False):
            modo = st.radio("Tipo de carga", ["diario", "historico"], key="gh_modo",
                            format_func={"diario": "Diaria · solo lo nuevo",
                                         "historico": "Completa · todo el historial"}.get)
            intento = st.text_input("Contraseña", type="password", key="gh_clave")
            enviar = st.form_submit_button("Actualizar ahora", key="gh_enviar", width="stretch")
        if enviar:
            if not intento or not hmac.compare_digest(intento.encode(), contrasena.encode()):
                time.sleep(1)
                st.error("Contraseña incorrecta.")
            else:
                try:
                    gh = actualizacion.ClienteGitHub(token, repo)
                    if actualizacion.hay_activa(gh.ejecuciones()):
                        st.warning("Ya hay una ejecución en curso: espera a que termine.")
                    else:
                        gh.lanzar(modo)
                        st.session_state["gh_lanzada"] = time.time()
                        ejecuciones_github.clear()
                        st.success("GitHub recibió la orden. La ejecución empieza en unos segundos.")
                except actualizacion.ErrorGitHub as err:
                    st.error(str(err))
        seguimiento(token, repo, completo=False)


def seccion_actualizacion(aviso: str | None) -> None:
    """Datos y método: qué hace el botón y la tabla de las últimas ejecuciones en GitHub."""
    token, repo, contrasena = credenciales_boton()
    html(c.seccion("Actualizar ahora", "El botón «Actualizar ahora» de la cabecera pide a GitHub Actions que corra ya "
                   "el mismo proceso programado: descarga los precios, recalcula todo y guarda en la base. Aquí se ve "
                   "cómo van las ejecuciones.", kicker="Actualización"))
    if not (token and repo and contrasena):
        html(c.nota("El botón no está configurado en este panel. El proceso igual corre solo cada día hábil a las "
                    "18:00 (hora de Perú)."))
        return
    if aviso == "ok":
        st.success("Actualización terminada: el panel ya muestra los datos nuevos.")
    elif aviso == "error":
        st.error("La ejecución terminó con error. Abre «Ver en GitHub» en la tabla para ver el detalle.")
    seguimiento(token, repo, completo=True)


def pantalla_de_aviso(mensaje: str, tipo: str = "info") -> None:
    html(c.cabecera(config.NOMBRE_PANEL, config.ROTULO_PANEL, config.LEMA_PANEL, []))
    (st.error if tipo == "error" else st.info)(mensaje)
    st.stop()


html(tema.css())
url, clave = credenciales()
if not url or not clave:
    pantalla_de_aviso(
        "Faltan las credenciales de lectura. Crea un archivo `.env` junto a `dashboard.py` con:\n\n"
        "```\nSUPABASE_URL=https://TU-PROYECTO.supabase.co\nSUPABASE_PUBLISHABLE_KEY=sb_publishable_...\n```\n"
        "La clave publicable está en Supabase → Settings → API Keys. En Streamlit Cloud van en *Secrets*.")
if f.es_clave_secreta(clave):
    pantalla_de_aviso("Esa es una clave SECRETA (da permiso de escritura). Este panel solo debe usar la clave "
                      "**publicable** (`sb_publishable_...`). Cámbiala en tu `.env` o en los Secrets.", "error")
try:
    D = cargar(url, clave)
except ErrorAPI as err:
    pantalla_de_aviso(f"No pude leer la base de datos: {err}\n\n401/403: revisa la clave. 404: ejecuta "
                      "`sql/schema.sql` o expón el esquema `public` en Supabase.", "error")
if D["precios"].empty or D["pesos"].empty or D["hoy"].empty:
    pantalla_de_aviso("La base de datos todavía no tiene datos. Carga el histórico una vez con "
                      "`python run_pipeline.py --modo historico` (o desde GitHub Actions) y recarga esta página.")

# ---------------------------------------------------------------------------
# Datos derivados
# ---------------------------------------------------------------------------
nombres = D["etfs"].set_index("ticker")["nombre"].to_dict() if not D["etfs"].empty else {}
clases = D["etfs"].set_index("ticker")["clase_activo"].to_dict() if not D["etfs"].empty else {}
hoy = D["hoy"].set_index("ticker")
hoy = hoy.reindex([t for t in config.TICKERS if t in hoy.index])
tickers = list(hoy.index)
fecha_datos = D["hoy"]["fecha"].max()
ind_hoy = D["ind_hoy"].set_index("ticker") if not D["ind_hoy"].empty else pd.DataFrame(columns=panel.COLS_IND)
ancho = panel.pesos_ancho(D["pesos"])
cambio = panel.cambio_de_peso(ancho, 30)
px = panel.matriz_precios(D["precios"])
universo = panel.resumen_universo(D["precios"], ind_hoy)
curvas = panel.curva_a_base100(D["curva"]).rename(
    columns={"estrategia": E, "pesos_iguales": IG, "spy": SPY}) if not D["curva"].empty else pd.DataFrame()
resumen = D["resumen"].set_index("estrategia").reindex([E, IG, SPY]).dropna(how="all") \
    if not D["resumen"].empty else pd.DataFrame()
hay_backtest = (not curvas.empty) and E in resumen.index

hay_modelo = not D["pred"].empty and not D["metricas"].empty and PRINCIPAL in set(D["metricas"]["modelo"])
curvas_ml = panel.curva_a_base100(D["ml_curva"]).rename(
    columns={"estrategia": E, "estrategia_ml": EML, "pesos_iguales": IG, "spy": SPY}) \
    if not D["ml_curva"].empty else pd.DataFrame()
resumen_ml = D["ml_resumen"].set_index("estrategia").reindex([E, EML, IG, SPY]).dropna(how="all") \
    if not D["ml_resumen"].empty else pd.DataFrame()
fecha_pred, prob_hoy = panel.prob_de_hoy(D["pred"]) if hay_modelo else (None, pd.DataFrame())
# El perfil se elige en la pestaña Prescriptivo; el resumen muestra la decisión de ese mismo perfil.
perfil_activo = st.session_state.get("perfil") or config.PERFIL_DEFECTO
rec_activa = panel.recomendacion_de(D["recs"], perfil_activo)
decision = panel.decision_del_mes(rec_activa, universo) if rec_activa else None
SIN_MODELO = ("Todavía no hay resultados del modelo predictivo. Puede faltar historia (necesita unos tres años de "
              "precios) o ejecutar `sql/migracion_ml_llm.sql` en Supabase y volver a correr el pipeline.")


def tabla_decision(d: pd.DataFrame, perfil: str, fecha_texto: str) -> None:
    """¿Compro, mantengo o vendo? La decisión en una línea y, debajo, una fila por ETF."""
    html(c.titular_decision(perfil, d["accion"].to_dict(), fecha_texto))
    def celda_cambio(r_) -> str:
        if pd.isna(r_["cambio"]):
            return '<span class="plano">—</span>'
        return c.delta(r_["cambio"], f.pp(r_["cambio"]), neutro=True)

    columnas = ["ETF", "Decisión", "Peso del perfil", "Mes anterior", "Cambio", "Último precio (USD)",
                "Nivel de alerta (USD)", "Distancia a la alerta"]
    filas = [[c.etiqueta_etf(t), c.etiqueta_accion(r_["accion"]), f'<b>{c.e(f.pct(r_["peso"]))}</b>',
              c.e(f.pct(r_["peso_previo"])), celda_cambio(r_), c.e(f.num(r_["cierre"], 2)),
              c.e(f.num(r_["alerta"], 2)), c.e(f.pct_signo(r_["margen"]))] for t, r_ in d.iterrows()]
    banda, minimo = f.num(config.BANDA_ACCION * 100, 0), f.pct(config.PESO_MIN, 0)
    html(c.tabla(columnas, filas, izquierda=(0, 1), ayudas={
        "Decisión": f"Comprar si el peso sube {banda} puntos o más frente al cierre del mes anterior; Vender si baja "
                    f"{banda} puntos o más (solo una parte: ningún ETF baja de {minimo}); si no, Mantener.",
        "Cambio": "Diferencia en puntos porcentuales frente al cierre del mes anterior.",
        "Nivel de alerta (USD)": f"Último cierre menos {f.num(config.ALERTA_DESVIOS, 0)} veces la volatilidad de un "
                                 "mes (volatilidad anual / raíz de 12).",
        "Distancia a la alerta": "Cuánto tendría que caer el precio para llegar al nivel de alerta.",
    }))
    html(c.nota(f"**Comprar / Vender:** el peso cambia {banda} puntos o más frente al mes anterior; vender es solo una "
                f"parte. **Nivel de alerta:** si el ETF cierra por debajo, revisa la posición; no es una venta "
                "automática."))


def calculadora(pesos: pd.Series, clave: str, archivo: str) -> None:
    """Convierte un reparto de pesos en órdenes de compra y venta. `clave` distingue los controles de cada pestaña."""
    a, b, d = st.columns([1.5, 1.2, 1.3])
    with a:
        partida = elegir("Punto de partida", ["Solo efectivo", "Ya tengo una cartera"], "Solo efectivo", f"{clave}partida")
    with d:
        banda = elegir("Banda de no operación", ["0 %", "0,5 %", "1 %", "2 %"], "1 %", f"{clave}banda")
    tolerancia = {"0 %": 0.0, "0,5 %": 0.005, "1 %": 0.01, "2 %": 0.02}[banda]
    if partida == "Solo efectivo":
        with b:
            capital = st.number_input("Capital a invertir (USD)", min_value=100.0, value=10_000.0, step=500.0,
                                      format="%.0f", key=f"{clave}capital")
        actuales = pd.Series({"Efectivo": capital})
    else:
        cols = st.columns(len(tickers) + 1)
        actuales = pd.Series({
            tk: col.number_input(f"{tk} (USD)", min_value=0.0, value=2_000.0, step=100.0, format="%.0f",
                                 key=f"{clave}act_{tk}")
            for tk, col in zip(tickers, cols)})
        actuales["Efectivo"] = cols[-1].number_input("Efectivo (USD)", min_value=0.0, value=0.0, step=100.0,
                                                     format="%.0f", key=f"{clave}act_efectivo")
    if float(actuales.sum()) <= 0:
        st.info("Ingresa algún monto para calcular las órdenes.")
        return
    orden, tot = panel.ordenes(pesos, universo["cierre"], actuales, tolerancia)
    etq = {"Comprar": "compra", "Vender": "venta", "Mantener": ""}
    filas = [[c.etiqueta_etf(tk), c.e(f.pct(r["peso"])), c.e(f.usd(r["actual"])), c.e(f.usd(r["objetivo"])),
              f'<span class="mt-etq {etq[r["operacion"]]}">{c.e(r["operacion"])}</span>',
              c.e(f.usd(abs(r["monto"]))) if r["operacion"] != "Mantener" else '<span class="plano">—</span>',
              c.e(f"{f.num(abs(r['acciones']), 2)} a {f.num(r['precio'], 2)}") if r["operacion"] != "Mantener"
              else '<span class="plano">—</span>']
             for tk, r in orden.iterrows()]
    html(c.tabla(["ETF", "Peso objetivo", "Monto actual", "Tamaño de la posición", "Operación", "Monto a operar",
                  "Acciones aprox. (al último cierre)"], filas))
    html(c.nota(
        f"Capital total **{f.usd(tot['capital'])}** · {tot['compras']} compras y {tot['ventas']} ventas por "
        f"**{f.usd(tot['negociado'])}** ({f.pct(tot['negociado'] / tot['capital'])} del capital) · costo estimado "
        f"al {f.pct(config.FRICCION)}: **{f.usd(tot['negociado'] * config.FRICCION, 2)}** · efectivo sin asignar: "
        f"{f.usd(tot['efectivo_final'])}. No se opera cuando la diferencia es menor al {banda} del capital. "
        "Cálculo ilustrativo de un proyecto académico: no es una recomendación de inversión."))
    salida = orden.reset_index().rename(columns={"index": "ticker"}).round(4)
    st.download_button("Descargar órdenes (CSV)", salida.to_csv(index=False).encode("utf-8-sig"),
                       file_name=archivo, mime="text/csv", key=f"{clave}dl_ordenes")


# ---------------------------------------------------------------------------
# Cabecera e indicadores clave
# ---------------------------------------------------------------------------
retraso = panel.dias_de_retraso(fecha_datos)
chips = [c.chip("Datos al día" if retraso <= 1 else f"Datos con {retraso} días hábiles de retraso",
                estado="ok" if retraso <= 1 else "aviso"),
         c.chip("Último cierre", f.fecha(fecha_datos))]
ultima_ok = D["corridas"][D["corridas"]["estado"] == "ok"].head(1) if not D["corridas"].empty else pd.DataFrame()
if not ultima_ok.empty:
    r = ultima_ok.iloc[0]
    chips.append(c.chip("Actualizado", f"{f.hace(r.get('fin'))} · {panel.origen_legible(r['origen'])}"))
# El aviso de una actualización recién terminada se lee una sola vez: aparece como notificación en
# cualquier pestaña y como mensaje en Datos y método.
AVISO_GH = st.session_state.pop("gh_aviso", None)
if AVISO_GH == "ok":
    st.toast("Datos actualizados: el panel ya muestra la corrida nueva.", icon=":material/check_circle:")
elif AVISO_GH == "error":
    st.toast("La actualización terminó con error. El detalle está en Datos y método.", icon=":material/error:")
html(c.cabecera(config.NOMBRE_PANEL, config.ROTULO_PANEL, config.LEMA_PANEL, [], linea=False))
# Barra de estado: cómo están los datos a la izquierda y, a la derecha, el botón para actualizarlos.
with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center", gap="small"):
    st.markdown(f'<div class="mt-estado barra">{"".join(chips)}</div>', unsafe_allow_html=True, width="content")
    if all(credenciales_boton()):
        boton_en_cabecera(*credenciales_boton())
html('<div class="mt-linea"></div>')

if hay_backtest:
    e = resumen.loc[E]
    ref = resumen.loc[IG] if IG in resumen.index else None

    def frente(metrica: str, formato, bueno_si_sube: bool = True) -> str:
        """Diferencia de la regla frente a «pesos iguales» en una métrica del backtest."""
        if ref is None:
            return ""
        dif = e[metrica] - ref[metrica]
        return f"{c.delta(dif, formato(dif), bueno_si_sube)} vs. pesos iguales"

    html(c.kpis([
        {"etiqueta": "Retorno anual", "valor": f.pct(e["cagr"]), "detalle": frente("cagr", f.pp),
         "grafico": c.sparkline(curvas[E].groupby(curvas.index.to_period("M")).last().tolist(), tema.T["tinta2"], 84, 30)},
        {"etiqueta": "Ratio de Sharpe", "valor": f.num(e["sharpe"], 2), "detalle": frente("sharpe", f.num_signo)},
        {"etiqueta": "Peor caída", "valor": f.pct(e["max_drawdown"]), "detalle": frente("max_drawdown", f.pp)},
        {"etiqueta": "Volatilidad anual", "valor": f.pct(e["volatilidad"]),
         "detalle": frente("volatilidad", f.pp, bueno_si_sube=False)},
        {"etiqueta": "Rotación anual", "valor": f.pct(e["turnover_anual"], 0),
         "detalle": c.e(f"{f.num(e['n_rebalanceos'], 0)} rebalanceos · costo {f.pct(e['friccion'])}")},
    ], pie=f"Simulación histórica de la regla, del {f.fecha(e['fecha_inicio'])} al {f.fecha(e['fecha_fin'])}, "
           "neta de costos. Rendimientos pasados no garantizan resultados futuros."))

t_resumen, t_descriptivo, t_predictivo, t_prescriptivo, t_asignacion, t_backtest, t_datos = st.tabs(
    ["Resumen", "Descriptivo", "Predictivo", "Prescriptivo", "Asignación", "Backtest", "Datos y método"])

# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
with t_resumen:
    proximo = panel.proximo_rebalanceo(fecha_datos)
    html(c.seccion("¿Compro, mantengo o vendo cada ETF este mes?",
                   f"Decisión para el perfil {perfil_activo.lower()} (se cambia en la pestaña Prescriptivo), con datos "
                   f"al {f.fecha(fecha_datos)}. El tamaño de cada posición en dinero está más abajo, en la calculadora.",
                   kicker="Decisión del mes"))
    if decision is None:
        html(c.nota("Aún no hay recomendaciones guardadas: corre el pipeline para ver la decisión del mes."))
    else:
        tabla_decision(decision, perfil_activo, f.fecha(fecha_datos))
    html(f'<div class="mt-duo">'
         f'{c.tarjeta_asignacion(hoy, nombres, clases, cambio, f.fecha(fecha_datos))}'
         f'{c.tarjeta_lectura(panel.lectura(hoy, ind_hoy, cambio), f.fecha(proximo), (proximo - fecha_datos).days)}'
         f'</div>')

    mejor = universo["ret_1m"].idxmax() if universo["ret_1m"].notna().any() else None
    dato_desc = f"En el último mes lidera **{mejor}** ({f.pct_signo(universo.at[mejor, 'ret_1m'])})." if mejor \
        else "Aún no hay un mes completo de datos."
    if hay_modelo:
        lider = prob_hoy[PRINCIPAL].idxmax()
        dato_pred = f"Mayor probabilidad de subida: **{lider}** ({f.pct(prob_hoy.at[lider, PRINCIPAL], 0)})."
    else:
        dato_pred = "El modelo aún no tiene resultados."
    if decision is not None:
        partes = [f"{a.lower()} **{', '.join(decision.index[decision['accion'] == a])}**"
                  for a in ("Comprar", "Vender") if (decision["accion"] == a).any()]
        dato_presc = f"Perfil {perfil_activo.lower()}: " + (" y ".join(partes) if partes else "mantener todo") + "."
    else:
        dato_presc = "Aún no hay recomendaciones guardadas."
    html(c.seccion("Tres reportes, una decisión", "Cada pestaña responde una pregunta distinta sobre la misma cartera.",
                   kicker="Cómo leer el panel"))
    html(c.reportes([
        ("Descriptivo", "¿Qué pasó?", "Precios en velas, retornos, volatilidad, correlación e indicadores técnicos "
                                      "de los cinco ETFs.", dato_desc),
        ("Predictivo", "¿Qué puede pasar?", "Un modelo de machine learning estima la probabilidad de que cada ETF "
                                            "suba en el próximo mes, y se mide sin trampas.", dato_pred),
        ("Prescriptivo", "¿Qué hacer?", "Para cada ETF: comprar, mantener o vender este mes, con el tamaño de la "
                                        "posición y un nivel de alerta. Los números salen de reglas; un modelo de "
                                        "lenguaje los explica.", dato_presc),
    ]))

    html(c.seccion("Tamaño de la posición",
                   f"Cuánto dinero va a cada ETF con los pesos del perfil {perfil_activo.lower()}. Indica con cuánto "
                   "cuentas o qué cartera tienes hoy, y obtén qué comprar y qué vender para llegar ahí.",
                   kicker="Calculadora"))
    calculadora(decision["peso"].reindex(tickers) if decision is not None else hoy["peso"], "",
                f"ordenes_{perfil_activo.lower()}_{fecha_datos:%Y-%m-%d}.csv")

    if hay_backtest:
        html(c.seccion("Cómo le habría ido", "El detalle completo está en la pestaña Backtest.", kicker="Histórico"))
        grafico(graficos.capital(curvas, "Crecimiento de 100 invertidos",
                                 "La regla frente a dos alternativas simples, con los mismos costos", alto=340))

# ---------------------------------------------------------------------------
# Descriptivo
# ---------------------------------------------------------------------------
with t_descriptivo:
    html(c.seccion("Panorama del universo", "Los cinco ETFs de la cartera con sus retornos (incluyen dividendos), "
                   "volatilidad y distancia a su máximo de 12 meses.", kicker="Descriptivo"))
    html(c.tabla_universo(universo))

    html(c.seccion("Gráfico de precios", "Velas japonesas del ETF elegido. El ETF y el rango también aplican a los "
                   "indicadores técnicos, la comparación, la correlación y la distribución de más abajo."))
    a, b, d = st.columns([1.4, 1.3, 1.5])
    with a:
        tk = elegir("ETF", tickers, tickers[0], "m_etf")
    with b:
        rango = elegir("Rango", list(panel.RANGOS_VELAS), "6 M", "m_rango")
    with d:
        capas = st.pills("Capas", ["Media 20", "Media 50", "Volumen"], selection_mode="multi",
                         default=["Media 20", "Media 50", "Volumen"], key="m_capas") or []
    dias = panel.RANGOS_VELAS[rango]
    intervalo, regla = panel.intervalo_velas(dias)
    diario = panel.ohlc(D["precios"], tk)
    v = panel.recortar(panel.velas(diario, regla), dias)

    html(c.ficha_etf(tk, nombres.get(tk, tk), clases.get(tk, ""), universo.loc[tk]))
    grafico(graficos.velas(
        v, intervalo, capas, f"{tk} · velas {intervalo}",
        f"Del {f.fecha(v.index[0])} al {f.fecha(v.index[-1])} · precios en USD sin ajustar por dividendos"))
    datos_del_grafico(
        "Ver los datos de las velas",
        ["Fecha", "Apertura", "Máximo", "Mínimo", "Cierre", "Variación", "Volumen"],
        [[c.e(f.fecha(d_)), *(c.e(f.num(r_[k], 2)) for k in ("apertura", "maximo", "minimo", "cierre")),
          c.delta(r_["variacion"], f.pct_signo(r_["variacion"], 2), flecha=False), c.e(f.compacto(r_["volumen"]))]
         for d_, r_ in v.iloc[::-1].iterrows()])

    tec = panel.tecnicos(diario)
    ult, lect = tec.iloc[-1], panel.lectura_tecnica(tec.iloc[-1])
    html(c.seccion("Indicadores técnicos", f"Calculados con los cierres diarios de {tk}. El RSI mide si el precio "
                   "subió o bajó demasiado rápido; el MACD compara dos medias móviles para ver hacia dónde va el "
                   "impulso; el ATR es cuánto se mueve el precio en un día típico."))
    html(c.kpis([
        {"etiqueta": "RSI (14 días)", "valor": f.num(ult["rsi"], 1), "detalle": c.e(lect["rsi"])},
        {"etiqueta": "MACD menos su señal", "valor": f.num_signo(ult["hist"], 2), "detalle": c.e(lect["macd"])},
        {"etiqueta": "ATR (14 días)", "valor": f.usd(ult["atr"], 2),
         "detalle": c.e(f"{f.pct(ult['atr_pct'])} del precio en un día típico")},
        {"etiqueta": "Volatilidad 60 días", "valor": f.pct(universo.at[tk, "vol_60d"]), "detalle": c.e("anualizada")},
    ], columnas=4))
    tec_r = panel.recortar(tec.dropna(subset=["rsi", "macd"]), dias)
    if len(tec_r) >= 2:
        grafico(graficos.tecnicos(tec_r, f"{tk} · RSI y MACD",
                                  "RSI sobre 70: sobrecompra; bajo 30: sobreventa · barras: MACD menos su señal"))
        datos_del_grafico(
            "Ver los datos de los indicadores técnicos (últimos 260 días del rango)",
            ["Fecha", "RSI", "MACD", "Señal", "MACD − señal", "ATR", "ATR / precio"],
            [[c.e(f.fecha(d_)), c.e(f.num(r_["rsi"], 1)), c.e(f.num_signo(r_["macd"], 3)), c.e(f.num_signo(r_["senal"], 3)),
              c.delta(r_["hist"], f.num_signo(r_["hist"], 3), flecha=False), c.e(f.num(r_["atr"], 2)),
              c.e(f.pct(r_["atr_pct"]))] for d_, r_ in tec_r.iloc[::-1].head(260).iterrows()])

    px_r = panel.recortar(px, dias)
    idx, corr = panel.base100(px_r), panel.correlaciones(px_r)
    a, b = st.columns([1.25, 1])
    with a:
        grafico(graficos.rendimiento(idx, "Rendimiento comparado",
                                     f"Precio con dividendos, base 100 al {f.fecha(idx.index[0])}"))
    with b:
        grafico(graficos.correlacion(corr, "Correlación entre ETFs",
                                     "Retornos diarios del rango · azul: juntos · ámbar: opuestos"))
    datos_del_grafico(
        "Ver los datos de rendimiento y correlación",
        ["ETF", "Base 100 hoy", "Retorno en el rango", *[f"Corr. con {t}" for t in corr.columns]],
        [[c.etiqueta_etf(t), c.e(f.num(idx[t].iloc[-1], 1)), c.delta(idx[t].iloc[-1] - 100,
                                                                       f.pct_signo(idx[t].iloc[-1] / 100 - 1), flecha=False),
          *(c.e(f.num(corr.at[t, o], 2)) for o in corr.columns)] for t in idx.columns])

    html(c.seccion("Distribución de los retornos diarios", "Qué tan seguido sube o baja cada ETF en un día y por "
                   "cuánto, en el rango elegido. Sirve para dimensionar un día «normal» y uno extremo."))
    r_tk = px_r[tk].pct_change().dropna()
    est = panel.estadisticas_retornos(px_r)
    a, b = st.columns([1, 1.3])
    with a:
        grafico(graficos.histograma(panel.histograma(r_tk), float(r_tk.mean()), tema.color_etf(tk),
                                    f"{tk} · retornos de un día", f"Número de días en cada tramo · {len(r_tk)} días"))
    with b:
        html(c.tabla(
            ["ETF", "Media diaria", "Desviación", "Días en positivo", "Mejor día", "Peor día", "Asimetría", "Curtosis"],
            [[c.etiqueta_etf(t), c.delta(r_["media"], f.pct_signo(r_["media"], 2), flecha=False),
              c.e(f.pct(r_["desviacion"], 2)), c.e(f.pct(r_["positivos"], 0)),
              c.delta(r_["mejor"], f.pct_signo(r_["mejor"]), flecha=False),
              c.delta(r_["peor"], f.pct_signo(r_["peor"]), flecha=False),
              c.e(f.num_signo(r_["asimetria"], 2)), c.e(f.num_signo(r_["curtosis"], 2))] for t, r_ in est.iterrows()]))
        html(c.nota("**Asimetría** negativa: las caídas grandes son más frecuentes que las subidas grandes. "
                    "**Curtosis** mayor que cero: hay más días extremos de los que tendría una distribución normal."))

# ---------------------------------------------------------------------------
# Predictivo
# ---------------------------------------------------------------------------
with t_predictivo:
    if not hay_modelo:
        st.info(SIN_MODELO)
    else:
        met = D["metricas"].set_index("modelo")
        mp = met.loc[PRINCIPAL]
        nombre_principal = config.NOMBRES_MODELOS[PRINCIPAL]
        html(c.seccion("¿Subirá cada ETF en el próximo mes?",
                       f"Un modelo de clasificación estima la probabilidad de que el precio de cada ETF (con dividendos) "
                       f"esté más alto dentro de {config.HORIZONTE_ML} días hábiles. Se entrena de nuevo cada fin de mes "
                       "y solo con datos anteriores a ese día.", kicker="Predictivo"))
        ref_ex = met.at["frecuencia", "exactitud"] if "frecuencia" in met.index else float("nan")
        html(c.kpis([
            {"etiqueta": "Exactitud", "valor": f.pct(mp["exactitud"]),
             "detalle": f"{c.delta(mp['exactitud'] - ref_ex, f.pp(mp['exactitud'] - ref_ex))} vs. referencia simple"},
            {"etiqueta": "AUC", "valor": f.num(mp["auc"], 2), "detalle": c.e("0,50 equivale a adivinar")},
            {"etiqueta": "F1", "valor": f.num(mp["f1"], 2),
             "detalle": c.e(f"precisión {f.pct(mp['precision_pos'], 0)} · sensibilidad {f.pct(mp['sensibilidad'], 0)}")},
            {"etiqueta": "Correlación con el retorno real", "valor": f.num_signo(mp["ic_spearman"], 2),
             "detalle": c.e("IC de Spearman")},
            {"etiqueta": "Predicciones evaluadas", "valor": f.num(mp["n"], 0),
             "detalle": c.e(f"{f.num(mp['n_fechas'], 0)} meses × {len(tickers)} ETFs")},
        ], pie=f"{nombre_principal}, del {f.fecha(mp['fecha_desde'])} al {f.fecha(mp['fecha_hasta'])}. Todo es fuera "
               "de muestra: cada predicción la hizo un modelo que no conocía ese mes."))

        a, b = st.columns([1.15, 1])
        with a:
            otro = "gb" if "gb" in prob_hoy else None
            grafico(graficos.probabilidades(
                prob_hoy, PRINCIPAL, otro, config.NOMBRES_MODELOS, "Probabilidad de subida a un mes",
                f"Al {f.fecha(fecha_pred)} · barra: {nombre_principal.lower()}" + (" · rombo: boosting" if otro else "")))
        with b:
            html(c.tarjeta("Qué tan confiable es", c.lista(panel.veredicto_del_modelo(D["metricas"], D["ml_resumen"])),
                           sub="Lectura directa de las métricas, sin adornos"))
        modelos_hoy = [m_ for m_ in ("logit", "gb", "momentum", "frecuencia") if m_ in prob_hoy]
        datos_del_grafico(
            "Ver las probabilidades de hoy por modelo",
            ["ETF", *[config.NOMBRES_MODELOS[m_] for m_ in modelos_hoy], "Retorno 6 meses", "Volatilidad 60 d"],
            [[c.etiqueta_etf(t), *(c.e(f.pct(r_[m_], 0)) for m_ in modelos_hoy),
              c.delta(universo.at[t, "ret_6m"], f.pct_signo(universo.at[t, "ret_6m"]), flecha=False),
              c.e(f.pct(universo.at[t, "vol_60d"]))] for t, r_ in prob_hoy.iterrows()])

        html(c.seccion("Desempeño fuera de muestra", "Los dos modelos entrenados frente a dos referencias que no "
                       "aprenden nada. Un modelo solo vale si les gana."))
        html(c.tabla(
            ["Modelo", "Predicciones", "Exactitud", "Precisión", "Sensibilidad", "F1", "AUC", "Brier", "IC de Spearman"],
            [[f'{c.e(config.NOMBRES_MODELOS.get(n_, n_))}'
              + (' <span class="mt-etq acento">principal</span>' if n_ == PRINCIPAL else ""),
              c.e(f.num(r_["n"], 0)), c.e(f.pct(r_["exactitud"])), c.e(f.pct(r_["precision_pos"])),
              c.e(f.pct(r_["sensibilidad"])), c.e(f.num(r_["f1"], 2)), c.e(f.num(r_["auc"], 2)),
              c.e(f.num(r_["brier"], 3)), c.e(f.num_signo(r_["ic_spearman"], 2))] for n_, r_ in met.iterrows()],
            ayudas=AYUDAS_METRICAS))
        html(c.nota(
            f"Se predice «sube» con una probabilidad de 50 % o más; pasa el mouse sobre cada métrica para ver qué mide. "
            f"En el periodo los ETFs subieron el {f.pct(mp['tasa_positivos'], 0)} de los meses, y las cinco predicciones "
            "de un mismo mes no son independientes: la muestra efectiva es menor de lo que parece."))

        cal = panel.calibracion(D["pred"], PRINCIPAL)
        a, b = st.columns([1.15, 1])
        with a:
            if len(cal) >= 2:
                grafico(graficos.calibracion(cal, "¿Sus probabilidades significan lo que dicen?",
                                             "Predicciones agrupadas en tramos de igual tamaño · sobre la diagonal, "
                                             "calibración perfecta"))
            else:
                html(c.nota("Todavía hay muy pocas predicciones resueltas para dibujar la calibración."))
        with b:
            html(c.tarjeta("Aciertos y errores", c.confusion(panel.matriz_confusion(D["pred"], PRINCIPAL)),
                           sub=f"{nombre_principal} · umbral de 50 %"))
        if len(cal) >= 2:
            datos_del_grafico("Ver los datos de calibración", ["Tramo", "Probabilidad media", "Veces que subió", "Casos"],
                              [[c.e(i + 1), c.e(f.pct(r_["prob_media"])), c.e(f.pct(r_["frecuencia"])),
                                c.e(f.num(r_["n"], 0))] for i, r_ in cal.iterrows()], izquierda=())

        imp = D["importancias"]
        if not imp.empty:
            html(c.seccion("Qué mira el modelo", "Las diez variables se calculan con precios hasta el día de la "
                           "predicción. Se muestra su peso en el último modelo entrenado."))
            a, b = st.columns(2)
            with a:
                grafico(graficos.importancia(imp[imp["modelo"] == "logit"], True, "Regresión logística",
                                             "Coeficientes (variables estandarizadas) · azul empuja a «sube», ámbar a «no sube»"))
            with b:
                grafico(graficos.importancia(imp[imp["modelo"] == "gb"], False, "Gradient boosting",
                                             "Parte de la mejora del modelo que aporta cada variable"))
            datos_del_grafico(
                "Ver el peso de cada variable", ["Variable", "Coeficiente (logística)", "Importancia (boosting)"],
                [[c.e(config.ETIQUETAS_VARIABLES.get(v_, v_)),
                  *(c.e(fmt(imp[(imp["modelo"] == m_) & (imp["variable"] == v_)]["importancia"].sum()))
                    for m_, fmt in (("logit", lambda x: f.num_signo(x, 3)), ("gb", lambda x: f.pct(x))))]
                 for v_ in config.ETIQUETAS_VARIABLES])

        if not curvas_ml.empty and EML in resumen_ml.index:
            html(c.seccion("¿Mejora a la regla?", "La prueba que importa: la misma regla de pesos, con y sin la "
                           "inclinación del modelo, en el mismo periodo, con las mismas fechas y los mismos costos."))
            grafico(graficos.capital(curvas_ml, "Crecimiento de 100 invertidos, con y sin modelo",
                                     f"Del {f.fecha(curvas_ml.index[0])} al {f.fecha(curvas_ml.index[-1])}, después de costos"))
            html(c.tabla(
                ["Estrategia", "Retorno total", "Retorno anual", "Volatilidad", "Sharpe", "Peor caída", "Rotación anual"],
                [[f'<span class="mt-tk"><i class="mt-sw" style="background:{graficos.ESTILO[n_]["color"]}"></i>{c.e(n_)}</span>',
                  c.e(f.pct(r_["retorno_total"])), c.e(f.pct(r_["cagr"])), c.e(f.pct(r_["volatilidad"])),
                  c.e(f.num(r_["sharpe"], 2)), c.e(f.pct(r_["max_drawdown"])), c.e(f.pct(r_["turnover_anual"], 0))]
                 for n_, r_ in resumen_ml.iterrows()]))
            html(c.nota(
                f"El modelo mueve cada peso como máximo ±{f.pct(config.INCLINACION_ML, 0)} antes de aplicar los límites. "
                f"Retornos netos de un costo de {f.pct(config.FRICCION)} por unidad negociada. Si «{EML}» no supera a "
                f"«{E}» en Sharpe, la recomendación de la pestaña Prescriptivo usa la regla sin el modelo."))

        html(c.seccion("Cómo se evita que el modelo haga trampa", kicker="Metodología"))
        dos_tarjetas(
            c.tarjeta("Validación", c.lista([
                "**Walk-forward.** Cada fin de mes se entrena un modelo nuevo con todo lo anterior y predice el mes "
                "siguiente. Nunca se evalúa con datos que usó para aprender.",
                f"**Embargo de {config.HORIZONTE_ML} días.** Saber si un ETF subió toma un mes; por eso solo se "
                "entrena con casos cuyo resultado ya se conocía el día de la predicción.",
                "**Preparación dentro del entrenamiento.** El escalado de las variables se ajusta en cada "
                "entrenamiento, no con toda la historia.",
                f"**Parámetros fijos.** Se definieron antes de ver resultados y no se optimizan; hacen falta "
                f"{f.num(config.MIN_ENTRENAMIENTO, 0)} días de historia para la primera predicción.",
            ])),
            c.tarjeta("Límites", c.lista([
                "**Predecir un mes es difícil.** En mercados líquidos lo normal es un AUC apenas sobre 0,50; un valor "
                "muy alto sería señal de error, no de talento.",
                "**Pocas observaciones independientes.** Hay una decisión por mes y los ETFs se mueven juntos.",
                "**Universo fijo.** Los cinco ETFs se eligieron hoy; el modelo no dice nada sobre otros activos.",
                "**No predice precios.** Solo estima la probabilidad de que el retorno del mes sea positivo.",
            ])))

# ---------------------------------------------------------------------------
# Prescriptivo
# ---------------------------------------------------------------------------
with t_prescriptivo:
    html(c.seccion("¿Compro, mantengo o vendo cada ETF este mes?",
                   "Los pesos, las decisiones y los niveles de alerta los calcula el código con reglas fijas. Un "
                   "modelo de lenguaje solo redacta la explicación, y su texto se revisa antes de publicarse.",
                   kicker="Prescriptivo"))
    perfil = elegir("Perfil de riesgo", list(config.PERFILES), config.PERFIL_DEFECTO, "perfil")
    rec = panel.recomendacion_de(D["recs"], perfil)
    if rec is None:
        st.info("Todavía no hay recomendaciones guardadas. Ejecuta `sql/migracion_ml_llm.sql` en Supabase (si no lo "
                "hiciste) y vuelve a correr el pipeline.")
    else:
        det = rec["detalle"]
        etfs = pd.DataFrame(det["etfs"]).set_index("ticker")
        minimo, maximo = det["rango"]
        por_llm = rec["fuente"] == "llm"
        etiquetas = [f'<span class="mt-etq acento">Redactado por {c.e(texto(rec.get("modelo_llm"), "LLM"))} · validado</span>'
                     if por_llm else '<span class="mt-etq">Texto por plantilla (sin LLM)</span>']
        if det.get("confianza"):
            etiquetas.append(f'<span class="mt-etq">Confianza del modelo: {c.e(det["confianza"])}</span>')
        etiquetas.append(f'<span class="mt-etq">{"Pesos con modelo" if det.get("usa_modelo") else "Pesos sin modelo"}</span>')
        html(c.tarjeta_recomendacion(
            f"Recomendación para el perfil {perfil.lower()}",
            f"Con datos al {f.fecha(rec['fecha'])} · renta variable {f.pct(det['renta_variable'])} "
            f"(el perfil admite de {f.pct(minimo, 0)} a {f.pct(maximo, 0)})",
            etiquetas, rec["resumen"],
            c.barra_segmentada([(t, float(r_["peso"]), tema.color_etf(t))
                                for t, r_ in etfs.sort_values("peso", ascending=False).iterrows()])))

        decision_perfil = panel.decision_del_mes(rec, universo)
        tabla_decision(decision_perfil, perfil, f.fecha(rec["fecha"]))

        html(c.seccion("Por qué, ETF por ETF", f"La probabilidad de subida es la del modelo predictivo a "
                       f"{config.HORIZONTE_ML} días hábiles."))
        html(c.tabla(["ETF", "Decisión", "Prob. de subida", "Por qué"],
                     [[c.etiqueta_etf(t), c.etiqueta_accion(r_["accion"]), c.e(f.pct(r_["prob_sube"], 0)),
                       c.e(r_["explicacion"])] for t, r_ in decision_perfil.iterrows()],
                     izquierda=(0, 1), texto=(3,)))

        dos_tarjetas(c.tarjeta("Riesgos", c.lista(det["riesgos"])),
                     c.tarjeta("Qué vigilar", c.lista(det["que_vigilar"])))
        html(c.aviso(det.get("aviso") or prescriptivo.AVISO_LEGAL))

        html(c.seccion("Tamaño de la posición", "Cuánto dinero va a cada ETF con los pesos del perfil elegido, y qué "
                       "comprar o vender para llegar ahí.", kicker="Calculadora"))
        calculadora(etfs["peso"].reindex(tickers), "p_", f"ordenes_{perfil.lower()}_{rec['fecha']:%Y-%m-%d}.csv")

        html(c.seccion("Cómo se generó", "Para auditar la recomendación: quién decide cada cosa, qué se revisa y cuánto "
                       "se usa el modelo de lenguaje.", kicker="Trazabilidad"))
        with st.expander("Quién decide qué y qué se revisa antes de publicar"):
            dos_tarjetas(
                c.tarjeta("Quién decide qué", c.lista([
                    "**El código** parte de los pesos de la regla (con la inclinación del modelo predictivo solo si "
                    "mejoró a la regla fuera de muestra), los lleva al rango de renta variable del perfil y respeta "
                    f"el mínimo de {f.pct(config.PESO_MIN, 0)} y el tope de {f.pct(config.PESO_MAX, 0)} por ETF.",
                    "**El código** decide si conviene comprar, mantener o vender cada ETF comparando con el cierre del "
                    "mes anterior, y calcula su nivel de alerta con la volatilidad.",
                    f"**El modelo de lenguaje** ({config.LLM_PROVEEDOR}) recibe esos números y redacta el resumen, la "
                    "explicación por ETF, los riesgos y qué vigilar. No puede cambiar pesos ni acciones.",
                    "**Si el modelo de lenguaje falla**, no hay clave o se llegó al límite de llamadas, el texto sale "
                    "de una plantilla fija con los mismos datos.",
                ])),
                c.tarjeta("Revisiones antes de publicar", c.lista([
                    *prescriptivo.VALIDACIONES,
                    "Si el texto no pasa, se pide una corrección indicando el motivo; si vuelve a fallar, se usa la plantilla.",
                    f"Máximo {config.LLM_LIMITE_DIA} llamadas por día y {config.LLM_LIMITE_CORRIDA} por ejecución, "
                    f"{f.num(config.LLM_MAX_TOKENS_SALIDA, 0)} tokens de respuesta y {config.LLM_TIEMPO_ESPERA} s de espera.",
                ])))

    log = D["llm_log"]
    with st.expander("Uso del modelo de lenguaje: llamadas, tokens y costo"):
        html(c.nota("Cada llamada queda registrada en la base de datos con su estado, sus tokens y su duración."))
        if log.empty:
            html(c.nota("Aún no hay llamadas registradas."))
        else:
            uso = panel.uso_del_llm(log)
            costo = panel.costo_por_cien(uso)
            tokens = uso["tokens_entrada"] + uso["tokens_salida"]
            html(c.kpis([
                {"etiqueta": "Llamadas hoy", "valor": f"{uso['hoy']} de {config.LLM_LIMITE_DIA}",
                 "detalle": c.e("límite diario, en horario UTC")},
                {"etiqueta": "Textos aceptados", "valor": f"{uso['aceptadas']} de {uso['llamadas']}",
                 "detalle": c.e("pasaron las cuatro revisiones")},
                {"etiqueta": "Tokens por recomendación", "valor": f.num(tokens, 0),
                 "detalle": c.e(f"{f.num(uso['tokens_entrada'], 0)} de entrada · {f.num(uso['tokens_salida'], 0)} de salida")},
                {"etiqueta": "Costo de 100 recomendaciones", "valor": f.usd(costo, 2) if costo is not None else f.VACIO,
                 "detalle": c.e("tarifa de pago; cero en la capa gratuita" if costo is not None
                                else "sin tarifa registrada para ese modelo" if uso["aceptadas"] else "sin textos aceptados")},
            ], columnas=4))
            estados_llm = {"ok": ("up", "✔ Aceptada"), "rechazada": ("down", "✖ Rechazada"), "error": ("down", "✖ Error"),
                           "limite": ("plano", "■ Límite alcanzado"), "sin_clave": ("plano", "■ Sin clave")}
            html(c.tabla(
                ["Registro", "Día", "Perfil", "Modelo", "Estado", "Tokens de entrada", "Tokens de salida", "Duración", "Detalle"],
                [[c.e(f"#{int(r_['id'])}"), c.e(f.fecha(r_["dia"])), c.e(texto(r_.get("perfil"), f.VACIO)),
                  c.e(texto(r_.get("modelo"), f.VACIO)),
                  f'<span class="{estados_llm.get(r_["estado"], ("plano", ""))[0]}">'
                  f'{c.e(estados_llm.get(r_["estado"], ("", r_["estado"]))[1])}</span>',
                  c.e(f.num(r_["tokens_entrada"], 0)), c.e(f.num(r_["tokens_salida"], 0)),
                  c.e(f"{f.num(r_['latencia_ms'] / 1000, 1)} s" if pd.notna(r_["latencia_ms"]) else f.VACIO),
                  c.e(texto(r_.get("mensaje")))] for _, r_ in log.head(15).iterrows()],
                izquierda=(0, 1, 2, 3, 4), texto=(8,), alto_max=360))

# ---------------------------------------------------------------------------
# Asignación
# ---------------------------------------------------------------------------
with t_asignacion:
    html(c.seccion("De dónde sale cada peso", "La regla parte de un peso por riesgo y lo ajusta por momentum, "
                   "siempre dentro de los límites.", kicker="Asignación"))
    riesgo = panel.contribucion_riesgo(hoy["peso"], px.pct_change().tail(config.VENTANA_VOL))
    a, b = st.columns(2)
    with a:
        grafico(graficos.base_vs_final(hoy, "Del peso por riesgo al peso final",
                                       "El círculo hueco es el peso solo por riesgo; el lleno, el peso sugerido"))
    with b:
        grafico(graficos.peso_vs_riesgo(hoy["peso"], riesgo, "Peso frente a aporte al riesgo",
                                        f"Parte de la volatilidad de la cartera que explica cada ETF ({config.VENTANA_VOL} días)"))

    def estado_limite(p: float) -> str:
        if p >= config.PESO_MAX - 1e-4:
            return '<span class="mt-etq acento">En el tope</span>'
        if p <= config.PESO_MIN + 1e-4:
            return '<span class="mt-etq acento">En el mínimo</span>'
        return '<span class="mt-etq">Libre</span>'

    def de_ind(tk_: str, col: str):
        return ind_hoy.at[tk_, col] if (tk_ in ind_hoy.index and col in ind_hoy) else float("nan")

    html(c.seccion("Señales de hoy", "Los datos que usa la regla para cada ETF y el resultado."))
    html(c.tabla(
        ["ETF", "Volatilidad 60 d", "Peso por riesgo", "Momentum", "Retorno 6 m", "Factor", "Límites",
         "Peso final", "Aporte al riesgo", "Explicación"],
        [[c.etiqueta_etf(t), c.e(f.pct(de_ind(t, "vol_60d"))), c.e(f.pct(r_["peso_base"])),
          c.delta(de_ind(t, "momentum"), f.pct_signo(de_ind(t, "momentum")), flecha=False),
          c.delta(de_ind(t, "ret_6m"), f.pct_signo(de_ind(t, "ret_6m")), flecha=False),
          c.e("×" + f.num(r_["factor_momentum"], 2)), estado_limite(r_["peso"]),
          f'<b>{c.e(f.pct(r_["peso"]))}</b>', c.e(f.pct(riesgo.get(t))), c.e(r_.get("motivo") or "")]
         for t, r_ in hoy.iterrows()], texto=(9,)))

    html(c.seccion("Cómo ha cambiado", "El periodo aplica a los dos gráficos siguientes."))
    periodo = elegir("Periodo", list(panel.PERIODOS), "3 A", "a_periodo")
    dias_p = panel.PERIODOS[periodo]
    mensual = panel.recortar(panel.fin_de_mes(ancho), dias_p)
    grafico(graficos.evolucion_pesos(mensual, "Reparto sugerido mes a mes",
                                     "Peso de cada ETF al cierre de cada mes, que es cuando la regla decide"))
    datos_del_grafico("Ver los pesos de cada mes", ["Mes", *mensual.columns],
                      [[c.e(f.mes_anio(d_)), *(c.e(f.pct(x)) for x in r_)] for d_, r_ in mensual.iloc[::-1].iterrows()])
    vol = panel.recortar(calcular_indicadores(px)["vol_60d"].dropna(how="any"), dias_p)
    grafico(graficos.volatilidad(vol, "Volatilidad de cada ETF",
                                 f"Anualizada, ventana de {config.VENTANA_VOL} días · a más volatilidad, menos peso base"))
    vol_mes = panel.fin_de_mes(vol)
    datos_del_grafico("Ver la volatilidad de cada mes", ["Mes", *vol_mes.columns],
                      [[c.e(f.mes_anio(d_)), *(c.e(f.pct(x)) for x in r_)] for d_, r_ in vol_mes.iloc[::-1].iterrows()])

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
with t_backtest:
    if not hay_backtest:
        st.info("Todavía no hay resultados de backtest.")
    else:
        e = resumen.loc[E]
        html(c.seccion("¿Habría funcionado?", "Simulación de la regla sobre el pasado, comparada con mantener los "
                       "cinco ETFs en partes iguales y con tener solo SPY.", kicker="Backtest"))
        grafico(graficos.capital(curvas, "Crecimiento de 100 invertidos",
                                 f"Del {f.fecha(curvas.index[0])} al {f.fecha(curvas.index[-1])}, después de costos"))
        html(c.tabla(
            ["Estrategia", "Retorno total", "Retorno anual", "Volatilidad", "Sharpe", "Peor caída", "Rotación anual",
             "Rebalanceos"],
            [[f'<span class="mt-tk"><i class="mt-sw" style="background:{graficos.ESTILO[n]["color"]}"></i>{c.e(n)}</span>',
              c.e(f.pct(r_["retorno_total"])), c.e(f.pct(r_["cagr"])), c.e(f.pct(r_["volatilidad"])),
              c.e(f.num(r_["sharpe"], 2)), c.e(f.pct(r_["max_drawdown"])), c.e(f.pct(r_["turnover_anual"], 0)),
              c.e(f.num(r_["n_rebalanceos"], 0))] for n, r_ in resumen.iterrows()]))
        html(c.nota(
            f"Se decide al cierre del último día hábil de cada mes y se opera al cierre del día hábil siguiente, con un "
            f"costo de {f.pct(e['friccion'])} por cada unidad de capital negociada. Las tres estrategias usan las mismas "
            "fechas y los mismos costos. Sharpe con tasa libre de riesgo igual a cero. Los parámetros de la regla se "
            "fijaron antes de ver estos resultados."))

        tabla_anual, parciales = panel.retornos_anuales(curvas)
        dd = panel.drawdowns(curvas)
        a, b = st.columns(2)
        with a:
            grafico(graficos.caidas(dd, "Caídas desde el máximo",
                                    "Cuánto se pierde en los peores momentos y cuánto tarda en recuperarse"))
        with b:
            grafico(graficos.anuales(tabla_anual, parciales, "Retorno por año",
                                     "Año calendario" + (" · * año incompleto" if parciales else "")))
        capital_fin = curvas.groupby(curvas.index.year).last()
        peor = dd.groupby(dd.index.year).min()
        datos_del_grafico(
            "Ver los datos por año",
            ["Año", *[f"Retorno · {n}" for n in tabla_anual.columns], *[f"Capital · {n}" for n in capital_fin.columns],
             *[f"Peor caída · {n}" for n in peor.columns]],
            [[c.e(f"{a_}*" if a_ in parciales else a_),
              *(c.delta(x, f.pct_signo(x), flecha=False) for x in tabla_anual.loc[a_]),
              *(c.e(f.num(x, 1)) for x in capital_fin.loc[a_]), *(c.e(f.pct(x)) for x in peor.loc[a_])]
             for a_ in tabla_anual.index[::-1]])

        r_mes, tabla_mes = panel.retornos_mensuales(curvas[E])
        m = panel.resumen_mensual(r_mes)
        html(c.seccion("Mes a mes", f"Retorno mensual de la estrategia «{E}»."))
        html(c.kpis([
            {"etiqueta": "Meses en positivo", "valor": f.pct(m["positivos"], 0),
             "detalle": c.e(f"{f.num(m['n'], 0)} meses analizados")},
            {"etiqueta": "Mejor mes", "valor": f.pct_signo(m["mejor"]), "detalle": c.e(f.mes_anio(m["mejor_mes"].to_timestamp()))},
            {"etiqueta": "Peor mes", "valor": f.pct_signo(m["peor"]), "detalle": c.e(f.mes_anio(m["peor_mes"].to_timestamp()))},
            {"etiqueta": "Retorno mensual medio", "valor": f.pct_signo(float(r_mes.mean()), 2),
             "detalle": c.e(f"mediana {f.pct_signo(float(r_mes.median()), 2)}")},
        ], columnas=4))
        grafico(graficos.calor_mensual(tabla_mes, "Retornos mensuales (%)",
                                       "Verde: el mes cerró en positivo · rojo: en negativo"))

# ---------------------------------------------------------------------------
# Datos y método
# ---------------------------------------------------------------------------
with t_datos:
    html(c.seccion("Cómo funciona el sistema", "Un proceso programado actualiza la base de datos; este panel solo lee.",
                   kicker="Arquitectura"))
    html(c.flujo([
        ("Extracción", "Cada día hábil a las 18:00 (hora de Perú), o al pulsar «Actualizar ahora», se descargan "
                       "los precios de cierre de los cinco ETFs.",
         "GitHub Actions · yfinance"),
        ("Almacenamiento", "Los precios se guardan en una base de datos relacional a través de su API. "
                           "Repetir una carga no duplica filas.", "Supabase · PostgreSQL · API REST"),
        ("Cálculo y modelo", "Con lo guardado se calculan indicadores, pesos y backtest, y se entrena el modelo que "
                             "estima la probabilidad de subida.", "Python · pandas · scikit-learn"),
        ("Recomendación", "Reglas fijas definen pesos y acciones por perfil; un modelo de lenguaje redacta la "
                          "explicación y el código la valida.", f"{config.LLM_PROVEEDOR} · API"),
        ("Publicación", "Este panel consulta la base con una clave de solo lectura y arma las vistas.",
         "Streamlit · Plotly"),
    ]))

    p = D["precios"]
    html(c.seccion("Cobertura de los datos"))
    html(c.kpis([
        {"etiqueta": "Filas de precios", "valor": f.num(len(p), 0), "detalle": c.e(f"{p['ticker'].nunique()} ETFs")},
        {"etiqueta": "Días con datos", "valor": f.num(p["fecha"].nunique(), 0), "detalle": c.e("ruedas de bolsa")},
        {"etiqueta": "Primer dato", "valor": f.fecha(p["fecha"].min()), "detalle": c.e("inicio del histórico")},
        {"etiqueta": "Último dato", "valor": f.fecha(p["fecha"].max()),
         "detalle": c.e("al día" if retraso <= 1 else f"{retraso} días hábiles de retraso")},
        {"etiqueta": "Decisiones registradas", "valor": f.num(len(D["pesos"]), 0), "detalle": c.e("filas de pesos sugeridos")},
    ]))

    seccion_actualizacion(AVISO_GH)

    html(c.seccion("Bitácora de ejecuciones", "Cada corrida del proceso queda registrada en la base de datos. "
                   "El panel guarda lo leído durante 10 minutos; el botón fuerza una lectura nueva."))
    if st.button("Volver a leer la base de datos", key="releer"):
        cargar.clear()
        st.rerun()
    if D["corridas"].empty:
        st.info("Aún no hay ejecuciones registradas.")
    else:
        estados = {"ok": ("up", "✔ Correcta"), "error": ("down", "✖ Con error"), "en_curso": ("plano", "● En curso")}
        html(c.tabla(
            ["Corrida", "Estado", "Modo", "Origen", "Inicio (hora de Perú)", "Duración", "Precios", "Indicadores", "Pesos",
             "Detalle"],
            [[c.e(f"#{int(r_['id'])}"),
              f'<span class="{estados.get(r_["estado"], ("plano", ""))[0]}">{c.e(estados.get(r_["estado"], ("", r_["estado"]))[1])}</span>',
              c.e({"historico": "Histórico", "diario": "Diario"}.get(r_["modo"], r_["modo"])), c.e(con_mayuscula(panel.origen_legible(r_["origen"]))),
              c.e(f.fecha_hora(r_.get("inicio"))), c.e(f.duracion(r_.get("inicio"), r_.get("fin"))),
              c.e(f.num(r_.get("filas_precios"), 0)), c.e(f.num(r_.get("filas_indicadores"), 0)),
              c.e(f.num(r_.get("filas_pesos"), 0)), c.e(texto(r_.get("mensaje")))]
             for _, r_ in D["corridas"].iterrows()], izquierda=(0, 1, 2, 3), texto=(9,), alto_max=360))

    html(c.seccion("Método", kicker="Metodología"))
    dos_tarjetas(c.tarjeta(
        "Cómo se calculan los pesos", c.lista([
            f"**Peso por riesgo.** Cada ETF recibe un peso proporcional a 1 / volatilidad ({config.VENTANA_VOL} días): "
            "los más estables pesan más.",
            "**Ajuste por momentum.** El momentum es el promedio de los retornos a 3, 6 y 12 meses. Se compara entre "
            "los cinco ETFs y multiplica el peso entre ×0,5 y ×1,5.",
            "**Filtro.** Si el retorno a 6 meses de un ETF es negativo, su peso se reduce a la mitad.",
            f"**Límites.** Ningún ETF baja de {f.pct(config.PESO_MIN, 0)} ni supera {f.pct(config.PESO_MAX, 0)}; "
            "los pesos siempre suman 100 %.",
        ])), c.tarjeta(
        "Alcance y límites", c.lista([
            "**Proyecto académico.** No ejecuta órdenes ni constituye asesoría de inversión.",
            "**Universo fijo.** Los cinco ETFs se eligieron hoy, conociendo su historia: eso favorece al backtest.",
            "**El pasado no se repite.** Los resultados dependen del periodo analizado.",
            "**Costos simplificados.** Se asume un costo fijo por operación; no hay impuestos ni deslizamiento.",
            "**Fuente de precios.** Yahoo Finance mediante yfinance, una librería no oficial de uso educativo.",
        ])))

    html(c.seccion("Fases CRISP-DM"))
    html(c.tabla(["Fase", "Qué se hizo", "Dónde está"], [
        ["Comprensión del negocio", "Definir la decisión: cómo repartir el capital entre cinco ETFs según el perfil", "README"],
        ["Comprensión de los datos", "Precios diarios, validación, cobertura y análisis descriptivo",
         "etf2/datos.py · tabla precios_diarios · pestaña Descriptivo"],
        ["Preparación de los datos", "Retornos, volatilidad, momentum, drawdown, RSI, MACD y ATR",
         "etf2/indicadores.py · etf2/tecnico.py"],
        ["Modelado", "Regla de pesos por riesgo y momentum; modelo que estima la probabilidad de subida a un mes",
         "etf2/pesos.py · etf2/predictivo.py"],
        ["Evaluación", "Backtest con costos, validación walk-forward, comparación con referencias y pruebas automáticas",
         "etf2/backtest.py · etf2/predictivo.py · tests/"],
        ["Despliegue", "Actualización programada, base de datos, recomendación con LLM validado y este panel",
         ".github/workflows · Supabase · etf2/prescriptivo.py · dashboard.py"],
    ], izquierda=(0, 1, 2), tenue=(1, 2)))
