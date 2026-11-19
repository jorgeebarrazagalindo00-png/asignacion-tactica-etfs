"""Pipeline de ETF2: extraer -> guardar en la base -> leer de la base -> calcular -> guardar.

El orden importa para la defensa del proyecto: los cálculos se hacen sobre lo que
hay GUARDADO en la base de datos (no sobre lo recién descargado), de modo que la
base es la fuente de verdad y el pipeline se puede repetir sin duplicar datos
(todas las escrituras son upserts).

Tres reportes salen de aquí:
  descriptivo  : precios, indicadores, pesos de la regla y su backtest.
  predictivo   : probabilidad de subida a un mes de cada ETF (modelo con validación walk-forward).
  prescriptivo : recomendación por perfil de riesgo; los números por reglas, el texto por un LLM validado.

Modos:
  historico : carga completa desde HISTORICO_INICIO (la primera vez).
  diario    : baja solo los últimos días, recalcula y actualiza (lo que corre el cron).
"""

import os
import traceback

import pandas as pd

from . import config, prescriptivo
from .api_db import ClienteSupabase, ErrorAPI, registros
from .backtest import correr_backtest
from .datos import cargar_csv, descargar_yfinance, matriz_precios, validar_precios
from .indicadores import a_largo, calcular_indicadores
from .pesos import pesos_anchos, pesos_historicos

SENALES = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_60d", "drawdown"]
COLUMNAS_CURVA_ML = {config.NOMBRE_ESTRATEGIA: "estrategia", config.NOMBRE_ESTRATEGIA_ML: "estrategia_ml",
                     config.NOMBRE_IGUALES: "pesos_iguales", config.NOMBRE_SPY: "spy"}
FALTAN_TABLAS = ("La base de datos todavía no tiene las tablas del modelo y de las recomendaciones: "
                 "ejecuta sql/migracion_ml_llm.sql en Supabase y vuelve a correr el pipeline.")


def _log(msg: str) -> None:
    print(f"[ETF2] {msg}", flush=True)


def _ahora() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _leer_precios_bd(cli: ClienteSupabase) -> pd.DataFrame:
    filas = cli.leer("precios_diarios", {
        "select": "ticker,fecha,apertura,maximo,minimo,cierre,cierre_ajustado,volumen",
        "order": "ticker.asc,fecha.asc",
    }, hilos=4)
    return validar_precios(pd.DataFrame(filas))


def origen_de_la_corrida() -> str:
    """Quién hizo arrancar el proceso: el horario de GitHub, el botón del panel, «Run workflow» o una PC."""
    if not os.getenv("GITHUB_ACTIONS"):
        return "pc"
    if os.getenv("GITHUB_EVENT_NAME") == "schedule":
        return "programada"
    return "panel" if os.getenv("ORIGEN_CORRIDA") == "panel" else "github_manual"


def ejecutar(modo: str = "diario", csv: str | None = None, sin_db: bool = False,
             cliente: ClienteSupabase | None = None, ml: bool = True, llm: bool = True,
             cliente_llm=None) -> dict:
    """Corre todo el pipeline y devuelve un diccionario con los resultados.

    ml=False omite el modelo predictivo; llm=False hace que las recomendaciones se redacten
    con la plantilla fija, sin llamar al modelo de lenguaje.
    """
    if modo not in ("historico", "diario"):
        raise ValueError("modo debe ser 'historico' o 'diario'")

    config.cargar_env()
    cli = None
    if not sin_db:
        cli = cliente or ClienteSupabase(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SECRET_KEY"))
    if llm and cliente_llm is None:
        cliente_llm = _cliente_llm()

    corrida_id = None
    if cli:
        origen = origen_de_la_corrida()
        corrida_id = cli.insertar_devolviendo(
            "corridas", {"estado": "en_curso", "modo": modo, "origen": origen})["id"]
        _log(f"Corrida #{corrida_id} iniciada (modo {modo}, origen {origen}).")

    try:
        res = _correr(modo, csv, cli, corrida_id, ml, llm, cliente_llm)
        if cli:
            mensaje = " · ".join([f"Último dato: {res['ultima_fecha'].date()}", *res["avisos"]])
            cli.actualizar("corridas", {"id": f"eq.{corrida_id}"}, {
                "estado": "ok", "fin": _ahora(),
                "filas_precios": res["n_precios"], "filas_indicadores": res["n_indicadores"],
                "filas_pesos": res["n_pesos"], "mensaje": mensaje[:500],
            })
        return res
    except Exception as e:
        _log(f"ERROR: {e}")
        if cli and corrida_id is not None:
            try:
                cli.actualizar("corridas", {"id": f"eq.{corrida_id}"}, {
                    "estado": "error", "fin": _ahora(),
                    "mensaje": f"{type(e).__name__}: {e}"[:500],
                })
            except Exception:
                traceback.print_exc()
        raise


def _correr(modo, csv, cli, corrida_id, ml, llm, cliente_llm) -> dict:
    # 1) EXTRAER -------------------------------------------------------------
    inicio = config.HISTORICO_INICIO
    if modo == "diario" and cli:
        ultima = cli.ultima_fecha("precios_diarios")
        if ultima is None:
            _log("La base está vacía: se hará la carga histórica completa.")
        else:
            inicio = (ultima - pd.Timedelta(days=config.DIAS_REFRESCO)).date().isoformat()
    _log(f"Descargando precios desde {inicio} ...")
    nuevos = cargar_csv(csv) if csv else descargar_yfinance(config.TICKERS, inicio)
    _log(f"Descargadas {len(nuevos)} filas de precios.")

    # 2) CARGAR (upsert) -----------------------------------------------------
    n_precios = len(nuevos)
    if cli:
        cli.insertar("etfs", [
            {"ticker": t, "nombre": n, "clase_activo": c, "descripcion": d}
            for t, (n, c, d) in config.ETFS.items()], conflicto="ticker")
        cli.insertar("precios_diarios", registros(nuevos), conflicto="ticker,fecha")
        _log("Precios guardados en la base de datos.")
        # 3) LEER de la base la serie completa, que es la fuente de verdad.
        precios = _leer_precios_bd(cli)
        _log(f"Leídas {len(precios)} filas de precios desde la base.")
    else:
        precios = nuevos

    # 4) CALCULAR ------------------------------------------------------------
    px = matriz_precios(precios)
    ind = calcular_indicadores(px)
    ind_largo = a_largo(ind)
    pesos_largo = pesos_historicos(ind)
    resumen, curvas = correr_backtest(px, pesos_anchos(pesos_largo))
    ultima_fecha = px.index.max()
    _log(f"Indicadores y pesos calculados hasta {ultima_fecha.date()}.")

    # 5) GUARDAR resultados --------------------------------------------------
    corte = pd.Timestamp.min if modo == "historico" else ultima_fecha - pd.Timedelta(days=config.DIAS_REFRESCO)
    ind_w = ind_largo[ind_largo["fecha"] >= corte]
    pes_w = pesos_largo[pesos_largo["fecha"] >= corte]
    if cli:
        cli.insertar("indicadores", registros(ind_w), conflicto="ticker,fecha")
        cli.insertar("pesos_sugeridos", registros(pes_w, corrida_id=corrida_id), conflicto="fecha,ticker")
        cli.insertar("backtest_resumen", registros(resumen, actualizado_en=_ahora()), conflicto="estrategia")
        curva_db = curvas.rename(columns={
            config.NOMBRE_ESTRATEGIA: "estrategia",
            config.NOMBRE_IGUALES: "pesos_iguales",
            config.NOMBRE_SPY: "spy"}).rename_axis("fecha").reset_index()
        cli.insertar("backtest_curva", registros(curva_db), conflicto="fecha")
        _log("Indicadores, pesos y backtest guardados.")

    # 6) PREDECIR y 7) RECOMENDAR --------------------------------------------
    # Si la base aún no tiene las tablas nuevas, lo anterior ya quedó guardado: se avisa y se sigue.
    modelo, recomendaciones, avisos = None, [], []
    try:
        if ml:
            modelo = _modelo_predictivo(px, ind, pesos_largo, cli)
        recomendaciones = _recomendar(ind, pesos_largo, modelo, cli, corrida_id, cliente_llm if llm else None, llm)
    except ErrorAPI as e:
        if e.codigo != 404:
            raise
        avisos.append(FALTAN_TABLAS)
        _log(f"AVISO: {FALTAN_TABLAS}")

    actuales = pesos_largo[pesos_largo["fecha"] == ultima_fecha].set_index("ticker")
    return {
        "ultima_fecha": ultima_fecha, "pesos_actuales": actuales, "backtest": resumen,
        "curvas": curvas, "n_precios": n_precios, "n_indicadores": len(ind_w),
        "n_pesos": len(pes_w), "corrida_id": corrida_id,
        "modelo": modelo, "recomendaciones": recomendaciones, "avisos": avisos,
    }


# --- Reporte predictivo ----------------------------------------------------------------
def _modelo_predictivo(px, ind, pesos_largo, cli) -> dict | None:
    from . import predictivo            # aquí, para que el panel no tenga que cargar scikit-learn

    _log("Entrenando el modelo predictivo (walk-forward mensual) ...")
    r = predictivo.entrenar_y_evaluar(px)
    if r is None:
        _log("Modelo predictivo: todavía no hay historia suficiente para entrenarlo.")
        return None
    r["pesos_ml"], r["resumen"], r["curvas"] = predictivo.efecto_del_modelo(px, ind, pesos_largo, r["predicciones"])
    principal = r["metricas"].set_index("modelo").loc[config.MODELO_PRINCIPAL]
    _log(f"Modelo evaluado en {int(principal['n'])} predicciones fuera de muestra "
         f"(exactitud {principal['exactitud']:.3f}, AUC {principal['auc']:.3f}).")

    if cli:
        predicciones = registros(r["predicciones"])
        for fila in predicciones:           # la columna es un entero 0/1 (o vacío si aún no se sabe)
            fila["subio"] = None if fila["subio"] is None else int(fila["subio"])
        cli.insertar("modelo_predicciones", predicciones, conflicto="fecha,ticker,modelo")
        cli.insertar("modelo_metricas", registros(r["metricas"], actualizado_en=_ahora()), conflicto="modelo")
        cli.insertar("modelo_importancias", registros(r["importancias"]), conflicto="modelo,variable")
        cli.insertar("pesos_ml", registros(r["pesos_ml"]), conflicto="fecha,ticker")
        cli.insertar("backtest_ml_resumen", registros(r["resumen"], actualizado_en=_ahora()), conflicto="estrategia")
        curva = r["curvas"].rename(columns=COLUMNAS_CURVA_ML).rename_axis("fecha").reset_index()
        cli.insertar("backtest_ml_curva", registros(curva), conflicto="fecha")
        _log("Predicciones, métricas y comparación con la regla guardadas.")
    return r


# --- Reporte prescriptivo --------------------------------------------------------------
def _cliente_llm():
    """Cliente de Gemini si hay clave en el entorno; None si no (se usará la plantilla)."""
    clave = os.getenv(config.LLM_VARIABLE_CLAVE)
    if not clave:
        return None
    elegido = os.getenv(config.LLM_VARIABLE_MODELO)
    modelos = ([elegido] if elegido else []) + [m for m in config.LLM_MODELOS if m != elegido]
    return prescriptivo.ClienteGemini(clave, modelos)


def _recomendar(ind, pesos_largo, modelo, cli, corrida_id, cliente_llm, llm_pedido: bool) -> list[dict]:
    """Una recomendación por perfil. El LLM solo redacta, y solo si hay clave y cupo de llamadas."""
    usa_modelo = modelo is not None and prescriptivo.conviene_usar_el_modelo(modelo["resumen"])
    serie = pesos_anchos(modelo["pesos_ml"] if usa_modelo else pesos_largo)
    fecha = serie.index[-1]
    senales = pd.DataFrame({k: ind[k].loc[fecha] for k in SENALES})
    confianza = None
    if modelo is not None:
        pred = modelo["predicciones"]
        hoy = pred[(pred["fecha"] == pred["fecha"].max()) & (pred["modelo"] == config.MODELO_PRINCIPAL)]
        senales["prob_sube"] = hoy.set_index("ticker")["prob_sube"]
        confianza = prescriptivo.confianza_del_modelo(modelo["metricas"])

    dia = pd.Timestamp.now(tz="UTC").date().isoformat()
    bitacora = []
    cupo = 0
    if cliente_llm is not None:
        usadas_hoy = 0
        if cli:
            hechas = cli.leer("llm_log", {"select": "estado", "dia": f"eq.{dia}"})
            usadas_hoy = sum(1 for x in hechas if x["estado"] in ("ok", "rechazada", "error"))
        cupo = max(0, min(config.LLM_LIMITE_CORRIDA, config.LLM_LIMITE_DIA - usadas_hoy))
    elif llm_pedido:
        bitacora.append({"estado": "sin_clave", "mensaje": f"No hay {config.LLM_VARIABLE_CLAVE}: texto por reglas."})

    filas, nuevas = [], []
    for perfil in config.PERFILES:
        plan = prescriptivo.armar_plan(serie, perfil, senales, confianza, usa_modelo)
        entrada = plan["entrada"]
        if cli:
            previa = cli.leer("recomendaciones", {"select": "*", "fecha": f"eq.{entrada['fecha']}",
                                                  "perfil": f"eq.{perfil}"})
            if previa and previa[0]["fuente"] == "llm" and previa[0]["entrada_hash"] == prescriptivo.huella(entrada):
                filas.append(previa[0])     # mismos datos y ya redactada por el LLM: ni se llama ni se pisa
                continue
        if cliente_llm is None:
            textos, fuente, modelo_llm, eventos = prescriptivo.texto_por_reglas(entrada), "reglas", None, []
        elif cupo < 1:
            textos, fuente, modelo_llm = prescriptivo.texto_por_reglas(entrada), "reglas", None
            eventos = [{"perfil": perfil, "estado": "limite",
                        "mensaje": f"Límite de llamadas alcanzado ({config.LLM_LIMITE_DIA} por día, "
                                   f"{config.LLM_LIMITE_CORRIDA} por corrida): texto por reglas."}]
        else:
            textos, fuente, modelo_llm, eventos = prescriptivo.redactar(entrada, cliente_llm, max_llamadas=min(2, cupo))
            cupo -= len(eventos)
        bitacora += eventos
        nuevas.append(prescriptivo.fila_de_recomendacion(plan, textos, fuente, modelo_llm) | {"creado_en": _ahora()})
        filas.append(nuevas[-1])

    if cli:
        cli.insertar("recomendaciones", nuevas, conflicto="fecha,perfil")
        campos = ("perfil", "modelo", "estado", "tokens_entrada", "tokens_salida", "latencia_ms", "mensaje")
        cli.insertar("llm_log", [{c: e.get(c) for c in campos} | {"dia": dia, "corrida_id": corrida_id}
                                 for e in bitacora])
    por_fuente = pd.Series([x["fuente"] for x in filas]).value_counts().to_dict()
    _log(f"Recomendaciones al {fecha.date()}: " + ", ".join(f"{n} por {k}" for k, n in por_fuente.items()) + ".")
    return filas
