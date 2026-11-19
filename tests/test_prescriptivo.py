"""Pruebas del reporte prescriptivo: perfiles, acciones y la revisión de lo que escribe el LLM."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import requests

from etf2 import config, prescriptivo as ps
from etf2.datos import matriz_precios
from etf2.indicadores import calcular_indicadores
from etf2.pesos import pesos_anchos, pesos_historicos

from llm_falso import LLMFalso, respuesta_valida
from sinteticos import generar


@pytest.fixture(scope="module")
def serie():
    px = matriz_precios(generar(700))
    return pesos_anchos(pesos_historicos(calcular_indicadores(px)))


@pytest.fixture(scope="module")
def senales():
    px = matriz_precios(generar(700))
    ind = calcular_indicadores(px)
    s = pd.DataFrame({k: ind[k].iloc[-1] for k in ("ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_60d", "drawdown")})
    s["prob_sube"] = pd.Series({"SPY": 0.61, "QQQ": 0.48, "IWM": 0.55, "GLD": 0.66, "TLT": 0.52})
    return s


CONFIANZA = {"nombre": "Regresión logística", "nivel": "baja", "auc": 0.513, "exactitud": 0.561,
             "exactitud_referencia": 0.584, "n": 185}


@pytest.fixture()
def plan(serie, senales):
    return ps.armar_plan(serie, "Moderado", senales, CONFIANZA, usa_modelo=True)


# --- Perfiles ---------------------------------------------------------------------------
def test_perfil_respeta_rango_y_limites_con_pesos_aleatorios():
    rng = np.random.default_rng(11)
    from etf2.pesos import aplicar_limites
    for _ in range(300):
        base = pd.Series(aplicar_limites(rng.uniform(0.02, 1.0, 5))[0], index=config.TICKERS)
        for perfil, (minimo, maximo) in config.PERFILES.items():
            w = ps.ajustar_a_perfil(base, perfil)
            assert w.sum() == pytest.approx(1.0)
            assert w.between(config.PESO_MIN - 1e-9, config.PESO_MAX + 1e-9).all()
            assert minimo - 1e-9 <= ps.peso_de_renta_variable(w) <= maximo + 1e-9
            if minimo <= ps.peso_de_renta_variable(base) <= maximo:
                assert np.allclose(w, base)                     # ya cumplía: no se toca


def test_perfil_mueve_la_renta_variable_al_borde_mas_cercano():
    base = pd.Series([0.30, 0.30, 0.20, 0.10, 0.10], index=config.TICKERS)       # 80 % en renta variable
    cons = ps.ajustar_a_perfil(base, "Conservador")
    assert ps.peso_de_renta_variable(cons) == pytest.approx(0.45)
    assert cons["SPY"] / cons["IWM"] == pytest.approx(1.5)                      # proporciones internas intactas
    assert cons["GLD"] == pytest.approx(cons["TLT"])
    agresivo = ps.ajustar_a_perfil(base, "Agresivo")
    assert np.allclose(agresivo, base)
    assert list(config.PERFILES) == ["Conservador", "Moderado", "Agresivo"]
    assert config.PERFIL_DEFECTO in config.PERFILES


def test_acciones_segun_la_banda():
    previo = pd.Series([0.20, 0.20, 0.20, 0.20, 0.20], index=config.TICKERS)
    objetivo = pd.Series([0.22, 0.18, 0.219, 0.181, 0.20], index=config.TICKERS)
    assert list(ps.clasificar_acciones(objetivo, previo)) == ["Comprar", "Vender", "Mantener", "Mantener", "Mantener"]
    assert set(ps.clasificar_acciones(objetivo, None)) == {"Mantener"}
    assert ps.ACCIONES == ["Comprar", "Vender", "Mantener"]
    assert set(ps.ACCIONES_ANTERIORES.values()) <= set(ps.ACCIONES)


def test_nivel_de_alerta_con_la_volatilidad_de_un_mes():
    cierre = pd.Series({"SPY": 600.0, "TLT": 90.0, "GLD": 300.0, "QQQ": 500.0, "IWM": 0.0})
    vol = pd.Series({"SPY": 0.18, "TLT": 0.12, "GLD": np.nan, "QQQ": 4.0, "IWM": 0.2})
    alerta = ps.nivel_de_alerta(cierre, vol)
    assert alerta["SPY"] == pytest.approx(600 * (1 - 2 * 0.18 / np.sqrt(12)))          # unos 10 % abajo
    assert alerta["TLT"] == pytest.approx(90 * (1 - 2 * 0.12 / np.sqrt(12)))
    assert alerta["SPY"] / cierre["SPY"] - 1 == pytest.approx(-0.1039, abs=1e-4)
    assert alerta[["GLD", "QQQ", "IWM"]].isna().all()       # sin volatilidad, caída imposible o sin precio: no hay nivel
    assert ps.nivel_de_alerta(cierre, vol, desvios=1)["SPY"] > alerta["SPY"]


def test_el_modelo_solo_se_usa_si_mejoro_a_la_regla():
    def resumen(regla, ml):
        return pd.DataFrame({"estrategia": [config.NOMBRE_ESTRATEGIA, config.NOMBRE_ESTRATEGIA_ML], "sharpe": [regla, ml]})
    assert ps.conviene_usar_el_modelo(resumen(0.80, 0.84)) is True
    assert ps.conviene_usar_el_modelo(resumen(0.80, 0.80)) is False
    assert ps.conviene_usar_el_modelo(resumen(0.80, 0.75)) is False
    assert ps.conviene_usar_el_modelo(None) is False
    assert ps.conviene_usar_el_modelo(pd.DataFrame()) is False


def test_confianza_segun_el_auc():
    def metricas(auc):
        return pd.DataFrame({"modelo": ["logit", "frecuencia"], "auc": [auc, np.nan], "exactitud": [0.56, 0.58], "n": [100, 100]})
    assert ps.confianza_del_modelo(metricas(0.50))["nivel"] == "baja"
    assert ps.confianza_del_modelo(metricas(0.55))["nivel"] == "media"
    assert ps.confianza_del_modelo(metricas(0.61))["nivel"] == "alta"
    assert ps.confianza_del_modelo(metricas(np.nan))["nivel"] == "baja"
    assert ps.confianza_del_modelo(metricas(0.55))["exactitud_referencia"] == 0.58
    assert ps.confianza_del_modelo(None) is None and ps.confianza_del_modelo(pd.DataFrame()) is None


# --- Plan y entrada del LLM ---------------------------------------------------------------
def test_plan_compara_con_el_cierre_del_mes_anterior(serie, plan):
    fecha = serie.index[-1]
    assert plan["fecha"] == fecha
    mes_pasado = serie[serie.index.to_period("M") < fecha.to_period("M")].iloc[-1]
    assert np.allclose(plan["previos"], ps.ajustar_a_perfil(mes_pasado, "Moderado")[config.TICKERS])
    assert list(plan["pesos"].index) == config.TICKERS
    assert plan["pesos"].sum() == pytest.approx(1.0)
    e = plan["entrada"]
    assert e["perfil"] == "Moderado" and e["fecha"] == f"{fecha:%Y-%m-%d}"
    assert [x["ticker"] for x in e["etfs"]] == config.TICKERS
    for x in e["etfs"]:
        assert x["accion"] == plan["acciones"][x["ticker"]]
        assert x["cambio_puntos"] == pytest.approx(x["peso_pct"] - x["peso_mes_anterior_pct"], abs=1e-9)
    rv = e["renta_variable"]
    assert rv["en_la_cartera_pct"] + rv["oro_y_bonos_pct"] == pytest.approx(100.0)
    assert e["modelo"]["inclina_los_pesos"] is True and e["modelo"]["auc"] == 0.51
    json.dumps(e)                                               # se puede enviar tal cual


def test_la_huella_cambia_solo_si_cambian_los_datos(serie, senales, plan):
    igual = ps.armar_plan(serie, "Moderado", senales, CONFIANZA, usa_modelo=True)
    assert ps.huella(igual["entrada"]) == ps.huella(plan["entrada"])
    otro_perfil = ps.armar_plan(serie, "Agresivo", senales, CONFIANZA, usa_modelo=True)
    sin_modelo = ps.armar_plan(serie, "Moderado", senales, CONFIANZA, usa_modelo=False)
    assert len({ps.huella(p["entrada"]) for p in (plan, otro_perfil, sin_modelo)}) == 3


# --- Plantilla de respaldo ------------------------------------------------------------------
@pytest.mark.parametrize("perfil", list(config.PERFILES))
def test_la_plantilla_solo_usa_cifras_de_la_entrada(serie, senales, perfil):
    for confianza in (CONFIANZA, None):
        entrada = ps.armar_plan(serie, perfil, senales, confianza, usa_modelo=confianza is not None)["entrada"]
        t = ps.texto_por_reglas(entrada)
        assert set(t["explicaciones"]) == set(config.TICKERS)
        assert 2 <= len(t["riesgos"]) <= 3 and 1 <= len(t["que_vigilar"]) <= 3
        assert ps.validar(respuesta_valida(entrada), entrada) == []
        if confianza is None:
            assert "modelo" not in json.dumps(t, ensure_ascii=False).lower()


def test_la_plantilla_funciona_sin_mes_anterior_y_sin_senales(serie):
    un_dia = serie.iloc[[-1]]
    p = ps.armar_plan(un_dia, "Moderado", pd.DataFrame(), None, usa_modelo=False)
    assert p["previos"] is None and set(p["acciones"]) == {"Mantener"}
    t = ps.texto_por_reglas(p["entrada"])
    assert "todavía no hay un mes anterior" in t["explicaciones"]["SPY"]
    fila = ps.fila_de_recomendacion(p, t, "reglas", None)
    assert fila["detalle"]["etfs"][0]["peso_previo"] is None
    json.dumps(fila)


# --- Revisión de la respuesta del LLM -------------------------------------------------------
def test_cifras_ajenas_tolera_redondeos_pero_no_inventos():
    entrada = {"a": 41.5, "b": -3.24, "nombre": "Russell 2000", "auc": 0.53, "fecha": "2025-02-18"}
    assert ps.cifras_ajenas("Pesa 41,5 % (unos 42 %), cae 3,2 % y el AUC es 0.53.", entrada) == []
    assert ps.cifras_ajenas("Los 5 ETFs, a 6 y 12 meses, el 100 % y el Russell 2000 al 18 de febrero de 2025.", entrada) == []
    assert ps.cifras_ajenas("Pesa 43 %.", entrada) == ["43"]
    assert ps.cifras_ajenas("Pesa 41,9 %.", entrada) == ["41,9"]
    assert ps.cifras_ajenas("Llegará a 650 dólares.", entrada) == ["650"]
    assert ps.cifras_ajenas("AUC de 0,58.", entrada) == ["0,58"]


def test_validar_acepta_una_respuesta_correcta(plan):
    assert ps.validar(respuesta_valida(plan["entrada"]), plan["entrada"]) == []


def test_validar_rechaza_lo_que_no_debe_pasar(plan):
    entrada = plan["entrada"]

    def con(cambio):
        r = copy.deepcopy(respuesta_valida(entrada))
        cambio(r)
        return " | ".join(ps.validar(r, entrada))

    otra = next(a for a in ps.ACCIONES if a != entrada["etfs"][0]["accion"])
    assert "la acción de SPY" in con(lambda r: r["etfs"][0].update(accion=otra))
    assert "exactamente los cinco" in con(lambda r: r["etfs"].pop())
    assert "exactamente los cinco" in con(lambda r: r["etfs"].append(dict(r["etfs"][0])))
    assert "exactamente los cinco" in con(lambda r: r["etfs"][1].update(ticker="AAPL"))
    assert "cifras que no están" in con(lambda r: r.update(resumen=r["resumen"] + " SPY llegará a 712,5 dólares."))
    assert "expresiones no permitidas" in con(lambda r: r.update(resumen=r["resumen"] + " Es una ganancia segura y garantizada."))
    assert "expresiones no permitidas" in con(lambda r: r["etfs"][0].update(explicacion="Conviene vender todo SPY ya."))
    assert "«resumen»" in con(lambda r: r.update(resumen="Corto."))
    assert "«resumen»" in con(lambda r: r.update(resumen="x" * (ps.MAX_RESUMEN + 1)))
    assert "«riesgos»" in con(lambda r: r.update(riesgos=[]))
    assert "«que_vigilar»" in con(lambda r: r.pop("que_vigilar"))
    assert "explicación de QQQ" in con(lambda r: r["etfs"][1].update(explicacion=""))
    assert ps.validar("no soy un objeto", entrada) == ["la respuesta no es un objeto JSON"]
    assert ps.validar({"resumen": "x" * 50, "riesgos": ["a" * 20] * 2, "que_vigilar": ["b" * 20], "etfs": "nada"},
                      entrada)[-1] == "«etfs» debe ser una lista de objetos"


def test_leer_json_tolera_el_bloque_de_codigo():
    assert ps.leer_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert ps.leer_json(' {"a": 1} ') == {"a": 1}
    assert ps.leer_json("esto no es json") is None and ps.leer_json("") is None and ps.leer_json(None) is None


# --- Redacción con reintento y respaldo -----------------------------------------------------
def test_redactar_acepta_la_respuesta_valida(plan):
    llm = LLMFalso(respuesta_valida(plan["entrada"]))
    textos, fuente, modelo, eventos = ps.redactar(plan["entrada"], llm)
    assert fuente == "llm" and modelo == "modelo-de-prueba" and len(llm.pedidos) == 1
    assert list(textos["explicaciones"]) == list(config.TICKERS)
    assert [e["estado"] for e in eventos] == ["ok"] and eventos[0]["tokens_entrada"] == 900
    pedido = llm.pedidos[0]
    assert pedido["sistema"] == ps.PROMPT_SISTEMA and pedido["esquema"] == ps.ESQUEMA_SALIDA
    assert json.loads(pedido["usuario"].split("\n", 1)[1]) == plan["entrada"]     # el LLM ve solo la entrada


def test_redactar_reintenta_una_vez_explicando_el_rechazo(plan):
    mala = respuesta_valida(plan["entrada"])
    mala["resumen"] += " El S&P subirá 37,2 % este año."
    llm = LLMFalso(mala, respuesta_valida(plan["entrada"]))
    textos, fuente, _, eventos = ps.redactar(plan["entrada"], llm)
    assert fuente == "llm" and [e["estado"] for e in eventos] == ["rechazada", "ok"]
    assert "37,2" in eventos[0]["mensaje"]
    assert "fue rechazada" in llm.pedidos[1]["usuario"] and "37,2" in llm.pedidos[1]["usuario"]
    assert "37,2" not in textos["resumen"]


def test_redactar_cae_a_la_plantilla_si_el_llm_insiste(plan):
    mala = respuesta_valida(plan["entrada"])
    mala["etfs"][0]["accion"] = next(a for a in ps.ACCIONES if a != mala["etfs"][0]["accion"])
    llm = LLMFalso(mala, "esto no es json", respuesta_valida(plan["entrada"]))
    textos, fuente, modelo, eventos = ps.redactar(plan["entrada"], llm)
    assert fuente == "reglas" and modelo is None and len(llm.pedidos) == 2        # no hay tercer intento
    assert [e["estado"] for e in eventos] == ["rechazada", "rechazada"]
    assert eventos[1]["mensaje"] == "la respuesta no es JSON válido"
    assert textos == ps.texto_por_reglas(plan["entrada"])


def test_redactar_no_insiste_si_la_api_falla(plan):
    llm = LLMFalso(ps.ErrorLLM("la API respondió 429: cuota agotada"), respuesta_valida(plan["entrada"]))
    textos, fuente, _, eventos = ps.redactar(plan["entrada"], llm)
    assert fuente == "reglas" and len(llm.pedidos) == 1
    assert eventos[0]["estado"] == "error" and "429" in eventos[0]["mensaje"]
    assert textos["resumen"].startswith("Para el perfil Moderado")


def test_fila_de_recomendacion(plan):
    textos = ps.texto_por_reglas(plan["entrada"])
    fila = ps.fila_de_recomendacion(plan, textos, "llm", "gemini-x")
    assert fila["fuente"] == "llm" and fila["modelo_llm"] == "gemini-x" and fila["perfil"] == "Moderado"
    assert fila["entrada_hash"] == ps.huella(plan["entrada"]) and len(fila["entrada_hash"]) == 64
    d = fila["detalle"]
    assert sum(e["peso"] for e in d["etfs"]) == pytest.approx(1.0)
    assert d["rango"] == list(config.PERFILES["Moderado"]) and d["usa_modelo"] is True and d["confianza"] == "baja"
    assert d["aviso"] == ps.AVISO_LEGAL and d["etfs"][0]["prob_sube"] == 0.61
    json.dumps(fila)


# --- Cliente HTTP de Gemini ------------------------------------------------------------------
class Respuesta:
    def __init__(self, codigo, cuerpo=None, texto=""):
        self.status_code, self._cuerpo, self.text = codigo, cuerpo, texto or json.dumps(cuerpo or {})
        self.ok = 200 <= codigo < 300

    def json(self):
        if self._cuerpo is None:
            raise ValueError("sin JSON")
        return self._cuerpo


def cuerpo_ok(texto='{"hola": 1}', **uso):
    return {"candidates": [{"content": {"parts": [{"text": "pensando", "thought": True}, {"text": texto}]},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 1200, "candidatesTokenCount": 350, **uso}}


class SesionFalsa:
    def __init__(self, *respuestas):
        self.respuestas, self.pedidos = list(respuestas), []

    def post(self, url, json=None, timeout=None, headers=None):
        self.pedidos.append({"url": url, "json": copy.deepcopy(json), "timeout": timeout, "headers": headers})
        r = self.respuestas.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def cliente(*respuestas, modelos=("modelo-a", "modelo-b")):
    sesion = SesionFalsa(*respuestas)
    return ps.ClienteGemini("clave-de-prueba", modelos=list(modelos), sesion=sesion, espera=0.0), sesion


def test_gemini_arma_la_peticion_y_lee_la_respuesta():
    cli, sesion = cliente(Respuesta(200, cuerpo_ok(thoughtsTokenCount=40)))
    r = cli.generar("sistema", "usuario", ps.ESQUEMA_SALIDA)
    assert r == {"texto": '{"hola": 1}', "modelo": "modelo-a", "tokens_entrada": 1200, "tokens_salida": 390}
    p = sesion.pedidos[0]
    assert p["url"].endswith("/v1beta/models/modelo-a:generateContent") and "clave" not in p["url"]
    assert p["headers"]["x-goog-api-key"] == "clave-de-prueba" and p["timeout"] == config.LLM_TIEMPO_ESPERA
    assert p["json"]["systemInstruction"]["parts"][0]["text"] == "sistema"
    assert p["json"]["contents"] == [{"role": "user", "parts": [{"text": "usuario"}]}]
    g = p["json"]["generationConfig"]
    assert g["responseMimeType"] == "application/json" and g["responseSchema"] == ps.ESQUEMA_SALIDA
    assert g["maxOutputTokens"] == config.LLM_MAX_TOKENS_SALIDA and g["temperature"] == config.LLM_TEMPERATURA


def test_gemini_pasa_al_siguiente_modelo_si_el_primero_no_existe():
    cli, sesion = cliente(Respuesta(404, {"error": {"message": "model not found"}}), Respuesta(200, cuerpo_ok()),
                          Respuesta(200, cuerpo_ok()))
    assert cli.generar("s", "u")["modelo"] == "modelo-b"
    assert cli.generar("s", "u")["modelo"] == "modelo-b" and len(sesion.pedidos) == 3    # ya no prueba el retirado
    solo, _ = cliente(Respuesta(404, {}), modelos=("modelo-a",))
    with pytest.raises(ps.ErrorLLM, match="modelo-a no existe"):
        solo.generar("s", "u")


def test_gemini_reintenta_sin_esquema_y_ante_saturacion():
    cli, sesion = cliente(Respuesta(400, {"error": {"message": "schema"}}), Respuesta(200, cuerpo_ok()))
    cli.generar("s", "u", ps.ESQUEMA_SALIDA)
    assert "responseSchema" in sesion.pedidos[0]["json"]["generationConfig"]
    assert "responseSchema" not in sesion.pedidos[1]["json"]["generationConfig"]
    assert sesion.pedidos[1]["json"]["generationConfig"]["responseMimeType"] == "application/json"

    cli, sesion = cliente(Respuesta(429, {}), Respuesta(200, cuerpo_ok()))
    assert cli.generar("s", "u")["texto"] and len(sesion.pedidos) == 2
    cli, sesion = cliente(Respuesta(503, {}), Respuesta(503, {"error": {"message": "sobrecargado"}}))
    with pytest.raises(ps.ErrorLLM, match="503: sobrecargado"):
        cli.generar("s", "u")
    assert len(sesion.pedidos) == 2


def test_gemini_convierte_los_fallos_en_errorllm():
    cli, _ = cliente(Respuesta(403, {"error": {"message": "API key not valid"}}))
    with pytest.raises(ps.ErrorLLM, match="403: API key not valid"):
        cli.generar("s", "u")
    cli, _ = cliente(Respuesta(200, {"candidates": [{"finishReason": "MAX_TOKENS"}]}))
    with pytest.raises(ps.ErrorLLM, match="respuesta vacía.*MAX_TOKENS"):
        cli.generar("s", "u")
    cli, _ = cliente(Respuesta(200, None, texto="<html>"))
    with pytest.raises(ps.ErrorLLM, match="no devolvió JSON"):
        cli.generar("s", "u")
    cli, _ = cliente(requests.ConnectionError("sin red https://x?key=no"))
    with pytest.raises(ps.ErrorLLM) as e:
        cli.generar("s", "u")
    assert "ConnectionError" in str(e.value) and "key=" not in str(e.value)
    with pytest.raises(ps.ErrorLLM, match=config.LLM_VARIABLE_CLAVE):
        ps.ClienteGemini("")


def test_gemini_por_http_de_verdad_contra_un_servidor_local(plan):
    """La misma llamada, pero viajando por HTTP (con `requests`) hasta un servidor local que imita la API."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    recibido = {}

    class Imitacion(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            cuerpo = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            recibido.update(ruta=self.path, clave=self.headers.get("x-goog-api-key"), cuerpo=cuerpo)
            entrada = json.loads(cuerpo["contents"][0]["parts"][0]["text"].split("\n", 1)[1])
            datos = json.dumps(cuerpo_ok(json.dumps(respuesta_valida(entrada), ensure_ascii=False))).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(datos)))
            self.end_headers()
            self.wfile.write(datos)

    srv = HTTPServer(("127.0.0.1", 0), Imitacion)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cli = ps.ClienteGemini("clave-de-prueba", modelos=["modelo-local"])
        cli.URL = f"http://127.0.0.1:{srv.server_port}/v1beta/models/{{modelo}}:generateContent"
        textos, fuente, modelo, eventos = ps.redactar(plan["entrada"], cli)
    finally:
        srv.shutdown()
        srv.server_close()
    assert fuente == "llm" and modelo == "modelo-local" and eventos[0]["estado"] == "ok"
    assert eventos[0]["tokens_entrada"] == 1200 and eventos[0]["tokens_salida"] == 350
    assert recibido["ruta"] == "/v1beta/models/modelo-local:generateContent" and recibido["clave"] == "clave-de-prueba"
    assert recibido["cuerpo"]["systemInstruction"]["parts"][0]["text"] == ps.PROMPT_SISTEMA
    assert textos["resumen"].startswith("Para el perfil Moderado")


def test_la_plantilla_habla_de_comprar_y_vender():
    etf = {"nombre": "x", "clase": "x", "prob_subida_pct": None, "retorno_1m_pct": 1.0, "retorno_3m_pct": 1.0,
           "retorno_6m_pct": 2.0, "retorno_12m_pct": 3.0, "volatilidad_anual_pct": 15.0, "distancia_al_maximo_pct": -1.0}
    entrada = {"perfil": "Moderado", "modelo": None,
               "renta_variable": {"rango_del_perfil_pct": [45, 65], "en_la_cartera_pct": 60.0, "oro_y_bonos_pct": 40.0},
               "etfs": [{**etf, "ticker": "SPY", "accion": "Comprar", "peso_pct": 30.0, "peso_mes_anterior_pct": 25.0,
                         "cambio_puntos": 5.0},
                        {**etf, "ticker": "TLT", "accion": "Vender", "peso_pct": 20.0, "peso_mes_anterior_pct": 25.0,
                         "cambio_puntos": -5.0},
                        {**etf, "ticker": "GLD", "accion": "Mantener", "peso_pct": 20.0, "peso_mes_anterior_pct": 20.5,
                         "cambio_puntos": -0.5}]}
    t = ps.texto_por_reglas(entrada)
    assert ("conviene comprar más SPY (sube su peso) y vender una parte de TLT (baja su peso); "
            "el resto se mantiene.") in t["resumen"]
    assert t["explicaciones"]["SPY"].startswith("Sube de 25,0 % a 30,0 %")
    assert t["explicaciones"]["TLT"].startswith("Baja de 25,0 % a 20,0 %")
    assert "Comprar | Vender | Mantener" in ps.PROMPT_SISTEMA and "nunca todo" in ps.PROMPT_SISTEMA
    entrada["etfs"] = entrada["etfs"][:2]                                     # sin ETFs que se mantengan
    assert "el resto se mantiene" not in ps.texto_por_reglas(entrada)["resumen"]


def test_el_resumen_debe_nombrar_lo_que_se_compra_y_se_vende(plan):
    entrada = copy.deepcopy(plan["entrada"])
    entrada["etfs"][0]["accion"], entrada["etfs"][3]["accion"] = "Comprar", "Vender"
    r = respuesta_valida(entrada)
    assert ps.validar(r, entrada) == []                    # la plantilla nombra a los dos
    r["resumen"] = "La cartera del perfil cambia poco y responde a las variaciones recientes de los activos."
    problema = " | ".join(ps.validar(r, entrada))
    t0, t3 = entrada["etfs"][0]["ticker"], entrada["etfs"][3]["ticker"]
    assert f"el resumen debe nombrar los ETFs que se compran o se venden: {t0}, {t3}" in problema


def test_cambiar_las_instrucciones_obliga_a_redactar_de_nuevo(plan, monkeypatch):
    antes = ps.huella(plan["entrada"])
    monkeypatch.setattr(ps, "PROMPT_SISTEMA", ps.PROMPT_SISTEMA + " Otra regla.")
    assert ps.huella(plan["entrada"]) != antes
    assert "con tildes y eñes" in ps.PROMPT_SISTEMA and "nunca «por ciento»" in ps.PROMPT_SISTEMA


def test_la_frase_del_modelo_la_escribe_el_codigo(plan):
    entrada = plan["entrada"]
    r = respuesta_valida(entrada)
    assert "modelo" not in r["resumen"].lower() and ps.validar(r, entrada) == []
    mala = dict(r, resumen=r["resumen"] + " El modelo de regresión logística se utiliza con una confianza baja.")
    assert "el resumen no debe hablar del modelo" in " | ".join(ps.validar(mala, entrada))
    textos = ps._ordenar(r, entrada)
    assert textos["resumen"].endswith(ps.frase_del_modelo(entrada["modelo"]))
    frase = ps.frase_del_modelo({**entrada["modelo"], "inclina_los_pesos": False})
    assert frase.endswith("y no se usa para inclinar los pesos.")
    assert ps._ordenar(r, {**entrada, "modelo": None})["resumen"] == r["resumen"]
    assert "nunca «se debe»" in ps.PROMPT_SISTEMA
    orden = dict(r, resumen=r["resumen"] + " Se debe comprar más de este ETF.")
    assert "expresiones no permitidas: debe comprar" in " | ".join(ps.validar(orden, entrada))
    causa = dict(r, resumen=r["resumen"] + " La baja se debe a su retorno negativo.")
    assert ps.validar(causa, entrada) == []                  # «se debe a» (una causa) sí está permitido

