"""Reporte prescriptivo: qué hacer con la cartera según el perfil de riesgo.

El trabajo está repartido para que el resultado sea confiable:

* El CÓDIGO decide todo lo que es número o acción: los pesos de cada perfil, el cambio frente
  al mes anterior, si corresponde comprar, vender o mantener cada ETF y su nivel de alerta.
* El MODELO DE LENGUAJE (LLM) solo redacta la explicación con esos datos. No elige pesos ni
  acciones, y lo que escribe se revisa antes de guardarlo: formato, que no cambie ninguna
  acción, que no use cifras que no recibió y que no prometa resultados.
* Si el LLM falla, falta la clave o se llegó al límite de llamadas, la explicación sale de una
  plantilla fija. El panel siempre tiene una recomendación.
"""

import hashlib
import json
import re
import time

import numpy as np
import pandas as pd
import requests

from . import config, formato as f
from .pesos import aplicar_limites

ACCIONES = ["Comprar", "Vender", "Mantener"]
# Nombres que usaban las recomendaciones guardadas antes del 18-nov-2026.
ACCIONES_ANTERIORES = {"Aumentar": "Comprar", "Reducir": "Vender"}
RENTA_VARIABLE = [t for t, grupo in config.GRUPO.items() if grupo == "Renta variable"]
AVISO_LEGAL = ("Proyecto académico. No es asesoría de inversión ni una oferta de compra o venta: "
               "los resultados pasados y las estimaciones del modelo no garantizan resultados futuros.")
VALIDACIONES = [
    "Formato: la respuesta es un JSON con los campos pedidos y los cinco ETFs, una vez cada uno.",
    "Acciones: la acción de cada ETF coincide con la que calculó el código, y el resumen nombra los ETFs que se "
    "compran y los que se venden. Si el modelo se usa o no lo escribe el código, no el LLM.",
    "Cifras: todo número del texto existe en los datos que se le enviaron.",
    "Lenguaje: no promete resultados, no da precios objetivo, no habla de vender todo ni ordena «se debe "
    "comprar» o «se debe vender».",
]

MAX_RESUMEN, MAX_EXPLICACION, MAX_FRASE = 900, 400, 300


# --- Lo que decide el código -----------------------------------------------------------
def ajustar_a_perfil(pesos: pd.Series, perfil: str) -> pd.Series:
    """Lleva la parte de renta variable al rango del perfil, sin romper los límites por ETF.

    Si la renta variable ya está dentro del rango, los pesos no cambian. Si no, se mueve al
    borde más cercano y cada grupo (renta variable por un lado, oro y bonos por otro)
    conserva sus proporciones internas.
    """
    minimo, maximo = config.PERFILES[perfil]
    rv = [t for t in pesos.index if t in RENTA_VARIABLE]
    resto = [t for t in pesos.index if t not in rv]
    parte_rv = float(np.clip(pesos[rv].sum(), minimo, maximo))
    final = pesos.astype(float).copy()
    final[rv] = aplicar_limites(pesos[rv], total=parte_rv)[0]
    final[resto] = aplicar_limites(pesos[resto], total=1.0 - parte_rv)[0]
    return final


def peso_de_renta_variable(pesos: pd.Series) -> float:
    return float(pesos[[t for t in pesos.index if t in RENTA_VARIABLE]].sum())


def clasificar_acciones(objetivo: pd.Series, previo: pd.Series | None) -> pd.Series:
    """Comprar / Vender / Mantener según el cambio de peso frente al mes anterior.

    Comprar = el peso sube al menos BANDA_ACCION (comprar más); Vender = baja al menos esa banda
    (vender una parte: ningún ETF baja del peso mínimo); si no, Mantener.
    """
    if previo is None:
        return pd.Series("Mantener", index=objetivo.index)
    # Se compara en puntos porcentuales con un decimal, igual que se muestra en el panel.
    cambio = (objetivo * 100).round(1) - (previo.reindex(objetivo.index) * 100).round(1)
    banda = config.BANDA_ACCION * 100 - 1e-9
    return cambio.apply(lambda c: "Comprar" if c >= banda else "Vender" if c <= -banda else "Mantener")


def nivel_de_alerta(cierre: pd.Series, vol_anual: pd.Series, desvios: float = config.ALERTA_DESVIOS) -> pd.Series:
    """Precio por debajo del cual conviene revisar cada ETF antes del próximo rebalanceo.

    Es el último cierre menos `desvios` veces la volatilidad de un mes (volatilidad anual / raíz
    de 12). No es un stop loss: la simulación histórica no vende cuando se cruza este nivel.
    """
    caida = desvios * vol_anual.astype(float) / np.sqrt(12)
    nivel = cierre.astype(float) * (1 - caida)
    return nivel.where((cierre > 0) & (vol_anual > 0) & (caida < 1))


def conviene_usar_el_modelo(resumen_ml: pd.DataFrame | None) -> bool:
    """El modelo solo inclina los pesos si mejoró a la regla en la prueba fuera de muestra."""
    if resumen_ml is None or resumen_ml.empty:
        return False
    r = resumen_ml.set_index("estrategia")["sharpe"]
    if config.NOMBRE_ESTRATEGIA_ML not in r or config.NOMBRE_ESTRATEGIA not in r:
        return False
    return bool(r[config.NOMBRE_ESTRATEGIA_ML] > r[config.NOMBRE_ESTRATEGIA])


def confianza_del_modelo(metricas: pd.DataFrame | None) -> dict | None:
    """Resume qué tan fiable es el modelo principal, según su AUC fuera de muestra."""
    if metricas is None or metricas.empty:
        return None
    m = metricas.set_index("modelo")
    if config.MODELO_PRINCIPAL not in m.index:
        return None
    auc = float(m.at[config.MODELO_PRINCIPAL, "auc"])
    bajo, alto = config.CORTES_CONFIANZA
    nivel = "baja" if (auc != auc or auc < bajo) else "media" if auc < alto else "alta"
    referencia = float(m.at["frecuencia", "exactitud"]) if "frecuencia" in m.index else float("nan")
    return {"nombre": config.NOMBRES_MODELOS[config.MODELO_PRINCIPAL], "nivel": nivel, "auc": auc,
            "exactitud": float(m.at[config.MODELO_PRINCIPAL, "exactitud"]),
            "exactitud_referencia": referencia, "n": int(m.at[config.MODELO_PRINCIPAL, "n"])}


def pesos_de_fecha_y_previos(serie: pd.DataFrame) -> tuple[pd.Timestamp, pd.Series, pd.Series | None]:
    """De una tabla de pesos (fechas x ETFs): los de la última fecha y los del cierre del mes anterior."""
    serie = serie[[t for t in config.TICKERS if t in serie.columns]]
    fecha = serie.index[-1]
    anteriores = serie[serie.index.to_period("M") < fecha.to_period("M")]
    return fecha, serie.iloc[-1], (anteriores.iloc[-1] if len(anteriores) else None)


def _p(x, decimales: int = 1):
    """Fracción -> porcentaje redondeado (0,2137 -> 21,4). None si no hay dato."""
    if x is None or x != x:
        return None
    valor = round(float(x) * 100.0, decimales)
    return int(valor) if decimales == 0 else valor


def armar_plan(serie: pd.DataFrame, perfil: str, senales: pd.DataFrame, confianza: dict | None,
               usa_modelo: bool) -> dict:
    """Calcula la recomendación de un perfil y prepara los datos que verá el LLM.

    serie: pesos de partida (fechas x ETFs). senales: una fila por ETF con ret_1m, ret_3m,
    ret_6m, ret_12m, vol_60d, drawdown y, si hay modelo, prob_sube.
    """
    fecha, hoy, antes = pesos_de_fecha_y_previos(serie)
    objetivo = ajustar_a_perfil(hoy, perfil)
    previo = ajustar_a_perfil(antes, perfil) if antes is not None else None
    acciones = clasificar_acciones(objetivo, previo)
    minimo, maximo = config.PERFILES[perfil]
    parte_rv = peso_de_renta_variable(objetivo)

    etfs = []
    for t in objetivo.index:
        s = senales.loc[t] if t in senales.index else {}
        etfs.append({
            "ticker": t, "nombre": config.ETFS[t][0], "clase": config.ETFS[t][1],
            "accion": acciones[t], "peso_pct": _p(objetivo[t]),
            "peso_mes_anterior_pct": _p(previo[t]) if previo is not None else None,
            "cambio_puntos": round(_p(objetivo[t]) - _p(previo[t]), 1) if previo is not None else None,
            "prob_subida_pct": _p(s.get("prob_sube"), 0),
            "retorno_1m_pct": _p(s.get("ret_1m")), "retorno_3m_pct": _p(s.get("ret_3m")),
            "retorno_6m_pct": _p(s.get("ret_6m")), "retorno_12m_pct": _p(s.get("ret_12m")),
            "volatilidad_anual_pct": _p(s.get("vol_60d")), "distancia_al_maximo_pct": _p(s.get("drawdown")),
        })
    entrada = {
        "fecha": f"{fecha:%Y-%m-%d}", "perfil": perfil,
        "renta_variable": {"rango_del_perfil_pct": [_p(minimo, 0), _p(maximo, 0)],
                           "en_la_cartera_pct": _p(parte_rv), "oro_y_bonos_pct": round(100.0 - _p(parte_rv), 1)},
        "reglas": {"peso_minimo_pct": _p(config.PESO_MIN, 0), "peso_maximo_pct": _p(config.PESO_MAX, 0),
                   "cambio_minimo_para_actuar_puntos": _p(config.BANDA_ACCION, 0),
                   "ventana_volatilidad_dias": config.VENTANA_VOL},
        "modelo": None if confianza is None else {
            "nombre": confianza["nombre"], "horizonte_dias": config.HORIZONTE_ML,
            "confianza": confianza["nivel"], "auc": round(confianza["auc"], 2),
            "exactitud_pct": _p(confianza["exactitud"]),
            "exactitud_referencia_simple_pct": _p(confianza["exactitud_referencia"]),
            "inclina_los_pesos": usa_modelo},
        "etfs": etfs,
    }
    return {"fecha": fecha, "perfil": perfil, "pesos": objetivo, "previos": previo, "acciones": acciones,
            "usa_modelo": usa_modelo, "confianza": confianza, "entrada": entrada}


def huella(entrada: dict) -> str:
    """Identifica una entrada y las instrucciones del LLM: si nada cambia, no hace falta volver a llamarlo."""
    contenido = {"entrada": entrada, "instrucciones": PROMPT_SISTEMA}
    return hashlib.sha256(json.dumps(contenido, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


# --- Lo que redacta el LLM -------------------------------------------------------------
PROMPT_SISTEMA = """Eres el redactor de un panel académico de asignación de cartera entre cinco ETFs.
Recibirás un JSON con los pesos ya calculados para un perfil de riesgo, la acción ya decidida para cada ETF \
y los indicadores que la sustentan. Tu única tarea es explicar esos datos en español claro, para una persona \
que no es experta en finanzas.

Reglas obligatorias:
1. No decides nada. Copia en "accion" exactamente la acción que trae cada ETF y no propongas pesos ni \
operaciones distintas de las que recibes. «Comprar» significa subir el peso del ETF (comprar más) y «Vender», \
bajar su peso (vender una parte, nunca todo, porque cada ETF conserva el peso mínimo).
2. Usa solo cifras que aparezcan en el JSON (puedes redondearlas a enteros). No calcules cifras nuevas ni \
inventes datos, noticias o eventos.
3. No predigas precios ni prometas resultados. No uses «garantiza», «seguro», «sin riesgo» ni precios objetivo.
4. En el resumen no hables del modelo predictivo: la frase sobre el modelo la agrega el código. En la \
explicación de cada ETF puedes citar su probabilidad, aclarando que es una estimación y respetando el campo \
"confianza". Si "modelo" es null, no lo menciones.
5. Sé concreto. En el resumen nombra por su ticker cada ETF que se compra y cada uno que se vende, con el dato \
del JSON que mejor lo explica (por ejemplo, su retorno a 6 meses, su volatilidad o su cambio de peso). Escribe \
«conviene comprar» o «conviene vender», nunca «se debe». No uses frases vagas como «responde a las variaciones \
recientes de los activos».
6. Explica cada ETF con sus propios indicadores: di de qué peso pasa a qué peso y por qué sube, baja o se mantiene.
7. Escribe en español correcto, con tildes y eñes. Escribe los porcentajes con el símbolo % y coma decimal, \
como en el panel; nunca «por ciento».
8. Tono sobrio y directo. Sin saludos, sin emojis y sin formato Markdown.

Responde únicamente con un JSON con esta forma:
{"resumen": "2 a 4 frases: cuánta renta variable queda frente al rango del perfil y qué ETFs comprar y cuáles \
vender, con su razón principal",
 "etfs": [{"ticker": "SPY", "accion": "Comprar | Vender | Mantener", "explicacion": "1 o 2 frases con el cambio \
de peso y el dato que lo explica"}],
 "riesgos": ["2 o 3 frases cortas"],
 "que_vigilar": ["1 a 3 frases cortas"]}
Incluye los cinco ETFs, una vez cada uno."""

ESQUEMA_SALIDA = {
    "type": "OBJECT", "required": ["resumen", "etfs", "riesgos", "que_vigilar"],
    "properties": {
        "resumen": {"type": "STRING"},
        "etfs": {"type": "ARRAY", "items": {
            "type": "OBJECT", "required": ["ticker", "accion", "explicacion"],
            "properties": {"ticker": {"type": "STRING"}, "accion": {"type": "STRING", "enum": ACCIONES},
                           "explicacion": {"type": "STRING"}}}},
        "riesgos": {"type": "ARRAY", "items": {"type": "STRING"}},
        "que_vigilar": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
}


def mensaje_usuario(entrada: dict) -> str:
    return "Datos para redactar:\n" + json.dumps(entrada, ensure_ascii=False, indent=1)


# --- Revisión de lo que devuelve el LLM ------------------------------------------------
FRASES_PROHIBIDAS = ["garantiz", "sin riesgo", "libre de riesgo", "seguro que", "con certeza", "con seguridad",
                     "no puede perder", "precio objetivo", "ganancia segura", "rentabilidad segura",
                     "vender todo", "vende todo", "venta total",
                     "debe comprar", "debe vender", "debe mantener", "deben comprar", "deben vender"]
_NUMERO = re.compile(r"\d+(?:[.,]\d+)?")


def _numeros_de(texto: str) -> list[tuple[float, int]]:
    """Números de un texto como (valor, decimales). Acepta coma o punto decimal."""
    salida = []
    for trozo in _NUMERO.findall(texto):
        decimales = len(re.split(r"[.,]", trozo)[1]) if re.search(r"[.,]", trozo) else 0
        salida.append((float(trozo.replace(",", ".")), decimales))
    return salida


def _cifras_permitidas(dato) -> set[float]:
    """Todos los números de la entrada (también los que van dentro de textos, como «S&P 500»)."""
    if isinstance(dato, bool) or dato is None:
        return set()
    if isinstance(dato, (int, float)):
        return {abs(float(dato))}
    if isinstance(dato, str):
        return {v for v, _ in _numeros_de(dato)}
    hijos = dato.values() if isinstance(dato, dict) else dato
    return set().union(*[_cifras_permitidas(h) for h in hijos]) if hijos else set()


def cifras_ajenas(texto: str, entrada: dict) -> list[str]:
    """Números del texto que no salen de la entrada (ni redondeando).

    Se toleran los enteros hasta 12 («cinco ETFs» escrito como 5, «6 meses») y el 100.
    """
    permitidas = _cifras_permitidas(entrada)
    ajenas = []
    for valor, decimales in _numeros_de(texto):
        if decimales == 0 and (valor <= 12 or valor == 100):
            continue
        margen = 0.5 * 10 ** (-decimales) + 1e-9
        if not any(abs(valor - p) <= margen for p in permitidas):
            ajenas.append(f.num(valor, decimales))
    return ajenas


def _es_texto(x, maximo: int, minimo: int = 10) -> bool:
    return isinstance(x, str) and minimo <= len(x.strip()) <= maximo


def validar(salida, entrada: dict) -> list[str]:
    """Revisa la respuesta del LLM. Devuelve la lista de problemas (vacía = se acepta)."""
    if not isinstance(salida, dict):
        return ["la respuesta no es un objeto JSON"]
    problemas = []
    if not _es_texto(salida.get("resumen"), MAX_RESUMEN, 40):
        problemas.append("falta «resumen» o su largo no es válido")
    for campo, lo, hi in (("riesgos", 2, 3), ("que_vigilar", 1, 3)):
        lista = salida.get(campo)
        if not (isinstance(lista, list) and lo <= len(lista) <= hi and all(_es_texto(x, MAX_FRASE) for x in lista)):
            problemas.append(f"«{campo}» debe traer de {lo} a {hi} frases")

    esperadas = {e["ticker"]: e["accion"] for e in entrada["etfs"]}
    etfs = salida.get("etfs")
    if not (isinstance(etfs, list) and all(isinstance(e, dict) for e in etfs)):
        return problemas + ["«etfs» debe ser una lista de objetos"]
    if sorted(str(e.get("ticker")) for e in etfs) != sorted(esperadas):
        problemas.append("«etfs» debe traer exactamente los cinco ETFs de la entrada, una vez cada uno")
    for e in etfs:
        t = e.get("ticker")
        if t in esperadas and e.get("accion") != esperadas[t]:
            problemas.append(f"la acción de {t} debe ser «{esperadas[t]}»")
        if not _es_texto(e.get("explicacion"), MAX_EXPLICACION, 20):
            problemas.append(f"falta la explicación de {t} o su largo no es válido")
    resumen = salida["resumen"] if isinstance(salida.get("resumen"), str) else ""
    sin_nombrar = [t for t, a in esperadas.items() if a in ("Comprar", "Vender") and t not in resumen]
    if sin_nombrar:
        problemas.append("el resumen debe nombrar los ETFs que se compran o se venden: " + ", ".join(sin_nombrar))
    if "modelo" in resumen.lower():
        problemas.append("el resumen no debe hablar del modelo: esa frase la agrega el código")
    if problemas:
        return problemas

    texto = " ".join([salida["resumen"], *salida["riesgos"], *salida["que_vigilar"],
                      *[e["explicacion"] for e in etfs]])
    ajenas = cifras_ajenas(texto, entrada)
    if ajenas:
        problemas.append("usa cifras que no están en los datos: " + ", ".join(sorted(set(ajenas))[:8]))
    usadas = [p for p in FRASES_PROHIBIDAS if p in texto.lower()]
    if usadas:
        problemas.append("usa expresiones no permitidas: " + ", ".join(usadas))
    return problemas


def frase_del_modelo(modelo: dict) -> str:
    """Si el modelo se usa o no es un dato, no una redacción: esta frase la escribe siempre el código."""
    uso = "se usa para inclinar los pesos" if modelo["inclina_los_pesos"] else "no se usa para inclinar los pesos"
    return f"El modelo predictivo tiene una confianza {modelo['confianza']} (AUC {f.num(modelo['auc'], 2)}) y {uso}."


def _ordenar(salida: dict, entrada: dict) -> dict:
    """Deja la respuesta en una forma fija: explicaciones por ticker, en el orden de la entrada, y la frase
    sobre el modelo al final del resumen."""
    por_ticker = {e["ticker"]: e["explicacion"].strip() for e in salida["etfs"]}
    resumen = salida["resumen"].strip()
    if entrada.get("modelo"):
        resumen = f"{resumen} {frase_del_modelo(entrada['modelo'])}"
    return {"resumen": resumen,
            "explicaciones": {e["ticker"]: por_ticker[e["ticker"]] for e in entrada["etfs"]},
            "riesgos": [x.strip() for x in salida["riesgos"]],
            "que_vigilar": [x.strip() for x in salida["que_vigilar"]]}


def leer_json(texto: str):
    """Interpreta la respuesta del LLM como JSON (tolera que venga envuelta en ```json ... ```)."""
    limpio = re.sub(r"^```(?:json)?\s*|\s*```$", "", (texto or "").strip())
    try:
        return json.loads(limpio)
    except (ValueError, TypeError):
        return None


# --- Plantilla de respaldo (sin LLM) ---------------------------------------------------
def _lista(nombres: list[str]) -> str:
    return nombres[0] if len(nombres) == 1 else ", ".join(nombres[:-1]) + " y " + nombres[-1]


def texto_por_reglas(entrada: dict) -> dict:
    """La misma explicación, armada con frases fijas a partir de los datos."""
    rv, etfs, modelo = entrada["renta_variable"], entrada["etfs"], entrada["modelo"]

    def n1(x) -> str:
        return f.num(x, 1)

    frases = [f"Para el perfil {entrada['perfil']}, la cartera queda con {n1(rv['en_la_cartera_pct'])} % en renta "
              f"variable (el perfil admite entre {f.num(rv['rango_del_perfil_pct'][0], 0)} % y "
              f"{f.num(rv['rango_del_perfil_pct'][1], 0)} %) y {n1(rv['oro_y_bonos_pct'])} % entre oro y bonos."]
    if all(e["accion"] == "Mantener" for e in etfs):
        frases.append("Frente al cierre del mes anterior, ningún peso cambia lo suficiente como para operar.")
    else:
        cambios = []
        for accion, verbo in (("Comprar", "comprar más {} (sube su peso)"),
                              ("Vender", "vender una parte de {} (baja su peso)")):
            grupo = [e["ticker"] for e in etfs if e["accion"] == accion]
            if grupo:
                cambios.append(verbo.format(_lista(grupo)))
        resto = "; el resto se mantiene" if any(e["accion"] == "Mantener" for e in etfs) else ""
        frases.append(f"Frente al cierre del mes anterior, conviene {' y '.join(cambios)}{resto}.")
    if modelo:
        frases.append(frase_del_modelo(modelo))

    explicaciones = {}
    for e in etfs:
        if e["cambio_puntos"] is None:
            partes = [f"Peso sugerido de {n1(e['peso_pct'])} %; todavía no hay un mes anterior para comparar."]
        elif e["accion"] == "Mantener":
            partes = [f"Se mantiene cerca de {n1(e['peso_pct'])} %: el cambio frente al mes anterior "
                      f"({f.num_signo(e['cambio_puntos'], 1)} puntos) no llega al mínimo para actuar."]
        else:
            partes = [f"{'Sube' if e['accion'] == 'Comprar' else 'Baja'} de {n1(e['peso_mes_anterior_pct'])} % a "
                      f"{n1(e['peso_pct'])} % ({f.num_signo(e['cambio_puntos'], 1)} puntos)."]
        if e["retorno_6m_pct"] is not None and e["volatilidad_anual_pct"] is not None:
            partes.append(f"Retorno a 6 meses de {f.num_signo(e['retorno_6m_pct'], 1)} % y volatilidad anual de "
                          f"{n1(e['volatilidad_anual_pct'])} %.")
        if modelo and e["prob_subida_pct"] is not None:
            partes.append(f"El modelo estima {f.num(e['prob_subida_pct'], 0)} % de probabilidad de que suba en el "
                          f"próximo mes.")
        explicaciones[e["ticker"]] = " ".join(partes)

    riesgos = ["La regla sigue tendencias: ante un giro brusco del mercado reacciona con retraso.",
               "Los pesos se revisan una vez al mes; entre revisiones la cartera puede alejarse del reparto sugerido."]
    con_vol = [e for e in etfs if e["volatilidad_anual_pct"] is not None]
    if con_vol:
        mayor = max(con_vol, key=lambda e: e["volatilidad_anual_pct"])
        riesgos.append(f"{mayor['ticker']} es hoy el ETF más volátil ({n1(mayor['volatilidad_anual_pct'])} % anual): "
                       f"sus movimientos pesan más en el resultado.")
    vigilar = ["El próximo cierre de mes: la regla vuelve a calcular los pesos con los datos de ese día."]
    negativos = [e["ticker"] for e in etfs if e["retorno_6m_pct"] is not None and e["retorno_6m_pct"] <= 0]
    if negativos:
        vigilar.append(f"El retorno a 6 meses de {_lista(negativos)}: mientras siga negativo, su peso va a la mitad.")
    if modelo and modelo["confianza"] == "baja":
        vigilar.append("El desempeño del modelo: su confianza es baja, así que conviene leer sus "
                       "probabilidades con cautela.")
    return {"resumen": " ".join(frases), "explicaciones": explicaciones, "riesgos": riesgos[:3],
            "que_vigilar": vigilar[:3]}


# --- Cliente del LLM -------------------------------------------------------------------
class ErrorLLM(RuntimeError):
    """El modelo de lenguaje no respondió o respondió algo inutilizable."""


class ClienteGemini:
    """Llamadas HTTP directas a la API de Gemini (sin SDK), para ver exactamente qué se envía.

    La clave viaja en una cabecera, nunca en la dirección. Si un modelo ya no existe (404)
    se pasa al siguiente de la lista.
    """

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent"

    def __init__(self, clave: str, modelos: list[str] | None = None, sesion=None, espera: float = 4.0):
        if not clave:
            raise ErrorLLM(f"Falta {config.LLM_VARIABLE_CLAVE}.")
        self.clave = clave
        self.modelos = list(modelos or config.LLM_MODELOS)
        self.s = sesion or requests.Session()
        self.espera = espera

    def _post(self, modelo: str, cuerpo: dict):
        try:
            return self.s.post(self.URL.format(modelo=modelo), json=cuerpo, timeout=config.LLM_TIEMPO_ESPERA,
                               headers={"x-goog-api-key": self.clave, "Content-Type": "application/json"})
        except requests.RequestException as e:
            raise ErrorLLM(f"no se pudo contactar a la API ({type(e).__name__})") from e

    def _llamar(self, modelo: str, sistema: str, usuario: str, esquema: dict | None):
        ajustes = {"temperature": config.LLM_TEMPERATURA, "maxOutputTokens": config.LLM_MAX_TOKENS_SALIDA,
                   "responseMimeType": "application/json"}
        if esquema:
            ajustes["responseSchema"] = esquema
        cuerpo = {"systemInstruction": {"parts": [{"text": sistema}]},
                  "contents": [{"role": "user", "parts": [{"text": usuario}]}],
                  "generationConfig": ajustes}
        r = self._post(modelo, cuerpo)
        if r.status_code == 400 and esquema:          # algunos modelos no aceptan este esquema: se pide solo JSON
            ajustes.pop("responseSchema")
            r = self._post(modelo, cuerpo)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(self.espera)
            r = self._post(modelo, cuerpo)
        return r

    def generar(self, sistema: str, usuario: str, esquema: dict | None = None) -> dict:
        """Devuelve {"texto", "modelo", "tokens_entrada", "tokens_salida"} o levanta ErrorLLM."""
        ultimo = "no hay modelos configurados"
        for modelo in list(self.modelos):
            r = self._llamar(modelo, sistema, usuario, esquema)
            if r.status_code == 404:                  # modelo retirado o mal escrito
                ultimo = f"el modelo {modelo} no existe"
                self.modelos.remove(modelo)
                continue
            if not r.ok:
                raise ErrorLLM(f"la API respondió {r.status_code}: {_mensaje_de_error(r)}")
            try:
                datos = r.json()
            except ValueError as e:
                raise ErrorLLM("la API no devolvió JSON") from e
            candidatos = datos.get("candidates") or [{}]
            partes = (candidatos[0].get("content") or {}).get("parts") or []
            texto = "".join(p.get("text", "") for p in partes if not p.get("thought"))
            if not texto.strip():
                raise ErrorLLM(f"respuesta vacía (motivo: {candidatos[0].get('finishReason', 'desconocido')})")
            uso = datos.get("usageMetadata") or {}
            return {"texto": texto, "modelo": modelo,
                    "tokens_entrada": int(uso.get("promptTokenCount") or 0),
                    "tokens_salida": int((uso.get("candidatesTokenCount") or 0) + (uso.get("thoughtsTokenCount") or 0))}
        raise ErrorLLM(ultimo)


def _mensaje_de_error(r) -> str:
    try:
        return str(r.json()["error"]["message"])[:200]
    except Exception:
        return (r.text or "")[:200]


# --- Redacción con revisión y respaldo -------------------------------------------------
def redactar(entrada: dict, cliente, max_llamadas: int = 2) -> tuple[dict, str, str | None, list[dict]]:
    """Pide la explicación al LLM, la revisa y, si hace falta, lo intenta una vez más.

    Devuelve (textos, fuente, modelo, eventos). fuente es "llm" si el texto pasó la revisión
    y "reglas" si se usó la plantilla. Cada llamada queda descrita en `eventos`.
    """
    eventos, correccion = [], ""
    for _ in range(max_llamadas):
        inicio = time.time()
        try:
            r = cliente.generar(PROMPT_SISTEMA, mensaje_usuario(entrada) + correccion, ESQUEMA_SALIDA)
        except ErrorLLM as e:
            eventos.append(_evento(entrada, "error", inicio, mensaje=str(e)))
            break                                      # un fallo de la API no se arregla pidiendo otra redacción
        salida = leer_json(r["texto"])
        problemas = ["la respuesta no es JSON válido"] if salida is None else validar(salida, entrada)
        if not problemas:
            eventos.append(_evento(entrada, "ok", inicio, r))
            return _ordenar(salida, entrada), "llm", r["modelo"], eventos
        eventos.append(_evento(entrada, "rechazada", inicio, r, "; ".join(problemas)))
        correccion = ("\n\nTu respuesta anterior fue rechazada por estos motivos: " + "; ".join(problemas) +
                      ". Corrígela y responde otra vez solo con el JSON.")
    return texto_por_reglas(entrada), "reglas", None, eventos


def _evento(entrada: dict, estado: str, inicio: float, r: dict | None = None, mensaje: str = "") -> dict:
    r = r or {}
    return {"perfil": entrada["perfil"], "modelo": r.get("modelo"), "estado": estado,
            "tokens_entrada": r.get("tokens_entrada"), "tokens_salida": r.get("tokens_salida"),
            "latencia_ms": int((time.time() - inicio) * 1000), "mensaje": mensaje[:500] or None}


def fila_de_recomendacion(plan: dict, textos: dict, fuente: str, modelo_llm: str | None) -> dict:
    """La recomendación de un perfil tal como se guarda en la base de datos."""
    previos, entrada = plan["previos"], plan["entrada"]
    etfs = [{"ticker": e["ticker"], "peso": float(plan["pesos"][e["ticker"]]),
             "peso_previo": float(previos[e["ticker"]]) if previos is not None else None,
             "accion": e["accion"], "prob_sube": None if e["prob_subida_pct"] is None else e["prob_subida_pct"] / 100.0,
             "explicacion": textos["explicaciones"][e["ticker"]]} for e in entrada["etfs"]]
    conf = plan["confianza"]
    detalle = {"etfs": etfs, "riesgos": textos["riesgos"], "que_vigilar": textos["que_vigilar"],
               "renta_variable": peso_de_renta_variable(plan["pesos"]), "rango": list(config.PERFILES[plan["perfil"]]),
               "usa_modelo": plan["usa_modelo"], "confianza": conf["nivel"] if conf else None,
               "auc": conf["auc"] if conf else None, "aviso": AVISO_LEGAL}
    return {"fecha": f"{plan['fecha']:%Y-%m-%d}", "perfil": plan["perfil"], "fuente": fuente,
            "modelo_llm": modelo_llm, "resumen": textos["resumen"], "detalle": detalle,
            "entrada_hash": huella(entrada)}
