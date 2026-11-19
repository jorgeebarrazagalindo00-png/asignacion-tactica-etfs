"""Modelos de lenguaje de mentira para las pruebas: ninguna llama a un servicio real."""

import json

from etf2 import prescriptivo as ps


def respuesta_valida(entrada: dict) -> dict:
    """Lo que devolvería un LLM obediente: mismas acciones y solo cifras de la entrada."""
    t = ps.texto_por_reglas(entrada)
    resumen = t["resumen"]
    if entrada.get("modelo"):           # la frase del modelo no la escribe el LLM: la agrega el código
        resumen = resumen.replace(ps.frase_del_modelo(entrada["modelo"]), "").strip()
    return {"resumen": resumen, "riesgos": t["riesgos"], "que_vigilar": t["que_vigilar"],
            "etfs": [{"ticker": e["ticker"], "accion": e["accion"], "explicacion": t["explicaciones"][e["ticker"]]}
                     for e in entrada["etfs"]]}


def entrada_del_mensaje(usuario: str) -> dict:
    """Recupera el JSON de entrada que viaja en el mensaje del usuario."""
    cuerpo = usuario.split("\n", 1)[1].split("\n\nTu respuesta anterior", 1)[0]
    return json.loads(cuerpo)


class LLMFalso:
    """Devuelve, en orden, las respuestas que se le cargan (un dict va como JSON; una excepción se lanza)."""

    def __init__(self, *respuestas):
        self.respuestas, self.pedidos = list(respuestas), []

    def generar(self, sistema, usuario, esquema=None):
        self.pedidos.append({"sistema": sistema, "usuario": usuario, "esquema": esquema})
        r = self.respuestas.pop(0)
        if isinstance(r, Exception):
            raise r
        texto = r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)
        return {"texto": texto, "modelo": "modelo-de-prueba", "tokens_entrada": 900, "tokens_salida": 300}


class LLMObediente:
    """Responde bien a cualquier entrada. Con `estropear`, altera cada respuesta antes de enviarla."""

    def __init__(self, estropear=None):
        self.pedidos, self.estropear = [], estropear

    def generar(self, sistema, usuario, esquema=None):
        self.pedidos.append(usuario)
        r = respuesta_valida(entrada_del_mensaje(usuario))
        if self.estropear:
            self.estropear(r)
        return {"texto": json.dumps(r, ensure_ascii=False), "modelo": "modelo-de-prueba",
                "tokens_entrada": 950, "tokens_salida": 320}
