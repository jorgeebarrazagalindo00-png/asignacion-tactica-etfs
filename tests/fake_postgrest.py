"""Servidor HTTP falso que imita lo mínimo de PostgREST (la API de Supabase).

Sirve para probar el pipeline completo sin internet ni cuenta de Supabase:
guarda las tablas en memoria y entiende POST (con on_conflict), GET (filtros eq,
order, limit, offset y el conteo de "Prefer: count=exact") y PATCH (filtro por id).
"""

import datetime
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

CLAVES = {
    "etfs": ["ticker"],
    "precios_diarios": ["ticker", "fecha"],
    "indicadores": ["ticker", "fecha"],
    "pesos_sugeridos": ["fecha", "ticker"],
    "backtest_resumen": ["estrategia"],
    "backtest_curva": ["fecha"],
    "modelo_predicciones": ["fecha", "ticker", "modelo"],
    "modelo_metricas": ["modelo"],
    "modelo_importancias": ["modelo", "variable"],
    "pesos_ml": ["fecha", "ticker"],
    "backtest_ml_resumen": ["estrategia"],
    "backtest_ml_curva": ["fecha"],
    "recomendaciones": ["fecha", "perfil"],
}
CON_ID = {"corridas", "llm_log"}       # tablas cuyo id lo pone la base de datos


class FakeSupabase:
    def __init__(self, max_filas: int = 1000):
        self.tablas: dict[str, list[dict]] = {}
        self.peticiones: list[dict] = []
        self.siguiente_id = {t: 1 for t in CON_ID}     # un contador por tabla, como en la base real
        self._indice: dict = {}      # (tabla, claves, valores) -> fila, para upserts rápidos
        self.max_filas = max_filas
        self.informa_total = True    # False imita un servidor que ignora "Prefer: count=exact"
        self.tablas_ausentes: set[str] = set()   # imita una base donde aún no se crearon esas tablas
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silencio
                pass

            def _responder(self, codigo, cuerpo=None, cabeceras=None):
                datos = b"" if cuerpo is None else json.dumps(cuerpo).encode()
                self.send_response(codigo)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(datos)))
                for k, v in (cabeceras or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(datos)

            def _tabla(self):
                u = urlparse(self.path)
                return u.path.split("/rest/v1/")[1], {k: v[0] for k, v in parse_qs(u.query).items()}

            def _registrar(self, cuerpo=None):
                tabla, q = self._tabla()
                outer.peticiones.append({"metodo": self.command, "tabla": tabla, "query": q,
                                         "headers": dict(self.headers), "cuerpo": cuerpo})
                return tabla, q

            def _falta(self, tabla):
                if tabla in outer.tablas_ausentes:
                    self._responder(404, {"code": "PGRST205",
                                          "message": f"Could not find the table 'public.{tabla}' in the schema cache"})
                    return True
                return False

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                cuerpo = json.loads(self.rfile.read(n) or b"[]")
                tabla, q = self._registrar(cuerpo)
                if self._falta(tabla):
                    return
                filas = cuerpo if isinstance(cuerpo, list) else [cuerpo]
                t = outer.tablas.setdefault(tabla, [])
                devueltas = []
                for f in filas:
                    f = dict(f)
                    if tabla in CON_ID:
                        f["id"] = outer.siguiente_id[tabla]
                        f.setdefault("inicio" if tabla == "corridas" else "creado_en",
                                     datetime.datetime.now(datetime.timezone.utc).isoformat())
                        outer.siguiente_id[tabla] += 1
                    claves = q.get("on_conflict", "").split(",") if q.get("on_conflict") else None
                    if claves:
                        llave = (tabla, tuple(claves), tuple(f.get(k) for k in claves))
                        existente = outer._indice.get(llave)
                        if existente is not None:
                            existente.update(f)
                            devueltas.append(existente)
                            continue
                        outer._indice[llave] = f
                    t.append(f)
                    devueltas.append(f)
                if "return=representation" in self.headers.get("Prefer", ""):
                    return self._responder(201, devueltas)
                self._responder(201)

            def do_GET(self):
                tabla, q = self._registrar()
                if self._falta(tabla):
                    return
                if tabla == "tabla_mala":      # simula un error 400 de PostgREST
                    return self._responder(400, {"message": "columna inexistente"})
                if tabla == "v_pesos_actuales":      # imita la vista de la base de datos
                    pes = outer.tablas.get("pesos_sugeridos", [])
                    ult = max((x["fecha"] for x in pes), default=None)
                    info = {e["ticker"]: e for e in outer.tablas.get("etfs", [])}
                    filas = [{**x, "nombre": info[x["ticker"]]["nombre"],
                              "clase_activo": info[x["ticker"]]["clase_activo"]}
                             for x in pes if x["fecha"] == ult]
                else:
                    filas = list(outer.tablas.get(tabla, []))
                for k, v in q.items():
                    if v.startswith("eq."):
                        filas = [f for f in filas if str(f.get(k)) == v[3:]]
                if "order" in q:
                    orden = [o.split(".") for o in q["order"].split(",")]
                    for col, *dir_ in reversed(orden):
                        filas.sort(key=lambda f: f.get(col), reverse=(dir_ == ["desc"]))
                off = int(q.get("offset", 0))
                lim = min(int(q.get("limit", outer.max_filas)), outer.max_filas)
                pagina = filas[off:off + lim]
                if "count=exact" in self.headers.get("Prefer", "") and outer.informa_total:
                    rango = f"{off}-{off + len(pagina) - 1}" if pagina else "*"
                    return self._responder(206 if len(pagina) < len(filas) else 200, pagina,
                                           {"Content-Range": f"{rango}/{len(filas)}"})
                self._responder(200, pagina)

            def do_PATCH(self):
                n = int(self.headers.get("Content-Length", 0))
                cambios = json.loads(self.rfile.read(n) or b"{}")
                tabla, q = self._registrar(cambios)
                for f in outer.tablas.get(tabla, []):
                    if all(str(f.get(k)) == v[3:] for k, v in q.items() if v.startswith("eq.")):
                        f.update(cambios)
                self._responder(204)

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def cerrar(self):
        self.srv.shutdown()
        self.srv.server_close()
