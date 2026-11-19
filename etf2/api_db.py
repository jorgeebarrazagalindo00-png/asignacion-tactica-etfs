"""Cliente mínimo para la API REST de Supabase (PostgREST).

Supabase genera automáticamente una API REST sobre cada tabla de la base de
datos. Aquí no se usa ninguna librería especial: son llamadas HTTP normales con
`requests`, así que se puede ver exactamente qué se envía:

    POST   {url}/rest/v1/<tabla>?on_conflict=col1,col2   -> insertar / actualizar (upsert)
    GET    {url}/rest/v1/<tabla>?select=...&order=...    -> leer
    PATCH  {url}/rest/v1/<tabla>?id=eq.5                 -> modificar filas

Seguridad: el pipeline escribe con la clave SECRETA del proyecto (guardada como
secreto en GitHub, jamás en el código). El dashboard solo debe usar la clave
PUBLICABLE (o `anon`), que por las políticas de la base solo puede LEER.

Tipos de clave: las nuevas (`sb_secret_...`, `sb_publishable_...`) no son JWT y
Supabase indica enviarlas SOLO en la cabecera `apikey`. Las antiguas (`service_role`,
`anon`) son JWT y además se envían como `Authorization: Bearer`.
"""

import datetime as dt
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests


class ErrorAPI(RuntimeError):
    """La API respondió con error o no se pudo contactar. `codigo` es el estado HTTP, si lo hubo."""

    def __init__(self, mensaje: str, codigo: int | None = None):
        super().__init__(mensaje)
        self.codigo = codigo


def a_json_seguro(x):
    """Convierte valores de pandas/numpy a algo que JSON entienda (NaN -> null)."""
    if x is None or x is pd.NaT or x is pd.NA:
        return None
    if isinstance(x, (pd.Timestamp, dt.datetime)):
        return x.date().isoformat()
    if isinstance(x, dt.date):
        return x.isoformat()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def registros(df: pd.DataFrame, **extra) -> list[dict]:
    """DataFrame -> lista de dicts lista para enviar como JSON."""
    filas = []
    for fila in df.to_dict("records"):
        limpio = {k: a_json_seguro(v) for k, v in fila.items()}
        limpio.update(extra)
        filas.append(limpio)
    return filas


class ClienteSupabase:
    def __init__(self, url: str, clave: str, sesion=None, reintentos: int = 3, espera: float = 1.5):
        if not url or not clave:
            raise ErrorAPI("Faltan SUPABASE_URL y/o SUPABASE_SECRET_KEY.")
        self.base = url.rstrip("/") + "/rest/v1"
        self.reintentos = reintentos
        self.espera = espera
        self.s = sesion or requests.Session()
        self.s.headers.update({"apikey": clave, "Content-Type": "application/json"})
        if clave.count(".") == 2:          # clave antigua (JWT): también va como Bearer
            self.s.headers["Authorization"] = f"Bearer {clave}"
        self._local = threading.local()

    # -- bajo nivel ----------------------------------------------------------
    def _pedir(self, metodo: str, ruta: str, sesion=None, **kw) -> requests.Response:
        s = sesion or self.s
        ultimo = None
        for intento in range(1, self.reintentos + 1):
            try:
                r = s.request(metodo, f"{self.base}/{ruta}", timeout=30, **kw)
            except requests.RequestException as e:
                ultimo = f"{type(e).__name__}"
                if intento == self.reintentos:
                    raise ErrorAPI(f"{metodo} {ruta}: no se pudo contactar a Supabase ({ultimo}).") from e
                time.sleep(self.espera * intento)
                continue
            if r.status_code in (429, 500, 502, 503, 504) and intento < self.reintentos:
                time.sleep(self.espera * intento)
                continue
            if not r.ok:
                raise ErrorAPI(f"{metodo} {ruta} -> {r.status_code}: {r.text[:300]}", codigo=r.status_code)
            return r
        raise ErrorAPI(f"{metodo} {ruta}: agotados los reintentos ({ultimo}).")

    # -- operaciones ---------------------------------------------------------
    def insertar(self, tabla: str, filas: list[dict], conflicto: str | None = None,
                 lote: int = 1000) -> int:
        """Inserta filas en lotes. Con `conflicto` (columnas de la clave) hace upsert:
        si la fila ya existe la actualiza en vez de fallar. Devuelve cuántas envió."""
        for i in range(0, len(filas), lote):
            trozo = filas[i:i + lote]
            if conflicto:
                params = {"on_conflict": conflicto}
                prefer = "resolution=merge-duplicates,return=minimal"
            else:
                params, prefer = None, "return=minimal"
            self._pedir("POST", tabla, params=params, headers={"Prefer": prefer},
                        data=json.dumps(trozo))
        return len(filas)

    def insertar_devolviendo(self, tabla: str, fila: dict) -> dict:
        """Inserta UNA fila y devuelve la fila creada (con su id)."""
        r = self._pedir("POST", tabla, headers={"Prefer": "return=representation"},
                        data=json.dumps(fila))
        datos = r.json()
        return datos[0] if isinstance(datos, list) else datos

    def actualizar(self, tabla: str, filtro: dict, cambios: dict) -> None:
        """PATCH: modifica las filas que cumplan filtro (p. ej. {'id': 'eq.5'})."""
        self._pedir("PATCH", tabla, params=filtro, headers={"Prefer": "return=minimal"},
                    data=json.dumps(cambios))

    def leer(self, tabla: str, params: dict | None = None, pagina: int = 1000,
             maximo: int | None = None, hilos: int = 1) -> list[dict]:
        """Lee las filas que cumplan params, de a `pagina` (Supabase limita cada respuesta,
        por defecto a 1000 filas).

        maximo: se detiene al juntar ese número de filas (con order=...desc trae las más recientes).
        hilos:  con más de 1, pide el total de filas y descarga las páginas en paralelo. Requiere
                un `order` que no deje empates, para que las páginas no se solapen.
        """
        params = dict(params or {})
        if hilos > 1 and maximo is None:
            return self._leer_en_paralelo(tabla, params, pagina, hilos)
        out, offset = [], 0
        while True:
            pedir = pagina if maximo is None else min(pagina, maximo - len(out))
            p = {**params, "limit": str(pedir), "offset": str(offset)}
            datos = self._pedir("GET", tabla, params=p).json()
            out.extend(datos)
            if len(datos) < pedir or (maximo is not None and len(out) >= maximo):
                return out
            offset += pedir

    def _sesion_del_hilo(self) -> requests.Session:
        s = getattr(self._local, "sesion", None)
        if s is None:
            s = requests.Session()
            s.headers.update(self.s.headers)
            self._local.sesion = s
        return s

    def _leer_en_paralelo(self, tabla: str, params: dict, pagina: int, hilos: int) -> list[dict]:
        r = self._pedir("GET", tabla, params={**params, "limit": str(pagina), "offset": "0"},
                        headers={"Prefer": "count=exact"})
        primera = r.json()
        m = re.search(r"/(\d+)$", r.headers.get("Content-Range", ""))
        if not m:                                  # el servidor no informó el total: modo normal
            return primera if len(primera) < pagina else self.leer(tabla, params, pagina)
        total, tam = int(m.group(1)), len(primera)
        if tam == 0 or total <= tam:
            return primera

        def traer(offset: int) -> list[dict]:
            p = {**params, "limit": str(tam), "offset": str(offset)}
            return self._pedir("GET", tabla, sesion=self._sesion_del_hilo(), params=p).json()

        with ThreadPoolExecutor(max_workers=hilos) as ex:
            partes = list(ex.map(traer, range(tam, total, tam)))
        for parte in partes:
            primera.extend(parte)
        return primera

    def ultima_fecha(self, tabla: str, columna: str = "fecha") -> pd.Timestamp | None:
        """Fecha más reciente guardada en una tabla (None si está vacía)."""
        datos = self._pedir("GET", tabla, params={"select": columna, "order": f"{columna}.desc",
                                                  "limit": "1"}).json()
        return pd.Timestamp(datos[0][columna]) if datos else None
