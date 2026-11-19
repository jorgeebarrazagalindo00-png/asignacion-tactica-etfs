"""Formato de números y fechas en español, y una guarda de seguridad para las claves."""

import base64
import json
import math

import pandas as pd

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
VACIO = "—"
NBSP = "\u00a0"


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) \
        and not (x is pd.NaT)


# --- números -------------------------------------------------------------------
def num(x, decimales: int = 1) -> str:
    """1234.5 -> '1.234,5' (miles con punto, decimales con coma)."""
    if not _ok(x):
        return VACIO
    s = f"{float(x):,.{decimales}f}"
    return s.replace(",", "\0").replace(".", ",").replace("\0", ".").replace("-", "−")


def pct(x, decimales: int = 1) -> str:
    """0.2405 -> '24,1 %'."""
    return VACIO if not _ok(x) else f"{num(float(x) * 100, decimales)}{NBSP}%"


def _signo(v: float, umbral: float) -> str:
    return "+" if v > umbral else ("−" if v < -umbral else "")


def pct_signo(x, decimales: int = 1) -> str:
    """Variación con signo: 0.0241 -> '+2,4 %', -0.013 -> '−1,3 %'."""
    if not _ok(x):
        return VACIO
    v = float(x) * 100
    return f"{_signo(v, 0.5 * 10 ** -decimales)}{num(abs(v), decimales)}{NBSP}%"


def pp(x, decimales: int = 1) -> str:
    """Diferencia en puntos porcentuales con signo: 0.032 -> '+3,2 pp'."""
    if not _ok(x):
        return VACIO
    v = float(x) * 100
    return f"{_signo(v, 0.5 * 10 ** -decimales)}{num(abs(v), decimales)}{NBSP}pp"


def num_signo(x, decimales: int = 2) -> str:
    if not _ok(x):
        return VACIO
    v = float(x)
    return f"{_signo(v, 0.5 * 10 ** -decimales)}{num(abs(v), decimales)}"


def usd(x, decimales: int = 0) -> str:
    """12345.6 -> 'USD 12.346'. Los negativos llevan el signo delante."""
    if not _ok(x):
        return VACIO
    v = float(x)
    return f"{'−' if v < 0 else ''}USD{NBSP}{num(abs(v), decimales)}"


def compacto(x) -> str:
    """Cantidades grandes: 39_985_300 -> '40,0 M'; 1_260 -> '1,3 k'."""
    if not _ok(x):
        return VACIO
    v = abs(float(x))
    signo = "−" if float(x) < 0 else ""
    if v >= 1e9:
        return f"{signo}{num(v / 1e9, 2)}{NBSP}mil{NBSP}M"
    if v >= 1e6:
        return f"{signo}{num(v / 1e6, 1)}{NBSP}M"
    if v >= 1e3:
        return f"{signo}{num(v / 1e3, 1)}{NBSP}k"
    return f"{signo}{num(v, 0)}"


def direccion(x, umbral: float = 0.0) -> str:
    """'sube', 'baja' o 'plano' según el signo (para elegir flecha y color)."""
    if not _ok(x) or abs(float(x)) <= umbral:
        return "plano"
    return "sube" if float(x) > 0 else "baja"


# --- fechas --------------------------------------------------------------------
def fecha(ts) -> str:
    if not _ok(ts):
        return VACIO
    ts = pd.Timestamp(ts)
    return f"{ts.day} {MESES[ts.month - 1]} {ts.year}"


def dia_mes(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.day} {MESES[ts.month - 1]}"


def mes_anio(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{MESES[ts.month - 1]} {ts.year}"


def _a_zona(ts, zona: str) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
    return ts.tz_convert(zona)


def fecha_hora(ts, zona: str = "America/Lima") -> str:
    """Fecha y hora local (por defecto hora de Perú)."""
    if not _ok(ts):
        return VACIO
    ts = _a_zona(ts, zona)
    return f"{fecha(ts)} {ts:%H:%M}"


def hace(ts, ahora=None) -> str:
    """Tiempo transcurrido en palabras: 'hace 5 min', 'hace 3 h', 'hace 2 días'."""
    if not _ok(ts):
        return VACIO
    ahora = pd.Timestamp.now(tz="UTC") if ahora is None else _a_zona(ahora, "UTC")
    seg = (ahora - _a_zona(ts, "UTC")).total_seconds()
    if seg < 90:
        return "hace un momento"
    if seg < 3600:
        return f"hace {seg / 60:.0f} min"
    if seg < 36 * 3600:
        return f"hace {seg / 3600:.0f} h"
    return f"hace {seg / 86400:.0f} días"


def duracion(inicio, fin) -> str:
    if not (_ok(inicio) and _ok(fin)):
        return VACIO
    seg = (pd.Timestamp(fin) - pd.Timestamp(inicio)).total_seconds()
    return f"{seg:.0f} s" if seg < 90 else f"{seg / 60:.1f} min".replace(".", ",")


# --- seguridad -----------------------------------------------------------------
def es_clave_secreta(clave: str) -> bool:
    """True si la clave da permiso de escritura (secret / service_role).

    El dashboard solo debe usar la clave publicable (o anon): una secreta se rechaza.
    """
    if not clave:
        return False
    if clave.startswith("sb_secret_"):
        return True
    partes = clave.split(".")
    if len(partes) == 3:                       # clave antigua (JWT)
        try:
            relleno = "=" * (-len(partes[1]) % 4)
            rol = json.loads(base64.urlsafe_b64decode(partes[1] + relleno)).get("role")
            return rol == "service_role"
        except Exception:
            return False
    return False


# --- gráficos ------------------------------------------------------------------
def separar_etiquetas(valores: dict, minimo: float) -> dict:
    """Separa etiquetas que se pisarían en vertical, conservando su orden.

    valores: {nombre: posición}. Devuelve {nombre: posición ajustada} con al menos
    `minimo` de distancia entre vecinas.
    """
    orden = sorted(valores, key=lambda k: valores[k])
    pos = {k: valores[k] for k in orden}
    for _ in range(50):
        movido = False
        for a, b in zip(orden, orden[1:]):
            falta = minimo - (pos[b] - pos[a])
            if falta > 1e-12:
                pos[a] -= falta / 2
                pos[b] += falta / 2
                movido = True
        if not movido:
            break
    return pos
