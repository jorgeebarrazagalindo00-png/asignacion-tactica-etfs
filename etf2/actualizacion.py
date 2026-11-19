"""Botón «Actualizar ahora» del panel.

El panel no ejecuta el proceso ni escribe en la base: le pide a GitHub Actions que corra
el mismo workflow programado (con sus propios secretos) y consulta en qué va.

API usada:
  POST /repos/{repo}/actions/workflows/{flujo}/dispatches   lanzar (responde 204)
  GET  /repos/{repo}/actions/workflows/{flujo}/runs         últimas ejecuciones
"""

import requests

from etf2 import config

API = "https://api.github.com"
EN_COLA = {"queued", "waiting", "requested", "pending"}
ACTIVOS = EN_COLA | {"in_progress"}
DISPAROS = {"schedule": "Programada", "workflow_dispatch": "Manual"}

_MENSAJES = {
    401: "GitHub rechazó el token: revisa que esté bien copiado y que no haya vencido.",
    403: "El token no tiene permiso para lanzar el workflow: necesita «Actions: Read and write» "
         "sobre el repositorio.",
    404: f"GitHub no encontró el repositorio o el workflow: revisa {config.VARIABLE_REPO_GITHUB} "
         "(usuario/repositorio) y que el token tenga acceso a ese repositorio.",
    422: f"GitHub no aceptó la orden: revisa que exista la rama «{config.GITHUB_RAMA}» y que el workflow "
         "tenga workflow_dispatch.",
}


class ErrorGitHub(Exception):
    pass


def normalizar_repo(repo: str) -> str:
    """Acepta 'usuario/repo' o la URL del repositorio y devuelve 'usuario/repo'."""
    r = (repo or "").strip()
    for prefijo in ("https://github.com/", "http://github.com/", "github.com/"):
        if r.lower().startswith(prefijo):
            r = r[len(prefijo):]
    r = r.strip("/").removesuffix(".git")
    partes = r.split("/")
    if len(partes) != 2 or not all(partes):
        raise ErrorGitHub(f"{config.VARIABLE_REPO_GITHUB} debe tener la forma usuario/repositorio.")
    return r


class ClienteGitHub:
    def __init__(self, token: str, repo: str, flujo: str = config.GITHUB_FLUJO, rama: str = config.GITHUB_RAMA,
                 base: str = API, espera: float = 15):
        self.repo, self.flujo, self.rama = normalizar_repo(repo), flujo, rama
        self.base, self.espera = base.rstrip("/"), espera
        self.sesion = requests.Session()
        self.sesion.headers.update({
            "Authorization": f"Bearer {token.strip()}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })

    def _pedir(self, metodo: str, sufijo: str, **kw) -> requests.Response:
        url = f"{self.base}/repos/{self.repo}/actions/workflows/{self.flujo}/{sufijo}"
        try:
            r = self.sesion.request(metodo, url, timeout=self.espera, **kw)
        except requests.RequestException as err:
            raise ErrorGitHub(f"No se pudo conectar con GitHub ({type(err).__name__}).") from err
        if r.status_code >= 400:
            raise ErrorGitHub(_MENSAJES.get(r.status_code, f"GitHub respondió con el código {r.status_code}."))
        return r

    def lanzar(self, modo: str = "diario") -> None:
        if modo not in ("diario", "historico"):
            raise ValueError("modo debe ser 'diario' o 'historico'")
        self._pedir("POST", "dispatches", json={"ref": self.rama, "inputs": {"modo": modo, "origen": "panel"}})

    def ejecuciones(self, n: int = 5) -> list[dict]:
        datos = self._pedir("GET", "runs", params={"per_page": n}).json()
        return [{
            "id": x.get("id"), "numero": x.get("run_number"), "estado": x.get("status"),
            "resultado": x.get("conclusion"), "evento": x.get("event"), "creada": x.get("created_at"),
            "actualizada": x.get("updated_at"), "url": x.get("html_url"),
        } for x in datos.get("workflow_runs", [])]


def activa(ejecucion: dict) -> bool:
    return ejecucion.get("estado") in ACTIVOS


def hay_activa(ejecuciones: list[dict]) -> bool:
    return any(activa(x) for x in ejecuciones)


def estado(ejecucion: dict) -> tuple[str, str]:
    """(clase de color, texto) de una ejecución, con las mismas etiquetas que la bitácora."""
    if ejecucion.get("estado") in EN_COLA:
        return "plano", "● En cola"
    if ejecucion.get("estado") == "in_progress":
        return "plano", "● En curso"
    return {
        "success": ("up", "✔ Correcta"),
        "failure": ("down", "✖ Con error"),
        "timed_out": ("down", "✖ Tiempo agotado"),
        "cancelled": ("plano", "Cancelada"),
    }.get(ejecucion.get("resultado"), ("plano", ejecucion.get("resultado") or ejecucion.get("estado") or "—"))
