"""Cliente de GitHub del botón «Actualizar ahora», contra un servidor local que imita la API."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from etf2 import actualizacion as act
from etf2.actualizacion import ClienteGitHub, ErrorGitHub

RUNS = {"total_count": 2, "workflow_runs": [
    {"id": 902, "run_number": 41, "status": "in_progress", "conclusion": None, "event": "workflow_dispatch",
     "created_at": "2026-11-18T15:02:10Z", "updated_at": "2026-11-18T15:03:00Z",
     "html_url": "https://github.com/elian/ETF2/actions/runs/902"},
    {"id": 901, "run_number": 40, "status": "completed", "conclusion": "success", "event": "schedule",
     "created_at": "2026-11-17T23:04:00Z", "updated_at": "2026-11-17T23:08:30Z",
     "html_url": "https://github.com/elian/ETF2/actions/runs/901"},
]}


class GitHubLocal:
    """Servidor HTTP mínimo: guarda cada petición y responde lo que se le indique."""

    def __init__(self):
        self.peticiones, self.codigo = [], None
        dueno = self

        class Manejador(BaseHTTPRequestHandler):
            def _responder(self, codigo, cuerpo=None):
                datos = b"" if cuerpo is None else json.dumps(cuerpo).encode()
                self.send_response(codigo)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(datos)))
                self.end_headers()
                self.wfile.write(datos)

            def _registrar(self):
                largo = int(self.headers.get("Content-Length") or 0)
                cuerpo = json.loads(self.rfile.read(largo)) if largo else None
                dueno.peticiones.append({"metodo": self.command, "ruta": self.path, "cabeceras": dict(self.headers),
                                         "cuerpo": cuerpo})

            def do_POST(self):
                self._registrar()
                self._responder(dueno.codigo or 204)

            def do_GET(self):
                self._registrar()
                codigo = dueno.codigo or 200
                self._responder(codigo, RUNS if codigo == 200 else {"message": "error"})

            def log_message(self, *a):
                pass

        self.servidor = ThreadingHTTPServer(("127.0.0.1", 0), Manejador)
        self.url = f"http://127.0.0.1:{self.servidor.server_address[1]}"
        threading.Thread(target=self.servidor.serve_forever, daemon=True).start()

    def cerrar(self):
        self.servidor.shutdown()
        self.servidor.server_close()


@pytest.fixture()
def github():
    g = GitHubLocal()
    yield g
    g.cerrar()


def test_lanzar_envia_la_orden_al_workflow(github):
    ClienteGitHub("github_pat_x ", "elian/ETF2", base=github.url).lanzar("historico")
    (p,) = github.peticiones
    assert p["metodo"] == "POST"
    assert p["ruta"] == "/repos/elian/ETF2/actions/workflows/diario.yml/dispatches"
    assert p["cuerpo"] == {"ref": "main", "inputs": {"modo": "historico", "origen": "panel"}}
    assert p["cabeceras"]["Authorization"] == "Bearer github_pat_x"
    assert p["cabeceras"]["Accept"] == "application/vnd.github+json"
    assert p["cabeceras"]["X-GitHub-Api-Version"] == "2022-11-28"


def test_lanzar_rechaza_un_modo_desconocido(github):
    with pytest.raises(ValueError):
        ClienteGitHub("t", "elian/ETF2", base=github.url).lanzar("todo")
    assert github.peticiones == []


def test_ejecuciones_se_traducen(github):
    ejs = ClienteGitHub("t", "elian/ETF2", base=github.url).ejecuciones(n=2)
    assert github.peticiones[0]["ruta"] == "/repos/elian/ETF2/actions/workflows/diario.yml/runs?per_page=2"
    assert [x["numero"] for x in ejs] == [41, 40]
    assert ejs[0] == {"id": 902, "numero": 41, "estado": "in_progress", "resultado": None,
                      "evento": "workflow_dispatch", "creada": "2026-11-18T15:02:10Z",
                      "actualizada": "2026-11-18T15:03:00Z", "url": "https://github.com/elian/ETF2/actions/runs/902"}
    assert act.hay_activa(ejs) and not act.hay_activa(ejs[1:])


@pytest.mark.parametrize("codigo, fragmento", [
    (401, "rechazó el token"), (403, "Actions: Read and write"), (404, "PANEL_GITHUB_REPO"),
    (422, "workflow_dispatch"), (500, "código 500"),
])
def test_errores_de_github_se_explican(github, codigo, fragmento):
    github.codigo = codigo
    with pytest.raises(ErrorGitHub, match=fragmento):
        ClienteGitHub("t", "elian/ETF2", base=github.url).lanzar()
    with pytest.raises(ErrorGitHub, match=fragmento):
        ClienteGitHub("t", "elian/ETF2", base=github.url).ejecuciones()


def test_sin_conexion():
    with pytest.raises(ErrorGitHub, match="No se pudo conectar"):
        ClienteGitHub("t", "elian/ETF2", base="http://127.0.0.1:9", espera=2).ejecuciones()


@pytest.mark.parametrize("entrada", ["elian/ETF2", " elian/ETF2/ ", "https://github.com/elian/ETF2",
                                     "https://github.com/elian/ETF2.git", "github.com/elian/ETF2"])
def test_repo_en_cualquier_forma_habitual(entrada):
    assert act.normalizar_repo(entrada) == "elian/ETF2"


@pytest.mark.parametrize("entrada", ["", "ETF2", "elian/ETF2/extra", "/ETF2", None])
def test_repo_mal_escrito(entrada):
    with pytest.raises(ErrorGitHub, match="usuario/repositorio"):
        act.normalizar_repo(entrada)


@pytest.mark.parametrize("estado, resultado, esperado", [
    ("queued", None, ("plano", "● En cola")),
    ("in_progress", None, ("plano", "● En curso")),
    ("completed", "success", ("up", "✔ Correcta")),
    ("completed", "failure", ("down", "✖ Con error")),
    ("completed", "cancelled", ("plano", "Cancelada")),
    ("completed", "skipped", ("plano", "skipped")),
])
def test_estado_legible(estado, resultado, esperado):
    assert act.estado({"estado": estado, "resultado": resultado}) == esperado


@pytest.mark.parametrize("entorno, esperado", [
    ({}, "pc"),
    ({"GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "schedule"}, "programada"),
    ({"GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "workflow_dispatch", "ORIGEN_CORRIDA": "panel"}, "panel"),
    ({"GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "workflow_dispatch", "ORIGEN_CORRIDA": "github"}, "github_manual"),
])
def test_origen_de_cada_corrida(monkeypatch, entorno, esperado):
    from etf2.pipeline import origen_de_la_corrida
    for var in ("GITHUB_ACTIONS", "GITHUB_EVENT_NAME", "ORIGEN_CORRIDA"):
        monkeypatch.delenv(var, raising=False)
    for var, valor in entorno.items():
        monkeypatch.setenv(var, valor)
    assert origen_de_la_corrida() == esperado


def test_origen_en_palabras():
    from etf2 import panel
    assert [panel.origen_legible(o) for o in ("programada", "panel", "github_manual", "pc")] == \
        ["automático", "botón del panel", "manual en GitHub", "desde una PC"]
    assert panel.origen_legible("github_actions") == "automático" and panel.origen_legible("manual") == "desde una PC"
    assert panel.origen_legible(None) == "—"


def test_el_workflow_recibe_el_origen_del_boton():
    texto = (__import__("pathlib").Path(__file__).resolve().parents[1] / ".github/workflows/diario.yml").read_text(encoding="utf-8")
    assert "origen:" in texto and "ORIGEN_CORRIDA: ${{ inputs.origen }}" in texto

