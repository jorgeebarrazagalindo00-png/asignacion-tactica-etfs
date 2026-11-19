"""Prueba de punta a punta: el panel completo se ejecuta contra un Supabase falso."""

import re
import uuid

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from etf2 import actualizacion, config
from etf2.api_db import ClienteSupabase
from etf2.pipeline import ejecutar

from fake_postgrest import FakeSupabase
from llm_falso import LLMObediente
from sinteticos import a_csv, generar

RAIZ = __import__("pathlib").Path(__file__).resolve().parents[1]
PESTANAS = ["Resumen", "Descriptivo", "Predictivo", "Prescriptivo", "Asignación", "Backtest", "Datos y método"]
TABLAS_DE_REPORTES = {"modelo_predicciones", "modelo_metricas", "modelo_importancias", "pesos_ml",
                      "backtest_ml_resumen", "backtest_ml_curva", "recomendaciones", "llm_log"}


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    return a_csv(generar(1100), tmp_path_factory.mktemp("datos") / "precios.csv")


@pytest.fixture(scope="module")
def supabase_con_datos(csv):
    """Base cargada por el pipeline sin clave de LLM: las recomendaciones salen de la plantilla."""
    fake = FakeSupabase()
    ejecutar(modo="historico", csv=csv, cliente=ClienteSupabase(fake.url, "sb_secret_x", espera=0.01))
    yield fake
    fake.cerrar()


def _app(monkeypatch, url=None, clave=None, boton: dict | None = None) -> AppTest:
    monkeypatch.chdir(RAIZ)
    monkeypatch.setattr(config, "cargar_env", lambda *a, **k: None)      # que no lea un .env real del disco
    variables = {"SUPABASE_URL": url, "SUPABASE_PUBLISHABLE_KEY": clave, config.VARIABLE_TOKEN_GITHUB: None,
                 config.VARIABLE_REPO_GITHUB: None, config.VARIABLE_CLAVE_PANEL: None, **(boton or {})}
    for var, valor in variables.items():
        monkeypatch.delenv(var, raising=False)
        if valor:
            monkeypatch.setenv(var, valor)
    return AppTest.from_file(str(RAIZ / "dashboard.py"), default_timeout=90)


def _html(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def _elegir(at: AppTest, clave: str, valor) -> None:
    """Marca una opción de un selector y vuelve a ejecutar el panel."""
    at.button_group(key=clave).set_value(valor).run()


def test_panel_completo(monkeypatch, supabase_con_datos):
    antes = len(supabase_con_datos.peticiones)
    at = _app(monkeypatch, supabase_con_datos.url, "sb_publishable_x").run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == PESTANAS
    html = _html(at)
    assert config.NOMBRE_PANEL in html and "Reparto de la regla" in html and "Lectura del día" in html
    assert config.NOMBRE_PANEL == "Asignación Táctica de ETFs" and config.ROTULO_PANEL in html
    assert "· desde una PC" in html and ">Desde una PC<" in html         # la corrida de la prueba fue local
    # La pregunta del mes, en el resumen y en el prescriptivo, con su tamaño de posición y nivel de alerta
    assert html.count("¿Compro, mantengo o vendo cada ETF este mes?") == 2
    assert "Decisión para el perfil moderado" in html and html.count("Nivel de alerta (USD)") == 2
    assert html.count("Tamaño de la posición") >= 4                       # título y columna, en las dos calculadoras
    assert "Aumentar" not in html and "Reducir" not in html
    for tk in config.TICKERS:
        assert f">{tk}</span>" in html
    assert "Retorno anual" in html and "Bitácora de ejecuciones" in html and "Fases CRISP-DM" in html
    assert "El botón no está configurado" in html and not at.text_input    # sin sus secretos, el botón no aparece
    assert "Tres reportes, una decisión" in html and "Mayor probabilidad de subida" in html
    # Descriptivo
    assert "Indicadores técnicos" in html and "RSI (14 días)" in html and "Distribución de los retornos diarios" in html
    # Predictivo
    assert "¿Subirá cada ETF en el próximo mes?" in html and "Desempeño fuera de muestra" in html
    assert "Regresión logística" in html and "Referencia: frecuencia histórica" in html
    assert "Matriz de confusión" in html and "Cómo se evita que el modelo haga trampa" in html
    # Prescriptivo
    assert "Recomendación para el perfil moderado" in html and "Texto por plantilla (sin LLM)" in html
    assert "Proyecto académico. No es asesoría de inversión" in html
    plegables = [e.label for e in at.expander]
    assert "Uso del modelo de lenguaje: llamadas, tokens y costo" in plegables
    assert "Quién decide qué y qué se revisa antes de publicar" in plegables
    assert html.count("Este mes, perfil moderado") == 2                    # titular de la decisión, resumen y prescriptivo
    assert 'title="Capacidad de ordenar' in html                           # ayuda al pasar el mouse sobre «AUC»
    assert "Sin clave" in html
    assert not re.search(r"[>\s(]nan[<\s%)]", html, flags=re.I)             # ningún dato faltante mostrado como «nan»
    assert not at.error and not at.info                                   # ninguna pantalla de aviso
    # El panel solo lee: ni una escritura contra la base.
    assert {p["metodo"] for p in supabase_con_datos.peticiones[antes:]} == {"GET"}


def test_calculadora_y_filtros(monkeypatch, supabase_con_datos):
    at = _app(monkeypatch, supabase_con_datos.url, "sb_publishable_x").run()
    assert "5 compras y 0 ventas" in _html(at)
    _elegir(at, "partida", "Ya tengo una cartera")
    at.number_input(key="act_SPY").set_value(9000.0).run()
    assert not at.exception
    assert "Vender" in _html(at)                                          # con SPY sobrado hay que vender
    at.number_input(key="act_SPY").set_value(0.0)
    for tk in config.TICKERS[1:]:
        at.number_input(key=f"act_{tk}").set_value(0.0)
    at.run()
    assert at.info and "Ingresa algún monto" in at.info[0].value          # cartera vacía: aviso, no error

    for etf in config.TICKERS:
        _elegir(at, "m_etf", etf)
        for rango in ("3 M", "1 A", "Máx."):
            _elegir(at, "m_rango", rango)
            assert not at.exception, (etf, rango, at.exception)
    _elegir(at, "m_capas", [])                                            # sin medias ni volumen
    assert not at.exception
    for periodo in ("1 A", "Máx."):
        _elegir(at, "a_periodo", periodo)
        assert not at.exception


def test_perfiles_y_calculadora_del_prescriptivo(monkeypatch, supabase_con_datos):
    at = _app(monkeypatch, supabase_con_datos.url, "sb_publishable_x").run()
    recs = {r["perfil"]: r for r in supabase_con_datos.tablas["recomendaciones"]}
    for perfil in config.PERFILES:
        _elegir(at, "perfil", perfil)
        assert not at.exception, (perfil, at.exception)
        html = _html(at)
        assert f"Recomendación para el perfil {perfil.lower()}" in html
        assert f"Decisión para el perfil {perfil.lower()}" in html               # el resumen sigue al perfil elegido
        assert recs[perfil]["resumen"][:60] in html                         # el texto guardado, tal cual
    _elegir(at, "perfil", None)                                             # desmarcar vuelve al perfil por defecto
    assert f"Recomendación para el perfil {config.PERFIL_DEFECTO.lower()}" in _html(at)

    at.number_input(key="p_capital").set_value(50_000.0).run()
    assert not at.exception and "USD\u00a050.000" in _html(at)
    _elegir(at, "p_partida", "Ya tengo una cartera")
    at.number_input(key="p_act_TLT").set_value(30_000.0).run()
    assert not at.exception and "Vender" in _html(at)
    assert at.number_input(key="capital").value == 10_000.0                 # la calculadora del resumen no se mezcla


def test_recomendacion_redactada_por_el_llm(monkeypatch, csv):
    fake = FakeSupabase()
    try:
        ejecutar(modo="historico", csv=csv, cliente=ClienteSupabase(fake.url, "sb_secret_x", espera=0.01),
                 cliente_llm=LLMObediente())
        at = _app(monkeypatch, fake.url, "sb_publishable_x").run()
        html = _html(at)
        assert not at.exception and not at.info
        assert "Redactado por modelo-de-prueba · validado" in html and "Aceptada" in html
        assert "3 de 3" in html and f"3 de {config.LLM_LIMITE_DIA}" in html   # aceptadas y llamadas de hoy
    finally:
        fake.cerrar()


def test_base_sin_las_tablas_de_los_reportes(monkeypatch, csv):
    """Si aún no se ejecutó la migración, lo descriptivo funciona y el resto lo explica."""
    fake = FakeSupabase()
    fake.tablas_ausentes = set(TABLAS_DE_REPORTES)
    try:
        ejecutar(modo="historico", csv=csv, cliente=ClienteSupabase(fake.url, "sb_secret_x", espera=0.01))
        at = _app(monkeypatch, fake.url, "sb_publishable_x").run()
        assert not at.exception and not at.error
        assert [t.label for t in at.tabs] == PESTANAS
        avisos = " ".join(i.value for i in at.info)
        assert "modelo predictivo" in avisos and "migracion_ml_llm.sql" in avisos
        html = _html(at)
        assert "Reparto de la regla" in html and "Indicadores técnicos" in html
        assert "Aún no hay recomendaciones guardadas: corre el pipeline" in html
        assert "El modelo aún no tiene resultados." in html and "Aún no hay llamadas registradas." in html
    finally:
        fake.cerrar()


def test_boton_de_releer_vuelve_a_consultar_la_base(monkeypatch, supabase_con_datos):
    at = _app(monkeypatch, supabase_con_datos.url, "sb_publishable_x").run()
    antes = len(supabase_con_datos.peticiones)
    at.run()
    assert len(supabase_con_datos.peticiones) == antes                    # segunda pasada: sale de la caché
    at.button(key="releer").click().run()
    assert not at.exception and len(supabase_con_datos.peticiones) > antes


def test_sin_credenciales_explica_que_falta(monkeypatch):
    at = _app(monkeypatch).run()
    assert not at.exception and "Faltan las credenciales" in at.info[0].value


def test_rechaza_una_clave_secreta_sin_usarla(monkeypatch, supabase_con_datos):
    antes = len(supabase_con_datos.peticiones)
    at = _app(monkeypatch, supabase_con_datos.url, "sb_secret_no_deberia_ir_aqui").run()
    assert not at.exception and "clave SECRETA" in at.error[0].value
    assert len(supabase_con_datos.peticiones) == antes                    # ni una sola petición con esa clave


def test_base_vacia_y_base_inalcanzable(monkeypatch):
    vacio = FakeSupabase()
    try:
        at = _app(monkeypatch, vacio.url, "sb_publishable_x").run()
        assert not at.exception and "todavía no tiene datos" in at.info[0].value
    finally:
        vacio.cerrar()
    at = _app(monkeypatch, "http://127.0.0.1:9", "sb_publishable_otra").run()
    assert not at.exception and "No pude leer la base de datos" in at.error[0].value


# ---------------------------------------------------------------------------
# Botón «Actualizar ahora»
# ---------------------------------------------------------------------------
def _ejecucion(id_, estado="completed", resultado="success", evento="schedule", creada="2026-11-17T23:04:00Z"):
    return {"id": id_, "numero": id_, "estado": estado, "resultado": resultado, "evento": evento,
            "creada": creada, "actualizada": "2026-11-17T23:08:30Z",
            "url": f"https://github.com/elian/ETF2/actions/runs/{id_}"}


class GitHubFalso:
    """Ocupa el lugar de ClienteGitHub: el panel nunca habla con GitHub de verdad en las pruebas."""
    ejs: list[dict] = []
    lanzadas: list[str] = []
    error: str | None = None

    def __init__(self, token, repo, **kw):
        actualizacion.normalizar_repo(repo)

    def ejecuciones(self, n=5):
        if GitHubFalso.error:
            raise actualizacion.ErrorGitHub(GitHubFalso.error)
        return list(GitHubFalso.ejs)

    def lanzar(self, modo="diario"):
        GitHubFalso.lanzadas.append(modo)


@pytest.fixture()
def boton(monkeypatch):
    GitHubFalso.ejs, GitHubFalso.lanzadas, GitHubFalso.error = [_ejecucion(40)], [], None
    monkeypatch.setattr(actualizacion, "ClienteGitHub", GitHubFalso)
    # Un token distinto en cada prueba: la lista de ejecuciones guardada en caché no se cruza entre pruebas.
    return {config.VARIABLE_TOKEN_GITHUB: f"github_pat_{uuid.uuid4().hex}", config.VARIABLE_REPO_GITHUB: "elian/ETF2",
            config.VARIABLE_CLAVE_PANEL: "clave-del-profe"}


def _pulsar(at: AppTest, contrasena: str, modo: str | None = None) -> AppTest:
    at.text_input(key="gh_clave").input(contrasena)
    if modo:
        at.radio(key="gh_modo").set_value(modo)
    return at.button(key="gh_enviar").click().run()


def test_boton_lanza_el_workflow_con_la_contrasena(monkeypatch, supabase_con_datos, boton):
    antes = len(supabase_con_datos.peticiones)
    at = _app(monkeypatch, supabase_con_datos.url, "sb_publishable_x", boton).run()
    assert not at.exception
    html = _html(at)
    assert "Actualizar ahora" in html and "Ver en GitHub" in html and "#40" in html and "Programada" in html
    assert GitHubFalso.lanzadas == []

    _pulsar(at, "otra-cosa")
    assert not at.exception and GitHubFalso.lanzadas == []
    assert [e.value for e in at.error] == ["Contraseña incorrecta."]

    _pulsar(at, "clave-del-profe", "historico")
    assert not at.exception and GitHubFalso.lanzadas == ["historico"]
    assert "GitHub recibió la orden" in at.success[0].value
    assert "Esperando que GitHub inicie la ejecución" in _html(at)
    # El botón no toca la base: solo lecturas contra Supabase.
    assert {p["metodo"] for p in supabase_con_datos.peticiones[antes:]} <= {"GET"}


def test_boton_no_lanza_si_ya_hay_una_en_curso(monkeypatch, supabase_con_datos, boton):
    GitHubFalso.ejs = [_ejecucion(41, "in_progress", None, "workflow_dispatch"), _ejecucion(40)]
    at = _app(monkeypatch, supabase_con_datos.url, "sb_publishable_x", boton).run()
    assert not at.exception and "● En curso" in _html(at)
    _pulsar(at, "clave-del-profe")
    assert not at.exception and GitHubFalso.lanzadas == []
    assert "Ya hay una ejecución en curso" in at.warning[0].value


def test_al_terminar_relee_la_base_y_avisa(monkeypatch, supabase_con_datos, boton):
    GitHubFalso.ejs = [_ejecucion(41, "in_progress", None, "workflow_dispatch"), _ejecucion(40)]
    at = _app(monkeypatch, supabase_con_datos.url, "sb_publishable_x", boton).run()
    assert not at.exception and at.session_state["gh_activa"]
    antes = len(supabase_con_datos.peticiones)

    GitHubFalso.ejs = [_ejecucion(41, "completed", "success", "workflow_dispatch"), _ejecucion(40)]
    st.cache_data.clear()                                                 # lo que haría el paso del tiempo
    at.run()
    assert not at.exception
    assert "Actualización terminada" in at.success[0].value
    assert len(supabase_con_datos.peticiones) > antes                     # volvió a leer la base
    assert not at.session_state["gh_activa"]

    GitHubFalso.ejs = [_ejecucion(42, "completed", "failure", "workflow_dispatch"), _ejecucion(41)]
    at.session_state["gh_vistas"] = {42}
    st.cache_data.clear()
    at.run()
    assert not at.exception and "terminó con error" in at.error[0].value


def test_boton_muestra_el_error_de_github(monkeypatch, supabase_con_datos, boton):
    GitHubFalso.error = "GitHub rechazó el token: revisa que esté bien copiado y que no haya vencido."
    at = _app(monkeypatch, supabase_con_datos.url, "sb_publishable_x", boton).run()
    assert not at.exception and "rechazó el token" in at.warning[0].value
    _pulsar(at, "clave-del-profe")
    assert not at.exception and "rechazó el token" in at.error[0].value and GitHubFalso.lanzadas == []
