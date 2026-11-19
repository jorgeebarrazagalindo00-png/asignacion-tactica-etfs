"""El pipeline completo con los reportes predictivo y prescriptivo, contra el Supabase falso."""

import pandas as pd
import pytest

from etf2 import config, prescriptivo as ps
from etf2.api_db import ClienteSupabase
from etf2.pipeline import FALTAN_TABLAS, ejecutar

from fake_postgrest import FakeSupabase
from llm_falso import LLMObediente
from sinteticos import a_csv, generar

TABLAS_NUEVAS = ["modelo_predicciones", "modelo_metricas", "modelo_importancias", "pesos_ml",
                 "backtest_ml_resumen", "backtest_ml_curva", "recomendaciones", "llm_log"]
PERFILES = list(config.PERFILES)


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    """1000 ruedas: alcanza para unos pocos meses de predicciones y corre rápido."""
    return a_csv(generar(1000), tmp_path_factory.mktemp("datos") / "precios.csv")


@pytest.fixture()
def fake():
    f = FakeSupabase()
    yield f
    f.cerrar()


@pytest.fixture()
def cli(fake):
    return ClienteSupabase(fake.url, "sb_secret_x", espera=0.01)


def estados(fake):
    return [x["estado"] for x in fake.tablas.get("llm_log", [])]


def test_sin_clave_del_llm_todo_funciona_con_la_plantilla(cli, fake, csv):
    res = ejecutar(modo="historico", csv=csv, cliente=cli)
    for tabla in TABLAS_NUEVAS:
        assert fake.tablas.get(tabla), f"la tabla {tabla} quedó vacía"

    pred = pd.DataFrame(fake.tablas["modelo_predicciones"])
    assert set(pred["modelo"]) == {"logit", "gb", "momentum", "frecuencia"}
    crudos = [x["subio"] for x in fake.tablas["modelo_predicciones"]]
    assert all(v is None or (type(v) is int and v in (0, 1)) for v in crudos)                 # entero o vacío
    assert None in crudos and 0 in crudos and 1 in crudos
    assert pred[pred["fecha"] == pred["fecha"].max()]["subio"].isna().all()
    assert len(fake.tablas["modelo_metricas"]) == 4 and len(fake.tablas["backtest_ml_resumen"]) == 4
    assert {x["estrategia"] for x in fake.tablas["backtest_ml_resumen"]} >= {config.NOMBRE_ESTRATEGIA_ML}
    assert set(fake.tablas["backtest_ml_curva"][0]) == {"fecha", "estrategia", "estrategia_ml", "pesos_iguales", "spy"}
    ml = pd.DataFrame(fake.tablas["pesos_ml"])
    assert ml.groupby("fecha")["peso"].sum().round(6).eq(1).all()

    recs = fake.tablas["recomendaciones"]
    assert [r["perfil"] for r in recs] == PERFILES and {r["fuente"] for r in recs} == {"reglas"}
    ultima = max(x["fecha"] for x in fake.tablas["precios_diarios"])
    for r in recs:
        d = r["detalle"]
        assert r["fecha"] == ultima and r["modelo_llm"] is None and len(r["entrada_hash"]) == 64
        assert sum(e["peso"] for e in d["etfs"]) == pytest.approx(1.0)
        minimo, maximo = config.PERFILES[r["perfil"]]
        assert minimo - 1e-9 <= d["renta_variable"] <= maximo + 1e-9
        assert all(e["accion"] in ps.ACCIONES and e["explicacion"] for e in d["etfs"])
        assert d["confianza"] in ("baja", "media", "alta") and d["aviso"] == ps.AVISO_LEGAL
    assert estados(fake) == ["sin_clave"]
    assert res["modelo"] is not None and len(res["recomendaciones"]) == 3 and res["avisos"] == []
    assert fake.tablas["corridas"][0]["estado"] == "ok"

    # Repetir la corrida no duplica nada.
    tamanos = {t: len(fake.tablas[t]) for t in TABLAS_NUEVAS if t != "llm_log"}
    ejecutar(modo="historico", csv=csv, cliente=cli)
    assert {t: len(fake.tablas[t]) for t in tamanos} == tamanos


def test_con_llm_se_guarda_su_texto_y_cada_llamada(cli, fake, csv):
    llm = LLMObediente()
    res = ejecutar(modo="historico", csv=csv, cliente=cli, cliente_llm=llm)
    recs = fake.tablas["recomendaciones"]
    assert {r["fuente"] for r in recs} == {"llm"} and {r["modelo_llm"] for r in recs} == {"modelo-de-prueba"}
    assert len(llm.pedidos) == 3
    log = fake.tablas["llm_log"]
    assert [x["estado"] for x in log] == ["ok"] * 3 and [x["perfil"] for x in log] == PERFILES
    hoy = pd.Timestamp.now(tz="UTC").date().isoformat()
    assert all(x["dia"] == hoy and x["corrida_id"] == res["corrida_id"] and x["tokens_entrada"] == 950
               and x["tokens_salida"] == 320 and x["latencia_ms"] >= 0 for x in log)

    # Segunda corrida con los mismos datos: la redacción ya existe, no se vuelve a llamar.
    creadas = [r["creado_en"] for r in recs]
    otra = ejecutar(modo="diario", csv=csv, cliente=cli, cliente_llm=llm)
    assert len(llm.pedidos) == 3 and len(fake.tablas["llm_log"]) == 3
    assert [r["creado_en"] for r in fake.tablas["recomendaciones"]] == creadas
    assert [r["fuente"] for r in otra["recomendaciones"]] == ["llm"] * 3

    # Una corrida posterior sin clave (o con --sin-llm) no pisa el texto ya validado.
    ejecutar(modo="diario", csv=csv, cliente=cli)
    ejecutar(modo="diario", csv=csv, cliente=cli, llm=False)
    assert [r["fuente"] for r in fake.tablas["recomendaciones"]] == ["llm"] * 3
    assert [r["creado_en"] for r in fake.tablas["recomendaciones"]] == creadas


def test_limite_diario_de_llamadas(cli, fake, csv):
    hoy = pd.Timestamp.now(tz="UTC").date().isoformat()
    fake.tablas["llm_log"] = [{"estado": "ok", "dia": hoy}] * (config.LLM_LIMITE_DIA - 1) + \
                             [{"estado": "sin_clave", "dia": hoy}, {"estado": "ok", "dia": "2020-01-01"}]
    llm = LLMObediente()
    ejecutar(modo="historico", csv=csv, cliente=cli, cliente_llm=llm)
    assert len(llm.pedidos) == 1                                   # solo quedaba una llamada para hoy
    assert [r["fuente"] for r in fake.tablas["recomendaciones"]] == ["llm", "reglas", "reglas"]
    nuevos = fake.tablas["llm_log"][config.LLM_LIMITE_DIA + 1:]
    assert [x["estado"] for x in nuevos] == ["ok", "limite", "limite"]
    assert "Límite de llamadas" in nuevos[1]["mensaje"]


def test_limite_de_llamadas_por_corrida(cli, fake, csv, monkeypatch):
    monkeypatch.setattr(config, "LLM_LIMITE_CORRIDA", 2)
    llm = LLMObediente(estropear=lambda r: r.update(resumen=r["resumen"] + " Subirá 87,3 % sin riesgo."))
    ejecutar(modo="historico", csv=csv, cliente=cli, cliente_llm=llm)
    assert len(llm.pedidos) == 2                                   # intento + reintento del primer perfil
    assert {r["fuente"] for r in fake.tablas["recomendaciones"]} == {"reglas"}
    assert estados(fake) == ["rechazada", "rechazada", "limite", "limite"]
    rechazo = fake.tablas["llm_log"][0]["mensaje"]
    assert "87,3" in rechazo and "sin riesgo" in rechazo


def test_sin_las_tablas_nuevas_lo_anterior_se_guarda_y_se_avisa(cli, fake, csv):
    fake.tablas_ausentes = set(TABLAS_NUEVAS)
    res = ejecutar(modo="historico", csv=csv, cliente=cli, cliente_llm=LLMObediente())
    assert res["avisos"] == [FALTAN_TABLAS] and res["recomendaciones"] == []
    for tabla in ("precios_diarios", "indicadores", "pesos_sugeridos", "backtest_resumen", "backtest_curva"):
        assert fake.tablas.get(tabla)
    assert not any(fake.tablas.get(t) for t in TABLAS_NUEVAS)
    corrida = fake.tablas["corridas"][0]
    assert corrida["estado"] == "ok" and "migracion_ml_llm.sql" in corrida["mensaje"]


def test_un_error_que_no_es_de_tablas_ausentes_si_detiene_la_corrida(cli, fake, csv, monkeypatch):
    from etf2 import pipeline
    from etf2.api_db import ErrorAPI

    def falla(*a, **k):
        raise ErrorAPI("POST modelo_predicciones -> 500: caído", codigo=500)
    monkeypatch.setattr(pipeline, "_modelo_predictivo", falla)
    with pytest.raises(ErrorAPI):
        ejecutar(modo="historico", csv=csv, cliente=cli)
    assert fake.tablas["corridas"][0]["estado"] == "error"


def test_sin_modelo_y_sin_llm(cli, fake, csv):
    llm = LLMObediente()
    res = ejecutar(modo="historico", csv=csv, cliente=cli, ml=False, llm=False, cliente_llm=llm)
    assert res["modelo"] is None and llm.pedidos == []
    assert not fake.tablas.get("modelo_predicciones") and not fake.tablas.get("llm_log")
    recs = fake.tablas["recomendaciones"]
    assert len(recs) == 3 and {r["fuente"] for r in recs} == {"reglas"}
    d = recs[0]["detalle"]
    assert d["usa_modelo"] is False and d["confianza"] is None and d["etfs"][0]["prob_sube"] is None
    assert "modelo" not in recs[0]["resumen"].lower()


def test_con_poca_historia_no_hay_modelo_pero_si_recomendacion(cli, fake, tmp_path):
    corto = a_csv(generar(600), tmp_path / "corto.csv")
    res = ejecutar(modo="historico", csv=corto, cliente=cli)
    assert res["modelo"] is None and not fake.tablas.get("modelo_predicciones")
    assert len(fake.tablas["recomendaciones"]) == 3 and fake.tablas["corridas"][0]["estado"] == "ok"


def test_sin_base_de_datos_tambien_puede_redactar_el_llm(csv):
    llm = LLMObediente()
    res = ejecutar(modo="historico", csv=csv, sin_db=True, cliente_llm=llm)
    assert [r["fuente"] for r in res["recomendaciones"]] == ["llm"] * 3 and len(llm.pedidos) == 3
    assert res["modelo"]["metricas"].set_index("modelo").at["logit", "n"] > 0


def test_el_modelo_de_gemini_se_puede_elegir_por_variable_de_entorno(monkeypatch):
    from etf2 import pipeline
    assert pipeline._cliente_llm() is None                          # sin clave no hay cliente
    monkeypatch.setenv(config.LLM_VARIABLE_CLAVE, "clave-de-prueba")
    assert pipeline._cliente_llm().modelos == config.LLM_MODELOS
    monkeypatch.setenv(config.LLM_VARIABLE_MODELO, "otro-modelo")
    assert pipeline._cliente_llm().modelos == ["otro-modelo", *config.LLM_MODELOS]
