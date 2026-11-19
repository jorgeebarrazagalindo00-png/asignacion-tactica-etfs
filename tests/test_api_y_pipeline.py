"""Pruebas de la capa de API y del pipeline completo contra un Supabase falso."""

import numpy as np
import pytest

from etf2.api_db import ClienteSupabase, ErrorAPI, a_json_seguro
from etf2.config import TICKERS
from etf2.pipeline import ejecutar

from fake_postgrest import FakeSupabase


@pytest.fixture()
def fake():
    f = FakeSupabase()
    yield f
    f.cerrar()


@pytest.fixture()
def cli(fake):
    return ClienteSupabase(fake.url, "clave-de-prueba", espera=0.01)


def test_json_seguro_convierte_nan_y_fechas():
    import pandas as pd
    assert a_json_seguro(float("nan")) is None
    assert a_json_seguro(np.float64("inf")) is None
    assert a_json_seguro(pd.Timestamp("2024-03-05")) == "2024-03-05"
    assert a_json_seguro(np.int64(7)) == 7 and isinstance(a_json_seguro(np.int64(7)), int)


def test_upsert_manda_on_conflict_y_cabeceras(cli, fake):
    cli.insertar("etfs", [{"ticker": "SPY", "nombre": "x", "clase_activo": "y"}], conflicto="ticker")
    p = fake.peticiones[-1]
    assert p["metodo"] == "POST" and p["query"]["on_conflict"] == "ticker"
    assert "merge-duplicates" in p["headers"]["Prefer"]
    assert p["headers"]["apikey"] == "clave-de-prueba"


def test_clave_nueva_solo_va_en_apikey_y_clave_jwt_tambien_como_bearer(fake):
    """Supabase: las claves sb_secret_... van SOLO en `apikey`; las antiguas (JWT) también como Bearer."""
    nueva = ClienteSupabase(fake.url, "sb_secret_abc123")
    nueva.leer("etfs")
    h = fake.peticiones[-1]["headers"]
    assert h["apikey"] == "sb_secret_abc123" and "Authorization" not in h

    jwt = "eyJhbGciOi.eyJyb2xlIjoi.firma"
    antigua = ClienteSupabase(fake.url, jwt)
    antigua.leer("etfs")
    h = fake.peticiones[-1]["headers"]
    assert h["apikey"] == jwt and h["Authorization"] == f"Bearer {jwt}"


def test_upsert_repetido_no_duplica(cli, fake):
    fila = {"ticker": "SPY", "nombre": "x", "clase_activo": "y"}
    cli.insertar("etfs", [fila], conflicto="ticker")
    cli.insertar("etfs", [{**fila, "nombre": "nuevo"}], conflicto="ticker")
    assert len(fake.tablas["etfs"]) == 1 and fake.tablas["etfs"][0]["nombre"] == "nuevo"


def test_lectura_pagina_hasta_traer_todo(cli, fake):
    fake.max_filas = 100
    fake.tablas["precios_diarios"] = [{"ticker": "SPY", "fecha": f"2024-01-{i:02d}"} for i in range(1, 29)] * 10
    filas = cli.leer("precios_diarios", {"select": "*"}, pagina=100)
    assert len(filas) == 280


def test_error_4xx_levanta_errorapi_con_el_mensaje(cli, fake):
    with pytest.raises(ErrorAPI, match="400.*columna inexistente"):
        cli.leer("tabla_mala")


def test_reintenta_ante_503_y_luego_funciona(fake):
    class SesionFalla:
        """Falla dos veces con 503 y luego responde bien."""
        def __init__(self):
            self.headers, self.n = {}, 0

        def request(self, metodo, url, **kw):
            self.n += 1

            class R:
                pass
            r = R()
            r.status_code = 503 if self.n < 3 else 200
            r.ok = r.status_code == 200
            r.text = "caído"
            r.json = lambda: []
            return r

    s = SesionFalla()
    c = ClienteSupabase("http://x", "k", sesion=s, espera=0.0)
    assert c.leer("etfs") == [] and s.n == 3


# --- Pipeline completo ------------------------------------------------------------
def test_pipeline_historico_y_luego_diario(cli, fake, csv_sintetico):
    r1 = ejecutar(modo="historico", csv=csv_sintetico, cliente=cli)

    # Quedó todo guardado en la "base de datos"
    for tabla in ["etfs", "precios_diarios", "indicadores", "pesos_sugeridos",
                  "backtest_resumen", "backtest_curva", "corridas"]:
        assert fake.tablas.get(tabla), f"la tabla {tabla} quedó vacía"
    assert {f["ticker"] for f in fake.tablas["etfs"]} == set(TICKERS)
    assert len(fake.tablas["precios_diarios"]) == 1600 * 5
    assert len(fake.tablas["backtest_resumen"]) == 3

    # Los pesos de la última fecha suman 1
    ult = max(f["fecha"] for f in fake.tablas["pesos_sugeridos"])
    pesos = [f["peso"] for f in fake.tablas["pesos_sugeridos"] if f["fecha"] == ult]
    assert len(pesos) == 5 and sum(pesos) == pytest.approx(1.0, abs=1e-6)

    # La corrida quedó registrada como OK
    corrida = fake.tablas["corridas"][0]
    assert corrida["estado"] == "ok" and corrida["modo"] == "historico" and corrida["filas_precios"] == 8000

    # Segunda corrida (idempotente): no duplica nada
    n_precios = len(fake.tablas["precios_diarios"])
    n_pesos = len(fake.tablas["pesos_sugeridos"])
    ejecutar(modo="historico", csv=csv_sintetico, cliente=cli)
    assert len(fake.tablas["precios_diarios"]) == n_precios
    assert len(fake.tablas["pesos_sugeridos"]) == n_pesos
    assert len(fake.tablas["corridas"]) == 2


def test_pipeline_diario_solo_reescribe_la_ventana_reciente(cli, fake, csv_sintetico):
    ejecutar(modo="historico", csv=csv_sintetico, cliente=cli)
    antes = len(fake.peticiones)
    ejecutar(modo="diario", csv=csv_sintetico, cliente=cli)
    nuevas = fake.peticiones[antes:]
    enviadas = sum(len(p["cuerpo"]) for p in nuevas
                   if p["metodo"] == "POST" and p["tabla"] == "pesos_sugeridos")
    assert 0 < enviadas <= 5 * 12          # ~10 días de calendario, no todo el histórico


def test_pipeline_marca_error_en_la_bitacora(cli, fake, tmp_path):
    malo = tmp_path / "malo.csv"
    malo.write_text("a,b\n1,2\n")
    with pytest.raises(Exception):
        ejecutar(modo="historico", csv=str(malo), cliente=cli)
    corrida = fake.tablas["corridas"][0]
    assert corrida["estado"] == "error" and "columnas" in corrida["mensaje"]


def test_modo_sin_db_no_necesita_credenciales(csv_sintetico, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)
    res = ejecutar(modo="historico", csv=csv_sintetico, sin_db=True)
    assert res["pesos_actuales"]["peso"].sum() == pytest.approx(1.0)
    assert len(res["backtest"]) == 3


# --- lectura en paralelo -------------------------------------------------------------
def _tabla_grande(fake, n=4321):
    fake.tablas["precios_diarios"] = [{"ticker": "SPY", "fecha": f"{i:06d}", "cierre": float(i)} for i in range(n)]
    return n


def test_lectura_en_paralelo_trae_lo_mismo_que_en_serie(cli, fake):
    n = _tabla_grande(fake)
    orden = {"select": "*", "order": "fecha.asc"}
    serie = cli.leer("precios_diarios", orden)
    antes = len(fake.peticiones)
    paralelo = cli.leer("precios_diarios", orden, hilos=6)
    assert paralelo == serie and len(paralelo) == n
    pedidos = fake.peticiones[antes:]
    assert len(pedidos) == 5                                             # 4321 filas en páginas de 1000
    assert "count=exact" in pedidos[0]["headers"]["Prefer"]
    assert sorted(int(p["query"]["offset"]) for p in pedidos) == [0, 1000, 2000, 3000, 4000]
    assert all(p["headers"]["apikey"] == "clave-de-prueba" for p in pedidos)   # los hilos llevan la clave


def test_lectura_en_paralelo_se_adapta_al_tope_del_servidor(cli, fake):
    n = _tabla_grande(fake, 1234)
    fake.max_filas = 300                                                 # el servidor devuelve menos de lo pedido
    assert len(cli.leer("precios_diarios", {"order": "fecha.asc"}, hilos=4)) == n


def test_lectura_en_paralelo_sin_conteo_cae_al_modo_normal(cli, fake):
    n = _tabla_grande(fake, 2500)
    fake.informa_total = False
    filas = cli.leer("precios_diarios", {"order": "fecha.asc"}, hilos=4)
    assert [f["fecha"] for f in filas] == [f"{i:06d}" for i in range(n)]


def test_lectura_en_paralelo_de_tablas_chicas_o_vacias(cli, fake):
    assert cli.leer("precios_diarios", {"order": "fecha.asc"}, hilos=4) == []
    _tabla_grande(fake, 7)
    antes = len(fake.peticiones)
    assert len(cli.leer("precios_diarios", {"order": "fecha.asc"}, hilos=4)) == 7
    assert len(fake.peticiones) - antes == 1                             # una sola petición basta
