import pandas as pd
import pytest

from etf2 import config
from sinteticos import a_csv, generar


@pytest.fixture(autouse=True, scope="session")
def entorno_de_pruebas():
    """Aísla las pruebas de la máquina donde corren.

    No se lee ningún .env real ni se usa una clave de Gemini del entorno (las pruebas nunca
    llaman a servicios externos), y el gradient boosting usa menos árboles: aquí se revisa
    que el modelo no haga trampa, no que acierte.
    """
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "cargar_env", lambda *a, **k: None)
    mp.setattr(config, "ARBOLES_GB", 12)
    mp.delenv(config.LLM_VARIABLE_CLAVE, raising=False)
    mp.delenv(config.LLM_VARIABLE_MODELO, raising=False)
    for var in ("GITHUB_ACTIONS", "GITHUB_EVENT_NAME", "ORIGEN_CORRIDA"):   # las corridas de prueba son «desde una PC»
        mp.delenv(var, raising=False)
    yield
    mp.undo()


@pytest.fixture(scope="session")
def precios_sinteticos() -> pd.DataFrame:
    """Cinco series de precios inventadas (semilla fija), 1600 ruedas."""
    return generar(1600)


@pytest.fixture()
def csv_sintetico(tmp_path, precios_sinteticos) -> str:
    return a_csv(precios_sinteticos, tmp_path / "precios.csv")
