"""Configuración central de ETF2.

Todos los parámetros del método viven aquí y se fijan ANTES de mirar resultados
(no se optimizan sobre el periodo de prueba). Si algún día se cambia uno, debe
quedar anotado en el README, porque cambia lo que significa el backtest.
"""

import os
from pathlib import Path

# --- Universo de ETFs -------------------------------------------------------
# ticker: (nombre, clase de activo, descripción corta)
ETFS = {
    "SPY": ("SPDR S&P 500 ETF Trust", "Acciones EE. UU. · gran capitalización",
            "Replica el índice S&P 500."),
    "QQQ": ("Invesco QQQ Trust", "Acciones EE. UU. · tecnología",
            "Replica el Nasdaq-100."),
    "IWM": ("iShares Russell 2000 ETF", "Acciones EE. UU. · pequeña capitalización",
            "Replica el índice Russell 2000."),
    "GLD": ("SPDR Gold Shares", "Materias primas · oro",
            "Refleja el precio del oro físico."),
    "TLT": ("iShares 20+ Year Treasury Bond ETF", "Renta fija · bonos del Tesoro a 20+ años",
            "Bonos del Tesoro de EE. UU. a más de 20 años."),
}
TICKERS = list(ETFS)
GRUPO = {"SPY": "Renta variable", "QQQ": "Renta variable", "IWM": "Renta variable",
         "GLD": "Oro", "TLT": "Bonos"}

# --- Presentación del panel ---------------------------------------------------
NOMBRE_PANEL = "Asignación Táctica de ETFs"
ROTULO_PANEL = "5 ETFs · rebalanceo mensual"
LEMA_PANEL = ("Renta variable, oro y bonos del Tesoro de EE. UU., con una decisión por mes según tu perfil "
              "de riesgo.")

# --- Datos ------------------------------------------------------------------
HISTORICO_INICIO = "2015-01-01"   # desde cuándo se carga el histórico
DIAS_REFRESCO = 10                # en modo diario se vuelven a bajar los últimos N días
                                  # (por si Yahoo corrigió algún cierre reciente)

# --- Indicadores (en días hábiles de bolsa) -----------------------------------
RETORNOS = {"ret_1m": 21, "ret_3m": 63, "ret_6m": 126, "ret_12m": 252}
VENTANA_VOL = 60                  # ventana de la volatilidad
DIAS_ANUALES = 252

# --- Regla de pesos -----------------------------------------------------------
INCLINACION_MOMENTUM = 0.5        # fuerza con que el momentum mueve el peso (0 = ignorarlo)
FACTOR_FILTRO_6M = 0.5            # si el retorno a 6 meses es <= 0, el peso se reduce a la mitad
PESO_MIN = 0.05                   # ningún ETF baja de 5 %
PESO_MAX = 0.40                   # ningún ETF supera 40 %

# --- Backtest -----------------------------------------------------------------
FRICCION = 0.001                  # 0,1 % de costo por cada unidad de capital negociada
NOMBRE_ESTRATEGIA = "Asignación táctica"     # la regla de este proyecto (rebalanceo mensual)
NOMBRE_IGUALES = "Pesos iguales"              # 20 % en cada ETF (rebalanceo mensual)
NOMBRE_SPY = "Solo SPY"                       # comprar y mantener el S&P 500
NOMBRE_ESTRATEGIA_ML = "Asignación + ML"      # la misma regla con la inclinación del modelo

# --- Modelo predictivo --------------------------------------------------------
# Pregunta que responde: ¿este ETF terminará los próximos HORIZONTE_ML días hábiles en positivo?
HORIZONTE_ML = 21                 # ≈ un mes de bolsa
MIN_ENTRENAMIENTO = 504           # días de historia (≈ 2 años) antes de la primera predicción
INCLINACION_ML = 0.3              # cuánto mueve la probabilidad al peso (0 = ignorar el modelo)
MODELO_PRINCIPAL = "logit"        # el que inclina los pesos; se fijó antes de ver resultados
NOMBRES_MODELOS = {
    "logit": "Regresión logística", "gb": "Gradient boosting",
    "momentum": "Referencia: retorno 6 m positivo", "frecuencia": "Referencia: frecuencia histórica",
}
ETIQUETAS_VARIABLES = {
    "ret_1m": "Retorno 1 mes", "ret_3m": "Retorno 3 meses", "ret_6m": "Retorno 6 meses",
    "ret_12m": "Retorno 12 meses", "vol_60d": "Volatilidad 60 d", "drawdown": "Distancia al máximo 12 m",
    "rsi_14": "RSI (14 d)", "macd_rel": "MACD (histograma / precio)", "dist_ma50": "Distancia a la media 50 d",
    "vol_corta_larga": "Volatilidad 20 d / 60 d",
}
ARBOLES_GB = 80                   # árboles del gradient boosting (poco profundos: 2 niveles)
CORTES_CONFIANZA = (0.52, 0.58)   # AUC fuera de muestra: menos de 0,52 baja; menos de 0,58 media; si no, alta
SEMILLA = 7

# --- Perfiles de riesgo (prescriptivo) ----------------------------------------
# Rango permitido de renta variable (SPY + QQQ + IWM) dentro de la cartera. Los rangos se tocan
# solo en el borde, para que los tres perfiles den carteras distintas con cualquier punto de partida.
PERFILES = {"Conservador": (0.20, 0.45), "Moderado": (0.45, 0.65), "Agresivo": (0.65, 0.85)}
PERFIL_DEFECTO = "Moderado"
BANDA_ACCION = 0.02               # un cambio de peso menor a 2 puntos se considera «Mantener»
# Nivel de alerta: precio tras una caída de ALERTA_DESVIOS veces la volatilidad de un mes.
# Es un aviso para revisar la posición, no una venta automática (el backtest no lo simula).
ALERTA_DESVIOS = 2.0

# --- Modelo de lenguaje (LLM) -------------------------------------------------
LLM_PROVEEDOR = "Google Gemini"
LLM_VARIABLE_CLAVE = "GEMINI_API_KEY"      # variable de entorno / secreto con la clave
LLM_VARIABLE_MODELO = "GEMINI_MODEL"       # opcional: cambiar de modelo sin tocar el código
LLM_MODELOS = ["gemini-3.5-flash-lite", "gemini-2.5-flash-lite"]   # si el primero no existe, prueba el siguiente
LLM_LIMITE_DIA = 12               # llamadas máximas por día (todas las corridas juntas)
LLM_LIMITE_CORRIDA = 6            # llamadas máximas en una sola corrida (incluye reintentos)
LLM_MAX_TOKENS_SALIDA = 2000
LLM_TIEMPO_ESPERA = 45            # segundos
LLM_TEMPERATURA = 0.2
# Tarifa de pago por millón de tokens (entrada, salida) en USD, leída de la página de precios de
# Google el 14/11/2026. Con la capa gratuita el costo es cero; sirve para estimar el peor caso.
LLM_PRECIOS_USD_M = {"gemini-3.5-flash-lite": (0.30, 2.50), "gemini-2.5-flash-lite": (0.10, 0.40)}

# --- Botón «Actualizar ahora» del panel -----------------------------------------
# El panel no corre el proceso ni escribe en la base: le pide a GitHub Actions que ejecute
# el mismo workflow programado. Sin estas tres variables el botón no aparece.
VARIABLE_TOKEN_GITHUB = "PANEL_GITHUB_TOKEN"   # token con permiso «Actions: Read and write» sobre el repo
VARIABLE_REPO_GITHUB = "PANEL_GITHUB_REPO"     # usuario/repositorio
VARIABLE_CLAVE_PANEL = "PANEL_CLAVE"           # contraseña que se pide antes de lanzar
GITHUB_FLUJO = "diario.yml"
GITHUB_RAMA = "main"
SEGUNDOS_SEGUIMIENTO = 15                      # cada cuánto se consulta a GitHub mientras corre


def cargar_env(ruta: str | Path = ".env") -> None:
    """Lee un archivo .env sencillo (CLAVE=valor) para correr el pipeline en tu PC.

    En GitHub Actions no se usa: allí las claves llegan como secretos del repositorio.
    Nunca pisa variables que ya estén definidas en el entorno.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        return
    # utf-8-sig: algunos editores de Windows guardan una marca invisible al inicio del archivo.
    for linea in ruta.read_text(encoding="utf-8-sig").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, valor = linea.split("=", 1)
        os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))
