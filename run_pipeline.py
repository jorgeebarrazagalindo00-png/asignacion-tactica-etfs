"""Punto de entrada: `python run_pipeline.py --modo historico|diario`."""

import argparse
import sys

import pandas as pd

from etf2 import config
from etf2.pipeline import ejecutar


def main() -> int:
    p = argparse.ArgumentParser(description="Pipeline de ETF2")
    p.add_argument("--modo", choices=["historico", "diario"], default="diario",
                   help="historico = carga completa (primera vez); diario = incremental")
    p.add_argument("--csv", help="leer precios de un CSV en vez de Yahoo Finance")
    p.add_argument("--sin-db", action="store_true",
                   help="no tocar la base de datos: solo calcular y mostrar resultados")
    p.add_argument("--sin-ml", action="store_true", help="omitir el modelo predictivo")
    p.add_argument("--sin-llm", action="store_true",
                   help="no llamar al modelo de lenguaje: las recomendaciones salen de la plantilla")
    a = p.parse_args()

    res = ejecutar(modo=a.modo, csv=a.csv, sin_db=a.sin_db, ml=not a.sin_ml, llm=not a.sin_llm)

    pd.options.display.float_format = "{:.4f}".format
    print(f"\nPesos sugeridos al {res['ultima_fecha'].date()}:")
    print(res["pesos_actuales"][["peso", "peso_base", "factor_momentum"]].to_string())
    print("\nBacktest (misma fricción y calendario para las tres):")
    cols = ["estrategia", "cagr", "volatilidad", "sharpe", "max_drawdown", "turnover_anual"]
    print(res["backtest"][cols].to_string(index=False))

    if res["modelo"]:
        m = res["modelo"]
        print("\nModelo predictivo, fuera de muestra (¿sube el ETF en el próximo mes?):")
        print(m["metricas"][["modelo", "n", "exactitud", "precision_pos", "sensibilidad", "f1", "auc"]]
              .to_string(index=False))
        print("\nLa regla con y sin el modelo, mismo periodo:")
        print(m["resumen"][cols].to_string(index=False))
    for r in res["recomendaciones"]:
        pesos = ", ".join(f"{e['ticker']} {e['peso']:.1%}" for e in r["detalle"]["etfs"])
        print(f"\nPerfil {r['perfil']} · texto por {r['fuente']}: {pesos}")
    for aviso in res["avisos"]:
        print(f"\nAVISO: {aviso}")
    if not a.sin_llm and not any(r["fuente"] == "llm" for r in res["recomendaciones"]):
        print(f"\n(Sin texto del LLM: revisa {config.LLM_VARIABLE_CLAVE} en tu .env o la bitácora llm_log.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
