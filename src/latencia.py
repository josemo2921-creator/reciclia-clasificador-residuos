"""Mide de nuevo la latencia de inferencia (sin otros procesos en ejecución) y actualiza resultados.json.

Uso:
    python src/latencia.py --runs runs --datos data/residuos5
"""
import argparse
import json
from pathlib import Path

from common import CarpetaResiduos, cargar_entrenado
from evaluate import latencia, latencia_preproceso

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs"); ap.add_argument("--datos", default="data/residuos5")
a = ap.parse_args()
rutas = [p for p, _ in CarpetaResiduos(a.datos, "test", None).items]
for m in ["mobilenetv3", "efficientnet_b0", "resnet50"]:
    f = Path(a.runs) / m / "resultados.json"
    r = json.load(open(f, encoding="utf-8"))
    modelo, _ = cargar_entrenado(Path(a.runs) / m / "mejor.pt")
    r["latencia"] = [latencia(modelo, 1, n=200), latencia(modelo, 2, n=200)]
    r["latencia_preproceso_ms"] = latencia_preproceso(rutas, n=200)
    json.dump(r, open(f, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(m, r["latencia"], round(r["latencia_preproceso_ms"], 1))
