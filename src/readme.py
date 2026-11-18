"""Genera README.md a partir de README.template.md y los resultados."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
ETQ = {"mobilenetv3": "MobileNetV3-Large", "efficientnet_b0": "EfficientNet-B0", "resnet50": "ResNet-50"}


def c(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


filas = []
for m, n in ETQ.items():
    r = json.load(open(R / "runs" / m / "resultados.json", encoding="utf-8"))
    filas.append(f"| {n} | {c(r['parametros_M'], 2)} M | {c(r['GMACs'], 2)} | {c(r['exactitud'] * 100)} % | "
                 f"{c(r['f1_macro'] * 100)} % | {c(r['ece_calibrado'] * 100)} % | {c(r['latencia'][0]['media_ms'])} ms |")
mn = json.load(open(R / "runs" / "mobilenetv3" / "resultados.json", encoding="utf-8"))
ds = json.load(open(R / "figuras" / "resumen_dataset.json", encoding="utf-8"))["dataset"]["particiones"]
n = {s: sum(v.values()) for s, v in ds.items()}
t = (R / "README.template.md").read_text(encoding="utf-8")
t = t.format(drive="_(pegar aquí el enlace de Google Drive)_", video="_(pegar aquí el enlace del video)_",
             nTest=n["test"], tabla="\n".join(filas), cob=c(mn["selectivo"]["cobertura"] * 100),
             sel=c(mn["selectivo"]["exactitud_aceptadas"] * 100), nTotal=sum(n.values()), nTrain=n["train"], nVal=n["val"])
(R / "README.md").write_text(t, encoding="utf-8")
print("README.md generado")
