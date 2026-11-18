"""Exporta los resultados a macros LaTeX (paper/resultados.tex) para que las cifras del
artículo salgan directamente de los experimentos.

Uso:
    python src/macros_latex.py --runs runs --figuras figuras --salida ../paper/resultados.tex
"""
import argparse
import json
from pathlib import Path

from common import CLASES

PREF = {"mobilenetv3": "MN", "efficientnet_b0": "EF", "resnet50": "RN"}
CL = {"plastico": "Pla", "papel_carton": "Pap", "vidrio": "Vid", "metal": "Met", "organico": "Org"}
RB = {"original": "Orig", "baja_iluminacion": "Oscuro", "sobreexposicion": "Claro", "bajo_contraste": "Contraste",
      "desenfoque": "Blur", "ruido": "Ruido", "jpeg_q15": "Jpeg", "rotacion_30": "Rot"}


def num(x, d=1):
    """Número con coma decimal (convención en español)."""
    return f"{x:.{d}f}".replace(".", "{,}")


def pc(x, d=1):
    return num(x * 100, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs"); ap.add_argument("--figuras", default="figuras")
    ap.add_argument("--salida", default="../paper/resultados.tex")
    a = ap.parse_args()
    L = ["% Archivo generado automáticamente por src/macros_latex.py — no editar a mano"]

    vistos = set()

    def d(nombre, valor):
        if nombre in vistos:
            return
        vistos.add(nombre)
        L.append(f"\\newcommand{{\\{nombre}}}{{{valor}}}")

    for m, p in PREF.items():
        r = json.load(open(Path(a.runs) / m / "resultados.json", encoding="utf-8"))
        h = json.load(open(Path(a.runs) / m / "historial.json", encoding="utf-8"))
        s = r["selectivo"]
        d(f"acc{p}", pc(r["exactitud"])); d(f"f{p}", pc(r["f1_macro"]))
        d(f"eceRaw{p}", pc(r["ece_sin_calibrar"])); d(f"eceCal{p}", pc(r["ece_calibrado"]))
        d(f"nllRaw{p}", num(r["nll_sin_calibrar"], 3)); d(f"nllCal{p}", num(r["nll_calibrado"], 3))
        d(f"T{p}", num(r["temperatura"], 2)); d(f"umbral{p}", pc(s["umbral"]))
        d(f"cob{p}", pc(s["cobertura"])); d(f"rech{p}", pc(s["tasa_rechazo"]))
        d(f"accSel{p}", pc(s["exactitud_aceptadas"])); d(f"errAuto{p}", s["errores_automaticos"])
        d(f"errRev{p}", s["errores_enviados_a_revision"]); d(f"okRev{p}", s["aciertos_enviados_a_revision"])
        d(f"errTot{p}", s["errores_automaticos"] + s["errores_enviados_a_revision"])
        d(f"latUno{p}", num(r["latencia"][0]["media_ms"])); d(f"latUnoPn{p}", num(r["latencia"][0]["p95_ms"]))
        d(f"latDos{p}", num(r["latencia"][1]["media_ms"])); d(f"latDosPn{p}", num(r["latencia"][1]["p95_ms"]))
        d(f"latPre{p}", num(r["latencia_preproceso_ms"]))
        d(f"params{p}", num(r["parametros_M"], 2)); d(f"gmacs{p}", num(r["GMACs"], 2))
        d(f"tam{p}", num(r["tamano_MB"])); d(f"epoca{p}", r["epoca_mejor"])
        d(f"epocas{p}", len(h["historial"])); d(f"min{p}", f"{h['minutos']:.0f}")
        d(f"fVal{p}", pc(r["f1_val"]))
        for c in CLASES:
            q = r["por_clase"][c]
            d(f"P{CL[c]}{p}", pc(q["precision"])); d(f"R{CL[c]}{p}", pc(q["recall"])); d(f"F{CL[c]}{p}", pc(q["f1"]))
            d(f"N{CL[c]}", q["n"])
        for f, q in r["por_fuente"].items():
            d(f"accFuente{'Tn' if f == 'trashnet' else 'Gc'}{p}", pc(q["exactitud"]))
            d(f"nFuente{'Tn' if f == 'trashnet' else 'Gc'}", q["n"])
        for k, v in r.get("robustez", {}).items():
            d(f"rob{RB[k]}{p}", pc(v["exactitud"]))
            d(f"robRech{RB[k]}{p}", pc(v["selectivo"]["tasa_rechazo"]))
            d(f"robSel{RB[k]}{p}", pc(v["selectivo"]["exactitud_aceptadas"]))
        for k, letra in (("0.97", "A"), ("0.98", "B"), ("0.99", "C")):
            q = r["puntos_operacion"][k]
            d(f"op{letra}Umbral{p}", pc(q["umbral"])); d(f"op{letra}Cob{p}", pc(q["cobertura"]))
            d(f"op{letra}Sel{p}", pc(q["exactitud_aceptadas"])); d(f"op{letra}Err{p}", q["errores_automaticos"])
            d(f"op{letra}Rev{p}", int(round(q["tasa_rechazo"] * r["n_test"])))
        d("nTest", r["n_test"]); d("objetivoSel", pc(r["objetivo_precision_val"], 0))
    est = json.load(open(Path(a.runs) / "estadistica.json", encoding="utf-8"))
    for m, p in PREF.items():
        lo, hi = est["ic95"][m]["exactitud"]; d(f"icAcc{p}", f"{pc(lo)}--{pc(hi)}")
        lo, hi = est["ic95"][m]["f1_macro"]; d(f"icF{p}", f"{pc(lo)}--{pc(hi)}")
    for k, v in est["mcnemar"].items():
        a1, a2 = k.split("_vs_")
        d(f"mcn{PREF[a1]}{PREF[a2]}", num(v["p_valor"], 2))
        d(f"mcnB{PREF[a1]}{PREF[a2]}", v["solo_acierta_1"]); d(f"mcnC{PREF[a1]}{PREF[a2]}", v["solo_acierta_2"])
    ds = json.load(open(Path(a.figuras) / "resumen_dataset.json", encoding="utf-8"))["dataset"]
    for s, v in ds["particiones"].items():
        for c, n in v.items():
            d(f"n{s.capitalize()}{CL[c]}", n)
        d(f"n{s.capitalize()}", sum(v.values()))
    for c, (tn, gc) in ds["fuentes"].items():
        d(f"tn{CL[c]}", tn); d(f"gc{CL[c]}", gc); d(f"tot{CL[c]}", tn + gc)
    d("nTotal", sum(tn + gc for tn, gc in ds["fuentes"].values()))
    d("nTotalTn", sum(tn for tn, _ in ds["fuentes"].values())); d("nTotalGc", sum(gc for _, gc in ds["fuentes"].values()))
    Path(a.salida).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"{len(L) - 1} macros -> {a.salida}")


if __name__ == "__main__":
    main()
