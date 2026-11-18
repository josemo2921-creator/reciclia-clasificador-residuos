"""Evaluación completa de un modelo entrenado.

Calcula: métricas en prueba, calibración por temperature scaling (ajustada en validación),
umbral de confianza para rechazo, curva riesgo-cobertura, exactitud por fuente,
robustez ante perturbaciones de captura, latencia en CPU y tamaño del modelo.

Uso:
    python src/evaluate.py --ckpt runs/mobilenetv3/mejor.pt --datos data/residuos5 --salida runs/mobilenetv3
"""
import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, Dataset
from torch.utils.flop_counter import FlopCounterMode

from common import CLASES, CarpetaResiduos, cargar_entrenado, tf_evaluacion

OBJETIVO_PRECISION = 0.97  # exactitud mínima exigida a las decisiones automáticas (en validación)


# ---------------------------------------------------------------- perturbaciones
def _ruido(im, sigma=0.08):
    a = np.asarray(im).astype(np.float32) / 255.0
    a = np.clip(a + np.random.default_rng(0).normal(0, sigma, a.shape), 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8))


def _jpeg(im, q=15):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0)
    return Image.open(b).convert("RGB")


PERTURBACIONES = {
    "original": lambda im: im,
    "baja_iluminacion": lambda im: ImageEnhance.Brightness(im).enhance(0.45),
    "sobreexposicion": lambda im: ImageEnhance.Brightness(im).enhance(1.7),
    "bajo_contraste": lambda im: ImageEnhance.Contrast(im).enhance(0.45),
    "desenfoque": lambda im: im.filter(ImageFilter.GaussianBlur(2.5)),
    "ruido": _ruido,
    "jpeg_q15": _jpeg,
    "rotacion_30": lambda im: im.rotate(30, fillcolor=(128, 128, 128)),
}


class Perturbado(Dataset):
    def __init__(self, base, f):
        self.base, self.f, self.tf = base, f, tf_evaluacion()

    def __len__(self):
        return len(self.base.items)

    def __getitem__(self, k):
        p, y = self.base.items[k]
        with Image.open(p) as im:
            return self.tf(self.f(im.convert("RGB"))), y


# ---------------------------------------------------------------- utilidades
@torch.no_grad()
def logits_de(m, ds, workers=2):
    L, Y = [], []
    for x, y in DataLoader(ds, batch_size=64, num_workers=workers):
        L.append(m(x)); Y.append(y)
    return torch.cat(L), torch.cat(Y)


def ajustar_temperatura(logits, y):
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)

    def cierre():
        opt.zero_grad()
        loss = F.cross_entropy(logits / logT.exp(), y)
        loss.backward()
        return loss
    opt.step(cierre)
    return float(logT.detach().exp())


def ece(prob, y, bins=15):
    conf, pred = prob.max(1)
    acc = (pred == y).float()
    bordes = torch.linspace(0, 1, bins + 1)
    e, tabla = 0.0, []
    for i in range(bins):
        msk = (conf > bordes[i]) & (conf <= bordes[i + 1])
        if msk.any():
            e += msk.float().mean().item() * abs(acc[msk].mean().item() - conf[msk].mean().item())
            tabla.append([float(bordes[i]), float(bordes[i + 1]), int(msk.sum()),
                          acc[msk].mean().item(), conf[msk].mean().item()])
    return e, tabla


def riesgo_cobertura(prob, y):
    conf, pred = prob.max(1)
    orden = torch.argsort(conf, descending=True)
    acierto = (pred[orden] == y[orden]).float()
    n = torch.arange(1, len(y) + 1).float()
    return (n / len(y)).tolist(), (acierto.cumsum(0) / n).tolist(), conf[orden].tolist()


def elegir_umbral(prob, y, objetivo):
    """Menor umbral cuya exactitud selectiva en validación alcanza el objetivo (máxima cobertura)."""
    conf, pred = prob.max(1)
    mejor = 1.0
    for t in np.unique(conf.numpy())[::-1]:
        msk = conf >= t
        if (pred[msk] == y[msk]).float().mean() >= objetivo:
            mejor = float(t)
    return mejor


def selectivo(prob, y, t):
    conf, pred = prob.max(1)
    msk = conf >= t
    return {"umbral": t, "cobertura": msk.float().mean().item(),
            "tasa_rechazo": 1 - msk.float().mean().item(),
            "exactitud_aceptadas": (pred[msk] == y[msk]).float().mean().item() if msk.any() else float("nan"),
            "errores_automaticos": int(((pred != y) & msk).sum()),
            "errores_enviados_a_revision": int(((pred != y) & ~msk).sum()),
            "aciertos_enviados_a_revision": int(((pred == y) & ~msk).sum())}


def latencia(m, hilos, n=100):
    torch.set_num_threads(hilos)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(10):
            m(x)
        t = []
        for _ in range(n):
            t0 = time.perf_counter(); m(x); t.append((time.perf_counter() - t0) * 1000)
    return {"hilos": hilos, "media_ms": float(np.mean(t)), "p95_ms": float(np.percentile(t, 95))}


def latencia_preproceso(rutas, n=100):
    tf = tf_evaluacion(); t = []
    for p in rutas[:n]:
        t0 = time.perf_counter()
        with Image.open(p) as im:
            tf(im.convert("RGB"))
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--datos", default="data/residuos5")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--sin_robustez", action="store_true")
    a = ap.parse_args()
    sal = Path(a.salida)
    m, ck = cargar_entrenado(a.ckpt)

    val = CarpetaResiduos(a.datos, "val", tf_evaluacion())
    test = CarpetaResiduos(a.datos, "test", tf_evaluacion())
    Lv, Yv = logits_de(m, val)
    Lt, Yt = logits_de(m, test)

    T = ajustar_temperatura(Lv, Yv)
    Pt_raw, Pt = F.softmax(Lt, 1), F.softmax(Lt / T, 1)
    Pv = F.softmax(Lv / T, 1)
    pred = Pt.argmax(1).numpy(); y = Yt.numpy()

    prec, rec, f1, sop = precision_recall_fscore_support(y, pred, labels=range(len(CLASES)), zero_division=0)
    fuentes = np.array([p.name.split("_")[0] for p, _ in test.items])
    por_fuente = {f: {"n": int((fuentes == f).sum()), "exactitud": float((pred[fuentes == f] == y[fuentes == f]).mean())}
                  for f in np.unique(fuentes)}
    ece_raw, _ = ece(Pt_raw, Yt)
    ece_cal, tabla_cal = ece(Pt, Yt)
    _, tabla_raw = ece(Pt_raw, Yt)
    t = elegir_umbral(Pv, Yv, OBJETIVO_PRECISION)
    cob, acc_sel, _ = riesgo_cobertura(Pt, Yt)

    res = {
        "modelo": ck["modelo"], "epoca_mejor": ck["epoca"], "f1_val": ck["f1_val"],
        "n_test": int(len(y)), "exactitud": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "por_clase": {c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "n": int(sop[i])}
                      for i, c in enumerate(CLASES)},
        "matriz_confusion": confusion_matrix(y, pred, labels=range(len(CLASES))).tolist(),
        "por_fuente": por_fuente,
        "temperatura": T, "ece_sin_calibrar": ece_raw, "ece_calibrado": ece_cal,
        "nll_sin_calibrar": float(F.cross_entropy(Lt, Yt)), "nll_calibrado": float(F.cross_entropy(Lt / T, Yt)),
        "fiabilidad_sin_calibrar": tabla_raw, "fiabilidad_calibrado": tabla_cal,
        "selectivo": selectivo(Pt, Yt, t), "objetivo_precision_val": OBJETIVO_PRECISION,
        "puntos_operacion": {str(o): selectivo(Pt, Yt, elegir_umbral(Pv, Yv, o)) for o in (0.97, 0.98, 0.99)},
        "riesgo_cobertura": {"cobertura": cob[::5], "exactitud": acc_sel[::5]},
        "parametros_M": sum(p.numel() for p in m.parameters()) / 1e6,
    }
    tam = sal / "_tmp.pt"; torch.save(m.state_dict(), tam)
    res["tamano_MB"] = tam.stat().st_size / 1e6; tam.unlink()
    with FlopCounterMode(display=False) as fc:
        m(torch.randn(1, 3, 224, 224))
    res["GMACs"] = fc.get_total_flops() / 2e9  # FlopCounter cuenta 2 FLOPs por MAC
    res["latencia"] = [latencia(m, 1), latencia(m, 2)]
    res["latencia_preproceso_ms"] = latencia_preproceso([p for p, _ in test.items])
    torch.set_num_threads(2)

    if not a.sin_robustez:
        base = CarpetaResiduos(a.datos, "test", None)
        res["robustez"] = {}
        for nombre, f in PERTURBACIONES.items():
            L, Y = logits_de(m, Perturbado(base, f))
            P = F.softmax(L / T, 1)
            res["robustez"][nombre] = {"exactitud": float((P.argmax(1) == Y).float().mean()),
                                       "selectivo": selectivo(P, Y, t)}
            print(nombre, res["robustez"][nombre]["exactitud"], flush=True)

    np.savez(sal / "logits.npz", val=Lv.numpy(), yval=Yv.numpy(), test=Lt.numpy(), ytest=Yt.numpy(),
             archivos_test=np.array([str(p) for p, _ in test.items]))
    json.dump(res, open(sal / "resultados.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in ["modelo", "exactitud", "f1_macro", "temperatura", "ece_sin_calibrar",
                                          "ece_calibrado", "selectivo", "latencia", "GMACs"]}, indent=1))


if __name__ == "__main__":
    main()
