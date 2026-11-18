"""Genera las figuras y tablas del informe a partir de runs/*/resultados.json.

Uso:
    python src/figuras.py --runs runs --datos data/residuos5 --salida figuras
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from common import CLASES, NOMBRES, cargar_entrenado, tf_evaluacion  # noqa: E402

MODELOS = ["mobilenetv3", "efficientnet_b0", "resnet50"]
ETIQ = {"mobilenetv3": "MobileNetV3-L", "efficientnet_b0": "EfficientNet-B0", "resnet50": "ResNet-50"}
COLOR = {"mobilenetv3": "#2a78d6", "efficientnet_b0": "#eb6834", "resnet50": "#1baf7a"}
TINTA, TINTA2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"
NC = [NOMBRES[c] for c in CLASES]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": TINTA2,
                     "axes.labelcolor": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": REJILLA, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


def guardar(fig, sal, nombre):
    fig.savefig(sal / f"{nombre}.pdf"); fig.savefig(sal / f"{nombre}.png", dpi=200); plt.close(fig)


def fig_dataset(datos, sal):
    filas = list(csv.DictReader(open(Path(datos) / "manifiesto.csv", encoding="utf-8")))
    cnt = Counter((r["clase"], r["fuente"]) for r in filas)
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    x = np.arange(len(CLASES))
    tn = np.array([cnt[(c, "trashnet")] for c in CLASES]); gc = np.array([cnt[(c, "gc12")] for c in CLASES])
    ax.bar(x, tn, 0.6, color="#2a78d6", label="TrashNet", edgecolor="white", linewidth=1)
    ax.bar(x, gc, 0.6, bottom=tn, color="#eb6834", label="Garbage Classification", edgecolor="white", linewidth=1)
    for i in range(len(CLASES)):
        ax.text(i, tn[i] + gc[i] + 15, str(tn[i] + gc[i]), ha="center", fontsize=7, color=TINTA)
    ax.set_xticks(x, NC, fontsize=7); ax.set_ylabel("Imágenes"); ax.set_ylim(0, 1100)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    guardar(fig, sal, "fig_dataset")
    tabla = {s: Counter(r["clase"] for r in filas if r["particion"] == s) for s in ("train", "val", "test")}
    fuentes = {c: (cnt[(c, "trashnet")], cnt[(c, "gc12")]) for c in CLASES}
    return {"particiones": {s: dict(v) for s, v in tabla.items()}, "fuentes": fuentes}


def fig_ejemplos(datos, sal):
    fig, axs = plt.subplots(2, 5, figsize=(7.0, 3.0))
    rng = np.random.default_rng(3)
    for j, c in enumerate(CLASES):
        for i, f in enumerate(["trashnet", "gc12"]):
            ax = axs[i, j]; ax.axis("off")
            cand = sorted((Path(datos) / "train" / c).glob(f"{f}_*.jpg"))
            if cand:
                im = Image.open(cand[rng.integers(len(cand))]).convert("RGB")
                w, h = im.size; s = min(w, h)
                ax.imshow(im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)))
            else:
                ax.text(0.5, 0.5, "sin imágenes\nen TrashNet", ha="center", va="center", fontsize=7, color=TINTA2)
            if i == 0:
                ax.set_title(NOMBRES[c], fontsize=8)
    axs[0, 0].text(-0.08, 0.5, "TrashNet", transform=axs[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    axs[1, 0].text(-0.08, 0.5, "Garbage Class.", transform=axs[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    fig.subplots_adjust(wspace=0.04, hspace=0.08)
    guardar(fig, sal, "fig_ejemplos")


def fig_curvas(runs, sal):
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for m in MODELOS:
        h = json.load(open(runs / m / "historial.json"))["historial"]
        ax.plot([r["epoca"] for r in h], [r["f1_val"] * 100 for r in h], "-o", ms=3, lw=1.6,
                color=COLOR[m], label=ETIQ[m])
    ax.axvline(3.5, color=TINTA2, lw=0.8, ls="--")
    ax.text(3.6, ax.get_ylim()[0] + 0.3, "inicio fase 2", fontsize=6.5, color=TINTA2)
    ax.set_xlabel("Época"); ax.set_ylabel("F1 macro en validación (%)"); ax.legend(fontsize=7)
    guardar(fig, sal, "fig_curvas")


def fig_confusion(res, sal):
    cm = np.array(res["matriz_confusion"]); n = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    ax.imshow(n, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            ax.text(j, i, f"{cm[i, j]}\n{n[i, j]:.0%}" if cm[i, j] else "0", ha="center", va="center",
                    fontsize=6.5, color="white" if n[i, j] > 0.55 else TINTA)
    ax.set_xticks(range(5), NC, rotation=30, ha="right", fontsize=7); ax.set_yticks(range(5), NC, fontsize=7)
    ax.set_xlabel("Clase predicha"); ax.set_ylabel("Clase real")
    for s in ax.spines.values():
        s.set_visible(False)
    guardar(fig, sal, "fig_confusion")


def fig_fiabilidad(res, sal):
    coma = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.9), sharey=True)
    for ax, k, tit, e in [(axs[0], "fiabilidad_sin_calibrar", "Sin calibrar", res["ece_sin_calibrar"]),
                          (axs[1], "fiabilidad_calibrado", f"T = {res['temperatura']:.2f}".replace(".", ","), res["ece_calibrado"])]:
        t = np.array(res[k])
        centros, acc, conf, n = (t[:, 0] + t[:, 1]) / 2, t[:, 3], t[:, 4], t[:, 2]
        ax.plot([0, 1], [0, 1], color=TINTA2, lw=0.8, ls="--")
        ax.bar(centros, acc, width=1 / 15 - 0.006, color="#2a78d6", alpha=0.85, label="Exactitud")
        ax.plot(conf, acc, "o", ms=2.5, color=TINTA)
        ax.set_title(f"{tit}\nECE = {e * 100:.1f} %".replace(".", ","), fontsize=7.5)
        ax.xaxis.set_major_formatter(coma); ax.yaxis.set_major_formatter(coma)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("Confianza", fontsize=7)
    axs[0].set_ylabel("Exactitud")
    guardar(fig, sal, "fig_fiabilidad")


def fig_riesgo(todos, sal):
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for m in MODELOS:
        rc = todos[m]["riesgo_cobertura"]
        ax.plot(np.array(rc["cobertura"]) * 100, np.array(rc["exactitud"]) * 100, lw=1.6, color=COLOR[m], label=ETIQ[m])
        s = todos[m]["selectivo"]
        ax.plot(s["cobertura"] * 100, s["exactitud_aceptadas"] * 100, "o", ms=5, color=COLOR[m],
                markeredgecolor="white", markeredgewidth=1)
    ax.set_xlabel("Cobertura: imágenes clasificadas automáticamente (%)"); ax.set_ylabel("Exactitud selectiva (%)")
    ax.set_xlim(70, 100.5); lo = min(todos[m]["exactitud"] for m in MODELOS) * 100
    ax.set_ylim(lo - 2, 100.5); ax.legend(fontsize=7, loc="lower left")
    guardar(fig, sal, "fig_riesgo_cobertura")


def fig_robustez(todos, sal):
    conds = list(todos[MODELOS[0]]["robustez"].keys())
    etq = {"original": "Original", "baja_iluminacion": "Baja\niluminación", "sobreexposicion": "Sobre-\nexposición",
           "bajo_contraste": "Bajo\ncontraste", "desenfoque": "Desen-\nfoque", "ruido": "Ruido",
           "jpeg_q15": "JPEG\nq=15", "rotacion_30": "Rotación\n30°"}
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    x = np.arange(len(conds))
    for k, m in enumerate(MODELOS):
        v = [todos[m]["robustez"][c]["exactitud"] * 100 for c in conds]
        ax.plot(x + (k - 1) * 0.18, v, "o", ms=6, color=COLOR[m], label=ETIQ[m], markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xticks(x, [etq[c] for c in conds], fontsize=7); ax.set_ylabel("Exactitud en prueba (%)")
    ax.set_xlim(-0.5, len(conds) - 0.5)
    ax.grid(axis="x", visible=False); ax.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    guardar(fig, sal, "fig_robustez")


def gradcam(m, x, clase):
    m.zero_grad()
    f = m.forward_features(x); f.retain_grad()
    out = m.forward_head(f)
    out[0, clase].backward()
    w = f.grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * f).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / (cam.max() + 1e-8)).detach().numpy(), out.detach()


def fig_gradcam(runs, sal, T, datos, n_ok=5, n_err=5):
    m, _ = cargar_entrenado(runs / "mobilenetv3" / "mejor.pt")
    z = np.load(runs / "mobilenetv3" / "logits.npz")
    L, y, arch = torch.tensor(z["test"]), z["ytest"], z["archivos_test"]
    P = F.softmax(L / T, 1); pred = P.argmax(1).numpy(); conf = P.max(1).values.numpy()
    rng = np.random.default_rng(7)
    ok = [int(rng.choice(np.where((pred == y) & (y == k))[0])) for k in range(5)][:n_ok]
    err_idx = np.where(pred != y)[0]
    err = list(err_idx[np.argsort(-conf[err_idx])][:n_err])
    tf = tf_evaluacion()
    fig, axs = plt.subplots(2, 5, figsize=(7.0, 3.3))
    for fila, idxs in enumerate([ok, err]):
        for j, i in enumerate(idxs):
            ruta = Path(datos) / "test" / Path(arch[i]).parent.name / Path(arch[i]).name
            im = Image.open(ruta).convert("RGB"); x = tf(im).unsqueeze(0).requires_grad_(False)
            cam, _ = gradcam(m, x, int(pred[i]))
            vis = x[0].permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            ax = axs[fila, j]; ax.imshow(np.clip(vis, 0, 1)); ax.imshow(cam, cmap="jet", alpha=0.4); ax.axis("off")
            ax.set_title(f"Real: {NOMBRES[CLASES[y[i]]]}\nPred: {NOMBRES[CLASES[pred[i]]]} ({conf[i]:.0%})", fontsize=6.5,
                         color=TINTA if pred[i] == y[i] else "#b3261e")
    fig.subplots_adjust(wspace=0.05, hspace=0.35)
    guardar(fig, sal, "fig_gradcam")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs"); ap.add_argument("--datos", default="data/residuos5")
    ap.add_argument("--salida", default="figuras")
    a = ap.parse_args()
    runs, sal = Path(a.runs), Path(a.salida); sal.mkdir(parents=True, exist_ok=True)
    todos = {m: json.load(open(runs / m / "resultados.json", encoding="utf-8")) for m in MODELOS}
    resumen = {"dataset": fig_dataset(a.datos, sal)}
    fig_ejemplos(a.datos, sal); fig_curvas(runs, sal)
    fig_confusion(todos["mobilenetv3"], sal); fig_fiabilidad(todos["mobilenetv3"], sal)
    fig_riesgo(todos, sal); fig_robustez(todos, sal)
    fig_gradcam(runs, sal, todos["mobilenetv3"]["temperatura"], a.datos)
    json.dump(resumen, open(sal / "resumen_dataset.json", "w"), indent=2, ensure_ascii=False)
    print("Figuras en", sal)


if __name__ == "__main__":
    main()
