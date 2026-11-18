"""Construye el dataset de 5 clases a partir de TrashNet y Garbage Classification (12 clases).

Pasos: mapeo de etiquetas -> deduplicación perceptual (pHash) -> muestreo balanceado
-> partición estratificada 70/15/15 por grupos de casi-duplicados (sin fuga).

Uso:
    python src/build_dataset.py --trashnet RUTA/dataset-resized --gc12 RUTA/garbage-big --out data/residuos5
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

CLASES = ["plastico", "papel_carton", "vidrio", "metal", "organico"]

MAPEO_TRASHNET = {
    "plastic": "plastico", "paper": "papel_carton", "cardboard": "papel_carton",
    "glass": "vidrio", "metal": "metal",
    # "trash" se excluye: mezcla de materiales sin clase clara
}
MAPEO_GC12 = {
    "plastic": "plastico", "paper": "papel_carton", "cardboard": "papel_carton",
    "brown-glass": "vidrio", "green-glass": "vidrio", "white-glass": "vidrio",
    "metal": "metal", "biological": "organico",
}
EXT = {".jpg", ".jpeg", ".png"}


def listar(trashnet, gc12):
    items = []
    for carpeta, clase in MAPEO_TRASHNET.items():
        for p in sorted((trashnet / carpeta).iterdir()):
            if p.suffix.lower() in EXT:
                items.append({"ruta": p, "clase": clase, "fuente": "trashnet", "orig": carpeta})
    for split in ("train", "val"):
        for carpeta, clase in MAPEO_GC12.items():
            for p in sorted((gc12 / split / carpeta).iterdir()):
                if p.suffix.lower() in EXT:
                    items.append({"ruta": p, "clase": clase, "fuente": "gc12", "orig": carpeta})
    return items


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trashnet", type=Path, required=True)
    ap.add_argument("--gc12", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--por_clase", type=int, default=1000)
    ap.add_argument("--max_trashnet", type=int, default=500)
    ap.add_argument("--umbral_hash", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed)

    items = listar(a.trashnet, a.gc12)
    print(f"Imágenes listadas: {len(items)}")

    # 1) Lectura y pHash; se descartan archivos corruptos o diminutos
    validos = []
    for it in items:
        try:
            with Image.open(it["ruta"]) as im:
                im = im.convert("RGB")
                if min(im.size) < 64:
                    continue
                it["hash"] = imagehash.phash(im)
            validos.append(it)
        except Exception:
            pass
    print(f"Válidas: {len(validos)} (descartadas {len(items) - len(validos)})")

    # 2) Agrupación de casi-duplicados (distancia de Hamming <= umbral)
    n = len(validos)
    uf = UnionFind(n)
    H = np.packbits(np.stack([it["hash"].hash.flatten() for it in validos]), axis=1)
    H = H.view(np.uint64).ravel()
    for i in range(n - 1):
        x = np.bitwise_xor(H[i + 1:], H[i])
        dist = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        for j in np.nonzero(dist <= a.umbral_hash)[0]:
            uf.union(i, i + 1 + int(j))
    grupos = defaultdict(list)
    for i in range(n):
        grupos[uf.find(i)].append(i)
    # Duplicados con etiquetas en conflicto se eliminan; los consistentes conservan un representante
    unicos = []
    conflictos = dup = 0
    for g in grupos.values():
        clases = {validos[i]["clase"] for i in g}
        if len(clases) > 1:
            conflictos += len(g)
            continue
        dup += len(g) - 1
        unicos.append(validos[g[0]])
    print(f"Duplicados eliminados: {dup}; eliminadas por etiqueta en conflicto: {conflictos}; únicas: {len(unicos)}")

    # 3) Muestreo balanceado: hasta max_trashnet de TrashNet y el resto de GC12
    por_clase = defaultdict(lambda: defaultdict(list))
    for it in unicos:
        por_clase[it["clase"]][it["fuente"]].append(it)
    seleccion = []
    for c in CLASES:
        tn = por_clase[c]["trashnet"][:]
        gc = por_clase[c]["gc12"][:]
        random.shuffle(tn); random.shuffle(gc)
        k_tn = min(len(tn), a.max_trashnet)
        k_gc = min(len(gc), a.por_clase - k_tn)
        seleccion += tn[:k_tn] + gc[:k_gc]
        print(f"  {c:13s} trashnet={k_tn:4d} gc12={k_gc:4d} total={k_tn + k_gc}")

    # 4) Partición estratificada por grupos (los grupos ya se colapsaron; se usa el índice como grupo)
    y = [CLASES.index(it["clase"]) for it in seleccion]
    idx = list(range(len(seleccion)))
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=a.seed)
    folds = [te for _, te in sgkf.split(idx, y, groups=idx)]
    asign = {}
    for f, te in enumerate(folds):
        s = "train" if f < 14 else ("val" if f < 17 else "test")
        for i in te:
            asign[i] = s

    # 5) Escritura: imágenes redimensionadas (lado mayor 384 px) y manifiesto CSV
    a.out.mkdir(parents=True, exist_ok=True)
    filas = []
    for i, it in enumerate(seleccion):
        s = asign[i]
        destino = a.out / s / it["clase"] / f"{it['fuente']}_{it['orig']}_{it['ruta'].stem}.jpg"
        destino.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(it["ruta"]) as im:
            im = im.convert("RGB")
            im.thumbnail((384, 384))
            im.save(destino, quality=92)
        filas.append({"archivo": str(destino.relative_to(a.out)), "clase": it["clase"],
                      "particion": s, "fuente": it["fuente"], "etiqueta_original": it["orig"]})
    with open(a.out / "manifiesto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)
    resumen = defaultdict(lambda: defaultdict(int))
    for r in filas:
        resumen[r["particion"]][r["clase"]] += 1
    for s in ("train", "val", "test"):
        print(s, dict(resumen[s]), sum(resumen[s].values()))


if __name__ == "__main__":
    main()
