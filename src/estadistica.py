"""Intervalos de confianza bootstrap (95 %) y prueba de McNemar entre modelos.

Uso:
    python src/estadistica.py --runs runs
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

MODELOS = ["mobilenetv3", "efficientnet_b0", "resnet50"]

ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs"); a = ap.parse_args()
pred, y = {}, None
for m in MODELOS:
    z = np.load(Path(a.runs) / m / "logits.npz")
    pred[m] = z["test"].argmax(1); y = z["ytest"]
rng = np.random.default_rng(0)
B = 2000
idx = rng.integers(0, len(y), (B, len(y)))
res = {"ic95": {}, "mcnemar": {}}
for m in MODELOS:
    acc = (pred[m][idx] == y[idx]).mean(1)
    f1 = np.array([f1_score(y[i], pred[m][i], average="macro") for i in idx])
    res["ic95"][m] = {"exactitud": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
                      "f1_macro": [float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))]}
for a1, a2 in combinations(MODELOS, 2):
    b = int(((pred[a1] == y) & (pred[a2] != y)).sum()); c = int(((pred[a1] != y) & (pred[a2] == y)).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    res["mcnemar"][f"{a1}_vs_{a2}"] = {"solo_acierta_1": b, "solo_acierta_2": c, "p_valor": float(p)}
json.dump(res, open(Path(a.runs) / "estadistica.json", "w"), indent=2)
print(json.dumps(res, indent=1))
