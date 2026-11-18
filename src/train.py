"""Entrenamiento por transfer learning en dos fases.

Fase 1: backbone congelado, solo se entrena la cabeza.
Fase 2: se descongelan las últimas etapas y se ajustan con tasa de aprendizaje menor.
Se guarda el checkpoint con mejor F1 macro en validación.

Uso:
    python src/train.py --modelo mobilenetv3 --datos data/residuos5 --pesos weights --salida runs
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from common import CarpetaResiduos, crear_modelo, partes_finetune, tf_entrenamiento, tf_evaluacion


def fijar_bn_congeladas(m, entrenables):
    """Pone en modo eval todas las capas BatchNorm que no estén en las partes entrenables."""
    ids = {id(mod) for p in entrenables for mod in p.modules()}
    for mod in m.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm) and id(mod) not in ids:
            mod.eval()


@torch.no_grad()
def evaluar(m, loader):
    m.eval()
    ys, ps, perdida = [], [], 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        o = m(x)
        perdida += ce(o, y).item()
        ys.append(y.numpy()); ps.append(o.argmax(1).numpy())
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    return perdida / len(ys), float((ys == ps).mean()), float(f1_score(ys, ps, average="macro"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="mobilenetv3")
    ap.add_argument("--datos", default="data/residuos5")
    ap.add_argument("--pesos", default="weights")
    ap.add_argument("--salida", default="runs")
    ap.add_argument("--epocas1", type=int, default=3)
    ap.add_argument("--epocas2", type=int, default=6)
    ap.add_argument("--lr1", type=float, default=3e-3)
    ap.add_argument("--lr2", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    salida = Path(a.salida) / a.modelo
    salida.mkdir(parents=True, exist_ok=True)

    tr = DataLoader(CarpetaResiduos(a.datos, "train", tf_entrenamiento()), batch_size=a.batch,
                    shuffle=True, num_workers=a.workers, drop_last=True)
    va = DataLoader(CarpetaResiduos(a.datos, "val", tf_evaluacion()), batch_size=64, num_workers=a.workers)

    m = crear_modelo(a.modelo, a.pesos)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    historial, mejor = [], -1.0
    fases = [(1, a.epocas1, a.lr1), (2, a.epocas2, a.lr2)]
    epoca_global = 0
    t_total = time.time()
    for fase, n_ep, lr in fases:
        for p in m.parameters():
            p.requires_grad = False
        partes = [m.get_classifier()] if fase == 1 else partes_finetune(a.modelo, m)
        params = [p for parte in partes for p in parte.parameters()]
        for p in params:
            p.requires_grad = True
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep * len(tr))
        for ep in range(n_ep):
            epoca_global += 1
            m.train(); fijar_bn_congeladas(m, partes)
            t0, suma, n = time.time(), 0.0, 0
            for x, y in tr:
                opt.zero_grad()
                loss = ce(m(x), y)
                loss.backward(); opt.step(); sched.step()
                suma += loss.item() * len(y); n += len(y)
            vl, vacc, vf1 = evaluar(m, va)
            reg = {"epoca": epoca_global, "fase": fase, "perdida_train": suma / n, "perdida_val": vl,
                   "exactitud_val": vacc, "f1_val": vf1, "segundos": time.time() - t0}
            historial.append(reg)
            print(json.dumps(reg), flush=True)
            if vf1 > mejor:
                mejor = vf1
                torch.save({"modelo": a.modelo, "estado": m.state_dict(), "epoca": epoca_global,
                            "f1_val": vf1}, salida / "mejor.pt")
    json.dump({"historial": historial, "mejor_f1_val": mejor, "minutos": (time.time() - t_total) / 60,
               "args": vars(a)}, open(salida / "historial.json", "w"), indent=2)
    print(f"Mejor F1 val: {mejor:.4f}")


if __name__ == "__main__":
    main()
