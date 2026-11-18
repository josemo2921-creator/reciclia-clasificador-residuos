"""Agente de clasificación de residuos: percepción -> decisión -> acción.

Uso por línea de comandos:
    python src/agente.py ruta/imagen.jpg [otra.jpg ...]
"""
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from common import CLASES, NOMBRES, cargar_entrenado, tf_evaluacion

RAIZ = Path(__file__).resolve().parents[1]

# Código de colores para separación en la fuente en Colombia (Resolución 2184 de 2019):
# blanco = aprovechables, verde = orgánicos aprovechables.
CONTENEDOR = {
    "plastico": ("Blanco (aprovechables)", "#f2f2f2"),
    "papel_carton": ("Blanco (aprovechables)", "#f2f2f2"),
    "vidrio": ("Blanco (aprovechables)", "#f2f2f2"),
    "metal": ("Blanco (aprovechables)", "#f2f2f2"),
    "organico": ("Verde (orgánicos aprovechables)", "#3a9d4a"),
}


class AgenteResiduos:
    def __init__(self, dir_modelo=RAIZ / "modelo", registro=RAIZ / "registro_clasificaciones.csv"):
        cfg = json.load(open(Path(dir_modelo) / "agente.json", encoding="utf-8"))
        self.modelo, _ = cargar_entrenado(Path(dir_modelo) / cfg["checkpoint"])
        self.T, self.umbral = cfg["temperatura"], cfg["umbral"]
        self.tf = tf_evaluacion()
        self.registro = Path(registro)

    # Percepción: imagen -> tensor normalizado
    def percibir(self, imagen):
        if not isinstance(imagen, Image.Image):
            imagen = Image.open(imagen)
        return self.tf(imagen.convert("RGB")).unsqueeze(0)

    # Decisión: clase, confianza calibrada y distribución completa
    @torch.no_grad()
    def decidir(self, x):
        p = F.softmax(self.modelo(x) / self.T, 1)[0]
        k = int(p.argmax())
        return CLASES[k], float(p[k]), {c: float(p[i]) for i, c in enumerate(CLASES)}

    # Acción: mostrar resultado / señal / revisión manual y registrar el evento
    def actuar(self, clase, conf, probs, origen=""):
        if conf >= self.umbral:
            contenedor, color = CONTENEDOR[clase]
            accion = {"estado": "clasificado", "clase": clase, "nombre": NOMBRES[clase],
                      "contenedor": contenedor, "color": color,
                      "mensaje": f"{NOMBRES[clase]} ({conf:.0%}) -> contenedor {contenedor}"}
        else:
            accion = {"estado": "incierto", "clase": None, "nombre": "Residuo incierto",
                      "contenedor": "Revisión manual", "color": "#e0a800",
                      "mensaje": f"Residuo incierto (confianza {conf:.0%} < {self.umbral:.0%}): solicite revisión manual"}
        accion.update({"confianza": conf, "probabilidades": probs})
        self._registrar(accion, origen)
        return accion

    def procesar(self, imagen, origen=""):
        t0 = time.perf_counter()
        clase, conf, probs = self.decidir(self.percibir(imagen))
        accion = self.actuar(clase, conf, probs, origen)
        accion["latencia_ms"] = (time.perf_counter() - t0) * 1000
        return accion

    def _registrar(self, accion, origen):
        nuevo = not self.registro.exists()
        with open(self.registro, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if nuevo:
                w.writerow(["fecha", "origen", "estado", "clase", "confianza"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), origen, accion["estado"],
                        accion["clase"] or "", f"{accion['confianza']:.4f}"])

    def estadisticas(self):
        if not self.registro.exists():
            return {}
        conteo = {}
        with open(self.registro, encoding="utf-8") as f:
            for fila in csv.DictReader(f):
                k = fila["clase"] or "incierto"
                conteo[k] = conteo.get(k, 0) + 1
        return conteo


if __name__ == "__main__":
    agente = AgenteResiduos()
    for ruta in sys.argv[1:]:
        r = agente.procesar(ruta, origen=ruta)
        print(f"{ruta}: {r['mensaje']}  [{r['latencia_ms']:.0f} ms]")
    print("Estadísticas acumuladas:", agente.estadisticas())
