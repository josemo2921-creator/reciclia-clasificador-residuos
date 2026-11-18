"""Demostración web del agente clasificador de residuos (Gradio).

Ejecutar desde la raíz del repositorio:
    python app/demo.py
y abrir http://127.0.0.1:7860
"""
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agente import AgenteResiduos  # noqa: E402
from common import NOMBRES  # noqa: E402

agente = AgenteResiduos()
EJEMPLOS = sorted(str(p) for p in (Path(__file__).parent / "ejemplos").glob("*.jpg"))


def clasificar(imagen):
    if imagen is None:
        return {}, "Cargue o capture una imagen.", ""
    r = agente.procesar(imagen, origen="demo")
    probs = {NOMBRES[c]: p for c, p in r["probabilidades"].items()}
    color = r["color"]
    texto = "#fff" if color == "#3a9d4a" else "#222"
    tarjeta = (f"<div style='background:{color};color:{texto};border:2px solid #333;border-radius:12px;"
               f"padding:18px;font-size:20px;text-align:center'>"
               f"<b>{r['nombre']}</b><br>Contenedor: {r['contenedor']}<br>"
               f"<span style='font-size:15px'>Confianza calibrada: {r['confianza']:.1%} · "
               f"umbral {agente.umbral:.0%} · {r['latencia_ms']:.0f} ms</span></div>")
    est = agente.estadisticas()
    resumen = " · ".join(f"{(NOMBRES.get(k, 'Incierto'))}: {v}" for k, v in est.items())
    return probs, tarjeta, f"Registro acumulado → {resumen}"


with gr.Blocks(title="ReciclIA: clasificador de residuos") as demo:
    gr.Markdown("# ♻️ ReciclIA — Agente de clasificación de residuos\n"
                "Suba una foto del residuo (o use la cámara). El agente predice la categoría, "
                "indica el contenedor y, si no está seguro, pide revisión manual.")
    with gr.Row():
        with gr.Column():
            entrada = gr.Image(type="pil", sources=["upload", "webcam"], label="Residuo")
            boton = gr.Button("Clasificar", variant="primary")
            if EJEMPLOS:
                gr.Examples(EJEMPLOS, inputs=entrada)
        with gr.Column():
            tarjeta = gr.HTML()
            probs = gr.Label(num_top_classes=5, label="Probabilidades calibradas")
            registro = gr.Markdown()
    boton.click(clasificar, entrada, [probs, tarjeta, registro])
    entrada.upload(clasificar, entrada, [probs, tarjeta, registro])

if __name__ == "__main__":
    demo.launch()
