# ♻️ ReciclIA — Clasificación automática de residuos reciclables

Agente inteligente que observa un residuo con una cámara, lo clasifica en **plástico, papel/cartón, vidrio, metal u orgánico**, indica el contenedor según el código de colores colombiano (Resolución 2184 de 2019) y **pide revisión manual cuando no está seguro**.

Proyecto del curso de Agentes Inteligentes — José David Moreno Ahumada, Ingeniería de Sistemas.

| | |
|---|---|
| 📄 Artículo | `paper/Articulo_ReciclIA.pdf` |
| 🗂️ Dataset procesado | https://drive.google.com/file/d/1hDcWeyB43BreaZ5l-OtWcR34YWTTz_ie/view?usp=sharing |
| 🎬 Video promocional | https://youtu.be/MqwA1bZAXf4 |

## Resultados (conjunto de prueba independiente, 687 imágenes)

| Modelo | Parámetros | GMACs | Exactitud | F1 macro | ECE calibrado | Latencia CPU (1 hilo) |
|---|---|---|---|---|---|---|
| MobileNetV3-Large | 4,21 M | 0,22 | 95,8 % | 95,7 % | 1,5 % | 13,5 ms |
| EfficientNet-B0 | 4,01 M | 0,38 | 96,8 % | 96,8 % | 1,7 % | 24,9 ms |
| ResNet-50 | 23,52 M | 4,09 | 96,4 % | 96,2 % | 2,3 % | 100,7 ms |

Con el umbral de confianza elegido en validación, MobileNetV3 decide automáticamente el **97,2 %** de los residuos con una exactitud de **97,3 %** y envía el resto a revisión manual.

## Estructura

```
src/
  build_dataset.py   # mapeo de etiquetas, deduplicación pHash, balance y partición sin fuga
  common.py          # clases, transformaciones, dataset y modelos (timm)
  train.py           # transfer learning en dos fases
  evaluate.py        # métricas, calibración, umbral, robustez, latencia
  latencia.py        # re-medición de latencia sin carga
  figuras.py         # figuras del artículo
  macros_latex.py    # exporta las cifras al artículo (paper/resultados.tex)
  agente.py          # agente: percibir → decidir → actuar (+ CLI)
app/
  demo.py            # demostración web con Gradio
  ejemplos/          # imágenes de ejemplo
modelo/
  mobilenetv3_residuos.pt   # modelo final
  agente.json               # temperatura y umbral calibrados
runs/                # historiales y resultados de los tres modelos
figuras/             # figuras generadas
paper/               # fuente LaTeX del artículo
```

## Uso rápido

```bash
pip install -r requirements.txt

# Clasificar imágenes desde la terminal
python src/agente.py app/ejemplos/plastico_gc12.jpg app/ejemplos/vidrio_ambiguo.jpg

# Demostración web (http://127.0.0.1:7860)
python app/demo.py
```

## Reproducir los experimentos

1. Descargar las fuentes:
   - TrashNet: https://github.com/garythung/trashnet (archivo `data/dataset-resized.zip`)
   - Garbage Classification (12 clases): Kaggle `mostafaabla/garbage-classification` (se usaron las carpetas `garbage-big/train` y `garbage-big/val` de la réplica `github.com/juanvelezmunoz32-ai/clearsky`).
   - O bien descargar directamente el dataset ya procesado desde Google Drive y omitir el paso 2.
2. Construir el dataset de 5 clases:
   ```bash
   python src/build_dataset.py --trashnet RUTA/dataset-resized --gc12 RUTA/garbage-big --out data/residuos5
   ```
3. Entrenar y evaluar (CPU; los pesos ImageNet se descargan solos):
   ```bash
   python src/train.py --modelo mobilenetv3 --datos data/residuos5 --epocas1 3 --epocas2 8
   python src/train.py --modelo efficientnet_b0 --datos data/residuos5 --epocas1 3 --epocas2 6
   python src/train.py --modelo resnet50 --datos data/residuos5 --epocas1 3 --epocas2 6
   for m in mobilenetv3 efficientnet_b0 resnet50; do python src/evaluate.py --ckpt runs/$m/mejor.pt --datos data/residuos5 --salida runs/$m; done
   python src/latencia.py --runs runs --datos data/residuos5
   python src/figuras.py --runs runs --datos data/residuos5 --salida figuras
   ```

## Datos

| Paso | Imágenes |
|---|---|
| Listadas (TrashNet + Garbage Classification, clases mapeadas) | 8 961 |
| Copias eliminadas por pHash (distancia ≤ 4) | 2 451 |
| Eliminadas por duplicados con etiqueta contradictoria | 32 |
| Únicas | 6 478 |
| Dataset final balanceado | 4587 (entren. 3213 / valid. 687 / prueba 687) |

Casi todo TrashNet está contenido en Garbage Classification (2 415 parejas idénticas): combinar ambas fuentes sin deduplicar mezcla las mismas fotos entre entrenamiento y prueba.

## Licencias

Código bajo licencia MIT. TrashNet se distribuye bajo licencia MIT; Garbage Classification se publicó en Kaggle — consulte sus términos antes de redistribuir las imágenes.
