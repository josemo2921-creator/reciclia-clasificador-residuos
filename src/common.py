"""Utilidades compartidas: clases, transformaciones, dataset y construcción de modelos."""
from pathlib import Path

import timm
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

CLASES = ["plastico", "papel_carton", "vidrio", "metal", "organico"]
NOMBRES = {"plastico": "Plástico", "papel_carton": "Papel/cartón", "vidrio": "Vidrio",
           "metal": "Metal", "organico": "Orgánico"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
IMG = 224

# Pesos ImageNet de timm publicados como "releases" en GitHub
MODELOS = {
    "mobilenetv3": {"timm": "mobilenetv3_large_100", "pesos": "mobilenetv3_large_100_ra-f55367f5.pth",
                    "url": "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth"},
    "efficientnet_b0": {"timm": "efficientnet_b0", "pesos": "efficientnet_b0_ra-3dd342df.pth",
                        "url": "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth"},
    "resnet50": {"timm": "resnet50", "pesos": "resnet50_a1_0-14fe96d1.pth",
                 "url": "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth"},
}


def tf_entrenamiento():
    return T.Compose([
        T.RandomResizedCrop(IMG, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.35, contrast=0.3, saturation=0.25, hue=0.02),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])


def tf_evaluacion():
    return T.Compose([T.Resize(int(IMG * 256 / 224)), T.CenterCrop(IMG), T.ToTensor(), T.Normalize(MEAN, STD)])


class CarpetaResiduos(Dataset):
    """Lee data/residuos5/<particion>/<clase>/*.jpg."""

    def __init__(self, raiz, particion, transform):
        self.items = []
        for i, c in enumerate(CLASES):
            for p in sorted((Path(raiz) / particion / c).glob("*.jpg")):
                self.items.append((p, i))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        p, y = self.items[k]
        with Image.open(p) as im:
            x = self.transform(im.convert("RGB"))
        return x, y


def crear_modelo(nombre, dir_pesos=None, num_clases=len(CLASES)):
    """Crea el modelo timm, carga pesos ImageNet (si se indica) y reemplaza el clasificador."""
    cfg = MODELOS[nombre]
    m = timm.create_model(cfg["timm"], pretrained=False)
    if dir_pesos is not None:
        ruta = Path(dir_pesos) / cfg["pesos"]
        if not ruta.exists():
            torch.hub.download_url_to_file(cfg["url"], str(ruta))
        m.load_state_dict(torch.load(ruta, map_location="cpu"))
    m.reset_classifier(num_clases)
    return m


def partes_finetune(nombre, m):
    """Módulos que se descongelan en la fase 2 (últimas etapas + cabeza)."""
    if nombre == "resnet50":
        return [m.layer4, m.fc]
    if nombre == "mobilenetv3":
        return [m.blocks[-2], m.blocks[-1], m.conv_head, m.classifier]
    return [m.blocks[-2], m.blocks[-1], m.conv_head, m.bn2, m.classifier]


def cargar_entrenado(ruta_ckpt):
    ck = torch.load(ruta_ckpt, map_location="cpu")
    m = crear_modelo(ck["modelo"])
    m.load_state_dict(ck["estado"])
    m.eval()
    return m, ck
