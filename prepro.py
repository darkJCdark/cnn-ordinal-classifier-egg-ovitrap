# -*- coding: utf-8 -*-
"""
prepro.py
=========
Preprocesamiento y carga de datos para el conteo de huevos de Aedes aegypti.

Unifica el pipeline que ya estaba validado de forma dispersa en los 3
notebooks originales (ResNet50 / InceptionV3 / EfficientNet-B3):

    1. Lectura del dataset en formato COCO (export de Roboflow).
    2. Split train/valid/test (70/20/10) sobre las imágenes ORIGINALES.
    3. Augmentation ONLINE (a nivel de tensor, vía torchvision.transforms):
       flips, rotación ±20°, ColorJitter ±15% y recorte aleatorio.
    4. Normalización ImageNet (mean/std de torchvision), la misma que se
       validó como correcta en el notebook de EfficientNet-B3. Es válida
       para los 3 backbones porque los tres están preentrenados en
       ImageNet con esa misma normalización en torchvision.

Nota sobre una decisión de diseño (a propósito, no es un descuido):
en los notebooks originales de InceptionV3 y EfficientNet-B3 el
augmentation era OFFLINE (se generaban 6 copias físicas por imagen de
train, con augmentation aplicado también a las bounding boxes). En tu
script de ResNet consolidado (el que ya soporta los 3 modelos y ya hace
5 iteraciones) el augmentation pasó a ser ONLINE vía
torchvision.transforms, y las bounding boxes ya no se necesitan porque
el target es solo el conteo (N), no la detección. Aquí seguimos
ESE enfoque (el más reciente) por ser más simple e idiomático en
PyTorch. Si en realidad querías mantener el augmentation offline con
bboxes, dilo y lo recupero del notebook de EfficientNet.

Bug corregido: la Tabla 1 pide "recorte aleatorio 90-100% del área,
relleno por reflexión". La versión anterior aplicaba Resize(224,224)
antes de RandomCrop(224,224); como ambos tamaños eran idénticos el
crop nunca recortaba nada (la salida era pixel-idéntica a la entrada).
Ahora se usa RandomResizedCrop(scale=(0.90, 1.0)), que recorta un área
aleatoria de 90-100% de la imagen ORIGINAL y la reescala directamente a
224x224 -- sin padding, como se pidió.
"""

import os
import json
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SEED = 42

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)          # mismo tamaño para los 3 backbones -> comparación justa
PCT_TRAIN, PCT_VALID, PCT_TEST = 0.70, 0.20, 0.10
BATCH_SIZE = 16

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Augmentation "fuerte" solo en train; val/test solo Resize + Normalize.
train_transforms = transforms.Compose([
    # Recorta un área aleatoria de 90-100% de la imagen ORIGINAL y la
    # reescala a 224x224 (sin padding) -- Tabla 1. Reemplaza al
    # Resize+RandomCrop anterior, que no recortaba nada porque ambos
    # tamaños eran idénticos.
    transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.90, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# 1. LECTURA DEL DATASET COCO
# ---------------------------------------------------------------------------
def cargar_coco(carpeta_dataset):
    """Lee `_annotations.coco.json` desde carpeta_dataset."""
    json_path = os.path.join(carpeta_dataset, '_annotations.coco.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"No se encontró el archivo de anotaciones en: {json_path}. "
            "Verifica la ruta de carpeta_dataset."
        )
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    print(f"Dataset cargado: {len(coco_data['images'])} imágenes, "
          f"{len(coco_data['annotations'])} huevos anotados.")
    return coco_data


# ---------------------------------------------------------------------------
# 2. SPLIT TRAIN/VALID/TEST (sobre la lista de imágenes, sin copiar archivos)
# ---------------------------------------------------------------------------
def dividir_dataset(coco_data, pct_train=PCT_TRAIN, pct_valid=PCT_VALID, seed=SEED):
    """Devuelve (lista_train, lista_val, lista_test): listas de entradas
    'images' del COCO. No mueve ni copia archivos en disco; el split es
    solo sobre la lista en memoria (las imágenes siguen leyéndose de
    carpeta_dataset). Semilla fija -> mismas imágenes de test siempre,
    para que las 5 iteraciones de cada modelo (y los 3 modelos entre sí)
    sean comparables."""
    assert abs(pct_train + pct_valid + PCT_TEST - 1.0) < 1e-6, \
        "Los porcentajes train/valid/test deben sumar 1.0"

    random.seed(seed)
    imagenes = list(coco_data['images'])
    random.shuffle(imagenes)

    n_total = len(imagenes)
    n_train = int(n_total * pct_train)
    n_valid = int(n_total * pct_valid)

    lista_train = imagenes[:n_train]
    lista_val = imagenes[n_train:n_train + n_valid]
    lista_test = imagenes[n_train + n_valid:]

    print(f"Split -> train: {len(lista_train)} | valid: {len(lista_val)} | "
          f"test: {len(lista_test)}")
    return lista_train, lista_val, lista_test


# ---------------------------------------------------------------------------
# 3. DATASET DE PYTORCH (imagen -> tensor, etiqueta -> conteo crudo)
# ---------------------------------------------------------------------------
class OvitrapDataset(Dataset):
    """Adaptador COCO -> PyTorch para el problema de conteo.

    image_list: lista de entradas 'images' del COCO (ya filtradas al
                split correspondiente por dividir_dataset()).
    annotations: lista COMPLETA de 'annotations' del COCO (se filtra
                internamente por image_id, así que se puede pasar la
                misma lista completa a los 3 splits).
    """

    def __init__(self, img_dir, image_list, annotations, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = image_list

        self.counts = {img['id']: 0 for img in self.images}
        for ann in annotations:
            if ann['image_id'] in self.counts:
                self.counts[ann['image_id']] += 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        count = self.counts[img_info['id']]
        label = np.float32(count)  # conteo crudo N (el modelo predice N̂ directo, sin log1p)
        return image, label


# ---------------------------------------------------------------------------
# 4. PIPELINE COMPLETO -> DATALOADERS (lo único que necesita llamar main.py)
# ---------------------------------------------------------------------------
def construir_dataloaders(carpeta_dataset, batch_size=BATCH_SIZE, seed=SEED):
    """Ejecuta lectura + split + construcción de los 3 DataLoaders
    (train/val/test), listos para entrenar cualquiera de los 3 backbones."""
    coco_data = cargar_coco(carpeta_dataset)
    lista_train, lista_val, lista_test = dividir_dataset(coco_data, seed=seed)

    dataset_train = OvitrapDataset(carpeta_dataset, lista_train,
                                    coco_data['annotations'], transform=train_transforms)
    dataset_val = OvitrapDataset(carpeta_dataset, lista_val,
                                  coco_data['annotations'], transform=val_transforms)
    dataset_test = OvitrapDataset(carpeta_dataset, lista_test,
                                   coco_data['annotations'], transform=val_transforms)

    dataloaders = {
        'train': DataLoader(dataset_train, batch_size=batch_size, shuffle=True),
        'val':   DataLoader(dataset_val, batch_size=batch_size, shuffle=False),
        'test':  DataLoader(dataset_test, batch_size=batch_size, shuffle=False),
    }

    print(f"DataLoaders listos -> train: {len(dataset_train)} | "
          f"val: {len(dataset_val)} | test: {len(dataset_test)}")
    return dataloaders
