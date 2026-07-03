# -*- coding: utf-8 -*-
"""
modelos.py
==========
Definición de la arquitectura única que soporta los 3 backbones
(ResNet50 / InceptionV3 / EfficientNet-B3) con la misma cabeza de
regresión, más las funciones de control de transfer learning (congelar
encoder / descongelar el último tercio para fine-tuning).

Se usa la API moderna de pesos de torchvision (`weights=...`) en lugar
de `pretrained=True` (deprecado), igual que ya se hizo en el notebook
de EfficientNet-B3.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    Inception_V3_Weights,
    EfficientNet_B3_Weights,
)

NOMBRES_SOPORTADOS = ('resnet50', 'inception_v3', 'efficientnet_b3')


class OvitrapCounterModel(nn.Module):
    """Encoder preentrenado en ImageNet + GAP + cabeza de regresión.

    Cabeza idéntica a la ecuación del documento:
        N̂ = ReLU(w·GAP(Fenc(I)) + b)
    es decir: Linear único (sin Dropout) sobre el GAP, seguido de un
    ReLU final que garantiza N̂ >= 0 de forma explícita. El modelo
    predice el conteo crudo N directamente (ya no log1p(N)); la única
    desviación respecto al documento es la función de pérdida
    (Huber en vez de MSE, ver entrenamiento.py), necesaria para
    estabilizar el entrenamiento con la distribución sesgada de conteos
    (max=543 huevos) sin tener que tocar la cabeza ni el target.
    """

    def __init__(self, model_name='efficientnet_b3', pretrained=True):
        super().__init__()
        if model_name not in NOMBRES_SOPORTADOS:
            raise ValueError(
                f"Modelo no soportado: '{model_name}'. Usa uno de {NOMBRES_SOPORTADOS}."
            )
        self.model_name = model_name

        if model_name == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            base_model = models.resnet50(weights=weights)
            # children(): conv1, bn1, relu, maxpool, layer1..layer4, avgpool, fc
            # quitamos avgpool y fc -> nos quedamos solo con los bloques conv.
            self.encoder = nn.Sequential(*list(base_model.children())[:-2])
            in_features = 2048

        elif model_name == 'inception_v3':
            weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
            # En versiones recientes de torchvision, pasar aux_logits=False
            # junto con `weights` pretrained lanza ValueError (antes de
            # cargar el checkpoint, torchvision fuerza aux_logits=True
            # internamente y antes simplemente lo sobrescribía en silencio;
            # ahora compara y revienta si el valor explícito no coincide).
            # Construimos con el default (aux_logits=True, forzado igual
            # mientras haya weights) y deshabilitamos la rama auxiliar
            # nosotros mismos justo después -- mismo resultado final que se
            # buscaba, sin pasar por el camino que ahora da error.
            base_model = models.inception_v3(weights=weights)
            base_model.aux_logits = False
            base_model.AuxLogits = None  # se quita de children(): ya no queda "suelto" en medio de la secuencia
            self.encoder = nn.Sequential(*list(base_model.children())[:-2])
            in_features = 2048

        else:  # efficientnet_b3
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.efficientnet_b3(weights=weights)
            self.encoder = base_model.features
            in_features = 1536

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.gap(x)
        x = self.regressor(x)
        x = torch.relu(x)  # N̂ = ReLU(w·GAP(Fenc(I))+b) -- igual que el documento
        return x

    # -- utilidades de transfer learning, atadas al propio modelo ----------
    def congelar_encoder(self):
        """Fase 1: congela TODO el encoder; solo entrena la cabeza."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def descongelar_ultimo_tercio(self):
        """Fase 2: descongela el último tercio de los bloques del encoder
        (mismo criterio para los 3 backbones, vía sus 'children()').

        Para efficientnet_b3 esto descongela los bloques 6, 7 y 8 de 9 --
        exactamente el mismo criterio que se justificó explícitamente en el
        notebook de EfficientNet-B3 ("último tercio de los 9 bloques")."""
        hijos = list(self.encoder.children())
        total = len(hijos)
        inicio = int(total * (2 / 3))
        for i, bloque in enumerate(hijos):
            if i >= inicio:
                for param in bloque.parameters():
                    param.requires_grad = True

    def parametros_entrenables(self):
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        return n_trainable, n_total


def crear_modelo(model_name, pretrained=True, device=None):
    """Fábrica simple: crea el modelo y opcionalmente lo manda a `device`."""
    modelo = OvitrapCounterModel(model_name=model_name, pretrained=pretrained)
    if device is not None:
        modelo = modelo.to(device)
    return modelo
