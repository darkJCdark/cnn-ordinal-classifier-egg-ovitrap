# -*- coding: utf-8 -*-
"""
entrenamiento.py
================
Bucle de entrenamiento en 2 fases (cabeza congelada -> fine-tuning del
último tercio del encoder, con early stopping) y evaluación final
(MAE / RMSE / MdAE + clasificación IDH según los rangos del MINSA).

El modelo (ver modelos.py) predice el conteo crudo N̂ directamente
(cabeza = Linear + ReLU final, sin Dropout, sin log1p), igual que la
ecuación del documento. La ÚNICA desviación intencional respecto al
documento es la función de pérdida: Huber en vez de MSE. Con conteos
de hasta ~500 huevos y mediana ~17, MSE crudo le da un gradiente enorme
a las pocas imágenes de alta densidad y empuja a los 3 modelos a
predecir básicamente la media; Huber es cuadrática para errores
pequeños (mismo comportamiento que MSE ahí) y lineal para errores
grandes, evitando que esas imágenes dominen el entrenamiento.

ATENCIÓN - discrepancia que encontré entre tus 2 scripts y que dejo
explícita en vez de decidir en silencio:
    - El notebook de EfficientNet-B3 usaba umbrales IDH (10, 50, 150).
    - Tu script de ResNet consolidado usaba (10, 50, 100).
  Ambos son "placeholders" marcados en el código original como
  pendientes de ajustar contra la norma técnica del MINSA. Aquí dejo
  (10, 50, 100) por ser el del script más reciente, pero como
  TAU1/TAU2/TAU3 al inicio del archivo para que los corrijas en un solo
  lugar apenas tengas los valores oficiales.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error

from modelos import crear_modelo

# ---------------------------------------------------------------------------
# Umbrales IDH (MINSA) -- AJUSTAR con los valores exactos de la norma técnica
# ---------------------------------------------------------------------------
TAU1, TAU2, TAU3 = 10, 50, 100  # Bajo < TAU1 <= Medio < TAU2 <= Alto < TAU3 <= Muy Alto


def asignar_clase_idh(n):
    """Clasificación determinística post-hoc sobre el conteo N̂ (no es un
    modelo adicional, solo umbralización)."""
    if n < TAU1:
        return 'Bajo'
    elif n < TAU2:
        return 'Medio'
    elif n < TAU3:
        return 'Alto'
    else:
        return 'Muy Alto'


# ---------------------------------------------------------------------------
# ENTRENAMIENTO EN 2 FASES
# ---------------------------------------------------------------------------
def entrenar_modelo(model_name, dataloaders, device,
                     epochs_fase1=10, lr_fase1=1e-3,
                     epochs_fase2=50, lr_fase2=1e-4,
                     patience=10, pretrained=True):
    """Crea un modelo nuevo (`model_name`) y lo entrena desde cero en 2 fases:

    Fase 1 (encoder congelado): se entrena solo la cabeza de regresión
            durante `epochs_fase1` épocas con `lr_fase1`, sin scheduler
            (calentamiento corto, no conviene que el LR baje aquí).
    Fase 2 (fine-tuning): se descongela el último tercio del encoder y se
            entrena hasta `epochs_fase2` épocas con `lr_fase2`,
            ReduceLROnPlateau(factor=0.5, patience=3) y early stopping
            (patience=`patience`) monitoreando val_loss; al final se
            restauran los pesos de la mejor época.

    Devuelve (modelo_entrenado, history) donde history tiene las listas
    train_loss/val_loss/train_mae/val_mae concatenando ambas fases.
    """
    print(f"\n{'=' * 60}\nENTRENANDO MODELO: {model_name.upper()}\n{'=' * 60}")

    model = crear_modelo(model_name, pretrained=pretrained, device=device)
    # Única desviación respecto al documento: Huber en vez de MSE (ver
    # docstring del módulo). delta=1.0 (default) -> cuadrática para
    # errores <1 huevo, lineal para el resto; si en la práctica el
    # gradiente queda demasiado chico frente a lr_fase1/lr_fase2, subir
    # delta (p.ej. 5-10) es el primer ajuste a probar.
    criterion = nn.HuberLoss()

    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}

    # --- FASE 1: cabeza, encoder congelado ---------------------------------
    print("-> Fase 1: encoder congelado, entrenando solo la cabeza...")
    model.congelar_encoder()
    optimizer = optim.Adam(model.regressor.parameters(), lr=lr_fase1)

    for epoch in range(1, epochs_fase1 + 1):
        train_loss, train_mae = _run_epoch(model, dataloaders['train'], criterion, optimizer, device)
        val_loss, val_mae = _run_epoch(model, dataloaders['val'], criterion, None, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        print(f"  [Fase 1] Epoch {epoch:3d}/{epochs_fase1} | "
              f"train_loss={train_loss:.3f} train_mae={train_mae:.2f} | "
              f"val_loss={val_loss:.3f} val_mae={val_mae:.2f}")

    # --- FASE 2: fine-tuning del último tercio, con early stopping ---------
    print("-> Fase 2: descongelando el último tercio del encoder...")
    model.descongelar_ultimo_tercio()
    n_trainable, n_total = model.parametros_entrenables()
    print(f"   Parámetros entrenables: {n_trainable:,} / {n_total:,}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_fase2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    epochs_sin_mejora = 0

    for epoch in range(1, epochs_fase2 + 1):
        train_loss, train_mae = _run_epoch(model, dataloaders['train'], criterion, optimizer, device)
        val_loss, val_mae = _run_epoch(model, dataloaders['val'], criterion, None, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        print(f"  [Fase 2] Epoch {epoch:3d}/{epochs_fase2} | "
              f"train_loss={train_loss:.3f} train_mae={train_mae:.2f} | "
              f"val_loss={val_loss:.3f} val_mae={val_mae:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_sin_mejora = 0
        else:
            epochs_sin_mejora += 1
            if epochs_sin_mejora >= patience:
                print(f"   Early stopping en epoch {epoch} (sin mejora en {patience} épocas)")
                break

    model.load_state_dict(best_state)
    return model, history


def _run_epoch(model, loader, criterion, optimizer, device):
    """Una pasada de train (si optimizer no es None) o eval. Devuelve
    (huber_loss, mae), ambos ya en escala real (conteo crudo) porque el
    modelo predice N̂ directamente (ReLU final, sin log1p/expm1)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_mae, n = 0.0, 0.0, 0

    with torch.set_grad_enabled(is_train):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().view(-1, 1)

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = X_batch.size(0)
            total_loss += loss.item() * bs
            total_mae += torch.abs(preds - y_batch).sum().item()
            n += bs

    return total_loss / n, total_mae / n


# ---------------------------------------------------------------------------
# EVALUACIÓN FINAL (test) + CLASIFICACIÓN IDH
# ---------------------------------------------------------------------------
def evaluar_modelo(model, dataloader, device):
    """Evalúa en el dataloader (normalmente 'test'). El modelo ya predice
    conteo crudo (ReLU final), así que no hace falta revertir log1p/expm1.
    Devuelve (mae, rmse, mdae, preds, reales, clases_pred, clases_real)."""
    model.eval()
    preds_all, reales_all = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy().flatten()
            reales = y_batch.float().numpy().flatten()

            preds_all.extend(preds)
            reales_all.extend(reales)

    preds_all = np.clip(np.array(preds_all), a_min=0, a_max=None)  # red de seguridad; ReLU ya lo garantiza
    reales_all = np.array(reales_all)

    mae = mean_absolute_error(reales_all, preds_all)
    rmse = np.sqrt(mean_squared_error(reales_all, preds_all))
    mdae = np.median(np.abs(preds_all - reales_all))

    clases_pred = [asignar_clase_idh(n) for n in preds_all]
    clases_real = [asignar_clase_idh(n) for n in reales_all]

    return mae, rmse, mdae, preds_all, reales_all, clases_pred, clases_real
