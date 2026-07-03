# -*- coding: utf-8 -*-
"""
main.py
=======
Punto de entrada. Para cada uno de los 3 modelos (ResNet50, InceptionV3,
EfficientNet-B3) entrena y evalúa NUM_ITERACIONES=5 veces (igual idea que
en tu ejemplo de evaluate_univar_onestep_mlp: se repite el entrenamiento
completo N veces y se promedian las métricas para tener una media y un
desvío, en vez de quedarte con el resultado de una sola corrida que
podría ser optimista o pesimista por azar de la inicialización).

Los DataLoaders (train/val/test) se construyen UNA sola vez y se
reutilizan en las 5 iteraciones de los 3 modelos: así la única fuente de
variabilidad entre iteraciones es la inicialización aleatoria de la
cabeza + la estocasticidad del entrenamiento (orden de batches, dropout),
no el split de datos -- esto es necesario para que las 5 corridas de un
mismo modelo, y los 3 modelos entre sí, sean comparables.

Salidas (en RESULTADOS_DIR), que luego lee analisis_comparativo.py:
    - resultados_metricas.csv      : MAE/RMSE/MdAE de cada (modelo, iteración)
    - predicciones_<modelo>.csv    : predicciones de la ÚLTIMA iteración
                                      de cada modelo (para scatter / matriz
                                      de confusión IDH)
    - pesos_<modelo>.pth           : pesos de la ÚLTIMA iteración de cada modelo
"""

import os

import pandas as pd
import torch

from prepro import construir_dataloaders
from entrenamiento import entrenar_modelo, evaluar_modelo

# ---------------------------------------------------------------------------
# CONFIGURACIÓN -- AJUSTAR LA RUTA DEL DATASET
# ---------------------------------------------------------------------------
CARPETA_DATASET = './Dataset'   # carpeta con las imágenes + _annotations.coco.json
RESULTADOS_DIR = 'resultados'

MODELOS_A_EVALUAR = ['resnet50', 'inception_v3', 'efficientnet_b3']
NUM_ITERACIONES = 5

EPOCHS_FASE1, LR_FASE1 = 10, 1e-3
EPOCHS_FASE2, LR_FASE2 = 50, 1e-4
PATIENCE = 10


def main():
    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo activo: {device}")

    dataloaders = construir_dataloaders(CARPETA_DATASET)

    metricas_todas = []  # filas: Modelo, Iteracion, MAE, RMSE, MdAE

    print(f"\n>>> INICIANDO PIPELINE: {len(MODELOS_A_EVALUAR)} MODELO(S) "
          f"x {NUM_ITERACIONES} ITERACIONES C/U")

    for nombre in MODELOS_A_EVALUAR:
        print(f"\n{'#' * 70}\nMODELO: {nombre.upper()}\n{'#' * 70}")

        ultima_evaluacion = None  # para guardar predicciones de la última iteración

        for i in range(1, NUM_ITERACIONES + 1):
            print(f"\n--- {nombre.upper()} | Iteración {i}/{NUM_ITERACIONES} ---")

            modelo, _history = entrenar_modelo(
                nombre, dataloaders, device,
                epochs_fase1=EPOCHS_FASE1, lr_fase1=LR_FASE1,
                epochs_fase2=EPOCHS_FASE2, lr_fase2=LR_FASE2,
                patience=PATIENCE,
            )

            mae, rmse, mdae, preds, reales, clases_pred, clases_real = evaluar_modelo(
                modelo, dataloaders['test'], device
            )

            print(f"  Resultado iteración {i}: MAE={mae:.2f} | RMSE={rmse:.2f} | MdAE={mdae:.2f}")

            metricas_todas.append({
                'Modelo': nombre, 'Iteracion': i,
                'MAE': mae, 'RMSE': rmse, 'MdAE': mdae,
            })

            ultima_evaluacion = (preds, reales, clases_pred, clases_real)

            if i == NUM_ITERACIONES:
                torch.save(modelo.state_dict(),
                           os.path.join(RESULTADOS_DIR, f'pesos_{nombre}.pth'))

        # Guarda las predicciones de la última iteración (para los gráficos
        # de dispersión y la matriz de confusión IDH en el análisis comparativo)
        preds, reales, clases_pred, clases_real = ultima_evaluacion
        df_pred = pd.DataFrame({
            'real': reales, 'prediccion': preds,
            'clase_real': clases_real, 'clase_pred': clases_pred,
        })
        df_pred.to_csv(os.path.join(RESULTADOS_DIR, f'predicciones_{nombre}.csv'), index=False)

        # Resumen rápido en consola para este modelo
        df_modelo = pd.DataFrame([m for m in metricas_todas if m['Modelo'] == nombre])
        print(f"\nResumen {nombre.upper()} ({NUM_ITERACIONES} iteraciones):")
        for metrica in ['MAE', 'RMSE', 'MdAE']:
            print(f"  {metrica}: {df_modelo[metrica].mean():.2f} ± {df_modelo[metrica].std():.2f}")

    # Guarda la tabla completa (todas las iteraciones de los 3 modelos)
    df_resultados = pd.DataFrame(metricas_todas)
    df_resultados.to_csv(os.path.join(RESULTADOS_DIR, 'resultados_metricas.csv'), index=False)

    print(f"\n{'=' * 70}\nLISTO. Resultados guardados en '{RESULTADOS_DIR}/'.")
    print("Ejecuta analisis_comparativo.py para ver tablas y gráficos comparativos.")
    print('=' * 70)


if __name__ == '__main__':
    main()
