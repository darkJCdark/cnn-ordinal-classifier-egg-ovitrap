# -*- coding: utf-8 -*-
"""
analisis_umbrales_minsa.py
==========================
Reasigna las predicciones de la última corrida (resultados/predicciones_*.csv)
a tres esquemas de umbrales y compara los modelos, sin reentrenar nada:

    1. Umbrales operativos del proyecto (10 / 50 / 100 huevos por imagen).
    2. Bandas de la Tabla 1 del protocolo del MINSA para el IDH (5 / 20 / 40).
       El IDH oficial es huevos totales / ovitrampas positivas de un área; aquí
       cada imagen se toma como una ovitrampa positiva, lo cual es una
       aproximación.
    3. Puntos calientes por ovitrampa del mismo protocolo (60 / 120 / 150).

Fuente de 2 y 3: MINSA, Protocolo Sanitario de Urgencia para el Reforzamiento
de la Vigilancia Entomológica del Vector Aedes aegypti mediante el Uso de
Ovitrampas (PSU N.° 11, R.M. N.° 010-2015/MINSA), Lima, 2016.

También compara ResNet-50 con EfficientNet-B3 mediante la prueba t de Welch
sobre las cinco corridas de resultados/resultados_metricas.csv.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

RESULTADOS_DIR = 'resultados'
MODELOS = [('resnet50', 'ResNet-50'), ('inception_v3', 'InceptionV3'),
           ('efficientnet_b3', 'EfficientNet-B3')]


def nivel_proyecto(n):
    return 0 if n < 10 else 1 if n < 50 else 2 if n < 100 else 3


def nivel_minsa_idh(n):
    return 0 if n < 5 else 1 if n < 20 else 2 if n < 40 else 3


def nivel_minsa_puntos_calientes(n):
    return 0 if n <= 60 else 1 if n <= 120 else 2 if n <= 150 else 3


ESQUEMAS = [
    ('Proyecto (10/50/100)', nivel_proyecto),
    ('MINSA, IDH (5/20/40; cada imagen = 1 ovitrampa positiva)', nivel_minsa_idh),
    ('MINSA, puntos calientes (60/120/150)', nivel_minsa_puntos_calientes),
]


def comparar_umbrales():
    for titulo, nivel in ESQUEMAS:
        print(f"\n{titulo}")
        for clave, nombre in MODELOS:
            df = pd.read_csv(os.path.join(RESULTADOS_DIR, f'predicciones_{clave}.csv'))
            real = df['real'].map(nivel).to_numpy()
            pred = df['prediccion'].map(nivel).to_numpy()
            print(f"  {nombre:16s} acierto {(real == pred).mean() * 100:5.1f} % | "
                  f"subestima {int((pred < real).sum()):2d} | "
                  f"sobreestima {int((pred > real).sum()):2d} | "
                  f"no adyacentes {int((np.abs(pred - real) > 1).sum()):2d}")


def comparar_corridas():
    df = pd.read_csv(os.path.join(RESULTADOS_DIR, 'resultados_metricas.csv'))
    print("\nWelch, ResNet-50 vs. EfficientNet-B3 (cinco corridas por modelo)")
    for metrica in ['MAE', 'RMSE', 'MdAE']:
        a = df.loc[df['Modelo'] == 'resnet50', metrica]
        b = df.loc[df['Modelo'] == 'efficientnet_b3', metrica]
        print(f"  {metrica:5s} p = {stats.ttest_ind(a, b, equal_var=False).pvalue:.3f}")


if __name__ == '__main__':
    comparar_umbrales()
    comparar_corridas()
