# -*- coding: utf-8 -*-
"""
analisis_comparativo.py
========================
Lee lo que dejó main.py en RESULTADOS_DIR y genera:

    1. Tabla final (media ± std de MAE/RMSE/MdAE) por modelo, sobre las
       NUM_ITERACIONES corridas de cada uno.
    2. Boxplot POR MODELO de sus 3 métricas a lo largo de las iteraciones
       (variabilidad interna de cada modelo -- lo que ya tenías).
    3. Boxplot COMPARATIVO ENTRE LOS 3 MODELOS, una figura por métrica
       (MAE / RMSE / MdAE), cada una con 3 cajas lado a lado -- esto es
       lo que faltaba para una comparación real entre arquitecturas
       (antes solo había una tabla de texto con medias y un boxplot por
       separado de cada modelo, pero nunca los 3 modelos en el mismo
       gráfico para la misma métrica).
    4. Dispersión (real vs. predicho) y matriz de confusión IDH por
       modelo, usando las predicciones de la ÚLTIMA iteración de cada uno.

No necesita PyTorch ni reentrenar nada: solo lee los CSV generados por
main.py, así que puedes correrlo cuantas veces quieras para ajustar los
gráficos sin volver a entrenar.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

RESULTADOS_DIR = 'resultados'   # debe coincidir con el de main.py
MODELOS = ['resnet50', 'inception_v3', 'efficientnet_b3']
ETIQUETAS_IDH = ['Bajo', 'Medio', 'Alto', 'Muy Alto']


# ---------------------------------------------------------------------------
# CARGA DE RESULTADOS
# ---------------------------------------------------------------------------
def cargar_resultados(resultados_dir=RESULTADOS_DIR):
    df_metricas = pd.read_csv(os.path.join(resultados_dir, 'resultados_metricas.csv'))

    predicciones = {}
    for nombre in MODELOS:
        path = os.path.join(resultados_dir, f'predicciones_{nombre}.csv')
        if os.path.exists(path):
            predicciones[nombre] = pd.read_csv(path)
        else:
            print(f"Aviso: no se encontró '{path}' (¿corriste main.py con este modelo?).")

    return df_metricas, predicciones


# ---------------------------------------------------------------------------
# 1. TABLA FINAL (media ± std por modelo)
# ---------------------------------------------------------------------------
def tabla_resumen(df_metricas):
    filas = []
    for nombre in df_metricas['Modelo'].unique():
        df_m = df_metricas[df_metricas['Modelo'] == nombre]
        fila = {'Modelo': nombre.upper()}
        for metrica in ['MAE', 'RMSE', 'MdAE']:
            fila[f'{metrica} (media ± std)'] = (
                f"{df_m[metrica].mean():.2f} ± {df_m[metrica].std():.2f}"
            )
        filas.append(fila)

    df_resumen = pd.DataFrame(filas)
    print("\n" + "=" * 80)
    print("TABLA COMPARATIVA FINAL (media ± std sobre las iteraciones)")
    print("=" * 80)
    print(df_resumen.to_string(index=False))
    return df_resumen


# ---------------------------------------------------------------------------
# 2. BOXPLOT POR MODELO (variabilidad interna de sus propias iteraciones)
# ---------------------------------------------------------------------------
def boxplot_por_modelo(df_metricas):
    for nombre in df_metricas['Modelo'].unique():
        df_m = df_metricas[df_metricas['Modelo'] == nombre]
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df_m[['MAE', 'RMSE', 'MdAE']], palette="Set2")
        plt.title(f'Variabilidad de métricas en {len(df_m)} iteraciones - {nombre.upper()}',
                   fontsize=13, fontweight='bold')
        plt.ylabel('Error (huevos)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# 3. BOXPLOT COMPARATIVO ENTRE LOS 3 MODELOS (una figura por métrica)
# ---------------------------------------------------------------------------
def boxplot_comparativo_entre_modelos(df_metricas):
    df_plot = df_metricas.copy()
    df_plot['Modelo'] = df_plot['Modelo'].str.upper()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Comparación entre arquitecturas (todas las iteraciones)',
                  fontsize=15, fontweight='bold')

    for ax, metrica in zip(axes, ['MAE', 'RMSE', 'MdAE']):
        sns.boxplot(data=df_plot, x='Modelo', y=metrica, hue='Modelo',
                    palette="Set2", legend=False, ax=ax)
        sns.stripplot(data=df_plot, x='Modelo', y=metrica, color='black',
                       alpha=0.5, size=4, ax=ax)
        ax.set_title(metrica)
        ax.set_xlabel('')
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. DISPERSIÓN + MATRIZ DE CONFUSIÓN IDH (última iteración de cada modelo)
# ---------------------------------------------------------------------------
def graficos_por_modelo(predicciones):
    for nombre, df_pred in predicciones.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Resultados en test (última iteración): {nombre.upper()}',
                      fontsize=15, fontweight='bold')

        reales = df_pred['real'].values
        preds = df_pred['prediccion'].values

        # A. Dispersión real vs. predicho
        axes[0].scatter(reales, preds, alpha=0.6, color='#1f77b4', edgecolor='black')
        max_val = max(reales.max(), preds.max()) if len(reales) else 100
        axes[0].plot([0, max_val], [0, max_val], color='red', linestyle='--',
                      label='Predicción perfecta')
        axes[0].set_title('Conteo real vs. predicción')
        axes[0].set_xlabel('Huevos reales (anotados)')
        axes[0].set_ylabel('Huevos predichos')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # B. Matriz de confusión IDH
        cm = confusion_matrix(df_pred['clase_real'], df_pred['clase_pred'],
                               labels=ETIQUETAS_IDH)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=ETIQUETAS_IDH, yticklabels=ETIQUETAS_IDH, ax=axes[1])
        axes[1].set_title('Matriz de confusión (riesgo IDH)')
        axes[1].set_xlabel('Riesgo predicho')
        axes[1].set_ylabel('Riesgo real')

        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    df_metricas, predicciones = cargar_resultados()

    tabla_resumen(df_metricas)
    boxplot_por_modelo(df_metricas)
    boxplot_comparativo_entre_modelos(df_metricas)
    graficos_por_modelo(predicciones)


if __name__ == '__main__':
    main()
