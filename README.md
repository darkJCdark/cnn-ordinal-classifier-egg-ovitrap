# Clasificación ordinal de huevos de Aedes aegypti en ovitrampas: Un análisis comparativo mediante CNN 🦟

## 👥 Autores
* Rodrigo Carbajal
* Jorge Chamorro
* Pedro Chavez
* Nassim Ramirez
* Gonzalo Ipanaque

---

## 📝 Descripción del Proyecto
Este proyecto aborda la complejidad de la vigilancia entomológica manual del mosquito transmisor del dengue (*Aedes aegypti*). Actualmente, el proceso de conteo de huevos depositados en paletas recolectoras (ovitrampas) es lento, agotador y propenso a errores humanos, además de requerir equipos costosos como microscopios.

Para solucionar esta problemática, este repositorio contiene una solución modular basada en **Visión Computacional y Deep Learning** implementada en **PyTorch** y **TensorFlow/Keras** que automatiza el conteo de huevos mediante regresión directa y clasifica el nivel de infestación según el **Índice de Densidad de Huevos (IDH)** alineado con los criterios del Ministerio de Salud del Perú (MINSA): Bajo, Medio, Alto y Muy Alto.

---

## 🎯 Objetivos
* **Objetivo General:** Comparar el desempeño de tres arquitecturas CNN con transfer learning (*ResNet-50*, *InceptionV3* y *EfficientNet-B3*) para el conteo automatizado de huevos de *Aedes aegypti* y su posterior mapeo al riesgo epidemiológico (IDH).
* **Objetivos Específicos:**
  * Preprocesar y adaptar un dataset de imágenes de ovitrampas utilizando técnicas de aumento de datos en línea (*Online Data Augmentation*) para evitar el sobreajuste.
  * Diseñar e implementar un esquema de entrenamiento en dos fases (*Freeze & Fine-tuning*) acoplando una cabeza de regresión lineal con activación ReLU final.
  * Evaluar rigurosamente la estabilidad de los modelos mediante múltiples iteraciones estadísticas usando métricas numéricas (MAE, RMSE, MdAE) y matrices de confusión sanitarias.
  * Identificar el modelo con mayor balance entre consistencia matemática y robustez operativa para su despliegue en entornos de vigilancia real.

---

## 📂 Estructura del Repositorio
Basado en la organización del código fuente, el proyecto está estructurado de la siguiente manera:

```text
├── Dataset/                      # Dataset original en formato COCO exportado de Roboflow
│   └── _annotations.coco.json    # Archivo de anotaciones JSON
├── train/                        # Imágenes destinadas al entrenamiento (70%)
├── valid/                        # Imágenes destinadas a la validación (20%)
├── test/                         # Imágenes destinadas a la prueba final (10%)
├── prepro.py                     # Carga de datos, división aleatoria y Data Augmentation Online
├── modelos.py                    # Definición de la arquitectura unificada y control de capas congeladas
├── entrenamiento.py              # Lógica del bucle de entrenamiento (2 Fases), Huber Loss e IDH
├── main.py                       # Punto de entrada para entrenar/evaluar los 3 modelos (5 iteraciones)
├── analisis_comparativo.py       # Post-procesamiento y generación automática de gráficos estadísticos
├── estructura.txt                # Registro detallado de la jerarquía de archivos
└── resultados/                   # Almacenamiento de logs, pesos y gráficos generados
    ├── resultados_metricas.csv   # Tabla consolidada con el MAE/RMSE/MdAE de todas las corridas
    ├── predicciones_<modelo>.csv # Predicciones detalladas de la última iteración
    ├── pesos_<modelo>.pth        # Pesos guardados del mejor estado de cada arquitectura
    └── Metricas/                 # Carpeta visual con los resultados gráficos
        ├── Boxplot-ALL.png       # Boxplot comparativo global entre las 3 redes
        ├── Boxplot-<MODELO>.png  # Variabilidad interna por métrica de cada arquitectura
        └── Real-Predicho_Matriz-de-Confusion_<MODELO>.png
