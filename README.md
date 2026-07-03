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

Para solucionar esta problemática, este repositorio contiene una solución basada en **Visión Computacional y Deep Learning** que automatiza el conteo de huevos y clasifica la infestación según el **Índice de Densidad de Huevos (IDH)** establecido por el Ministerio de Salud del Perú (MINSA): Bajo, Medio, Alto y Muy Alto.

## 🎯 Objetivos
* **Objetivo General:** Comparar el desempeño de tres arquitecturas CNN con transfer learning (ResNet-50, InceptionV3 y EfficientNet-B3) para el conteo de huevos de *Aedes aegypti* y su clasificación según el riesgo epidemiológico (IDH).
* **Objetivos Específicos:**
  * Preprocesar y adaptar un dataset de imágenes de ovitrampas (de un laboratorio brasileño) utilizando técnicas de aumento de datos (*Data Augmentation*) para evitar el sobreajuste.
  * Entrenar los modelos empleando *fine-tuning* con una capa de regresión adaptada.
  * Evaluar el desempeño utilizando métricas de regresión numéricas (MAE, RMSE, MdAE) y matrices de confusión para validar el riesgo epidemiológico.
  * Identificar el modelo más eficiente y adecuado para su implementación en zonas de escasos recursos.

---

## 🧠 Arquitecturas Evaluadas
Se aplicó *Transfer Learning* sobre tres modelos preentrenados, adaptando su última capa para la tarea de regresión (conteo continuo) y mapeando luego dichos resultados a umbrales ordinales (IDH).
1. **ResNet-50**
2. **InceptionV3**
3. **EfficientNet-B3**

---

## 📊 Resultados Principales
El análisis comparativo arrojó las siguientes conclusiones determinantes:
* ❌ **InceptionV3:** Resultó ser el modelo más inestable del estudio, presentando un Error Absoluto Medio (MAE) de **24.78** y subestimando drásticamente densidades altas.
* ⚖️ **ResNet-50:** Obtuvo el menor error global estándar, alcanzando un MAE de **14.12**, lo que lo hace muy consistente en densidades bajas y medias.
* 🏆 **EfficientNet-B3 (Mejor Modelo):** Demostró superioridad técnica y clínica frente a las densidades críticas (clústeres masivos de huevos). En escenarios extremos (ej. solapamiento real de 382 huevos), EfficientNet logró predecir 339 huevos, mostrando el mejor balance operativo. 

**Validación Epidemiológica:** Las desviaciones matemáticas de EfficientNet-B3 ocurren exclusivamente entre categorías de IDH adyacentes (ej. de Medio a Bajo), evitando **falsos negativos extremos** (ej. de Muy Alto a Bajo). Esto garantiza que, operativamente, el sistema detone las alertas epidemiológicas correctas sin comprometer la salud pública.

---

## 💻 Requisitos e Instalación

```bash
# Clonar este repositorio
git clone [https://github.com/darkJCdark/cnn-ordinal-classifier-egg-ovitrap](https://github.com/darkJCdark/cnn-ordinal-classifier-egg-ovitrap)
cd repo-aedes-aegypti

# Crear entorno virtual e instalar dependencias
pip install -r requirements.txt
