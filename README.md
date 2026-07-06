# Ordinal Classification of Aedes aegypti Eggs in Ovitraps: A Comparative CNN-Based Analysis 🦟

## 👥 Authors
* Rodrigo Carbajal
* Jorge Chamorro
* Pedro Chavez
* Nassim Ramirez
* Gonzalo Ipanaque

---

## 📝 Project Description
This repository contains a modular deep learning framework implemented in **PyTorch** designed to automate entomological surveillance for dengue vector control. Traditionally, the quantification of *Aedes aegypti* eggs deposited on wooden ovitrap paddles is conducted manually by public health technicians. This process is highly labor-intensive, slow, prone to fatigue-induced human error, and dependent on specialized laboratory hardware (microscopes).

To optimize this critical task, this project presents an automated computer vision solution that counts eggs via direct regression and translates these numbers into actionable epidemiological risk categories based on the **Egg Density Index (IDH / EDI)** framework set by the Ministry of Health of Peru (MINSA): **Low, Medium, High, and Very High**.

---

## 🎯 Core Objectives
* **General Objective:** Perform a rigorous comparative evaluation of three pre-trained Convolutional Neural Network (CNN) architectures (*ResNet-50*, *InceptionV3*, and *EfficientNet-B3*) adapted for direct regression egg counting and subsequent post-hoc ordinal mapping.
* **Specific Objectives:**
  * Engineer a high-performance input pipeline with **Online Data Augmentation** to prevent overfitting and handle extreme class imbalance and density variations.
  * Implement a standardized, reproducible **Two-Phase Training Protocol (Freeze & Fine-tuning)** across all network backbones.
  * Evaluate model stability by executing multiple statistical trials, measuring error metrics (MAE, RMSE, MdAE), and analyzing clinical confusion matrices.
  * Identify the most robust architecture suitable for deployment in resource-limited sanitary jurisdictions.

---

## 📂 Repository Architecture
The workspace is structured into specialized modules to separate data preparation, model construction, execution, and graphical statistical reporting:


```
├── Dataset/                      # Raw image data in COCO format (exported from Roboflow)
│   └── _annotations.coco.json    # Object detection bounding box and metadata file
├── train/                        # Directory for training split images (70%)
├── valid/                        # Directory for validation split images (20%)
├── test/                         # Directory for final test split images (10%)
├── prepro.py                     # Metadata parser, dataset partitioner, and Online Augmentation
├── modelos.py                    # Unified network factory with structural regression heads
├── entrenamiento.py              # Two-phase training loops, Huber Loss criteria, and evaluation logic
├── main.py                       # Experimental orchestrator (executes 5 cross-validation runs per network)
├── analisis_comparativo.py       # Post-processing analytics script (offline plotting and metric aggregation)
├── estructura.txt                # Repository file layout descriptor
└── resultados/                   # Directory for training artifacts, logs, and graphics
    ├── resultados_metricas.csv   # Aggregated evaluation metrics for all 15 experimental runs
    ├── predicciones_<model>.csv  # Per-sample predictions from the final execution run
    ├── pesos_<model>.pth         # Optimal state serialization (best validation loss checkpoint)
    └── Metricas/                 # Visual reporting folder
        ├── Boxplot-ALL.png       # Comparative boxplot illustrating model error distributions
        ├── Boxplot-<model>.png   # Metric variance charts per individual architecture
        └── Real-Predicho_Matriz-de-Confusion_<model>.png # Scatter plots and IDH confusion matrices

```

---

## ⚙️ End-to-End Deep Learning Pipeline

The project implements a streamlined, fully automated pipeline that transitions from raw image data to epidemiological risk assessment charts.

```text
 [ Raw Dataset (COCO) ] ──> [ Deterministic Split (70/20/10) ]
                                            │
                                            ▼
 [ Statistical Evaluation ] <── [ Preprocessing & Online Augmentation ]
            │                               │
            ▼                               ▼
 [ IDH Confusion Matrices ] <── [ Two-Phase Training (Huber Loss) ]

```

### 1. Data Ingestion & Partitioning (`prepro.py`)

* **Metadata Parsing:** The script ingests `_annotations.coco.json`, mapping each image identifier to its corresponding ground-truth egg count $N$ by aggregating its bounding box instances.
* **Proportional Splitting:** Images are stratified and split into **Training (70%)**, **Validation (20%)**, and **Testing (10%)** sets. Crucially, the data partitioning is decoupled from the data loaders and fixed using a constant random seed (`SEED = 42`) to ensure that all models are exposed to the exact same data distributions.

### 2. On-the-Fly Tensor Augmentation (`prepro.py`)

To ensure robustness against variations in field capture environments (illumination shifts, varying camera sensors, structural noise on paddles), the training data loader applies **Online Data Augmentation** at the tensor level using `torchvision.transforms`:

* **Spatial Transformations:** Random horizontal and vertical flips combined with a random rotation of $\pm20^\circ$.
* **Photometric Alterations:** Color jittering (*ColorJitter*) modifying brightness, contrast, saturation, and hue by $\pm15\%$.
* **Scale Invariance:** Random cropping to force features to be location-independent.
* **Standardized Normalization:** Pixel arrays are scaled and normalized utilizing the strict ImageNet channel-wise distribution parameters:

$$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$



### 3. Unified Regression Architecture & Structural Safety Net (`modelos.py`)

To isolate performance differences to the backbone architectures alone, all models utilize a mathematically identical, customized regression network head:


$$\hat{N} = \text{ReLU}\left(W \cdot \text{GAP}(F_{enc}(I)) + b\right)$$

* **Feature Extraction ($F_{enc}$):** Pre-trained ImageNet weights are loaded onto the selected encoder (*ResNet-50*, *InceptionV3*, or *EfficientNet-B3*).
* **Dimensionality Reduction (GAP):** A *Global Average Pooling* layer flattens the multidimensional feature maps into a 1D vector while preserving spatial activation intensity.
* **Linear Regression Layer:** A single dense layer (`nn.Linear`) maps features directly to a continuous numerical prediction without intermediate Dropout layers to maintain gradient flow integrity.
* **Structural Safety Net (ReLU):** A final *Rectified Linear Unit (ReLU)* activation is appended at the output. This guarantees that predicted values are strictly non-negative ($\hat{N} \ge 0$), matching the physical reality of egg counts.

### 4. Dual-Phase Fine-Tuning Strategy (`entrenamiento.py`)

Training is structured into two distinct operational phases to avoid devastating pre-trained weight distortion:

* **Phase 1: Encoder Warm-up (10 Epochs):** All weights within the convolutional backbone ($F_{enc}$) are frozen. Optimization is restricted exclusively to the regression head ($W, b$). A high learning rate (`LR = 1e-3`) is managed by the Adam optimizer to establish a stable starting trajectory.
* **Phase 2: Targeted Fine-Tuning (50 Epochs):** The **last third of the encoder blocks** is unfrozen (e.g., blocks 6, 7, and 8 for *EfficientNet-B3*). Optimization proceeds with a conservative learning rate (`LR = 1e-4`) under a `ReduceLROnPlateau` scheduler (decay factor $= 0.5$, patience $= 3$ epochs). An **Early Stopping** mechanism monitors validation loss with a patience window of 10 epochs to prevent overfitting.

### 5. Robust Optimization: Huber Loss Criteria

The dataset exhibits extreme skewness: while the median count is ~17 eggs, peak dense samples contain massive clusters reaching up to ~500 eggs. Optimizing with standard Mean Squared Error (MSE) forces the networks to generate massive gradients on high-density outliers, destroying the model's accuracy on low-to-medium ranges. To mitigate this, **Huber Loss** is implemented:


$$L_{\delta}(N, \hat{N}) = \begin{cases} 
\frac{1}{2}(N - \hat{N})^2 & \text{for } |N - \hat{N}| \le \delta \\
\delta \left(|N - \hat{N}| - \frac{1}{2}\delta\right) & \text{otherwise} 
\end{cases}$$


Set at $\delta = 1.0$, it penalizes small errors quadratically (like MSE) and extreme outlier errors linearly (like Mean Absolute Error), stabilizing convergence across all density levels.

---

## 📊 Comparative Results & Epidemiological Validation

To measure structural stability against weight initialization, the orchestration script `main.py` performs **5 complete, independent training iterations** for each architecture.

The consolidated cross-validation results demonstrate key architectural differences:

* ❌ **InceptionV3:** Exhibited high instability and poor error metrics, yielding a global Mean Absolute Error (MAE) of **24.78** eggs and systematically underestimating dense clusters.
* ⚖️ **ResNet-50:** Showed highly consistent and balanced behavior on low-to-moderate counts, securing a standard global MAE of **14.12** eggs.
* 🏆 **EfficientNet-B3 (Top Performer):** Demonstrated superior mathematical and clinical performance, particularly within extreme density scenarios. In test instances containing massive overlapping clusters (e.g., ground truth of 382 eggs), EfficientNet-B3 successfully predicted 339 eggs.

### Post-Hoc Ordinal IDH Mapping

Continuous network outputs are mapped into discrete health alert thresholds post-evaluation using MINSA's standardized parameters:

* **Low (Bajo):** $\hat{N} < 10$ eggs.
* **Medium (Medio):** $10 \le \hat{N} < 50$ eggs.
* **High (Alto):** $50 \le \hat{N} < 100$ eggs.
* **Very High (Muy Alto):** $\hat{N} \ge 100$ eggs.

**Public Health Safeguards:** Confusion matrix analytics confirmed that the predictive errors of **EfficientNet-B3** occur exclusively between **adjacent risk categories** (e.g., misclassifying a 'Medium' paddle as 'Low'). Crucially, the model completely eliminated extreme false negatives (e.g., misclassifying a 'Very High' risk zone as 'Low'). This ensures that public health systems can dependably trigger containment protocols and vector control alerts without endangering vulnerable communities.

---

## 💻 Installation & Usage Guide

### Prerequisites

* Python 3.11 or higher
* CUDA-compliant GPU environment (Highly recommended; automatic fallback to CPU is included)

### Setup and Environment Configuration

```bash
# 1. Clone the project repository
git clone [https://github.com/darkJCdark/cnn-ordinal-classifier-egg-ovitrap](https://github.com/darkJCdark/cnn-ordinal-classifier-egg-ovitrap)
cd repo-aedes-aegypti

# 2. Initialize a clean virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\\Scripts\\activate

# 3. Install core dependencies
pip install opencv-python-headless pillow numpy matplotlib pandas seaborn scikit-learn torch torchvision

```

### Execution Protocol

The execution pipeline is split into two independent stages to isolate resource-heavy model computation from downstream graphic report rendering:

1. **Massive Model Training & Cross-Validation:**
Executes the full orchestration sequence, running 5 independent training routines for all three models sequentially. Weights, loss track logs, and continuous testing predictions are exported directly to `resultados/`.
```bash
python main.py

```


2. **Offline Statistical Report Generation:**
Reads the saved metrics files locally (operates instantly on CPU without requiring PyTorch or GPU resources) to compute averages, plot comparative boxplots, draw prediction-vs-ground-truth scatter curves, and render IDH matrices.
```bash
python analisis_comparativo.py

```
