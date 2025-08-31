# DL-PhawIA: Detección de Retinopatía Diabética

## 📋 Descripción

Este proyecto implementa un sistema de clasificación automática para la detección de retinopatía diabética utilizando técnicas de Deep Learning. Se comparan tres arquitecturas de redes neuronales convolucionales (CNN) para evaluar su efectividad en la clasificación de imágenes de retina en diferentes grados de severidad.

## 🔗 Repositorio del Proyecto

**Enlace público**: https://github.com/haroldeustaquio/Proyecto_EustaquioHarold_Clasificacion

El código completo, notebooks, modelos entrenados y resultados están disponibles en el repositorio de GitHub.

## 🗂️ Ruta Elegida y Dataset

### Dataset Utilizado
- **Fuente**: APTOS 2019 Blindness Detection (Kaggle)
- **Link**: https://www.kaggle.com/competitions/aptos2019-blindness-detection
- **Tipo**: Imágenes de retina del concurso APTOS 2019 – Blindness Detection
- **Clases**: Clasificación en cinco niveles de severidad (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR)
- **Reagrupación**: Las clases se reagrupan en 3 categorías finales:
  - Clase 0: Sin retinopatía (original: 0)
  - Clase 1: Retinopatía leve/moderada (original: 1, 2)
  - Clase 2: Retinopatía severa/proliferativa (original: 3, 4)
- **Formato**: Formato JPG
- **Preprocesamiento**: Recorte centrado, CLAHE, filtro bilateral
- **División**: División estratificada: 60% entrenamiento, 20% validación, 20% prueba

### Fuente y Licencia
- **Licencia**: Uso académico/no comercial únicamente (según términos de Kaggle)
- **Restricciones**: Prohibido uso comercial según los términos del concurso APTOS 2019
- **Uso permitido**: Exclusivamente para investigación académica y educativa
- **Cumplimiento ético**: Este proyecto se adhiere estrictamente a los términos de uso del dataset

## 🚀 Cómo Ejecutar

### Requisitos Previos
- Python 3.8+
- GPU NVIDIA con CUDA (recomendado)
- Jupyter Notebook o Google Colab

### Configuración del Entorno

#### Opción 1: Google Colab (Recomendado)
1. Abre los notebooks en Google Colab
2. **Activa GPU**: Runtime → Change runtime type → GPU → T4/A100
3. Los notebooks están optimizados para Colab con montaje automático de Google Drive

#### Opción 2: Local con Jupyter
```bash
# Clonar el repositorio
git clone https://github.com/haroldeustaquio/Proyecto_EustaquioHarold_Clasificacion.git
cd Proyecto_EustaquioHarold_Clasificacion

# Instalar dependencias
pip install -r requirements.txt

# Iniciar Jupyter
jupyter notebook
```

### Estructura de Notebooks
- `00_preprocessing.ipynb`: Preprocesamiento completo de datos
- `01_training.ipynb`: Entrenamiento de los tres modelos
- `02_eval_inference.ipynb`: Evaluación y análisis de resultados

## 🎯 Cómo Entrenar y Evaluar

### Paso 1: Preprocesamiento de Datos
```python
# Ejecutar notebook: 00_preprocessing.ipynb
# Incluye:
# - Reagrupación de clases (5 → 3 categorías)
# - Conversión PNG → JPG
# - Recorte centrado de retina
# - Realce de contraste CLAHE
# - Filtro bilateral para reducción de ruido
# - División estratificada train/val/test
# - Balanceo de clases con data augmentation
```

### Paso 2: Entrenamiento de Modelos
```python
# Ejecutar notebook: 01_training.ipynb
# Arquitecturas implementadas:
# - ResNet50
# - EfficientNet-B3  
# - MobileNet-V3-Large

# Configuración de entrenamiento:
# - Learning rates diferenciales (head: 1e-3, backbone: 1e-4)
# - Mixed Precision Training (AMP)
# - Label Smoothing (α=0.05)
# - Early Stopping basado en F1-macro
# - ReduceLROnPlateau scheduler
```

### Paso 3: Evaluación y Análisis
```python
# Ejecutar notebook: 02_eval_inference.ipynb
# Incluye:
# - Métricas comprensivas (Accuracy, F1, AUC, Kappa, MCC)
# - Matrices de confusión
# - Curvas de entrenamiento
# - Inferencias en imágenes de prueba
# - Comparación entre arquitecturas
```

## 📊 Arquitecturas Implementadas

### 🏗️ ResNet50
- **Características**: Conexiones residuales, 50 capas
- **Ventajas**: Baseline robusto, evita vanishing gradients
- **Uso**: Clasificación de referencia

### ⚡ EfficientNet-B3  
- **Características**: Compound scaling, squeeze-and-excitation
- **Ventajas**: Mejor eficiencia computacional, menos parámetros
- **Uso**: Mejor balance precisión/eficiencia

### 📱 MobileNet-V3-Large
- **Características**: Depthwise separable convolutions, hard-swish
- **Ventajas**: Optimizado para dispositivos móviles
- **Uso**: Deployment en producción con recursos limitados

## 📈 Cómo Generar Métricas y Visualizaciones

### Métricas Calculadas
El sistema calcula automáticamente las siguientes métricas:

**Métricas Básicas:**
- Accuracy y Balanced Accuracy
- F1-Score macro
- Precision y Recall macro

**Métricas Avanzadas:**
- ROC-AUC macro
- Average Precision (PR-AUC) macro
- Cohen's Kappa
- Matthews Correlation Coefficient (MCC)

**Visualizaciones:**
- Matrices de confusión
- Curvas de entrenamiento (loss, F1, accuracy, AUC)
- Ejemplos de inferencia con predicciones top-k

### Generar Tabla de Resultados
```python
# En 02_eval_inference.ipynb, al final:
# Se genera automáticamente una tabla comparativa con todas las métricas

# Ejemplo de tabla generada:
metrics_comparison = pd.concat([
    resnet_metrics_df, 
    mobilenet_metrics_df, 
    efficientnet_metrics_df
])
print(metrics_comparison)
```

### Generar Gráficos de Métricas
```python
# Las funciones de visualización están incluidas en 02_eval_inference.ipynb:

# Curvas de entrenamiento
plot_series(f1_scores, title="F1-Macro Evolution", ylabel="F1-Score")
plot_series(accuracy_scores, title="Accuracy Evolution", ylabel="Accuracy")

# Matrices de confusión
plot_confusion(cm, class_names, title="Confusion Matrix - Test Set")

# Comparación entre modelos
plot_model_comparison(metrics_df)
```

## 🔧 Configuraciones Técnicas

### Optimizaciones Implementadas
- **CUDA**: Optimizaciones cudnn.benchmark y allow_tf32
- **Mixed Precision**: Automatic Mixed Precision (AMP) para acelerar entrenamiento
- **DataLoaders**: Optimizados con pin_memory, persistent_workers, prefetch_factor
- **Compilación**: torch.compile para mayor eficiencia
- **Memory Format**: channels_last para mejor rendimiento en GPU

### Hiperparámetros Principales
```python
BATCH_SIZE = 32
EPOCHS = 30 (con early stopping)
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
LABEL_SMOOTHING = 0.05
```

## 📁 Estructura del Proyecto

```
DL-PhawIA/
├── data/
│   ├── original/           # Datos originales
│   ├── processed/          # Datos procesados
│   ├── structured/         # Datos organizados por clases
│   └── structured-balanced/ # Datos balanceados
├── notebooks/
│   ├── 00_preprocessing.ipynb    # Preprocesamiento
│   ├── 01_training.ipynb         # Entrenamiento
│   └── 02_eval_inference.ipynb   # Evaluación
├── models/                 # Checkpoints guardados
├── results/               # Resultados y visualizaciones
└── requirements.txt       # Dependencias
```

## 📋 Resultados Principales

Los modelos entrenados alcanzan métricas competitivas en la clasificación de retinopatía diabética:

| Modelo | F1_macro_Val | F1_macro_Test | Balanced_Accuracy_Val | Balanced_Accuracy_Test | ROC_AUC_macro_Val | ROC_AUC_macro_Test |
|--------|--------------|---------------|----------------------|------------------------|-------------------|-------------------|
| ResNet-50 | 0.821 | 0.838 | 0.816 | 0.835 | 0.954 | 0.953 |
| MobileNetV3-Large | 0.851 | 0.853 | 0.852 | 0.855 | 0.949 | 0.954 |
| EfficientNet-B3 | 0.849 | 0.853 | 0.843 | 0.847 | 0.941 | 0.950 |

## 🔍 Análisis y Conclusiones

El proyecto proporciona:
1. **Pipeline completo**: Desde preprocesamiento hasta evaluación
2. **Comparación rigurosa**: Tres arquitecturas con métricas comprehensivas
3. **Optimizaciones modernas**: Mixed precision, data augmentation, early stopping
4. **Visualizaciones detalladas**: Matrices de confusión, curvas de entrenamiento
5. **Código reproducible**: Semillas fijas y configuración determinística

## 📞 Contacto

Para consultas o colaboraciones, consultar notebooks o resultados en `results/`.

---

**Nota**: Proyecto exclusivamente para investigación académica. No apto para uso clínico sin supervisión profesional. Este trabajo cumple con los términos de uso académico/no comercial del dataset APTOS 2019 Blindness Detection de Kaggle, respetando las restricciones éticas y legales establecidas para el uso responsable de datos médicos.