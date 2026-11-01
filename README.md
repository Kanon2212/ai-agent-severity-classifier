# Clasificador de Incidentes de Agentes de IA

## 1. Descripción del proyecto

Este proyecto implementa un pipeline base de entrenamiento y validación utilizando **PyTorch** para clasificar incidentes relacionados con agentes de inteligencia artificial.

El objetivo es construir un clasificador multiclase que permita predecir la severidad de un incidente a partir de características estructuradas del dataset.

### Objetivo de clasificación

La variable objetivo es `severity`, que contiene cinco categorías:

- `critical`
- `high`
- `info`
- `low`
- `medium`

## 2. Tecnologías utilizadas

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- CUDA

### Entorno de ejecución

| Componente | Configuración |
|---|---|
| PyTorch | 2.14.1+cu126 |
| CUDA | 12.6 |
| GPU | NVIDIA GeForce RTX 2070 SUPER |
| Memoria GPU | 8 GB |
| Optimizador | Adam |
| Learning rate | 0.001 |
| Épocas | 50 |
| Tamaño del lote | 32 |
| Semilla aleatoria | 42 |

## 3. Estructura del repositorio

```text
ag-news-classifier/
│
├── data/
│   ├── raw/
│   │   ├── ai_agent_incidents_full.csv
│   │   └── README.md
│   │
│   └── processed/
│       ├── ai_agent_incidents_clean.csv
│       └── README.md
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── device.py
│   ├── evaluate.py
│   ├── evaluation.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── train.py
│   └── vectorization.py
│
├── results/
│   ├── mejor_modelo.pth
│   ├── curva_loss.png
│   ├── curva_accuracy.png
│   └── matriz_confusion.png
│
├── README.md
├── requirements.txt
└── requirements-lock.txt
```

## 4. Pipeline de entrenamiento

El proyecto está organizado en diferentes módulos para separar las responsabilidades.

### Detección del dispositivo

`device.py`

Detecta automáticamente el dispositivo disponible y selecciona CUDA, MPS o CPU.

### Preparación del dataset

`dataset.py`

Realiza las siguientes operaciones:

- Carga del dataset procesado.
- Selección de variables predictoras y variable objetivo.
- Codificación de etiquetas.
- División estratificada en entrenamiento y validación.
- Normalización de variables numéricas.
- Codificación One-Hot de variables categóricas.
- Conversión de los datos a tensores de PyTorch.
- Cre