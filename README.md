# AI Agent Severity Classifier

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
ai-agent-severity-classifier/
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
│   └── curva_accuracy.png
│
├── README.md
├── requirements.txt
└── requirements-lock.txt
```

Los archivos de `results/` se generan durante la ejecución. Los modelos y gráficos están excluidos del seguimiento de Git mediante `.gitignore`.

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
- Creación de los DataLoaders para entrenamiento y validación.

### Arquitectura del modelo

`model.py`

Se implementa una red neuronal multicapa (MLP) mediante `torch.nn.Module`.

Arquitectura:

```text
Entrada: 104 características
          |
    Capa lineal: 64
          |
        ReLU
          |
    Capa lineal: 32
          |
        ReLU
          |
    Capa de salida: 5
          |
    Predicción multiclase
```

La salida contiene cinco valores, uno por cada categoría de severidad.

### Entrenamiento

`train.py`

El entrenamiento utiliza:

- Función de pérdida: `CrossEntropyLoss`.
- Optimizador: Adam.
- Tasa de aprendizaje: 0.001.
- 50 épocas.
- Tamaño de lote: 32.
- Retropropagación mediante `loss.backward()`.
- Actualización de parámetros mediante `optimizer.step()`.

Durante el entrenamiento se calcula la pérdida y accuracy, tanto para entrenamiento como para validación.

También se guarda el modelo con la menor pérdida de validación.

### Evaluación

`evaluate.py`

Se utiliza el mejor modelo guardado para evaluar el rendimiento sobre el conjunto de validación.

Se calculan las siguientes métricas:

- Accuracy.
- Balanced Accuracy.
- F1 Macro.
- F1 Weighted.
- Precision, Recall y F1-score por clase.
- Matriz de confusión.

## 5. Resultados del entrenamiento

El conjunto de datos se dividió de la siguiente manera:

| Conjunto | Registros |
|---|---:|
| Entrenamiento | 316 |
| Validación | 79 |
| Total | 395 |

Después del preprocesamiento se obtuvieron 104 características de entrada y cinco clases de salida.

### Métricas de evaluación

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.7215 |
| Balanced Accuracy | 0.5937 |
| F1 Macro | 0.5870 |
| F1 Weighted | 0.7105 |

### Resultados por categoría

| Clase | Precision | Recall | F1-score |
|---|---:|---:|---:|
| critical | 0.64 | 0.70 | 0.67 |
| high | 0.66 | 0.74 | 0.70 |
| info | 1.00 | 1.00 | 1.00 |
| low | 0.00 | 0.00 | 0.00 |
| medium | 0.62 | 0.53 | 0.57 |

La clase `low` presenta resultados nulos porque solamente cuenta con dos registros en el conjunto de validación. Por este motivo, sus métricas deben interpretarse con cautela.

## 6. Interpretación de las curvas de aprendizaje

Durante las primeras épocas se observa una disminución progresiva de la pérdida de entrenamiento y validación.

La menor pérdida de validación se obtuvo en la época 19, con un valor de 0.6734.

A partir de ese punto, la pérdida de entrenamiento continúa disminuyendo mientras que la pérdida de validación comienza a aumentar.

Este comportamiento evidencia un posible sobreajuste (*overfitting*): el modelo mejora su ajuste sobre los datos de entrenamiento, pero pierde capacidad de generalización sobre los datos de validación.

Por esta razón, se conserva el checkpoint correspondiente a la menor pérdida de validación.

## 7. Visualizaciones generadas

Durante la ejecución del pipeline se generan las siguientes visualizaciones:

- `curva_loss.png`: evolución de la pérdida durante el entrenamiento y la validación.
- `curva_accuracy.png`: evolución de la precisión durante el entrenamiento y la validación.

Las imágenes se guardan en el directorio `results/`.

## 8. Instalación y ejecución

### Clonar el repositorio

```bash
git clone https://github.com/Kanon2212/ai-agent-severity-classifier.git
cd ai-agent-severity-classifier
```

### Instalar dependencias

```bash
python -m pip install -r requirements.txt
```

La instalación de PyTorch con soporte CUDA puede requerir seleccionar el paquete compatible con el sistema operativo, controlador y versión de CUDA del equipo.

### Ejecutar el entrenamiento

Desde la raíz del proyecto:

```bash
python -m src.train
```

### Ejecutar la evaluación

```bash
python -m src.evaluate
```

## 9. Conclusión

Se implementó un pipeline funcional de clasificación multiclase utilizando PyTorch, incorporando preparación de datos, detección automática del dispositivo, construcción de una red neuronal, entrenamiento, validación y evaluación.

Los resultados permiten observar el comportamiento del modelo y detectar problemas de generalización, como el sobreajuste y el efecto del desequilibrio entre clases.

El proyecto cumple con el objetivo de construir una infraestructura base reproducible para entrenar y evaluar modelos de clasificación utilizando PyTorch.
