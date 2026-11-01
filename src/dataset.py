
# ============================================================
# DATASET — PREPARACIÓN DE DATOS PARA PYTORCH
# ============================================================

import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer


# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------

RUTA_DATASET = "data/processed/ai_agent_incidents_clean.csv"

VARIABLE_OBJETIVO = "severity"

VARIABLES_NUMERICAS = [
    "year"
]

VARIABLES_CATEGORICAS = [
    "month",
    "attack_type_code",
    "incident_kind",
    "confidence_level",
    "ai_involvement",
    "region"
]

VARIABLES_BOOLEANAS = [
    "real_harm_confirmed",
    "is_landmark_event"
]

TAMANO_LOTE = 32
SEMILLA = 42


# ------------------------------------------------------------
# CARGA Y PREPARACIÓN DEL DATASET
# ------------------------------------------------------------

def crear_dataloaders():

    # Cargar los datos procesados
    df = pd.read_csv(RUTA_DATASET)

    # Seleccionar las características y la variable objetivo
    columnas = (
        VARIABLES_NUMERICAS
        + VARIABLES_CATEGORICAS
        + VARIABLES_BOOLEANAS
    )

    X = df[columnas].copy()
    y = df[VARIABLE_OBJETIVO].copy()

    # Convertir variables booleanas a valores numéricos
    X[VARIABLES_BOOLEANAS] = (
        X[VARIABLES_BOOLEANAS].astype(int)
    )

    # Codificar las clases objetivo
    codificador_etiquetas = LabelEncoder()
    y_codificado = codificador_etiquetas.fit_transform(y)

    # Separar entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_codificado,
        test_size=0.20,
        random_state=SEMILLA,
        stratify=y_codificado
    )

    # Definir las transformaciones
    preprocesador = ColumnTransformer(
        transformers=[
            (
                "numericas",
                StandardScaler(),
                VARIABLES_NUMERICAS
            ),
            (
                "categoricas",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                VARIABLES_CATEGORICAS
            ),
            (
                "booleanas",
                "passthrough",
                VARIABLES_BOOLEANAS
            )
        ]
    )

    # Ajustar el preprocesador únicamente con entrenamiento
    X_train_procesado = preprocesador.fit_transform(X_train)

    # Aplicar las mismas transformaciones a validación
    X_val_procesado = preprocesador.transform(X_val)

    # Convertir las características a tensores
    X_train_tensor = torch.tensor(
        X_train_procesado,
        dtype=torch.float32
    )

    X_val_tensor = torch.tensor(
        X_val_procesado,
        dtype=torch.float32
    )

    # Convertir las etiquetas a tensores enteros
    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long
    )

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.long
    )

    # Crear los conjuntos de datos
    dataset_train = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    dataset_val = TensorDataset(
        X_val_tensor,
        y_val_tensor
    )

    # Crear los cargadores de datos
    train_loader = DataLoader(
        dataset_train,
        batch_size=TAMANO_LOTE,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=TAMANO_LOTE,
        shuffle=False
    )

    return (
        train_loader,
        val_loader,
        X_train_tensor.shape[1],
        len(codificador_etiquetas.classes_),
        codificador_etiquetas.classes_,
        preprocesador
    )


# ------------------------------------------------------------
# PRUEBA DE PREPARACIÓN
# ------------------------------------------------------------

if __name__ == "__main__":

    (
        train_loader,
        val_loader,
        input_size,
        num_classes,
        clases,
        preprocesador
    ) = crear_dataloaders()

    print("Preparación del dataset completada.")
    print("\nRegistros de entrenamiento:", len(train_loader.dataset))
    print("Registros de validación:", len(val_loader.dataset))
    print("Cantidad de características:", input_size)
    print("Cantidad de clases:", num_classes)
    print("Clases:", list(clases))

    # Inspeccionar un lote de entrenamiento
    datos, etiquetas = next(iter(train_loader))

    print("\nForma del lote de datos:", datos.shape)
    print("Forma del lote de etiquetas:", etiquetas.shape)
    print("Tipo de datos:", datos.dtype)
    print("Tipo de etiquetas:", etiquetas.dtype)
