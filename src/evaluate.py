
# ============================================================
# EVALUACIÓN — CLASIFICADOR MLP
# ============================================================

import os
import torch
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)

from src.dataset import crear_dataloaders
from src.model import ClasificadorMLP
from src.device import DEVICE


# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------

RUTA_MODELO = "results/mejor_modelo.pth"


# ------------------------------------------------------------
# CARGA DE DATOS
# ------------------------------------------------------------

(
    train_loader,
    val_loader,
    input_size,
    num_classes,
    clases,
    preprocesador
) = crear_dataloaders()


# ------------------------------------------------------------
# CARGA DEL MODELO ENTRENADO
# ------------------------------------------------------------

modelo = ClasificadorMLP(
    input_size=input_size,
    hidden_size=64,
    num_classes=num_classes
).to(DEVICE)

modelo.load_state_dict(
    torch.load(RUTA_MODELO, map_location=DEVICE)
)

modelo.eval()


# ------------------------------------------------------------
# OBTENER PREDICCIONES
# ------------------------------------------------------------

etiquetas_reales = []
etiquetas_predichas = []

with torch.no_grad():

    for datos, etiquetas in val_loader:

        datos = datos.to(DEVICE)

        salidas = modelo(datos)

        predicciones = torch.argmax(salidas, dim=1)

        etiquetas_reales.extend(
            etiquetas.numpy()
        )

        etiquetas_predichas.extend(
            predicciones.cpu().numpy()
        )


# ------------------------------------------------------------
# MÉTRICAS GENERALES
# ------------------------------------------------------------

accuracy = accuracy_score(
    etiquetas_reales,
    etiquetas_predichas
)

f1_macro = f1_score(
    etiquetas_reales,
    etiquetas_predichas,
    average="macro",
    zero_division=0
)

f1_weighted = f1_score(
    etiquetas_reales,
    etiquetas_predichas,
    average="weighted",
    zero_division=0
)

accuracy_balanceada = balanced_accuracy_score(
    etiquetas_reales,
    etiquetas_predichas
)


# ------------------------------------------------------------
# INFORME DE CLASIFICACIÓN
# ------------------------------------------------------------

print("=" * 60)
print("INFORME DE EVALUACIÓN")
print("=" * 60)

print("\nAccuracy:", round(accuracy, 4))
print("Balanced Accuracy:", round(accuracy_balanceada, 4))
print("F1 Macro:", round(f1_macro, 4))
print("F1 Weighted:", round(f1_weighted, 4))

print("\nMÉTRICAS POR CLASE")

print(
    classification_report(
        etiquetas_reales,
        etiquetas_predichas,
        labels=list(range(num_classes)),
        target_names=list(clases),
        zero_division=0
    )
)


# ------------------------------------------------------------
# MATRIZ DE CONFUSIÓN
# ------------------------------------------------------------

matriz = confusion_matrix(
    etiquetas_reales,
    etiquetas_predichas,
    labels=list(range(num_classes))
)

print("\nMATRIZ DE CONFUSIÓN")
print(matriz)

visualizacion = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=clases
)

visualizacion.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Matriz de confusión - Clasificador MLP")
plt.tight_layout()

plt.savefig(
    "results/matriz_confusion.png",
    dpi=150
)

plt.show()

print("\nMatriz guardada en results/matriz_confusion.png")
