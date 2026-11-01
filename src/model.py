
# ============================================================
# MODELO — CLASIFICADOR MLP CON PYTORCH
# ============================================================

import torch
import torch.nn as nn


class ClasificadorMLP(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes
    ):
        super().__init__()

        # Definición de la arquitectura neuronal
        self.red = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):

        # Propagación hacia adelante
        return self.red(x)


# ------------------------------------------------------------
# PRUEBA DEL MODELO
# ------------------------------------------------------------

if __name__ == "__main__":

    # Dimensiones reales del dataset
    input_size = 104
    hidden_size = 64
    num_classes = 5

    # Crear el modelo
    modelo = ClasificadorMLP(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes
    )

    # Generar un lote de prueba
    datos = torch.randn(32, input_size)

    # Obtener las predicciones
    resultados = modelo(datos)

    print("Arquitectura del modelo:")
    print(modelo)

    print("\nForma de entrada:", datos.shape)
    print("Forma de salida:", resultados.shape)

    print(
        "\nCantidad de parámetros:",
        sum(p.numel() for p in modelo.parameters())
    )
