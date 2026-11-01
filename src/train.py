
# ============================================================
# ENTRENAMIENTO — CLASIFICADOR MLP CON PYTORCH
# ============================================================

import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import Adam

from src.dataset import crear_dataloaders
from src.model import ClasificadorMLP
from src.device import DEVICE


# ------------------------------------------------------------
# CONFIGURACIÓN DEL ENTRENAMIENTO
# ------------------------------------------------------------

SEMILLA = 42
TASA_APRENDIZAJE = 0.001
EPOCAS = 50
TAMANO_CAPA_OCULTA = 64

RUTA_RESULTADOS = "results"


# ------------------------------------------------------------
# REPRODUCIBILIDAD
# ------------------------------------------------------------

random.seed(SEMILLA)
np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEMILLA)


# ------------------------------------------------------------
# CARGA DE DATOS
# ------------------------------------------------------------

train_loader, val_loader, input_size, num_classes, clases, preprocesador = (
    crear_dataloaders()
)


# ------------------------------------------------------------
# INICIALIZACIÓN DEL MODELO
# ------------------------------------------------------------

modelo = ClasificadorMLP(
    input_size=input_size,
    hidden_size=TAMANO_CAPA_OCULTA,
    num_classes=num_classes
).to(DEVICE)


# ------------------------------------------------------------
# FUNCIÓN DE PÉRDIDA Y OPTIMIZADOR
# ------------------------------------------------------------

criterio = nn.CrossEntropyLoss()

optimizador = Adam(
    modelo.parameters(),
    lr=TASA_APRENDIZAJE
)


# ------------------------------------------------------------
# FUNCIÓN DE VALIDACIÓN
# ------------------------------------------------------------

def evaluar_modelo(modelo, val_loader):

    modelo.eval()

    perdida_total = 0.0
    predicciones_correctas = 0
    total_registros = 0

    with torch.no_grad():

        for datos, etiquetas in val_loader:

            datos = datos.to(DEVICE)
            etiquetas = etiquetas.to(DEVICE)

            salidas = modelo(datos)
            perdida = criterio(salidas, etiquetas)

            perdida_total += perdida.item() * datos.size(0)

            predicciones = torch.argmax(salidas, dim=1)

            predicciones_correctas += (
                predicciones == etiquetas
            ).sum().item()

            total_registros += etiquetas.size(0)

    perdida_promedio = perdida_total / total_registros

    exactitud = (
        predicciones_correctas / total_registros
    )

    return perdida_promedio, exactitud


# ------------------------------------------------------------
# CICLO DE ENTRENAMIENTO
# ------------------------------------------------------------

historial = {
    "perdida_entrenamiento": [],
    "perdida_validacion": [],
    "exactitud_entrenamiento": [],
    "exactitud_validacion": []
}

mejor_perdida_validacion = float("inf")

os.makedirs(RUTA_RESULTADOS, exist_ok=True)

print("=" * 60)
print("INICIO DEL ENTRENAMIENTO")
print("=" * 60)

print("Dispositivo:", DEVICE)
print("PyTorch:", torch.__version__)
print("Características:", input_size)
print("Clases:", num_classes)
print("Tasa de aprendizaje:", TASA_APRENDIZAJE)
print("Épocas:", EPOCAS)

for epoca in range(EPOCAS):

    modelo.train()

    perdida_total = 0.0
    predicciones_correctas = 0
    total_registros = 0

    for datos, etiquetas in train_loader:

        datos = datos.to(DEVICE)
        etiquetas = etiquetas.to(DEVICE)

        # Reiniciar los gradientes
        optimizador.zero_grad()

        # Propagación hacia adelante
        salidas = modelo(datos)

        # Calcular la pérdida
        perdida = criterio(salidas, etiquetas)

        # Retropropagación automática
        perdida.backward()

        # Actualizar los parámetros
        optimizador.step()

        perdida_total += perdida.item() * datos.size(0)

        predicciones = torch.argmax(salidas, dim=1)

        predicciones_correctas += (
            predicciones == etiquetas
        ).sum().item()

        total_registros += etiquetas.size(0)

    # Métricas de entrenamiento
    perdida_entrenamiento = perdida_total / total_registros

    exactitud_entrenamiento = (
        predicciones_correctas / total_registros
    )

    # Evaluación independiente
    perdida_validacion, exactitud_validacion = evaluar_modelo(
        modelo,
        val_loader
    )

    # Guardar historial
    historial["perdida_entrenamiento"].append(
        perdida_entrenamiento
    )

    historial["perdida_validacion"].append(
        perdida_validacion
    )

    historial["exactitud_entrenamiento"].append(
        exactitud_entrenamiento
    )

    historial["exactitud_validacion"].append(
        exactitud_validacion
    )

    # Guardar el mejor modelo según la pérdida de validación
    if perdida_validacion < mejor_perdida_validacion:

        mejor_perdida_validacion = perdida_validacion

        torch.save(
            modelo.state_dict(),
            os.path.join(RUTA_RESULTADOS, "mejor_modelo.pth")
        )

    # Mostrar resultados
    print(
        f"Época [{epoca + 1:02d}/{EPOCAS}] "
        f"| Loss Train: {perdida_entrenamiento:.4f} "
        f"| Loss Val: {perdida_validacion:.4f} "
        f"| Acc Train: {exactitud_entrenamiento:.4f} "
        f"| Acc Val: {exactitud_validacion:.4f}"
    )

# ------------------------------------------------------------
# RESTAURAR EL MEJOR MODELO
# ------------------------------------------------------------

ruta_modelo = os.path.join(
    RUTA_RESULTADOS,
    "mejor_modelo.pth"
)

modelo.load_state_dict(
    torch.load(ruta_modelo, map_location=DEVICE)
)

# ------------------------------------------------------------
# GENERAR CURVAS DE APRENDIZAJE
# ------------------------------------------------------------

epocas = range(1, EPOCAS + 1)

# Curva de pérdida
plt.figure(figsize=(10, 5))

plt.plot(
    epocas,
    historial["perdida_entrenamiento"],
    label="Entrenamiento"
)

plt.plot(
    epocas,
    historial["perdida_validacion"],
    label="Validación"
)

plt.title("Evolución de la pérdida")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(RUTA_RESULTADOS, "curva_loss.png"),
    dpi=150
)

plt.show()


# Curva de exactitud
plt.figure(figsize=(10, 5))

plt.plot(
    epocas,
    historial["exactitud_entrenamiento"],
    label="Entrenamiento"
)

plt.plot(
    epocas,
    historial["exactitud_validacion"],
    label="Validación"
)

plt.title("Evolución de la exactitud")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(RUTA_RESULTADOS, "curva_accuracy.png"),
    dpi=150
)

plt.show()

print("\nCurvas guardadas en la carpeta results.")


# ------------------------------------------------------------
# RESUMEN FINAL
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("ENTRENAMIENTO FINALIZADO")
print("=" * 60)

print("Mejor pérdida de validación:", round(
    mejor_perdida_validacion, 4
))

print(
    "Modelo guardado en:",
    os.path.join(RUTA_RESULTADOS, "mejor_modelo.pth")
)
