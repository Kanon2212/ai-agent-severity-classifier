
# ============================================================
# DETECCIÓN AUTOMÁTICA DEL DISPOSITIVO
# ============================================================

import torch


def obtener_dispositivo():
    """
    Detecta automáticamente el dispositivo disponible
    para ejecutar los modelos de PyTorch.
    """

    if torch.cuda.is_available():
        dispositivo = torch.device("cuda")

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dispositivo = torch.device("mps")

    else:
        dispositivo = torch.device("cpu")

    return dispositivo


# Detectar dispositivo
DEVICE = obtener_dispositivo()


if __name__ == "__main__":

    print("PyTorch:", torch.__version__)
    print("Dispositivo seleccionado:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "Memoria total:",
            round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2),
            "GB"
        )
