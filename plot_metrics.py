import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F
import numpy as np

# ============================================================
# 0. CONFIGURACIÓN INICIAL
# ============================================================
# Nombres de las clases del modelo (ordenados según el entrenamiento)
class_names = ["elliptical", "spiral", "irregular"]
num_classes = len(class_names)

# ============================================================
# 1. CARGA DEL HISTORIAL COMPLETO DEL ENTRENAMIENTO
# ============================================================
# Se asume que el archivo incluye:
#   - train_loss, val_loss
#   - train_acc, val_acc
#   - val_probs : probabilidades predichas por la red en validación
#   - val_labels : etiquetas verdaderas (índices: 0,1,2)

try:
    # Cargar historial completo (se requiere desactivar weights_only)
    history = torch.load("galaxy_cnn_full_history.pth", weights_only=False)

    # ----- Métricas de entrenamiento -----
    train_loss = history["train_loss"]
    val_loss   = history["val_loss"]
    train_acc  = history["train_acc"]
    val_acc    = history["val_acc"]

    # ----- Datos para curva ROC -----
    val_probs  = history["val_probs"]
    val_labels = history["val_labels"]

    # Convertir tensores a NumPy si es necesario
    if isinstance(val_probs, torch.Tensor):
        val_probs = val_probs.cpu().numpy()
    if isinstance(val_labels, torch.Tensor):
        val_labels = val_labels.cpu().numpy()

except KeyError as e:
    print(f"Error: Falta la clave {e} en el archivo 'history'. Asegúrate de guardar val_probs y val_labels.")
    exit()
except FileNotFoundError:
    print("Error: No se encontró el archivo 'galaxy_cnn_full_history.pth'.")
    exit()

# ============================================================
# 2. CURVAS DE ENTRENAMIENTO (LOSS & ACCURACY)
# ============================================================

# ----------- CURVA DE LOSS -----------
plt.figure(figsize=(7,5))
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Validation Loss")
plt.title("Curva de Pérdida (Loss)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ----------- CURVA DE ACCURACY -----------
plt.figure(figsize=(7,5))
plt.plot(train_acc, label="Train Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.title("Curva de Exactitud (Accuracy)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()

# ============================================================
# 3. CURVA ROC MULTICLASE (One-vs-Rest) 📈
# ============================================================
# Se convierte val_labels a formato binario para calcular una ROC por clase.

from sklearn.preprocessing import label_binarize

# Crear matriz binaria: cada columna = clase independiente
y_true_bin = label_binarize(val_labels, classes=range(num_classes))

plt.figure(figsize=(8, 8))

# ----- Calcular ROC por clase -----
for i in range(num_classes):
    # fpr: false positive rate
    # tpr: true positive rate (recall)
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], val_probs[:, i])

    # Área bajo la curva
    roc_auc = auc(fpr, tpr)

    # Graficar curva
    plt.plot(fpr, tpr, label=f"ROC {class_names[i]} (AUC = {roc_auc:.2f})")

# Línea base (clasificación aleatoria)
plt.plot([0, 1], [0, 1], "k--", label="Clasificación aleatoria (AUC = 0.50)")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR) / Recall")
plt.title("Curvas ROC — Clasificación Multiclase (One-vs-Rest)")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
# ============================================================