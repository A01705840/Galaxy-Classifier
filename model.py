#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento básico en PyTorch para clasificación de galaxias.
===============================================================

Este script entrena una red convolucional simple (CNN) sobre un dataset de galaxias,
con imágenes organizadas en carpetas por clase, por ejemplo:

dataset/
├── train/
│   ├── spiral/
│   ├── elliptical/
│   └── irregular/
└── val/
    ├── spiral/
    ├── elliptical/
    └── irregular/

Autor: [Tu Nombre]
Fecha: 2025-11-07
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import torch.nn.functional as F

import os
from datetime import datetime
from collections import Counter
import numpy as np
# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

data_dir = r"C:\Users\majos\OneDrive\Documentos\Galaxias\images_E_S_SB_299x299_a_03"
learning_rate = 1e-3
num_classes = 3   # Cambia según tus clases reales
batch_size = 8   # Ajuste ideal para 3090 sin saturar VRAM
max_total_images = 15000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # Optimiza convoluciones en GPU

print(f"\n Entrenando en dispositivo: {device}")
if device.type == "cuda":
    print(f"🔹 GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"🔹 Memoria disponible: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB\n")


# ============================================================
# TRANSFORMACIONES Y CARGA DE DATOS
# ============================================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),   # Redimensionar imágenes
    transforms.ToTensor(),           # Convertir a tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalizar a rango [-1, 1]
])

# Dataset original
train_dataset_full = datasets.ImageFolder(os.path.join(data_dir, "images_E_S_SB_299x299_a_03_train"), transform=transform)
test_dataset_full = datasets.ImageFolder(os.path.join(data_dir, "images_E_S_SB_299x299_a_03_test"), transform=transform)

# ============================================================
# DIVISIÓN DE TRAIN / VAL (80/10) Y LIMITACIÓN DE IMÁGENES
# ============================================================

# Combinar train y test si deseas limitar total de imágenes
full_train = train_dataset_full
full_test = test_dataset_full

# Barajar índices antes de limitar
rng = np.random.default_rng(seed=42)
train_indices = np.arange(len(full_train))
rng.shuffle(train_indices)

test_indices = np.arange(len(full_test))
rng.shuffle(test_indices)

# Limitar cantidad total de imágenes
max_train = min(len(full_train), int(max_total_images * 0.8))
max_test = min(len(full_test), int(max_total_images * 0.2))

# Aplicar límite y crear subsets aleatorios
train_subset = torch.utils.data.Subset(full_train, train_indices[:max_train])
test_subset = torch.utils.data.Subset(full_test, test_indices[:max_test])

# Ahora dividir train_subset en train y val (90/10)
train_size = int(0.9 * len(train_subset))
val_size = len(train_subset) - train_size
train_dataset, val_dataset = random_split(train_subset, [train_size, val_size])

test_dataset = test_subset

# Revisar tamaños
print(f" Tamaño total limitado a {max_total_images} imágenes")
print(f" - Entrenamiento: {len(train_dataset)}")
print(f" - Validación: {len(val_dataset)}")
print(f" - Test: {len(test_dataset)}")

def get_targets(dataset):
    """Obtiene las etiquetas sin importar si es Subset anidado o ImageFolder."""
    if isinstance(dataset, Subset):
        return np.array(get_targets(dataset.dataset))[dataset.indices]
    elif hasattr(dataset, "targets"):  # ImageFolder o dataset similar
        return np.array(dataset.targets)
    else:
        raise AttributeError("El dataset no tiene atributo 'targets'.")

def count_classes(dataset, name):
    targets = get_targets(dataset)
    print(f"{name}: {Counter(targets)}")

print("\n Distribución de clases por subconjunto:")
count_classes(train_dataset, "Train")
count_classes(val_dataset, "Val")
count_classes(test_dataset, "Test")



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print(f" Clases detectadas: {train_dataset_full.classes}\n")


# ============================================================
# DEFINICIÓN DEL MODELO CNN
# ============================================================

class GalaxyCNN(nn.Module):
    """
    Arquitectura CNN simple para clasificación de galaxias.
    - Tres bloques convolucionales con ReLU + MaxPool.
    - Dos capas completamente conectadas al final.
    """

    def __init__(self, num_classes):
        super(GalaxyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Bloque convolucional 1
        x = self.pool(F.relu(self.conv1(x)))
        # Bloque convolucional 2
        x = self.pool(F.relu(self.conv2(x)))
        # Bloque convolucional 3
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten
        x = x.view(x.size(0), -1)
        # Capas completamente conectadas
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = GalaxyCNN(num_classes=num_classes).to(device)

# ============================================================
# DEFINICIÓN DEL OPTIMIZADOR Y CRITERIO
# ============================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ============================================================
# FUNCIÓN DE ENTRENAMIENTO
# ============================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """
    Entrena el modelo y evalúa en el conjunto de validación.
    """

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total, correct = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()                # Reiniciar gradientes
            outputs = model(images)              # Forward
            loss = criterion(outputs, labels)    # Calcular pérdida
            loss.backward()                      # Backpropagation
            optimizer.step()                     # Actualizar pesos

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total

        # Evaluación
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)

        print(f" Época [{epoch+1}/{num_epochs}] "
              f"| Pérdida Entrenamiento: {train_loss:.4f} "
              f"| Acc. Entrenamiento: {train_acc:.2f}% "
              f"| Pérdida Validación: {val_loss:.4f} "
              f"| Acc. Validación: {val_acc:.2f}%")

    print("\nEntrenamiento finalizado.")


def evaluate_model(model, val_loader, criterion):
    """
    Evalúa el modelo en el conjunto de validación.
    """
    model.eval()
    val_loss = 0.0
    total, correct = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = 100 * correct / total
    return val_loss, val_acc

# ============================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================

if __name__ == "__main__":
    start_time = datetime.now()
    train_model(model, train_loader, val_loader, criterion, optimizer, 50)
    print(f"Tiempo total: {datetime.now() - start_time}")

    torch.save(model.state_dict(), "galaxy_cnn.pth")
    print("Modelo guardado en 'galaxy_cnn.pth'")

