#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
import warnings
# Suprimir advertencias de PyTorch/NumPy si aparecen
warnings.filterwarnings("ignore", category=UserWarning) 

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

data_dir = r"images_E_S_SB_299x299_a_03"
learning_rate = 1e-3
weight_decay_rate = 1e-4 # MEJORA 3: Tasa de Decaimiento de Peso (Regularización L2)
num_classes = 3 
batch_size = 8
max_total_images = 15000
num_epochs_to_run = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ============================================================
# TRANSFORMACIONES Y CARGA DE DATOS (MEJORADO) 📏
# ============================================================
# MEJORA 1: Data Augmentation (solo para el conjunto de entrenamiento)
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    # Augmentation:
    transforms.RandomHorizontalFlip(), # Volteo horizontal aleatorio
    transforms.RandomRotation(15),  # Rotación leve (hasta 15 grados)
    # transforms.ColorJitter(brightness=0.1, contrast=0.1), # Opcional: cambios de color
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Transformaciones para Validación y Prueba (solo redimensionamiento y normalización)
val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset original
train_dataset_full = datasets.ImageFolder(os.path.join(data_dir, "images_E_S_SB_299x299_a_03_train"), transform=train_transform) # Usar train_transform
test_dataset_full = datasets.ImageFolder(os.path.join(data_dir, "images_E_S_SB_299x299_a_03_test"), transform=val_test_transform) # Usar val_test_transform

# ============================================================
# CÁLCULO DE PESOS DE CLASE Y DIAGNÓSTICO (NUEVO) ⚖️
# ============================================================
print("--- Diagnóstico de Carga y Pesos ---")
# Conteo de targets en el dataset completo (basado en el output anterior)
targets_full = train_dataset_full.targets
class_counts_dict = Counter(targets_full)

# Asegurar el orden de los conteos según los índices 0, 1, 2
# Conteos: Clase 0 (E): 81547, Clase 1 (S): 29907, Clase 2 (SB): 8977
class_counts_tensor = torch.tensor([
    class_counts_dict.get(0, 0),
    class_counts_dict.get(1, 0),
    class_counts_dict.get(2, 0)
], dtype=torch.float)

total_samples = class_counts_tensor.sum()

# Calcular pesos inversos (Inverso de la frecuencia de aparición)
# Esto asigna un peso mayor a las clases minoritarias.
class_weights = total_samples / class_counts_tensor
    
# Normalizar los pesos para que no afecten la magnitud de la pérdida
class_weights = class_weights / class_weights.sum() * num_classes 

print(f"Mapeo de Clases: {train_dataset_full.class_to_idx}")
print(f"Total de imágenes cargadas (full): {len(train_dataset_full)}")
print(f"Pesos de Clase Calculados (0:E, 1:S, 2:SB): {class_weights}")
print("-" * 50)

# Mover los pesos al dispositivo (CPU/GPU)
class_weights = class_weights.to(device) 


# ============================================================
# DIVISIÓN DE TRAIN / VAL / TEST (Sin cambios)
# ============================================================

full_train = train_dataset_full
full_test = test_dataset_full

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

# Re-aplicar el val_test_transform al dataset de prueba completo antes de crear el subset
# Nota: train_dataset_full ya usa el train_transform. Necesitamos asegurar que el val/test use el transform correcto.
# Dado que ImageFolder aplica el transform en la carga, la forma más limpia de aplicar 
# el transform de validación es creando un dataset separado para la validación.
val_dataset_base = datasets.ImageFolder(os.path.join(data_dir, "images_E_S_SB_299x299_a_03_train"), transform=val_test_transform)
test_dataset_base = datasets.ImageFolder(os.path.join(data_dir, "images_E_S_SB_299x299_a_03_test"), transform=val_test_transform)


# Ahora, crear los subsets con la indexación correcta
# Para Train (ya usa train_transform):
train_subset = torch.utils.data.Subset(train_dataset_full, train_indices[:max_train])

# Para Test (usando el base con val_test_transform):
test_subset = torch.utils.data.Subset(test_dataset_base, test_indices[:max_test])

# Ahora dividir train_subset en train y val (90/10)
# Para la validación, usaremos el mismo conjunto de índices que train, 
# pero aplicado al dataset base con val_test_transform para evitar Data Augmentation.
val_indices_full = train_indices[:max_train] # Usamos el mismo set de imágenes que train_subset

train_size = int(0.9 * len(train_subset))
val_size = len(train_subset) - train_size
train_dataset, val_dataset = random_split(train_subset, [train_size, val_size])
# Dividir los índices en train_final y val_final (90/10)
train_final_indices = val_indices_full[:train_size]
val_final_indices = val_indices_full[train_size:]

# Creación final de datasets usando los índices y los transforms correctos
train_dataset = torch.utils.data.Subset(train_dataset_full, train_final_indices) # Con Augmentation
val_dataset = torch.utils.data.Subset(val_dataset_base, val_final_indices) # Sin Augmentation

test_dataset = test_subset # Sin Augmentation

# Definición de DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


# ============================================================
# DEFINICIÓN DEL MODELO CNN (MEJORADO) 🧠
# ... (Sin cambios en el modelo)
# ============================================================

class GalaxyCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(GalaxyCNN, self).__init__()

        # Bloque 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=6, padding=2)
        self.bn1 = nn.BatchNorm2d(32)  # MEJORA 2: Batch Norm
        self.dropout1 = nn.Dropout(0.2) # MEJORA 2: Dropout reducido (0.5 -> 0.2)

        # Bloque 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)  # MEJORA 2: Batch Norm
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.2) # MEJORA 2: Dropout reducido (0.5 -> 0.2)

        # Bloque 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) # MEJORA 2: Batch Norm
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.2) # MEJORA 2: Dropout reducido (0.5 -> 0.2)

        # Bloque 4 (Última Conv)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128) # MEJORA 2: Batch Norm
        self.dropout4 = nn.Dropout(0.5)  # MEJORA 2: Dropout alto (0.5) antes de FC

        self.fc1 = None
        self.fc2 = None
        self.num_classes = num_classes

    def _get_flattened_size(self, x):
        with torch.no_grad():
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.dropout1(x)

            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)
            x = self.dropout2(x)

            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool3(x)
            x = self.dropout3(x)

            x = F.relu(self.bn4(self.conv4(x)))
            x = self.dropout4(x)
            return x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        if self.fc1 is None:
            flattened_size = self._get_flattened_size(x)
            self.fc1 = nn.Linear(flattened_size, 64).to(x.device)
            self.fc2 = nn.Linear(64, self.num_classes).to(x.device)

        # CONV -> BN -> ReLU -> Dropout
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        # CONV -> BN -> ReLU -> Pool -> Dropout
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # CONV -> BN -> ReLU -> Pool -> Dropout
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # CONV -> BN -> ReLU -> Dropout
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout4(x)
        
        # Flatten y FC
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = GalaxyCNN(num_classes=num_classes).to(device)

# ============================================================
# DEFINICIÓN DEL OPTIMIZADOR Y CRITERIO (CORREGIDO) 💰
# ============================================================

# ¡CORRECCIÓN CLAVE! Usar los pesos calculados para CrossEntropyLoss
criterion = nn.CrossEntropyLoss(weight=class_weights) 
# MEJORA 3: Aplicación de Decaimiento de Peso (Weight Decay) al optimizador
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)

# ============================================================
# CLASE EARLY STOPPING (Sin cambios)
# ============================================================

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ============================================================
# FUNCIONES DE EVALUACIÓN Y ENTRENAMIENTO (Sin cambios funcionales mayores)
# ============================================================

def evaluate_model(model, loader, criterion):
    """Evalúa el modelo y retorna métricas básicas (Loss, Acc, MSE)."""
    model.eval()
    avg_loss = 0.0
    total, correct = 0, 0
    mse_total = 0.0
    mse_loss_fn = nn.MSELoss()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # MSE
            one_hot = F.one_hot(labels, num_classes=3).float()
            mse_total += mse_loss_fn(outputs, one_hot).item()

            avg_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss /= len(loader.dataset)
    avg_acc = 100 * correct / total
    avg_mse = mse_total / len(loader) 
    return avg_loss, avg_acc, avg_mse

def full_evaluate_model(model, loader):
    """Realiza una evaluación completa para obtener probabilidades y etiquetas."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            
            # Obtener probabilidades (Softmax)
            probabilities = F.softmax(outputs, dim=1)
            
            all_probs.append(probabilities.cpu())
            all_labels.append(labels.cpu())

    # Concatenar y convertir a NumPy
    final_val_probs = torch.cat(all_probs).numpy()
    final_val_labels = torch.cat(all_labels).numpy()
    
    return final_val_probs, final_val_labels

# ============================================================
# FUNCIÓN DE ENTRENAMIENTO
# ============================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    train_mse_history = []
    val_mse_history = []
    
    final_val_probs = None
    final_val_labels = None

    mse_loss_fn = nn.MSELoss()
    early_stopper = EarlyStopping(patience=15, min_delta=0.0)
    best_val_loss = float("inf") # Inicialización para el guardado condicional

    for epoch in range(1, num_epochs + 1):
        # ------------------------------
        # ENTRENAMIENTO
        # ------------------------------
        model.train()
        running_loss = 0.0
        total, correct = 0, 0
        mse_epoch = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # MSE Train
            one_hot = F.one_hot(labels, num_classes=3).float()
            mse_epoch += mse_loss_fn(outputs, one_hot).item()

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total
        train_mse = mse_epoch / len(train_loader)


        # ------------------------------
        # VALIDACIÓN
        # ------------------------------
        val_loss, val_acc, val_mse = evaluate_model(model, val_loader, criterion)

        # Guardar historia
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_mse_history.append(train_mse)
        val_mse_history.append(val_mse)

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% ")

        # ------------------------------
        # GUARDADO Y EARLY STOPPING (Lógica Corregida) 🏆
        # ------------------------------
        
        # Guardado condicional del mejor modelo
        if val_loss < best_val_loss:
            print(f"   >>> Mejora en la pérdida de validación ({best_val_loss:.4f} -> {val_loss:.4f}). Guardando 'best_galaxy_cnn.pth'...")
            best_val_loss = val_loss
            # Guardar el estado del modelo para el uso posterior
            torch.save(model.state_dict(), "best_galaxy_cnn.pth")
            
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("\n⛔ Early stopping activado: no hubo mejora en 15 epochs.")
            break

    print("\nEntrenamiento terminado.")

    # ------------------------------
    # EVALUACIÓN FINAL para ROC
    # ------------------------------
    # Cargar el mejor modelo (el que se guardó condicionalmente) para la evaluación final
    try:
        model.load_state_dict(torch.load("best_galaxy_cnn.pth", map_location=device))
        print("Cargando el mejor modelo guardado para la evaluación final...")
    except FileNotFoundError:
        print("Advertencia: No se encontró 'best_galaxy_cnn.pth'. Usando el estado final del modelo.")
    
    final_val_probs, final_val_labels = full_evaluate_model(model, val_loader)

    # Guardar historial COMPLETO para plot_metrics.py
    torch.save({
        "model_state_dict": model.state_dict(),
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "train_acc": train_acc_history,
        "val_acc": val_acc_history,
        "train_mse": train_mse_history,
        "val_mse": val_mse_history,
        # --- Datos para ROC ---
        "val_probs": final_val_probs,
        "val_labels": final_val_labels
    }, "galaxy_cnn_full_history.pth")

    print("\nHistorial completo guardado en 'galaxy_cnn_full_history.pth'")
    return train_loss_history, val_loss_history, train_acc_history, val_acc_history

# ============================================================
# PUNTO DE ENTRADA PRINCIPAL (Guardado Final Incondicional Eliminado)
# ============================================================

if __name__ == "__main__":
    start_time = datetime.now()
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs_to_run)
    print(f"\nTiempo total de entrenamiento: {datetime.now() - start_time}")
    # Nota: El modelo ya se guarda condicionalmente dentro de train_model,
    # por lo que eliminamos el guardado incondicional aquí.
    print(" ✅ Proceso completado. El mejor modelo ha sido guardado como 'best_galaxy_cnn.pth'.")