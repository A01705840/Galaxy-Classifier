import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURACIÓN Y RUTAS DE ARCHIVOS 📂
# ============================================================
model_path = r"./best_galaxy_cnn.pth"

# Imagen de prueba 
#img_path = r"./images_E_S_SB_299x299_a_03/images_E_S_SB_299x299_a_03_test/E/1927.jpg"
#img_path = r"./images_E_S_SB_299x299_a_03/images_E_S_SB_299x299_a_03_test/S/9290.jpg"
img_path = r"./images_E_S_SB_299x299_a_03/images_E_S_SB_299x299_a_03_test/SB/67477.jpg"

# Nombre de las clases del modelo
class_names = ["elliptical", "spiral", "spiral barred"]

# Tamaño de entrada esperado por la CNN después de las transformaciones
INPUT_SIZE = 128 

# Selección de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. MODELO CNN (Arquitectura con cálculo automático de FC) 🧠
# ============================================================
class GalaxyCNN(nn.Module):
    """
    CNN para clasificación de galaxias con extracción opcional de
    activaciones intermedias. Calcula dinámicamente el tamaño de la
    capa plana para evitar errores si cambia la arquitectura convolucional.
    """
    def __init__(self, num_classes=3):
        super(GalaxyCNN, self).__init__()

        # Bloque 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=6, padding=2)
        self.dropout1 = nn.Dropout(0.5)

        # Bloque 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        # Bloque 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)

        # Bloque 4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dropout4 = nn.Dropout(0.25)

        # 🔹 Cálculo automático del tamaño de entrada a la capa FC
        with torch.no_grad():
            dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
            x = self._forward_features(dummy)
            flattened_size = x.view(1, -1).shape[1]

        # Capas totalmente conectadas
        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def _forward_features(self, x):
        """
        Forward parcial: procesa solo las capas convolucionales para
        calcular el tamaño plano o extraer características.
        """
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = F.relu(self.conv4(x))
        x = self.dropout4(x)

        return x

    def forward(self, x, return_activations=False):
        """
        Forward completo.
        Si return_activations=True, devuelve también un diccionario con
        las activaciones intermedias de varias capas conv/pool.
        """
        activations = {}

        # Capa 1
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        activations["conv1"] = x.clone()

        # Capa 2 + max-pooling
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        activations["pool2"] = x.clone()

        # Capa 3 + max-pooling
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        activations["pool3"] = x.clone()

        # Capa 4 (última conv)
        x = F.relu(self.conv4(x))
        x = self.dropout4(x)
        activations["conv4"] = x.clone()

        # Aplanado + FCs
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return (x, activations) if return_activations else x


# ============================================================
# 3. FUNCIÓN PARA VISUALIZAR MAPAS DE CARACTERÍSTICAS 🖼️
# ============================================================
def plot_feature_maps(feature_maps, layer_name, max_channels=16):
    """
    Muestra un subconjunto (hasta 16) de mapas de características de
    una capa convolucional/pooling en formato grid.
    """
    feature_maps_np = feature_maps.squeeze(0).cpu().numpy()
    num_channels = feature_maps_np.shape[0]
    channels_to_plot = min(num_channels, max_channels)

    # Cálculo del grid cuadrado
    grid_size = int(channels_to_plot**0.5)
    if grid_size * grid_size < channels_to_plot:
        grid_size += 1

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    fig.suptitle(f"Activaciones de la capa: {layer_name} ({num_channels} canales)", fontsize=12)

    for i, ax in enumerate(axes_flat):
        if i < channels_to_plot:
            ax.imshow(feature_maps_np[i], cmap="viridis")
            ax.set_title(f"Canal {i+1}", fontsize=8)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# ============================================================
# 4. TRANSFORMACIONES 📏
# ============================================================
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),  # Tamaño de entrada
    transforms.ToTensor(),
    # Normalización utilizada durante el entrenamiento
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ============================================================
# 5. INICIALIZACIÓN DEL MODELO Y PREDICCIÓN 🚀
# ============================================================

# Inicializar modelo
num_classes = len(class_names)
model = GalaxyCNN(num_classes=num_classes).to(device)

# Inicializar capas FC mediante una pasada dummy
dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
_ = model(dummy_input)

# Cargar pesos
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Cargar imagen
image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# Obtener predicción y activaciones
with torch.no_grad():
    outputs, activations = model(input_tensor, return_activations=True)
    probs = F.softmax(outputs, dim=1)
    confidence, class_idx = torch.max(probs, 1)

    pred_label = class_names[class_idx.item()]
    conf = confidence.item() * 100

# ============================================================
# 6. RESULTADOS Y VISUALIZACIÓN ✨
# ============================================================

print("\n--- Probabilidades por clase ---")
for i, cls in enumerate(class_names):
    print(f"{cls:12s}: {probs[0][i].item()*100:.2f}%")

print(f"\nPredicción final: {pred_label} (Confianza: {conf:.2f}%)")

# Imagen original
plt.figure(figsize=(5, 5))
plt.imshow(image)
plt.axis("off")
plt.title(f"Resultado\n{pred_label} ({conf:.2f}%)")
plt.show()

# Activaciones
print("\n--- Visualización de mapas de características ---")
for layer_name, fmap in activations.items():
    plot_feature_maps(fmap, layer_name, max_channels=16)
