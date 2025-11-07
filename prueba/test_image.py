import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# --- CONFIGURACIÓN ---
model_path = r"C:\Users\majos\OneDrive\Documentos\Galaxias\galaxy_cnn.pth"  # ruta del modelo guardado
img_path = r"C:\Users\majos\OneDrive\Documentos\Galaxias\prueba\778.jpg"      # imagen a probar
class_names = ["elliptical", "spiral", "irregular"]  # tus clases
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DEFINICIÓN DEL MODELO ---
class GalaxyCNN(nn.Module):
    def __init__(self, num_classes):
        super(GalaxyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # aplanar
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- RECREAR Y CARGAR EL MODELO ---
num_classes = len(class_names)
model = GalaxyCNN(num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# --- TRANSFORMACIONES ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # asegúrate de usar el tamaño que entrenaste
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- CARGAR IMAGEN ---
image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# --- PREDICCIÓN ---
with torch.no_grad():
    outputs = model(input_tensor)
    probs = F.softmax(outputs, dim=1)
    confidence, predicted_class = torch.max(probs, 1)

pred_label = class_names[predicted_class.item()]
conf = confidence.item() * 100

# --- VISUALIZACIÓN ---
plt.figure(figsize=(5, 5))
plt.imshow(image)
plt.axis("off")
plt.title(f"🔭 Pred: {pred_label}\n📈 Confianza: {conf:.2f}%")
plt.show()