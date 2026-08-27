import io
import os

import streamlit as st
from PIL import ImageOps, Image
from torch import nn
import numpy as np
import torch
import torchvision.transforms as transforms


class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes):
        super(BrainTumorCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class PadToSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return ImageOps.pad(img, self.size, method=Image.LANCZOS, color=(0,0,0))

class CropBlackSpace: #crops black area around brain in the scan
    def __init__(self, tolerance = 15):
        self.tolerance = tolerance

    def __call__(self, img):
        if  img.mode != "RGB":
            img = img.convert("RGB")

        np_image = np.array(img.convert("L"))
        mask = np_image > self.tolerance

        coordinates = np.argwhere(mask)

        if coordinates.size > 0:
            y0,x0 = coordinates.min(axis=0)
            y1, x1 = coordinates.max(axis=0)
            return img.crop((x0,y0,x1,y1))
        return img

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "brain_classifier_model.pth")
NUM_CLASSES = 4
CLASSES = ["Tumor: Glioma", "Tumor: Meningioma", "No tumor", "Tumor: Pituitary"]

model = BrainTumorCNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
model.eval()

transform = transforms.Compose([
    CropBlackSpace(),
    PadToSize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

def process_image(image_bytes: bytes, filename: str):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor_img = transform(image)
    return tensor_img.unsqueeze(0)


st.set_page_config(
    page_title="Detekcja guzów mózgu",
    page_icon="🧠",
    layout="centered"
)

st.title("System klasyfikacji nowotworów mózgu")
st.caption("""
1. Prześlij zdjęcia MRI w formacie JPG lub PNG
2. Kliknij przycisk 'Predict'
3. Sprawdź wyniki
---
""")

uploaded_files = st.file_uploader("Wybierz zdjęcia rezonansu magnetycznego w formatach (JPG, PNG)...", type=["jpg", "jpeg", "png"],
                                  accept_multiple_files=True)

if uploaded_files:
    st.info(f"📁 {len(uploaded_files)} zdjęcie/a do analizy.")
    st.markdown("---")

    _, btn_col, _ = st.columns([1, 1, 1])
    with btn_col:
        analyze_button = st.button("Analizuj zdjęcia", type="primary", use_container_width=True)

    if analyze_button:
        with st.spinner("Przesyłanie zdjęć do modelu i ich analiza..."):

            st.subheader("Wyniki diagnozy")
            cols = st.columns(2)

            for index, file in enumerate(uploaded_files):
                col = cols[index%2]

                with col:
                    try:
                        st.image(Image.open(file), caption=file.name, use_container_width=True)

                        image_bytes = file.getvalue()
                        input_tensor = process_image(image_bytes, filename=file.name)

                        with torch.no_grad():
                            outputs = model(input_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                            predicted_index = torch.argmax(probabilities).item()

                            predicted_class = CLASSES[predicted_index]
                            confidence = probabilities[predicted_index].item() * 100


                        if confidence > 60:
                            st.error(f"{predicted_class}: {confidence:.2f}%")
                        elif predicted_class == "No tumor":
                            st.success(f"{predicted_class}: {confidence:.2f}%")
                        else:
                            st.error(f"{predicted_class}: {confidence:.2f}%")

                    except Exception as e:
                        st.error(f"Wystąpił błąd: {e}")

                    st.markdown("---")