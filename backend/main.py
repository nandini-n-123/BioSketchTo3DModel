from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import os

# =========================
# INIT APP
# =========================
app = FastAPI(title="BioSketch-3D API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD MODEL (ONLY ONCE)
# =========================
MODEL_PATH = "biosketch_model.pt"

model = efficientnet_b0()
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Class labels
classes = ['brain', 'hibiscus']

# =========================
# TRANSFORM (same as test.py)
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# OPTIONAL: OpenCV preprocessing
# =========================
def preprocess_for_ai(img):
    height, width = img.shape[:2]

    if width > 1000:
        scale = 1000 / width
        img = cv2.resize(img, (int(width * scale), int(height * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 7
    )

    inverted = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(thresh.shape[1], x + w + padding)
        y2 = min(thresh.shape[0], y + h + padding)
        cropped = thresh[y1:y2, x1:x2]
    else:
        cropped = thresh

    resized = cv2.resize(cropped, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    return rgb

# =========================
# API ROUTE
# =========================
@app.post("/api/classify-sketch")
async def classify_sketch(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        return {"status": "error", "message": "Invalid file type"}

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Preprocess image
    processed_img = preprocess_for_ai(raw_img)

    # Save debug image
    os.makedirs("debug_results", exist_ok=True)
    cv2.imwrite("debug_results/last_processed.jpg", processed_img)

    # Convert to PIL
    pil_image = Image.fromarray(processed_img)

    # Apply transforms
    input_tensor = transform(pil_image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    prediction = classes[predicted_idx.item()]
    confidence_score = round(confidence.item() * 100, 2)

    return {
        "status": "success",
        "prediction": prediction,
        "confidence": confidence_score,
        "message": "Real AI prediction complete"
    }