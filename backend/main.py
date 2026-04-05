from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import random
import cv2
import numpy as np

# Initialize the FastAPI app
app = FastAPI(title="BioSketch-3D API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Simplified for local testing
    allow_credentials=True,
    allow_methods=["*"],       
    allow_headers=["*"],       
)

# ==========================================
# OPEN CV PREPROCESSING LOGIC
# ==========================================
def preprocess_for_ai(img):
    """
    Cleans the raw photo for the AI: Grayscale -> Blur -> Threshold -> Crop -> Resize.
    """
    height, width = img.shape[:2]
    if width > 1000:
        scale = 1000 / width
        img = cv2.resize(img, (int(width * scale), int(height * scale)))

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. PRE-BLUR (NEW STEP)
    # We blur the image *before* thresholding. This smudges the thin notebook lines 
    # into the white paper so the algorithm ignores them, but keeps your thick pen strokes.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Adjusted Adaptive Thresholding
    # Increased block size from 11 to 21 (looks at a wider area)
    # Increased 'C' constant from 2 to 7 (requires the ink to be MUCH darker than the paper to turn black)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 7
    )

    # 4. Smart Cropping (Find the drawing)
    inverted = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(thresh.shape[1], x + w + padding)
        y2 = min(thresh.shape[0], y + h + padding)
        cropped = thresh[y1:y2, x1:x2]
    else:
        cropped = thresh

    # 5. Format for MobileNetV2 (224x224 RGB)
    final_resized = cv2.resize(cropped, (224, 224))
    final_rgb = cv2.cvtColor(final_resized, cv2.COLOR_GRAY2RGB)
    

    return final_rgb
# ==========================================
# THE UPLOAD ROUTE
# ==========================================
@app.post("/api/classify-sketch")
async def classify_sketch(file: UploadFile = File(...)):
    
    if not file.content_type.startswith("image/"):
        return {"status": "error", "message": "File provided is not an image."}

    # --- NEW: Read the image into OpenCV ---
    # 1. Read the raw bytes from the uploaded file
    contents = await file.read()
    # 2. Convert bytes to a numpy array
    nparr = np.frombuffer(contents, np.uint8)
    # 3. Decode the array into an OpenCV image format
    raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # --- NEW: Process the image ---
    cleaned_img = preprocess_for_ai(raw_img)

    # --- DEBUG: Save the image to your laptop to see if it worked! ---
    cv2.imwrite("debug_results/last_processed.jpg", cleaned_img)
    # ----------------------------------------------------------------

    # MOCK Logic (Waiting for Member 1)
    await asyncio.sleep(1.5)
    predictions = ["brain", "hibiscus"]
    fake_ai_result = random.choice(predictions)

    return {
        "status": "success",
        "filename": file.filename,
        "prediction": fake_ai_result,
        "message": "Image cleaned successfully! Check your project folder for debug_cleaned_output.jpg"
    }