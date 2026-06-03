# BioSketch3D

BioSketch3D is a web-based educational application that converts hand-drawn biological organ sketches into interactive 3D models. The system combines a React + Vite frontend, a FastAPI backend, an EfficientNet-B0 organ classifier, OpenCV-based image preprocessing, and a deterministic lattice-cage Free-Form Deformation (FFD) pipeline.

BioSketch3D does **not** generate new anatomy from scratch. It starts from validated baseline 3D organ assets and reshapes the selected model according to the extracted outline of the student's sketch. This keeps the output controlled, repeatable, and suitable for classroom demonstrations.

---

## Table of contents

1. Project overview
2. Supported organs and current limitations
3. How the pipeline works
4. Repository structure
5. Required files before running
6. Prerequisites
7. First-time setup
8. Running the project locally
9. Phone-camera QR workflow over the same Wi-Fi network
10. Laptop webcam notes
11. Testing the backend
12. Testing the frontend
13. Debug files and result logs
14. Common problems and fixes
15. Git and file-sharing notes
16. Useful commands
17. Final teammate checklist

---

## 1. Project overview

The repository contains three main modules:

```text
BioSketchTo3D/
├── backend/          # FastAPI API + deformation pipeline
├── frontend/         # React + Vite UI, crop flow, QR workflow, and 3D viewer
└── model_scripts/    # EfficientNet-B0 classifier code and trained checkpoint
```

The normal user flow is:

```text
User uploads a sketch, captures it using the laptop webcam,
or sends a phone-camera image through the same Wi-Fi network
        ↓
Frontend shows the image and allows crop adjustment
        ↓
Frontend sends the cropped original image to FastAPI
        ↓
Backend saves the uploaded image
        ↓
Classifier branch preprocesses the image for EfficientNet-B0
        ↓
CNN predicts brain, heart, or lungs
        ↓
Low-confidence or unsupported sketches are rejected
        ↓
Deformation branch preprocesses the image for contour extraction
        ↓
Backend selects the matching validated GLB organ model
        ↓
Ordered contour correspondence + depth-locked lattice-cage FFD
reshape the model according to the sketch outline
        ↓
Backend saves the generated GLB inside backend/outputs/
        ↓
Frontend loads the returned model URL and renders the 3D model
```

### Important preprocessing design

The project currently uses **two OpenCV preprocessing branches on the backend**, because classification and deformation require different outputs:

```text
Classifier preprocessing:
    model_scripts/src/biosketch_classifier/preprocessing.py
    → CNN-friendly cleaned and padded image

Deformation preprocessing:
    backend/deform/app/vision/sketch_parser.py
    → high-resolution binary image for contour extraction
```

Do not replace both branches with the exact same final image without retesting the classifier. The two pipelines have different purposes. A future refactor may share only the initial shadow-removal and denoising stage.

The frontend does **not** apply scan-style enhancement. It sends the cropped original image to the backend.

---

## 2. Supported organs and current limitations

The current validated organ classes are:

```text
brain
heart
lungs
```

If an uploaded sketch does not confidently match these supported organs, the backend should stop before deformation and return an unsupported-organ or low-confidence message.

For the current version, use:

```text
- Plain unruled white sheet
- Dark pencil, pen, or marker
- Flat paper surface
- Bright and even lighting
- Minimal strong shadows
- Sketch filling most of the camera frame
- Standard front-view orientation
```

### Current deformation limitations

The stable implementation uses:

```text
global ordered contour correspondence
+
regular rectangular lattice cage
+
depth-locked weighted FFD
```

The system mainly matches the **outer silhouette** of the sketch. It does not automatically understand every anatomical sub-part.

Typical behavior:

```text
Brain:
    Performs well because the major outer regions are comparatively distinct.

Lungs:
    Left and right lobe resizing works reasonably well.
    Independent trachea stretching is harder because nearby upper lobe regions
    can share the same cage influence.

Heart:
    Most challenging organ because the vessels, atria, and ventricle are
    geometrically close and interconnected. Pulling one outline region may
    also affect nearby structures.
```

Semantic landmarks, TPS-lite, and direct demo-emphasis experiments may remain in an experimental folder, but they should not be imported into the stable pipeline unless they have been tested again.

Ruled notebook sheets, graph sheets, faint erased marks, and strong shadows may interfere with preprocessing. The current sketch parser removes many scattered artifacts and ignores residual noise when it does not appear in the final extracted contour.

---

## 3. How the pipeline works

### 3.1 Frontend capture and crop

The frontend supports:

```text
1. Upload File
2. Laptop Camera
3. Phone Camera QR
```

The user can adjust the crop box before submitting.

The frontend sends the cropped original image to the backend. It does not convert the image into a black-and-white preview before upload.

### 3.2 Classifier preprocessing

Classifier preprocessing is located at:

```text
model_scripts/src/biosketch_classifier/preprocessing.py
```

Its goal is to create a stable input image for EfficientNet-B0. It may include:

```text
- image loading
- shadow or lighting correction
- thresholding or scan-style cleanup
- content cropping
- square padding
- conversion to PIL / tensor transforms
```

The trained checkpoint should be present at:

```text
model_scripts/checkpoints/organ_classifier/best_model.pt
```

The classifier wrapper is located at:

```text
model_scripts/src/biosketch_classifier/predictor.py
```

### 3.3 Deformation preprocessing

Deformation preprocessing is located at:

```text
backend/deform/app/vision/sketch_parser.py
```

Its goal is to preserve the contour geometry needed by FFD. It may include:

```text
- grayscale conversion
- shadow normalization
- adaptive thresholding
- noise and connected-component filtering
- contour extraction
- contour simplification
- normalized sketch coordinates
```

Generated debug images are stored under:

```text
backend/deform/presentation_debug/
```

The presentation debug image contains:

```text
1. Original Sketch
2. Preprocessed Binary
3. Extracted Border + Anchors
```

Small residual dots in the binary panel are acceptable when they do not appear in the extracted border and anchor panel, because only the final selected contour is sent to the deformation stage.

### 3.4 Unsupported-organ rejection

The classifier is trained for a closed set of supported classes. Without a confidence gate, an unsupported drawing could be forced into one of the known classes.

The minimum automatic confidence value is configured in:

```text
backend/main.py
```

Look for a variable such as:

```python
MIN_AUTO_CONFIDENCE = 0.60
```

The exact value may be tuned after testing:

```text
Increase it if unsupported sketches pass too easily.
Reduce it slightly if valid organ sketches are rejected too often.
```

### 3.5 Deterministic deformation

The deformation pipeline is located at:

```text
backend/deform/app/pipeline.py
```

It loads a validated GLB model from:

```text
backend/deform/assets/3d_models/
```

The current stable flow is:

```text
extract sketch contour
→ load and normalize selected GLB
→ extract projected model silhouette
→ build regular lattice cage
→ compute ordered closed-contour correspondence
→ solve depth-locked weighted cage deformation
→ export deformed GLB
```

The deformation is **depth-locked**: the FFD solver primarily changes the front-view sketch plane while preserving the model depth behavior.

The generated model is stored in:

```text
backend/outputs/
```

The frontend receives a URL such as:

```text
http://127.0.0.1:8000/outputs/heart_xxxxxxxxx_deformed.glb
```

and renders the GLB in the browser.

---

## 4. Repository structure

Expected structure:

```text
BioSketchTo3D/
├── README.md
├── .gitignore
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── uploads/                         # runtime images, ignored by Git
│   ├── outputs/                         # generated GLBs, ignored by Git
│   └── deform/
│       ├── app/
│       │   ├── pipeline.py
│       │   ├── result_table.py
│       │   ├── config/
│       │   │   └── organ_presets.py
│       │   ├── deformation/
│       │   ├── geometry/
│       │   │   └── mesh_prep.py
│       │   └── vision/
│       │       └── sketch_parser.py
│       ├── assets/
│       │   ├── 3d_models/
│       │   │   ├── brain.glb
│       │   │   ├── heart.glb
│       │   │   └── lungs.glb
│       │   └── sketches/
│       └── presentation_debug/          # generated debug files
│
├── frontend/
│   ├── package.json
│   ├── package-lock.json
│   ├── .env.example                     # optional overrides only
│   ├── index.html
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       ├── index.css
│       ├── UploadSection.jsx
│       ├── MobileScanPage.jsx
│       ├── mobile-scan.css
│       └── ModelViewer.jsx
│
└── model_scripts/
    ├── requirements.txt                 # optional if dependencies are merged
    ├── scripts/
    ├── src/
    │   └── biosketch_classifier/
    │       ├── predictor.py
    │       ├── preprocessing.py
    │       ├── models.py
    │       ├── constants.py
    │       └── utils.py
    └── checkpoints/
        └── organ_classifier/
            └── best_model.pt
```

---

## 5. Required files before running

The application will not fully run unless these files exist and are not empty:

```text
backend/deform/assets/3d_models/brain.glb
backend/deform/assets/3d_models/heart.glb
backend/deform/assets/3d_models/lungs.glb
model_scripts/checkpoints/organ_classifier/best_model.pt
```

After starting the backend, check:

```text
http://127.0.0.1:8000/api/health
```

The health response should confirm that the required assets are available.

---

## 6. Prerequisites

Install:

```text
Python 3.10 or 3.11
Node.js 18+
npm 9+
Git
```

Check versions:

```bash
python --version
node -v
npm -v
git --version
```

For NVIDIA GPU support during classification, install a CUDA-enabled PyTorch build that matches the machine's CUDA setup. The classifier wrapper can select CUDA automatically when `torch.cuda.is_available()` returns `True`.

Verify GPU visibility:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 7. First-time setup

Clone the repository:

```bash
git clone <repository-url>
cd BioSketchTo3D
```

### 7.1 Backend setup using a virtual environment

Windows Command Prompt:

```cmd
cd backend
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 7.2 Backend setup using Conda

```bash
conda create -n deepleaf python=3.11
conda activate deepleaf
cd backend
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If classifier dependencies have **not** yet been merged into `backend/requirements.txt`, also run:

```bash
pip install -r ..\model_scripts\requirements.txt
```

Recommended repository cleanup:

```text
Keep one combined backend/requirements.txt containing:
    FastAPI dependencies
    classifier dependencies
    deformation dependencies
```

This makes teammate setup simpler.

### 7.3 Frontend setup

```bash
cd frontend
npm install
```

The frontend dependency list includes:

```text
qrcode.react
```

Teammates do not need to install it separately when the updated `package.json` and `package-lock.json` are committed. Running `npm install` is enough.

If npm uses an incorrect registry:

```bash
npm config set registry https://registry.npmjs.org/
```

---

## 8. Running the project locally

Use two terminals.

### Terminal 1: backend

Windows example:

```bash
cd C:\BioSketchTo3D\backend
conda activate biosketch3d
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

If using `.venv`:

```bash
cd C:\BioSketchTo3D\backend
.venv\Scripts\activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Terminal 2: frontend

```bash
cd C:\BioSketchTo3D\frontend
npm run dev -- --host 0.0.0.0
```

Vite should print:

```text
Local:   http://localhost:5173/
Network: http://<laptop-ip>:5173/
```

Example:

```text
Local:   http://localhost:5173/
Network: http://172.20.10.11:5173/
```

---

## 9. Phone-camera QR workflow over the same Wi-Fi network

The application supports phone-camera capture without manually changing the backend URL each time.

### 9.1 Connect both devices

Use one of these:

```text
Option A: connect laptop and phone to the same Wi-Fi router
Option B: turn on phone hotspot and connect the laptop to it
```

A phone hotspot is often more reliable on networks that isolate devices.

### 9.2 Open the Vite Network URL

On the laptop, open:

```text
http://<laptop-ip>:5173/
```

Example:

```text
http://172.20.10.11:5173/
```

Do not use `localhost` for the phone QR workflow because a phone cannot access the laptop through the laptop's localhost address.

### 9.3 Generate and use the QR code

On the laptop:

```text
1. Open Upload Biological Diagram.
2. Select Phone Camera QR.
3. Click Generate Phone QR Code.
4. Scan the QR code using the phone.
5. Tap Capture Sketch.
6. Photograph the sketch using the phone's rear camera.
7. Review the phone preview.
8. Tap Send to Laptop.
9. Return to the laptop.
10. Adjust the crop box if needed.
11. Click Submit Sketch.
```

The frontend should derive the backend host from the browser hostname.

Example:

```text
Laptop opens:  http://172.20.10.11:5173/
Backend calls: http://172.20.10.11:8000/
QR code uses:  http://172.20.10.11:5173/mobile-scan/<session-id>
```

Normally, you do not need to edit `.env` when the IP address changes.

### 9.4 Test connectivity manually

Before using QR capture, open these URLs on the phone:

```text
http://<laptop-ip>:5173/
http://<laptop-ip>:8000/api/health
```

If both pages open, the QR flow should work.

### 9.5 Windows Firewall

When Windows asks whether Python or Node.js can communicate through the firewall, allow access on Private networks.

If the phone cannot access the laptop:

```text
- Confirm both devices are on the same Wi-Fi or hotspot
- Confirm backend uses --host 0.0.0.0
- Confirm frontend uses --host 0.0.0.0
- Check the new Vite Network URL after switching networks
- Allow Python and Node.js on Private networks
- Try the phone hotspot instead of guest Wi-Fi
```

---

## 10. Laptop webcam notes

Laptop webcam capture uses the browser camera API.

For laptop webcam mode, open:

```text
http://localhost:5173/
```

Browsers generally allow webcam access on localhost during local development. A plain LAN HTTP address such as:

```text
http://172.20.10.11:5173/
```

may not be treated as a secure context for browser webcam access.

Use:

```text
localhost URL     → laptop webcam mode
Network URL       → phone-camera QR mode
```

If laptop camera access fails:

```text
1. Open http://localhost:5173/
2. Allow camera permission in the browser.
3. Check Windows Settings → Privacy & security → Camera.
4. Allow camera access for desktop apps.
5. Close Zoom, Teams, Meet, Camera app, or other webcam users.
6. Refresh the page.
```

---

## 11. Testing the backend

### 11.1 Health check

Laptop-local URL:

```text
http://127.0.0.1:8000/api/health
```

Network URL:

```text
http://<laptop-ip>:8000/api/health
```

### 11.2 Test only the classifier

Windows CMD:

```cmd
curl -X POST "http://127.0.0.1:8000/api/classify-sketch" ^
  -F "file=@path\to\sketch.jpg"
```

macOS/Linux:

```bash
curl -X POST "http://127.0.0.1:8000/api/classify-sketch" \
  -F "file=@path/to/sketch.jpg"
```

### 11.3 Test full deformation endpoint

Automatic organ prediction:

Windows CMD:

```cmd
curl -X POST "http://127.0.0.1:8000/api/deform-sketch" ^
  -F "file=@path\to\sketch.jpg" ^
  -F "save_debug=true"
```

Manual organ override for testing:

```cmd
curl -X POST "http://127.0.0.1:8000/api/deform-sketch" ^
  -F "file=@path\to\sketch.jpg" ^
  -F "organ=heart" ^
  -F "save_debug=true"
```

macOS/Linux:

```bash
curl -X POST "http://127.0.0.1:8000/api/deform-sketch" \
  -F "file=@path/to/sketch.jpg" \
  -F "organ=heart" \
  -F "save_debug=true"
```

A successful response should include a generated model URL, such as:

```json
{
  "status": "success",
  "organ": "heart",
  "model_url": "http://127.0.0.1:8000/outputs/..._deformed.glb"
}
```

### 11.4 Test deformation pipeline directly

From the deformation root:

```bash
cd C:\BioSketchTo3D\backend\deform
python -m app.pipeline
```

Using `python -m app.pipeline` is recommended because it runs the file as a package module and avoids import-path errors.

---

## 12. Testing the frontend

### Upload file mode

```text
1. Open http://localhost:5173/
2. Click Get Started.
3. Choose Upload File.
4. Select an image.
5. Adjust the crop box.
6. Click Submit Sketch.
7. Wait for the GLB model to appear.
```

### Laptop camera mode

```text
1. Open http://localhost:5173/
2. Choose Laptop Camera.
3. Allow camera permission.
4. Capture the sketch.
5. Adjust crop.
6. Submit.
```

### Phone QR mode

```text
1. Run backend and frontend using 0.0.0.0.
2. Open the Vite Network URL on the laptop.
3. Choose Phone Camera QR.
4. Generate QR.
5. Scan using the phone.
6. Capture and send.
7. Crop on laptop.
8. Submit.
```

---

## 13. Debug files and result logs

### 13.1 Deformation debug files

Generated debug files are stored under:

```text
backend/deform/presentation_debug/
```

Typical files include:

```text
*_sketch_border.png
*_correspondence.png
cage_views/
result_tables/
timing_logs/
```

These files are useful for project analysis but should normally remain ignored by Git.

### 13.2 Result-analysis logs

Deformation metrics may be appended to:

```text
backend/deform/presentation_debug/result_tables/deformation_results.jsonl
```

Typical metrics include:

```text
rms_before
rms_after
rms_improvement_percent
similarity_before_percent
similarity_after_percent
mean_deformation_percent
max_deformation_percent
mean_depth_deformation_percent
```

### 13.3 Timing logs

Timing logs may be stored in:

```text
backend/deform/presentation_debug/timing_logs/timing_results.jsonl
```

Typical request timing values include:

```text
request_id
timestamp
status
organ
prediction_source
confidence
input_filename
output_filename
upload_save_time_ms
prediction_time_ms
deformation_time_ms
processing_time_ms
```

---

## 14. Common problems and fixes

### Problem: QR button shows `Failed to fetch`

Check the browser console.

If the frontend still calls an old IP address, delete the local frontend `.env` file:

Windows CMD:

```cmd
cd C:\BioSketchTo3D\frontend
del .env
```

Then restart Vite:

```cmd
npm run dev -- --host 0.0.0.0
```

Open the newly printed Network URL.

Also confirm backend is running with:

```cmd
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Problem: laptop webcam permission error

Open:

```text
http://localhost:5173/
```

Do not use the Network URL for laptop webcam mode.

### Problem: phone cannot open QR page

Test manually on the phone:

```text
http://<laptop-ip>:5173/
http://<laptop-ip>:8000/api/health
```

If these do not open, check firewall and network isolation.

### Problem: `No model generated yet`

Possible causes:

```text
1. Backend did not return model_url.
2. Backend generated the file but /outputs is not mounted.
3. Frontend is passing the wrong prop to ModelViewer.
4. Backend URL is incorrect.
```

### Problem: classifier checkpoint missing or empty

Check:

```text
model_scripts/checkpoints/organ_classifier/best_model.pt
```

It must exist and must not be zero bytes.

### Problem: GLB model missing

Check:

```text
backend/deform/assets/3d_models/brain.glb
backend/deform/assets/3d_models/heart.glb
backend/deform/assets/3d_models/lungs.glb
```

### Problem: `ModuleNotFoundError: No module named 'app'`

Run the deformation pipeline from:

```bash
cd C:\BioSketchTo3D\backend\deform
python -m app.pipeline
```

### Problem: generated model looks distorted

Possible causes:

```text
- sketch outline differs too much from the baseline model
- organ preset allows excessive displacement
- smoothing value is too low
- part constraints are enabled for a region that pulls nearby geometry
- outline includes unwanted protrusions or shadow artifacts
```

Check:

```text
backend/deform/app/config/organ_presets.py
backend/deform/app/deformation/part_constraints.py
backend/deform/presentation_debug/
```

### Problem: binary panel still shows small dots

Check the third panel:

```text
Extracted Border + Anchors
```

If the unwanted dots do not appear as contour anchors, they will not affect FFD. Avoid making preprocessing too aggressive because valid faint pencil strokes may be removed.

### Problem: phone or webcam sketch is poor

Use:

```text
- dark pencil, pen, or marker
- plain unruled sheet
- bright and even light
- flat paper
- minimal shadows
- sketch filling most of the frame
```

### Problem: `npm install` uses an invalid registry

Set the public registry:

```bash
npm config set registry https://registry.npmjs.org/
```

If needed:

```cmd
del package-lock.json
rmdir /s /q node_modules
npm install
```

---

## 15. Git and file-sharing notes

Do not commit generated/runtime files:

```text
backend/uploads/
backend/outputs/
backend/deform/presentation_debug/
frontend/node_modules/
frontend/dist/
backend/.venv/
frontend/.env
```

Keep and commit:

```text
frontend/.env.example
frontend/package.json
frontend/package-lock.json
backend/requirements.txt
```

`frontend/.env.example` is optional documentation for manual override cases. The normal same-Wi-Fi flow should work without creating `frontend/.env`.

Your `.gitignore` should include:

```gitignore
.env
.env.*
!.env.example

frontend/.env
frontend/.env.*
!frontend/.env.example

node_modules/
frontend/node_modules/
frontend/dist/

backend/uploads/
backend/outputs/
backend/deform/presentation_debug/

.venv/
backend/.venv/

.vscode/
backend/deform/.vscode/
model_scripts/.vscode/

__pycache__/
*.pyc
```

If required GLB files or the trained checkpoint are not committed, share them separately and ask teammates to place them in the correct folders.

---

## 16. Useful commands

### Run backend for local + network access

```bash
cd C:\BioSketchTo3D\backend
conda activate biosketch3d
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Run frontend for local + network access

```bash
cd C:\BioSketchTo3D\frontend
npm run dev -- --host 0.0.0.0
```

### Install frontend packages

```bash
cd C:\BioSketchTo3D\frontend
npm install
```

### Test deformation pipeline directly

```bash
cd C:\BioSketchTo3D\backend\deform
python -m app.pipeline
```

### Check current Wi-Fi IPv4 address

```cmd
ipconfig
```

### Delete stale frontend `.env`

```cmd
cd C:\BioSketchTo3D\frontend
del .env
```

### Check backend health locally

```text
http://127.0.0.1:8000/api/health
```

### Build frontend

```bash
cd C:\BioSketchTo3D\frontend
npm run build
```

### Pull latest code

```bash
git pull origin main
```

### Push changes directly to main

```bash
git add .
git status
git commit -m "Update BioSketch3D"
git push origin main
```

Always inspect `git status` before committing.

---

## 17. Final teammate checklist

Before running:

```text
[ ] Clone or pull the latest main branch
[ ] Confirm brain.glb, heart.glb, and lungs.glb exist
[ ] Confirm best_model.pt exists and is not empty
[ ] Install backend requirements
[ ] Install model_scripts requirements too if they are not yet merged
[ ] Run npm install inside frontend
[ ] Run backend using --host 0.0.0.0 --port 8000
[ ] Run frontend using --host 0.0.0.0
[ ] Use http://localhost:5173 for laptop webcam
[ ] Use the Vite Network URL for phone-camera QR flow
[ ] Use a plain unruled sheet and dark pencil, pen, or marker
[ ] Check /api/health if anything fails
```
