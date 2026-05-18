# BioSketch3D

BioSketch3D is a web-based application that converts hand-drawn biological sketches into interactive 3D organ models. The project combines a React frontend, a FastAPI backend, an EfficientNet-based organ classifier, OpenCV sketch preprocessing, and a deterministic Free-Form Deformation (FFD) pipeline.

Unlike generative image-to-3D systems, this project does not hallucinate new anatomy. It starts from validated baseline 3D organ assets and deforms them according to the outline of the user's sketch. This keeps the final model fast, stable, and more suitable for biological education.

---

## Table of contents

1. [Project overview](#project-overview)
2. [How the pipeline works](#how-the-pipeline-works)
3. [Repository structure](#repository-structure)
4. [Required files before running](#required-files-before-running)
5. [Prerequisites](#prerequisites)
6. [First-time setup](#first-time-setup)
7. [Running the project](#running-the-project)
8. [Testing the backend](#testing-the-backend)
9. [Testing the frontend](#testing-the-frontend)
10. [Common problems and fixes](#common-problems-and-fixes)
11. [Git and file sharing notes](#git-and-file-sharing-notes)
12. [Useful commands](#useful-commands)

---

## Project overview

The application has three main parts:

```text
BioSketchTo3D/
├── backend/          # FastAPI API + deformation pipeline entry point
├── frontend/         # React + Vite user interface and 3D viewer
└── model_scripts/    # CNN classifier code and trained checkpoint
```

The user flow is:

```text
User captures/uploads sketch
        ↓
Frontend lets user crop and optionally enhance the sketch
        ↓
Frontend sends cropped image to FastAPI backend
        ↓
Backend classifies organ using model_scripts CNN predictor
        ↓
Backend selects correct baseline 3D asset
        ↓
OpenCV extracts sketch contour
        ↓
FFD/lattice deformation modifies the baseline GLB
        ↓
Backend saves generated .glb in backend/outputs
        ↓
Frontend loads the returned model_url and displays the 3D model
```

Supported organs at the moment:

```text
heart
brain
lungs
```

---

## How the pipeline works

### 1. Frontend capture and crop

The frontend lets users either upload an image or capture one from the laptop webcam. The user can manually crop the sketch before submitting. This is important because the backend deformation works best when the sketch area is clean and centered.

The frontend can also show an enhanced scan-style preview. This helps with poor webcam images by converting the cropped sketch into a cleaner white-background, dark-line image before sending it to the backend.

### 2. Backend classification

The backend loads the classifier from:

```text
model_scripts/src/biosketch_classifier/predictor.py
```

The trained checkpoint should be placed at:

```text
model_scripts/checkpoints/organ_classifier/best_model.pt
```

If the frontend sends a manual organ value, the backend can skip automatic classification and directly use that selected organ.

### 3. OpenCV preprocessing

The backend processes the sketch image using OpenCV to extract the visible contour/outline of the biological diagram. Debug images are saved under:

```text
backend/deform/presentation_debug/
```

These debug outputs are useful for checking whether the border and anchor extraction worked correctly.

### 4. Deterministic deformation

Instead of generating a new 3D model from scratch, the backend loads a validated baseline GLB model from:

```text
backend/deform/assets/3d_models/
```

Then it deforms the model using a lattice cage and Free-Form Deformation. The result is saved in:

```text
backend/outputs/
```

The backend returns a URL like:

```text
http://127.0.0.1:8000/outputs/heart_xxxxxxxxx_deformed.glb
```

The frontend uses this URL to render the 3D model.

---

## Repository structure

Expected structure:

```text
BioSketchTo3D/
├── README.md
├── .gitignore
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── uploads/                     # generated at runtime, ignored by Git
│   ├── outputs/                     # generated GLB files, ignored by Git
│   └── deform/
│       ├── app/
│       │   ├── pipeline.py
│       │   ├── config/
│       │   ├── deformation/
│       │   ├── geometry/
│       │   └── vision/
│       ├── assets/
│       │   └── 3d_models/
│       │       ├── brain.glb        # required, usually not committed
│       │       ├── heart.glb        # required, usually not committed
│       │       └── lungs.glb        # required, usually not committed
│       └── presentation_debug/      # generated debug files, ignored by Git
│
├── frontend/
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   ├── .env.example
│   ├── .env                         # local only, ignored by Git
│   ├── index.html
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       ├── index.css
│       ├── UploadSection.jsx
│       └── ModelViewer.jsx
│
└── model_scripts/
    ├── scripts/
    │   ├── 02_train.py
    │   └── 03_test_image.py
    ├── src/
    │   └── biosketch_classifier/
    └── checkpoints/
        └── organ_classifier/
            └── best_model.pt        # required, usually not committed
```

---

## Required files before running

The project will not fully run unless these files exist and are not empty:

```text
backend/deform/assets/3d_models/brain.glb
backend/deform/assets/3d_models/heart.glb
backend/deform/assets/3d_models/lungs.glb
model_scripts/checkpoints/organ_classifier/best_model.pt
```

Important checks:

```text
1. The .glb files must be real 3D model files.
2. best_model.pt must be the trained classifier checkpoint.
3. A 0-byte best_model.pt will fail.
4. If these files are too large for GitHub, share them using Google Drive, OneDrive, GitHub Releases, or Git LFS.
```

After placing the files, check their status from the backend health endpoint:

```text
http://127.0.0.1:8000/api/health
```

The response should show `exists: true` and `size_bytes` greater than 0 for every required asset.

---

## Prerequisites

Install these before running the project:

### 1. Python

Recommended:

```text
Python 3.10 or 3.11
```

Check version:

```bash
python --version
```

or on some systems:

```bash
python3 --version
```

### 2. Node.js and npm

Recommended:

```text
Node.js 18+
npm 9+
```

Check versions:

```bash
node -v
npm -v
```

### 3. Git

Check version:

```bash
git --version
```

---

## First-time setup

Clone the repository:

```bash
git clone <your-repository-url>
cd BioSketchTo3D
```

If you downloaded the project as a zip, extract it and open the root folder in VS Code.

---

## Running the project

You need two terminals:

```text
Terminal 1: backend
Terminal 2: frontend
```

---

### Terminal 1: Run backend

#### Windows PowerShell

From the project root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

If PowerShell blocks activation, run this once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again:

```powershell
.\.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
cd backend
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

#### macOS/Linux

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Backend should show something like:

```text
Uvicorn running on http://127.0.0.1:8000
```

Open this in the browser:

```text
http://127.0.0.1:8000/api/health
```

---

### Terminal 2: Run frontend

Open a second terminal from the project root.

```bash
cd frontend
npm install
```

Create the frontend environment file.

#### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

#### Windows Command Prompt

```cmd
copy .env.example .env
```

#### macOS/Linux

```bash
cp .env.example .env
```

Open `frontend/.env` and make sure it contains:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Start the frontend:

```bash
npm run dev
```

Frontend should show something like:

```text
Local: http://localhost:5173/
```

Open:

```text
http://localhost:5173/
```

---

## Testing the backend

### 1. Health check

Open:

```text
http://127.0.0.1:8000/api/health
```

Expected:

```json
{
  "status": "success",
  "backend": "running",
  "predictor_available": true
}
```

If `predictor_available` is false, check:

```text
model_scripts/checkpoints/organ_classifier/best_model.pt
```

It must exist and must not be empty.

### 2. Test only classifier

Use `/api/classify-sketch` if you want to test the CNN without deformation.

With curl:

```bash
curl -X POST "http://127.0.0.1:8000/api/classify-sketch" ^
  -F "file=@path/to/sketch.jpg"
```

On macOS/Linux:

```bash
curl -X POST "http://127.0.0.1:8000/api/classify-sketch" \
  -F "file=@path/to/sketch.jpg"
```

### 3. Test full deformation endpoint

```bash
curl -X POST "http://127.0.0.1:8000/api/deform-sketch" ^
  -F "file=@path/to/sketch.jpg" ^
  -F "organ=heart" ^
  -F "save_debug=true"
```

On macOS/Linux:

```bash
curl -X POST "http://127.0.0.1:8000/api/deform-sketch" \
  -F "file=@path/to/sketch.jpg" \
  -F "organ=heart" \
  -F "save_debug=true"
```

If successful, the response should include:

```json
{
  "status": "success",
  "organ": "heart",
  "model_url": "http://127.0.0.1:8000/outputs/..._deformed.glb"
}
```

Copy the `model_url` into your browser. If the file downloads, the backend is serving generated GLB files correctly.

---

## Testing the frontend

After both backend and frontend are running:

1. Open the frontend URL.
2. Click **Get Started**.
3. Choose **Upload File** or **Use Camera**.
4. Upload/capture a sketch.
5. Adjust the crop box.
6. Submit the cropped sketch.
7. Wait for backend processing.
8. The generated 3D model should appear in the viewer.

For webcam testing:

```text
Use a dark pen or marker.
Keep the paper flat.
Use bright lighting.
Fill most of the camera frame with the sketch.
Avoid shadows and glare.
```

Laptop webcams often struggle with faint pencil lines. For demos, use a black pen sketch on white paper.

---

## Common problems and fixes

### Problem: `No model generated yet` appears in frontend

Possible causes:

```text
1. Backend did not return model_url.
2. Frontend is passing wrong prop to ModelViewer.
3. Backend generated file but /outputs is not mounted.
4. VITE_API_BASE_URL is missing or incorrect.
```

Fix checklist:

```text
1. Check browser DevTools Console.
2. Check backend terminal logs.
3. Confirm response from /api/deform-sketch contains model_url.
4. Open model_url directly in browser.
5. Make sure frontend/.env has VITE_API_BASE_URL=http://127.0.0.1:8000.
```

### Problem: Backend says classifier checkpoint is missing or empty

Check this file:

```text
model_scripts/checkpoints/organ_classifier/best_model.pt
```

It must be the real trained checkpoint, not a placeholder.

### Problem: Backend cannot find GLB model

Check these files:

```text
backend/deform/assets/3d_models/brain.glb
backend/deform/assets/3d_models/heart.glb
backend/deform/assets/3d_models/lungs.glb
```

The filenames must match exactly.

### Problem: CORS or failed fetch error in frontend

Make sure backend is running:

```text
http://127.0.0.1:8000/api/health
```

Make sure frontend `.env` has:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Then restart Vite:

```bash
npm run dev
```

### Problem: `ModuleNotFoundError` in backend

Make sure you are running the command from the `backend` folder:

```bash
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Do not run backend from inside `backend/deform`.

### Problem: Torch installation is slow or fails

Try upgrading pip first:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If you need a specific CPU/CUDA PyTorch version, install PyTorch using the official command for your system, then run:

```bash
pip install -r requirements.txt
```

### Problem: Model appears cropped or too zoomed in

Use the updated `ModelViewer.jsx` that normalizes the model using its bounding box. Also make sure the viewer container CSS gives the canvas a real height:

```css
.model-canvas-wrapper {
  width: 100%;
  height: 100%;
}
```

### Problem: Webcam result is poor

Use a dark pen/marker instead of pencil. The frontend scan preview helps, but blurry images cannot be fully recovered.

---

## Git and file sharing notes

Do not commit generated or heavy runtime files:

```text
backend/uploads/
backend/outputs/
backend/deform/presentation_debug/
frontend/node_modules/
frontend/dist/
backend/.venv/
```

Usually, do not commit these large required assets directly:

```text
*.pt
*.pth
*.glb
*.obj
*.fbx
```

Recommended options for large files:

```text
1. Git LFS
2. Google Drive shared folder
3. OneDrive shared folder
4. GitHub Releases
```

If using Git LFS:

```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.glb"
git add .gitattributes
git add backend/deform/assets/3d_models/*.glb
git add model_scripts/checkpoints/organ_classifier/best_model.pt
git commit -m "Add model assets with Git LFS"
```

If not using Git LFS, keep those files outside Git and ask teammates to manually place them in the required folders.

---

## Useful commands

### Backend

```bash
cd backend
.venv\Scripts\activate      # Windows CMD
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

macOS/Linux:

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Frontend production build test

```bash
cd frontend
npm run build
npm run preview
```

### Check backend health

```text
http://127.0.0.1:8000/api/health
```

### Clear generated backend files

Windows PowerShell:

```powershell
Remove-Item backend\uploads\* -Force -ErrorAction SilentlyContinue
Remove-Item backend\outputs\* -Force -ErrorAction SilentlyContinue
```

macOS/Linux:

```bash
rm -rf backend/uploads/* backend/outputs/*
```

---

## Development notes for teammates

### Frontend responsibilities

```text
- Capture/upload sketch
- Allow crop adjustment
- Optional scan-style preview/enhancement
- Send image to backend
- Render returned GLB model URL
```

### Backend responsibilities

```text
- Save uploaded sketch
- Classify organ or accept manual organ input
- Run OpenCV contour extraction
- Run deterministic deformation pipeline
- Save generated GLB
- Return frontend-accessible model_url
```

### model_scripts responsibilities

```text
- Train/test organ classifier
- Provide predictor used by backend
- Store/load trained checkpoint
```

---

## Final checklist before pushing to GitHub

Run this before committing:

```bash
git status
```

Make sure you are not committing:

```text
node_modules/
.venv/
__pycache__/
backend/uploads/
backend/outputs/
large .pt/.glb files unless using Git LFS
```

Recommended commit contents:

```text
README.md
.gitignore
backend/main.py
backend/requirements.txt
backend/deform/app/**/*.py
frontend/package.json
frontend/package-lock.json
frontend/src/**/*.jsx
frontend/src/**/*.css
frontend/.env.example
model_scripts/src/**/*.py
model_scripts/scripts/**/*.py
```
