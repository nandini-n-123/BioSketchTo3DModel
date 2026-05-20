import os
import sys
import uuid
import time
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


# ============================================================
# Project paths
# ============================================================

# main.py is inside:
# backend/main.py
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

DEFORM_ROOT = BACKEND_DIR / "deform"
MODEL_SCRIPTS_ROOT = PROJECT_ROOT / "model_scripts"
MODEL_SCRIPTS_SRC = MODEL_SCRIPTS_ROOT / "src"

UPLOADS_DIR = BACKEND_DIR / "uploads"
OUTPUTS_DIR = BACKEND_DIR / "outputs"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Allow importing:
#   from app.pipeline import deform_organ
#   from biosketch_classifier.predictor import OrganClassifier
if str(DEFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFORM_ROOT))

if str(MODEL_SCRIPTS_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_SCRIPTS_SRC))


# ============================================================
# Import deformation pipeline
# ============================================================

try:
    from app.pipeline import deform_organ, DEFAULT_DEBUG_DIR
except Exception as e:
    raise RuntimeError(
        "Could not import deformation pipeline. "
        "Check that backend/deform/app/pipeline.py exists and imports correctly."
    ) from e


# ============================================================
# Import AI predictor from model_scripts
# ============================================================

PREDICTOR_AVAILABLE = True
CLASSIFIER = None

try:
    from biosketch_classifier.predictor import OrganClassifier

    CHECKPOINT_PATH = (
        MODEL_SCRIPTS_ROOT
        / "checkpoints"
        / "organ_classifier"
        / "best_model.pt"
    )

    if not CHECKPOINT_PATH.exists() or CHECKPOINT_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            f"Classifier checkpoint is missing or empty: {CHECKPOINT_PATH}"
        )

    CLASSIFIER = OrganClassifier(
        checkpoint_path=CHECKPOINT_PATH,
        confidence_threshold=0.65,
        use_scan_preprocessing=True,
    )

except Exception as e:
    PREDICTOR_AVAILABLE = False
    CLASSIFIER = None
    print(f"[WARNING] Organ classifier could not be loaded: {e}")


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(title="BioSketch-3D API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # okay for college/demo; restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve generated GLB files to frontend.
app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUTS_DIR)),
    name="outputs",
)

# Serve presentation/debug images if needed.
app.mount(
    "/presentation_debug",
    StaticFiles(directory=str(DEFAULT_DEBUG_DIR)),
    name="presentation_debug",
)


# ============================================================
# Helpers
# ============================================================

VALID_ORGANS = {"heart", "brain", "lungs"}


def _file_status(path: Path) -> Dict[str, Any]:
    """Small diagnostics helper used by /api/health."""
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def _asset_status() -> Dict[str, Any]:
    model_dir = DEFORM_ROOT / "assets" / "3d_models"
    checkpoint_path = (
        MODEL_SCRIPTS_ROOT / "checkpoints" / "organ_classifier" / "best_model.pt"
    )

    return {
        "checkpoint": _file_status(checkpoint_path),
        "models": {
            organ: _file_status(model_dir / f"{organ}.glb")
            for organ in sorted(VALID_ORGANS)
        },
    }


def _normalize_organ_name(value: str) -> str:
    """
    Normalize classifier/manual label to expected organ name.
    """
    if value is None:
        raise ValueError("Organ name is missing.")

    organ = str(value).lower().strip()

    aliases = {
        "lung": "lungs",
        "lungs": "lungs",
        "heart": "heart",
        "brain": "brain",
    }

    if organ not in aliases:
        raise ValueError(
            f"Invalid organ '{value}'. Expected one of: heart, brain, lungs."
        )

    return aliases[organ]


def _extract_prediction_result(prediction_output: Any) -> Dict[str, Any]:
    """
    Supports different predictor return styles.

    Accepted:
        "heart"

        {"organ": "heart", "confidence": 95.2}

        {"prediction": "heart", "confidence": 95.2}

        {"class": "heart", "probability": 0.95}
    """
    if isinstance(prediction_output, str):
        organ = _normalize_organ_name(prediction_output)
        return {
            "organ": organ,
            "confidence": None,
            "raw": prediction_output,
        }

    if isinstance(prediction_output, dict):
        organ_value = (
            prediction_output.get("organ")
            or prediction_output.get("prediction")
            or prediction_output.get("class")
            or prediction_output.get("label")
        )

        organ = _normalize_organ_name(organ_value)

        confidence = (
            prediction_output.get("confidence")
            or prediction_output.get("probability")
            or prediction_output.get("score")
        )

        return {
            "organ": organ,
            "confidence": confidence,
            "raw": prediction_output,
        }

    raise ValueError(
        "predict_organ(...) must return either a string or a dictionary."
    )


def _predict_from_model_scripts(sketch_path: str) -> Dict[str, Any]:
    """
    Calls model_scripts/src/biosketch_classifier/predictor.py.
    """
    if not PREDICTOR_AVAILABLE or CLASSIFIER is None:
        raise RuntimeError(
            "Organ classifier is not available. "
            "Check model_scripts/src/biosketch_classifier/predictor.py "
            "and the checkpoint path."
        )

    prediction_output = CLASSIFIER.predict(sketch_path)
    return _extract_prediction_result(prediction_output)


def _safe_file_extension(filename: str, content_type: Optional[str]) -> str:
    """
    Choose safe image extension.
    """
    suffix = Path(filename or "").suffix.lower()

    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return suffix

    if content_type == "image/png":
        return ".png"

    if content_type == "image/webp":
        return ".webp"

    return ".jpg"


async def _save_uploaded_image(file: UploadFile) -> Path:
    """
    Save uploaded sketch to backend/uploads.
    """
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    ext = _safe_file_extension(file.filename, file.content_type)
    filename = f"sketch_{uuid.uuid4().hex}{ext}"

    save_path = UPLOADS_DIR / filename

    with open(save_path, "wb") as f:
        f.write(contents)

    return save_path


def _url_for_file(request: Request, file_path: Optional[str], base_dir: Path, route_prefix: str) -> Optional[str]:
    """
    Convert local file path into frontend-accessible URL.
    """
    if not file_path:
        return None

    path = Path(file_path).resolve()
    base = base_dir.resolve()

    try:
        relative = path.relative_to(base).as_posix()
    except ValueError:
        return None

    base_url = str(request.base_url).rstrip("/")
    return f"{base_url}/{route_prefix}/{relative}"


def _build_metrics_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return only frontend/report-friendly metrics.
    """
    return {
        "rms_before": result.get("rms_before"),
        "rms_after": result.get("rms_after"),
        "rms_improvement_percent": result.get("rms_improvement_percent"),
        "similarity_before_percent": result.get("similarity_before_percent"),
        "similarity_after_percent": result.get("similarity_after_percent"),
        "mean_deformation_percent": result.get("mean_deformation_percent"),
        "max_deformation_percent": result.get("max_deformation_percent"),
        "mean_depth_deformation_percent": result.get("mean_depth_deformation_percent"),
        "max_depth_deformation_percent": result.get("max_depth_deformation_percent"),
    }


# ============================================================
# Routes
# ============================================================

@app.get("/")
def root():
    return {
        "status": "success",
        "message": "BioSketch-3D backend is running.",
    }


@app.get("/api/health")
def health():
    return {
        "status": "success",
        "backend": "running",
        "predictor_available": PREDICTOR_AVAILABLE,
        "valid_organs": sorted(list(VALID_ORGANS)),
        "assets": _asset_status(),
        "endpoints": {
            "main_pipeline": "/api/deform-sketch",
            "classifier_only": "/api/classify-sketch",
            "outputs": "/outputs/{generated_file.glb}",
        },
    }


@app.post("/api/classify-sketch")
async def classify_sketch(file: UploadFile = File(...)):
    """
    Compatibility / quick-test endpoint.

    The connected frontend uses /api/deform-sketch because it needs the final
    GLB, but this route is useful for testing only the CNN classifier.
    """
    sketch_path = await _save_uploaded_image(file)

    try:
        prediction_info = _predict_from_model_scripts(str(sketch_path))
        predicted_organ = prediction_info["organ"]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Organ prediction failed: {str(e)}",
        )

    return {
        "status": "success",
        "organ": predicted_organ,
        "prediction": predicted_organ,
        "confidence": prediction_info.get("confidence"),
        "raw_prediction": prediction_info.get("raw"),
    }


@app.post("/api/deform-sketch")
async def deform_sketch(
    request: Request,
    file: UploadFile = File(...),
    organ: Optional[str] = Form(None),
    save_debug: bool = Form(True),
):
    """
    Main frontend endpoint.

    Frontend sends:
        file: sketch image

    Optional:
        organ: heart / brain / lungs

    If organ is not provided:
        backend uses model_scripts/predictor.py to classify the sketch.

    Returns:
        model_url: frontend-loadable URL to deformed GLB
        organ: predicted/selected organ
        metrics: deformation result metrics
    """

    request_start_time = time.perf_counter()

    # 1. Save uploaded sketch.
    upload_save_start_time = time.perf_counter()
    sketch_path = await _save_uploaded_image(file)
    upload_save_time_ms = round((time.perf_counter() - upload_save_start_time) * 1000, 2)

    # 2. Get organ from manual input or classifier.
    prediction_start_time = time.perf_counter()
    try:
        if organ:
            normalized_organ = _normalize_organ_name(organ)
            prediction_info = {
                "organ": normalized_organ,
                "confidence": None,
                "source": "manual",
            }
        else:
            prediction_info = _predict_from_model_scripts(str(sketch_path))
            prediction_info["source"] = "model_scripts"

        predicted_organ = prediction_info["organ"]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Organ prediction failed: {str(e)}",
        )

    prediction_time_ms = round((time.perf_counter() - prediction_start_time) * 1000, 2)

    # 3. Prepare exact output path.
    output_filename = f"{predicted_organ}_{uuid.uuid4().hex}_deformed.glb"
    output_path = OUTPUTS_DIR / output_filename

    # 4. Run deterministic deformation pipeline.
    deformation_start_time = time.perf_counter()
    try:
        result = deform_organ(
            sketch_path=str(sketch_path),
            organ=predicted_organ,
            output_path=str(output_path),
            save_debug=save_debug,
            apply_debug_colors=False,
            log_result=True,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Deformation pipeline failed: {str(e)}",
        )

    deformation_time_ms = round((time.perf_counter() - deformation_start_time) * 1000, 2)

    # 5. Build URLs for frontend.
    model_url = _url_for_file(
        request=request,
        file_path=result.get("output_path"),
        base_dir=OUTPUTS_DIR,
        route_prefix="outputs",
    )

    debug_url = _url_for_file(
        request=request,
        file_path=result.get("debug_path"),
        base_dir=DEFAULT_DEBUG_DIR,
        route_prefix="presentation_debug",
    )

    sketch_debug_url = _url_for_file(
        request=request,
        file_path=result.get("sketch_debug_path"),
        base_dir=DEFAULT_DEBUG_DIR,
        route_prefix="presentation_debug",
    )

    if not model_url:
        raise HTTPException(
            status_code=500,
            detail="Deformation completed, but backend could not build a model URL.",
        )

    processing_time_ms = round((time.perf_counter() - request_start_time) * 1000, 2)

    return {
        "status": "success",
        "message": "Sketch classified and deformed successfully.",
        "organ": predicted_organ,
        "prediction": {
            "organ": predicted_organ,
            "confidence": prediction_info.get("confidence"),
            "source": prediction_info.get("source"),
        },
        "model_url": model_url,
        "output_filename": Path(result["output_path"]).name,
        "processing_time_ms": processing_time_ms,
        "timing": {
            "processing_time_ms": processing_time_ms,
            "upload_save_time_ms": upload_save_time_ms,
            "prediction_time_ms": prediction_time_ms,
            "deformation_time_ms": deformation_time_ms,
        },
        "debug": {
            "correspondence_url": debug_url,
            "sketch_border_url": sketch_debug_url,
        },
        "metrics": _build_metrics_response(result),
    }


@app.get("/api/download/{filename}")
def download_output(filename: str):
    """
    Optional direct download endpoint for a generated GLB.
    """
    file_path = OUTPUTS_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="File not found.",
        )

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="model/gltf-binary",
    )


@app.get("/api/export-results-table")
def export_results_table():
    """
    Optional endpoint to export the accumulated CSV table.
    """
    try:
        from app.result_table import export_csv
        csv_path = export_csv()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not export CSV: {str(e)}",
        )

    return {
        "status": "success",
        "csv_path": csv_path,
    }