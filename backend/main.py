import json
import sys
import uuid
import time
from datetime import datetime
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
# Project constants
# ============================================================

VALID_ORGANS = {"heart", "brain", "lungs"}

# Auto-detected sketches must reach this confidence before the backend deforms
# a 3D model. Increase this value if unsupported sketches are still passing.
# Decrease it if genuine brain/heart/lungs sketches are being rejected too often.
MIN_AUTO_CONFIDENCE = 0.60

UNSUPPORTED_LABELS = {
    "unsupported",
    "unknown",
    "other",
    "none",
    "not_supported",
    "not supported",
    "out_of_distribution",
    "out of distribution",
}

UNSUPPORTED_ORGAN_MESSAGE = (
    "Sorry, this sketch does not match the organs currently supported by "
    "BioSketch3D. At present, the system supports only Brain, Heart, and Lungs "
    "because validated 3D models are available only for these organs."
)


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


# Timing logs are stored with the existing presentation/debug outputs.
TIMING_LOG_DIR = Path(DEFAULT_DEBUG_DIR) / "timing_logs"
TIMING_LOG_PATH = TIMING_LOG_DIR / "timing_results.jsonl"
TIMING_LOG_DIR.mkdir(parents=True, exist_ok=True)


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

# Serve presentation/debug images and timing logs if needed.
app.mount(
    "/presentation_debug",
    StaticFiles(directory=str(DEFAULT_DEBUG_DIR)),
    name="presentation_debug",
)


# ============================================================
# Helpers
# ============================================================

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
    Normalize manual labels to expected organ names.
    This is used for manual organ selection and known classifier labels.
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


def _confidence_to_float(confidence: Any) -> Optional[float]:
    """
    Converts confidence into 0.0 - 1.0 range.

    Supports:
        0.82
        "0.82"
        82
        "82%"
    """
    if confidence is None:
        return None

    try:
        value = str(confidence).replace("%", "").strip()
        value = float(value)
    except Exception:
        return None

    if value > 1:
        value = value / 100.0

    return max(0.0, min(1.0, value))


def _extract_prediction_result(prediction_output: Any) -> Dict[str, Any]:
    """
    Supports different predictor return styles.

    Accepted:
        "heart"

        {"organ": "heart", "confidence": 95.2}

        {"prediction": "heart", "confidence": 95.2}

        {"class": "heart", "probability": 0.95}

        {"organ": "unsupported", "confidence": 0.41}

    If the classifier ever returns an unexpected label such as "kidney",
    it is treated as unsupported instead of being forced into deformation.
    """
    if isinstance(prediction_output, str):
        raw_label = prediction_output.lower().strip()

        if raw_label in UNSUPPORTED_LABELS or raw_label not in VALID_ORGANS:
            return {
                "organ": "unsupported",
                "confidence": None,
                "raw": prediction_output,
            }

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

        confidence = (
            prediction_output.get("confidence")
            or prediction_output.get("probability")
            or prediction_output.get("score")
        )

        raw_label = str(organ_value).lower().strip() if organ_value is not None else ""

        if raw_label in UNSUPPORTED_LABELS or raw_label not in VALID_ORGANS:
            organ = "unsupported"
        else:
            organ = _normalize_organ_name(organ_value)

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


def _build_unsupported_detail(
    prediction_info: Dict[str, Any],
    prediction_time_ms: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Returns a structured unsupported-organ error if the auto classifier is not
    confident enough or if it returns an unsupported/unknown label.

    This check only applies to auto-detection. Manual organ selection is treated
    as an intentional override by the user.
    """
    if prediction_info.get("source") != "model_scripts":
        return None

    predicted_organ = prediction_info.get("organ")
    confidence_float = _confidence_to_float(prediction_info.get("confidence"))

    is_unknown_label = predicted_organ not in VALID_ORGANS
    is_low_confidence = (
        confidence_float is not None and confidence_float < MIN_AUTO_CONFIDENCE
    )

    if not is_unknown_label and not is_low_confidence:
        return None

    return {
        "code": "UNSUPPORTED_ORGAN",
        "message": UNSUPPORTED_ORGAN_MESSAGE,
        "predicted_organ": predicted_organ,
        "confidence_percent": (
            round(confidence_float * 100, 2)
            if confidence_float is not None
            else None
        ),
        "minimum_required_percent": round(MIN_AUTO_CONFIDENCE * 100, 2),
        "supported_organs": sorted(list(VALID_ORGANS)),
        "prediction_time_ms": prediction_time_ms,
        "raw_prediction": prediction_info.get("raw"),
    }


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


def _url_for_file(
    request: Request,
    file_path: Optional[str],
    base_dir: Path,
    route_prefix: str,
) -> Optional[str]:
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


def _append_timing_log(record: Dict[str, Any]) -> None:
    """
    Store one backend timing record as JSONL.
    JSONL = one JSON object per line, useful for experiment logs/report analysis.
    """
    try:
        TIMING_LOG_DIR.mkdir(parents=True, exist_ok=True)

        with open(TIMING_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"[WARNING] Could not write timing log: {e}")


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
        "minimum_auto_confidence_percent": round(MIN_AUTO_CONFIDENCE * 100, 2),
        "assets": _asset_status(),
        "timing_log": {
            "path": str(TIMING_LOG_PATH),
            "exists": TIMING_LOG_PATH.exists(),
        },
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
        prediction_info["source"] = "model_scripts"
        unsupported_detail = _build_unsupported_detail(prediction_info)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Organ prediction failed: {str(e)}",
        )

    return {
        "status": "success",
        "organ": predicted_organ,
        "prediction": predicted_organ,
        "is_supported": unsupported_detail is None,
        "unsupported_detail": unsupported_detail,
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

    If auto-detection confidence is too low or the sketch is unsupported:
        backend returns HTTP 422 and does not run deformation.

    Returns:
        model_url: frontend-loadable URL to deformed GLB
        organ: predicted/selected organ
        metrics: deformation result metrics
        timing: backend timing breakdown
    """

    request_id = uuid.uuid4().hex
    request_start_time = time.perf_counter()
    timestamp = datetime.now().isoformat(timespec="seconds")

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
                "raw": organ,
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

    unsupported_detail = _build_unsupported_detail(
        prediction_info=prediction_info,
        prediction_time_ms=prediction_time_ms,
    )

    if unsupported_detail is not None:
        processing_time_ms = round((time.perf_counter() - request_start_time) * 1000, 2)

        _append_timing_log(
            {
                "request_id": request_id,
                "timestamp": timestamp,
                "status": "unsupported",
                "organ": predicted_organ,
                "prediction_source": prediction_info.get("source"),
                "confidence": prediction_info.get("confidence"),
                "input_filename": sketch_path.name,
                "output_filename": None,
                "upload_save_time_ms": upload_save_time_ms,
                "prediction_time_ms": prediction_time_ms,
                "deformation_time_ms": None,
                "processing_time_ms": processing_time_ms,
                "reason": unsupported_detail,
            }
        )

        raise HTTPException(status_code=422, detail=unsupported_detail)

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
        base_dir=Path(DEFAULT_DEBUG_DIR),
        route_prefix="presentation_debug",
    )

    sketch_debug_url = _url_for_file(
        request=request,
        file_path=result.get("sketch_debug_path"),
        base_dir=Path(DEFAULT_DEBUG_DIR),
        route_prefix="presentation_debug",
    )

    if not model_url:
        raise HTTPException(
            status_code=500,
            detail="Deformation completed, but backend could not build a model URL.",
        )

    processing_time_ms = round((time.perf_counter() - request_start_time) * 1000, 2)
    metrics = _build_metrics_response(result)

    timing = {
        "processing_time_ms": processing_time_ms,
        "upload_save_time_ms": upload_save_time_ms,
        "prediction_time_ms": prediction_time_ms,
        "deformation_time_ms": deformation_time_ms,
    }

    _append_timing_log(
        {
            "request_id": request_id,
            "timestamp": timestamp,
            "status": "success",
            "organ": predicted_organ,
            "prediction_source": prediction_info.get("source"),
            "confidence": prediction_info.get("confidence"),
            "input_filename": sketch_path.name,
            "output_filename": Path(result["output_path"]).name,
            **timing,
            "rms_before": result.get("rms_before"),
            "rms_after": result.get("rms_after"),
            "similarity_after_percent": result.get("similarity_after_percent"),
            "mean_deformation_percent": result.get("mean_deformation_percent"),
        }
    )

    return {
        "status": "success",
        "message": "Sketch classified and deformed successfully.",
        "request_id": request_id,
        "organ": predicted_organ,
        "prediction": {
            "organ": predicted_organ,
            "confidence": prediction_info.get("confidence"),
            "source": prediction_info.get("source"),
        },
        "model_url": model_url,
        "output_filename": Path(result["output_path"]).name,
        "processing_time_ms": processing_time_ms,
        "timing": timing,
        "debug": {
            "correspondence_url": debug_url,
            "sketch_border_url": sketch_debug_url,
        },
        "metrics": metrics,
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
