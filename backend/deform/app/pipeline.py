"""
pipeline.py

Clean callable deformation pipeline for backend/frontend integration.

This file is meant to be called by backend/main.py, not run as an
interactive test script.

Main function:
    deform_organ(
        sketch_path="path/to/uploaded/sketch.jpg",
        organ="heart",
        model_path=None,
        output_path=None,
        save_debug=True,
    )

Returns a dictionary containing:
    - organ
    - model_path
    - sketch_path
    - output_path
    - debug_path
    - sketch_debug_path
    - rms_before
    - rms_after
"""

"""
pipeline.py

Clean callable deformation pipeline for backend/frontend integration.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# ============================================================
# Fix Python import path
# ============================================================

# pipeline.py is inside:
# backend/deform/app/pipeline.py
APP_DIR = Path(__file__).resolve().parent
DEFORM_ROOT = APP_DIR.parent              # backend/deform
BACKEND_ROOT = DEFORM_ROOT.parent         # backend

# Add backend/deform to Python path so imports like
# from app.vision.sketch_parser import ... work correctly.
if str(DEFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFORM_ROOT))

import cv2
import numpy as np
import trimesh

from app.vision.sketch_parser import (
    preprocess_sketch,
    extract_contours,
    optimize_vectors,
    normalize_coordinates,
    get_primary_contour_points,
)

from app.geometry.mesh_prep import load_and_normalize_mesh

from app.deformation.lattice import (
    LatticeFFD,
    bounds_from_vertices,
)

from app.deformation.cage_weights import calculate_weight_matrix

from app.deformation.solver import projected_mesh_outline

from app.deformation.correspondence import (
    build_landmark_correspondence,
    find_nearest_mesh_vertices_for_outline,
    save_correspondence_debug,
)

from app.deformation.weighted_solver import solve_depth_locked_weighted_cage

from app.deformation.part_constraints import build_part_constraints

from app.config.organ_presets import get_organ_preset


from app.deformation.metrics import (
    rms_improvement_percent,
    contour_similarity_percent,
    deformation_stats,
)

from app.result_table import append_result
from app.deformation.cage_visualization import export_scene_with_cage
from app.deformation.demo_emphasis import emphasize_brainstem_from_sketch
# ============================================================
# Paths
# ============================================================

# pipeline.py is inside:

ASSETS_DIR = DEFORM_ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "3d_models"

DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "outputs"
DEFAULT_DEBUG_DIR = DEFORM_ROOT / "presentation_debug"


# ============================================================
# Axis convention
# ============================================================

# For exported/downloaded GLB/glTF front-facing assets:
# X/Y = visible 2D sketch plane
# Z = depth
PLANE_AXES = (0, 1)
DEPTH_AXIS = 2


# ============================================================
# Organ default models
# ============================================================

DEFAULT_MODEL_FILES = {
    "heart": "heart.glb",
    "brain": "brain.glb",
    "lungs": "lungs.glb",
}


# ============================================================
# Helper functions
# ============================================================

def _validate_organ(organ: str) -> str:
    organ = organ.lower().strip()

    if organ not in DEFAULT_MODEL_FILES:
        raise ValueError(
            f"Unknown organ '{organ}'. Expected one of: "
            f"{list(DEFAULT_MODEL_FILES.keys())}"
        )

    return organ


def _resolve_model_path(organ: str, model_path: Optional[str] = None) -> Path:
    """
    Resolve model path.

    If model_path is None:
        use backend/deform/assets/3d_models/{organ}.glb
    """
    organ = _validate_organ(organ)

    if model_path is None:
        path = MODELS_DIR / DEFAULT_MODEL_FILES[organ]
    else:
        path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"3D model not found: {path}")

    return path


def _make_output_path(
    organ: str,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """
    Create output path for final deformed GLB.
    """
    organ = _validate_organ(organ)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    if output_dir is None:
        out_dir = DEFAULT_OUTPUT_DIR
    else:
        out_dir = Path(output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    filename = f"{organ}_deformed_{timestamp}.glb"

    return out_dir / filename


def _make_debug_path(
    organ: str,
    sketch_path: str,
    model_path: str,
    debug_dir: Optional[str] = None,
) -> Path:
    """
    Create clean correspondence debug image path.
    """
    organ = _validate_organ(organ)

    if debug_dir is None:
        dbg_dir = DEFAULT_DEBUG_DIR
    else:
        dbg_dir = Path(debug_dir)

    dbg_dir.mkdir(parents=True, exist_ok=True)

    sketch_base = Path(sketch_path).stem
    model_base = Path(model_path).stem

    return dbg_dir / f"{organ}_{model_base}_{sketch_base}_correspondence.png"


def _make_sketch_debug_path(
    organ: str,
    sketch_path: str,
    model_path: str,
    debug_dir: Optional[str] = None,
) -> Path:
    """
    Create clean sketch-border debug image path.
    """
    organ = _validate_organ(organ)

    if debug_dir is None:
        dbg_dir = DEFAULT_DEBUG_DIR
    else:
        dbg_dir = Path(debug_dir)

    dbg_dir.mkdir(parents=True, exist_ok=True)

    sketch_base = Path(sketch_path).stem
    model_base = Path(model_path).stem

    return dbg_dir / f"{organ}_{model_base}_{sketch_base}_sketch_border.png"

def _make_cage_debug_paths(
    organ: str,
    sketch_path: str,
    model_path: str,
    debug_dir: Optional[str] = None,
):
    """
    Create before/after cage visualization paths.
    """
    organ = _validate_organ(organ)

    if debug_dir is None:
        dbg_dir = DEFAULT_DEBUG_DIR
    else:
        dbg_dir = Path(debug_dir)

    cage_dir = dbg_dir / "cage_views"
    cage_dir.mkdir(parents=True, exist_ok=True)

    sketch_base = Path(sketch_path).stem
    model_base = Path(model_path).stem

    before_path = cage_dir / f"{organ}_{model_base}_{sketch_base}_before_cage.glb"
    after_path = cage_dir / f"{organ}_{model_base}_{sketch_base}_after_cage.glb"

    return before_path, after_path


def _save_sketch_border_debug(
    sketch_path: str,
    processed_img: np.ndarray,
    optimized_contours: dict,
    output_path: str,
) -> None:
    """
    Save a presentation-friendly visualization of sketch preprocessing.

    Output image shows:
      left  = original sketch
      middle = thresholded/preprocessed binary image
      right = extracted outer contour + anchor points
    """
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    original = cv2.imread(str(sketch_path))

    if original is None:
        raise FileNotFoundError(f"Could not read sketch for debug visualization: {sketch_path}")

    # Make processed image 3-channel for side-by-side display.
    if len(processed_img.shape) == 2:
        processed_bgr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    else:
        processed_bgr = processed_img.copy()

    border_canvas = np.zeros_like(original)

    contours = optimized_contours.get("outer", [])

    if contours:
        # Draw extracted outer contour.
        cv2.drawContours(
            border_canvas,
            contours,
            -1,
            (0, 255, 0),
            2,
        )

        # Draw anchor/control points.
        for cnt in contours:
            for pt in cnt:
                x = int(pt[0][0])
                y = int(pt[0][1])
                cv2.circle(
                    border_canvas,
                    (x, y),
                    5,
                    (255, 255, 0),
                    -1,
                )

    # Resize all panels to same height.
    display_height = 600

    def _resize_to_height(img, height):
        h, w = img.shape[:2]
        scale = height / max(h, 1)
        new_w = int(w * scale)
        return cv2.resize(img, (new_w, height))

    original_panel = _resize_to_height(original, display_height)
    processed_panel = _resize_to_height(processed_bgr, display_height)
    border_panel = _resize_to_height(border_canvas, display_height)

    # Add labels.
    def _add_label(img, label):
        labeled = img.copy()
        cv2.rectangle(
            labeled,
            (0, 0),
            (labeled.shape[1], 40),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            labeled,
            label,
            (12, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return labeled

    original_panel = _add_label(original_panel, "1. Original Sketch")
    processed_panel = _add_label(processed_panel, "2. Preprocessed Binary")
    border_panel = _add_label(border_panel, "3. Extracted Border + Anchors")

    combined = cv2.hconcat([
        original_panel,
        processed_panel,
        border_panel,
    ])

    cv2.imwrite(output_path, combined)

    print(f"[DEBUG] Saved sketch border image: {output_path}")


def _stack_scene_vertices(scene: trimesh.Scene) -> np.ndarray:
    """
    Stack all geometry vertices into one global vertex array.

    This lets the whole anatomy deform through one shared cage.
    """
    if not scene.geometry:
        raise ValueError("Scene has no geometry.")

    return np.vstack([
        geom.vertices.copy()
        for geom in scene.geometry.values()
    ])


def _overwrite_scene_vertices(scene: trimesh.Scene, vertices: np.ndarray) -> None:
    """
    Write global vertex array back into scene geometries.

    This preserves the original scene structure while applying one global
    deformation result.

    Important:
    After editing vertices, clear trimesh caches so normals/bounds are rebuilt.
    """
    offset = 0

    for geom in scene.geometry.values():
        n = len(geom.vertices)

        new_vertices = vertices[offset: offset + n]

        # Keep vertices as float32/float64 numeric arrays.
        geom.vertices = np.asarray(new_vertices, dtype=np.float64)

        # Clear cached normals/bounds because vertex positions changed.
        try:
            geom._cache.clear()
        except Exception:
            pass

        offset += n

    if offset != len(vertices):
        raise ValueError(
            f"Vertex overwrite mismatch. Used {offset}, "
            f"but got {len(vertices)} vertices."
        )


def _apply_plane_bbox_alignment(
    vertices: np.ndarray,
    target_2d: np.ndarray,
    plane_axes=PLANE_AXES,
    depth_axis=DEPTH_AXIS,
) -> np.ndarray:
    """
    Align model bounding box to sketch bounding box before FFD.

    This handles simple translation and scale first.
    The lattice then only handles residual shape deformation.
    """
    v = vertices.copy()
    src = v[:, plane_axes]

    src_min = src.min(axis=0)
    src_max = src.max(axis=0)

    tgt_min = target_2d.min(axis=0)
    tgt_max = target_2d.max(axis=0)

    src_size = np.maximum(src_max - src_min, 1e-8)
    tgt_size = np.maximum(tgt_max - tgt_min, 1e-8)

    scale_2d = tgt_size / src_size

    src_center = (src_min + src_max) / 2.0
    tgt_center = (tgt_min + tgt_max) / 2.0

    v[:, plane_axes[0]] = (
        (v[:, plane_axes[0]] - src_center[0])
        * scale_2d[0]
        + tgt_center[0]
    )

    v[:, plane_axes[1]] = (
        (v[:, plane_axes[1]] - src_center[1])
        * scale_2d[1]
        + tgt_center[1]
    )

    # Preserve depth, but scale it proportionally to the 2D scaling.
    depth_scale = float(np.mean(scale_2d))
    depth_center = (
        v[:, depth_axis].min()
        + v[:, depth_axis].max()
    ) / 2.0

    v[:, depth_axis] = (
        v[:, depth_axis] - depth_center
    ) * depth_scale

    print(f"[ALIGN] Plane scale: {scale_2d}; depth scale: {depth_scale:.4f}")

    return v


def _compute_constraint_rms(
    vertices_3d: np.ndarray,
    pilot_indices: np.ndarray,
    target_points_2d: np.ndarray,
    plane_axes=PLANE_AXES,
) -> float:
    """
    Compute RMS error between selected pilot vertices and 2D target points.
    """
    projected = vertices_3d[pilot_indices][:, plane_axes]
    diff = projected - target_points_2d

    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def _prepare_scene_for_smooth_export(scene: trimesh.Scene) -> None:
    """
    Prepare deformed meshes for smooth GLB export.

    The FFD changes vertex positions. After that, normals need to be rebuilt;
    otherwise Blender/Three.js may show visible triangular flat faces.
    """
    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue

        try:
            # Remove unused vertices if any exist.
            geom.remove_unreferenced_vertices()
        except Exception:
            pass

        try:
            # Fix winding/normal orientation.
            geom.fix_normals()
        except Exception:
            pass

        try:
            # Force trimesh to recompute smooth vertex normals.
            geom._cache.clear()
            _ = geom.vertex_normals
        except Exception:
            pass

        try:
            # Helps avoid backface/light artifacts in some viewers.
            if hasattr(geom.visual, "material") and geom.visual.material is not None:
                geom.visual.material.doubleSided = True
        except Exception:
            pass

def _set_optional_debug_colors(scene: trimesh.Scene, organ: str) -> None:
    """
    Optional: viewer/debug colors.

    For actual frontend output, you usually want to preserve original materials.
    This function is not used by default.
    """
    for geom in scene.geometry.values():
        if organ == "heart":
            geom.visual.face_colors = [190, 130, 115, 255]
        elif organ == "brain":
            geom.visual.face_colors = [220, 170, 150, 255]
        elif organ == "lungs":
            geom.visual.face_colors = [180, 40, 70, 210]
        else:
            geom.visual.face_colors = [190, 160, 150, 255]


# ============================================================
# Main pipeline function
# ============================================================

def deform_organ(
    sketch_path: str,
    organ: str,
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    save_debug: bool = True,
    debug_dir: Optional[str] = None,
    apply_debug_colors: bool = False,
    use_part_constraints: Optional[bool] = None,
    log_result: bool = True,
) -> Dict[str, Any]:
    """
    Run the full deterministic BioSketch deformation pipeline.
    """

    organ = _validate_organ(organ)

    sketch_path_obj = Path(sketch_path)
    if not sketch_path_obj.exists():
        raise FileNotFoundError(f"Sketch image not found: {sketch_path}")

    model_path_obj = _resolve_model_path(organ, model_path)
    output_path_obj = _make_output_path(
        organ=organ,
        output_path=output_path,
        output_dir=output_dir,
    )

    preset = get_organ_preset(organ)

    resolution = preset["resolution"]
    alpha = preset["alpha"]
    beta = preset["beta"]
    max_displacement = preset["max_displacement"]
    total_samples = preset["total_samples"]

    if use_part_constraints is None:
        # Current best behavior:
        # - heart: use anatomical handle constraints
        # - brain: use light constraints
        # - lungs: no part constraints; coarse global FFD works best
        use_part_constraints = organ == "brain"

    print("\n=============================================")
    print("BIOSKETCH PIPELINE")
    print("=============================================")
    print(f"[ORGAN] {organ}")
    print(f"[MODEL] {model_path_obj}")
    print(f"[SKETCH] {sketch_path_obj}")
    print(f"[OUTPUT] {output_path_obj}")
    print(f"[PRESET] {preset['description']}")
    print(
        "[PRESET] "
        f"resolution={resolution}, "
        f"alpha={alpha}, "
        f"beta={beta}, "
        f"max_disp={max_displacement}, "
        f"samples={total_samples}"
    )
    print(f"[PART CONSTRAINTS] {use_part_constraints}")

    # --------------------------------------------------------
    # 1. Read sketch contour
    # --------------------------------------------------------
    print("\n[1] Reading sketch contour")

    processed = preprocess_sketch(str(sketch_path_obj))
    h, w = processed.shape

    contours = extract_contours(processed)
    optimized = optimize_vectors(contours, initial_epsilon_factor=0.002)
    normalized = normalize_coordinates(optimized, w, h)

    sketch_points = get_primary_contour_points(normalized)

    print(f"[VISION] Sketch anchors: {len(sketch_points)}")

    if len(sketch_points) < 10:
        raise RuntimeError(
            f"Too few sketch contour points found: {len(sketch_points)}"
        )

    sketch_debug_path_obj = None

    if save_debug:
        sketch_debug_path_obj = _make_sketch_debug_path(
            organ=organ,
            sketch_path=str(sketch_path_obj),
            model_path=str(model_path_obj),
            debug_dir=debug_dir,
        )

        _save_sketch_border_debug(
            sketch_path=str(sketch_path_obj),
            processed_img=processed,
            optimized_contours=optimized,
            output_path=str(sketch_debug_path_obj),
        )

    # --------------------------------------------------------
    # 2. Load and align mesh
    # --------------------------------------------------------
    print("\n[2] Loading and pre-aligning mesh")

    scene = load_and_normalize_mesh(str(model_path_obj))

    working_vertices = _stack_scene_vertices(scene)

    working_vertices = _apply_plane_bbox_alignment(
        working_vertices,
        sketch_points,
        plane_axes=PLANE_AXES,
        depth_axis=DEPTH_AXIS,
    )

    temp_scene = scene.copy()
    _overwrite_scene_vertices(temp_scene, working_vertices)

    temp_mesh = trimesh.util.concatenate(
        tuple(temp_scene.geometry.values())
    )

    # --------------------------------------------------------
    # 3. Extract projected mesh silhouette
    # --------------------------------------------------------
    print("\n[3] Extracting projected source silhouette")

    source_outline = projected_mesh_outline(
        temp_mesh,
        plane_axes=PLANE_AXES,
    )

    print(f"[MESH] Source outline samples before resampling: {len(source_outline)}")

    if len(source_outline) < 10:
        raise RuntimeError(
            f"Too few source silhouette points found: {len(source_outline)}"
        )

    # --------------------------------------------------------
    # 4. Build FFD cage
    # --------------------------------------------------------
    print("\n[4] Building dynamic cage")

    cage_min, cage_max = bounds_from_vertices(
        working_vertices,
        padding=0.08,
    )

    cage = LatticeFFD(
        resolution=resolution,
        bounds=(cage_min, cage_max),
    )

    cage_points = cage.get_points()

    W = calculate_weight_matrix(
        working_vertices,
        cage.min_bounds,
        cage.max_bounds,
        resolution,
        clip=False,
    )

    cage_before_path_obj = None
    cage_after_path_obj = None

    if save_debug:
        cage_before_path_obj, cage_after_path_obj = _make_cage_debug_paths(
            organ=organ,
            sketch_path=str(sketch_path_obj),
            model_path=str(model_path_obj),
            debug_dir=debug_dir,
        )

        # This is the aligned model before deformation + original cage.
        export_scene_with_cage(
            scene=temp_scene,
            cage_points=cage_points,
            resolution=resolution,
            output_path=cage_before_path_obj,
            line_color=(255, 0, 0, 255),
            point_color=(0, 255, 255, 255),
            make_model_transparent=False,
        )

    reconstructed = W @ cage_points
    recon_error = float(np.max(np.abs(reconstructed - working_vertices)))

    print(f"[CHECK] FFD identity reconstruction max error: {recon_error:.8f}")

    if recon_error > 1e-4:
        print("[WARNING] FFD reconstruction error is high.")
        print("[WARNING] Some vertices may be outside the cage.")

    # --------------------------------------------------------
    # 5. Build contour correspondence
    # --------------------------------------------------------
    print("\n[5] Building contour correspondence")

    source_corr_2d, target_corr_2d, corr_weights, corr_debug = build_landmark_correspondence(
        source_contour_2d=source_outline,
        sketch_contour_2d=sketch_points,
        organ=organ,
        total_samples=total_samples,
    )

    debug_path_obj = None

    if save_debug:
        debug_path_obj = _make_debug_path(
            organ=organ,
            sketch_path=str(sketch_path_obj),
            model_path=str(model_path_obj),
            debug_dir=debug_dir,
        )

        save_correspondence_debug(
            source_corr_2d,
            target_corr_2d,
            debug_path=str(debug_path_obj),
            source_landmarks=corr_debug["source_landmarks"],
            target_landmarks=corr_debug["target_landmarks"],
            line_stride=max(4, total_samples // 45),
        )

        print(f"[DEBUG] Saved correspondence image: {debug_path_obj}")

    # --------------------------------------------------------
    # 6. Pilot vertices and optional part constraints
    # --------------------------------------------------------
    print("\n[6] Finding pilot vertices")

    pilot_indices = find_nearest_mesh_vertices_for_outline(
        vertices_3d=working_vertices,
        source_outline_2d=source_corr_2d,
        plane_axes=PLANE_AXES,
    )

    extra_pilots = np.asarray([], dtype=np.int64)
    extra_targets = np.zeros((0, 2), dtype=np.float32)
    extra_weights = np.asarray([], dtype=np.float32)

    if use_part_constraints:
        extra_pilots, extra_targets, extra_weights = build_part_constraints(
            scene=temp_scene,
            vertices_3d=working_vertices,
            sketch_points_2d=sketch_points,
            organ=organ,
            plane_axes=PLANE_AXES,
        )

    if len(extra_pilots) > 0:
        pilot_indices = np.concatenate([pilot_indices, extra_pilots])
        target_corr_2d = np.vstack([target_corr_2d, extra_targets])
        corr_weights = np.concatenate([corr_weights, extra_weights])

    print(f"[SOLVER] Silhouette constraints: {len(source_corr_2d)}")
    print(f"[SOLVER] Extra part constraints: {len(extra_pilots)}")
    print(f"[SOLVER] Total constraints: {len(pilot_indices)}")

    # --------------------------------------------------------
    # 7. Solve depth-locked weighted cage
    # --------------------------------------------------------
    print("\n[7] Solving depth-locked weighted cage")

    rms_before = _compute_constraint_rms(
        vertices_3d=working_vertices,
        pilot_indices=pilot_indices,
        target_points_2d=target_corr_2d,
        plane_axes=PLANE_AXES,
    )

    C_new = solve_depth_locked_weighted_cage(
        pilot_indices=pilot_indices,
        target_points_2d=target_corr_2d,
        vertices_3d=working_vertices,
        weight_matrix=W,
        original_cage=cage_points,
        resolution=resolution,
        plane_axes=PLANE_AXES,
        depth_axis=DEPTH_AXIS,
        point_weights=corr_weights,
        alpha=alpha,
        beta=beta,
        max_displacement=max_displacement,
    )

    deformed_vertices = W @ C_new
    # Optional presentation-safe emphasis for small parts.
# This makes the brainstem deformation visibly clear for demo/report.
    deformed_vertices = emphasize_brainstem_from_sketch(
    vertices_3d=deformed_vertices,
    sketch_points_2d=sketch_points,
    organ=organ,
    plane_axes=PLANE_AXES,
    depth_axis=DEPTH_AXIS,
    strength=0.45,
    )

    if save_debug and cage_after_path_obj is not None:
        after_scene = scene.copy()
        _overwrite_scene_vertices(after_scene, deformed_vertices)

        # This is the deformed model + deformed cage.
        export_scene_with_cage(
            scene=after_scene,
            cage_points=C_new,
            resolution=resolution,
            output_path=cage_after_path_obj,
            line_color=(0, 255, 0, 255),
            point_color=(255, 255, 0, 255),
            make_model_transparent=False,
        )

    # --------------------------------------------------------
    # Result analysis metrics
    # --------------------------------------------------------

    deformation_summary = deformation_stats(
        original_vertices=working_vertices,
        deformed_vertices=deformed_vertices,
        plane_axes=PLANE_AXES,
        depth_axis=DEPTH_AXIS,
    )

    # Build temporary deformed mesh to extract its projected silhouette.
    metric_scene = temp_scene.copy()
    _overwrite_scene_vertices(metric_scene, deformed_vertices)

    metric_mesh = trimesh.util.concatenate(
        tuple(metric_scene.geometry.values())
    )

    deformed_outline = projected_mesh_outline(
        metric_mesh,
        plane_axes=PLANE_AXES,
    )

    similarity_before = contour_similarity_percent(
        model_outline_2d=source_outline,
        sketch_outline_2d=sketch_points,
    )

    similarity_after = contour_similarity_percent(
        model_outline_2d=deformed_outline,
        sketch_outline_2d=sketch_points,
    )

    rms_after = _compute_constraint_rms(
        vertices_3d=deformed_vertices,
        pilot_indices=pilot_indices,
        target_points_2d=target_corr_2d,
        plane_axes=PLANE_AXES,
    )

    fit_improvement = rms_improvement_percent(
        rms_before=rms_before,
        rms_after=rms_after,
    )

    print("\n[RESULT ANALYSIS]")
    print(f"[RESULT] RMS improvement       : {fit_improvement:.2f}%")
    print(f"[RESULT] Similarity before    : {similarity_before:.2f}%")
    print(f"[RESULT] Similarity after     : {similarity_after:.2f}%")
    print(
        "[RESULT] Mean deformation    : "
        f"{deformation_summary['mean_deformation_percent']:.2f}% of model size"
    )
    print(
        "[RESULT] Max deformation     : "
        f"{deformation_summary['max_deformation_percent']:.2f}% of model size"
    )
    print(
        "[RESULT] Mean depth movement : "
        f"{deformation_summary['mean_depth_deformation_percent']:.4f}% of model size"
    )

    print(f"[METRIC] RMS before: {rms_before:.5f}")
    print(f"[METRIC] RMS after : {rms_after:.5f}")

    # --------------------------------------------------------
    # 8. Write vertices back and export GLB
    # --------------------------------------------------------
    print("\n[8] Exporting deformed GLB")

    _overwrite_scene_vertices(scene, deformed_vertices)

    if apply_debug_colors:
        _set_optional_debug_colors(scene, organ)

        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Rebuild normals before exporting so Blender/Three.js does not show
    # strong triangular flat-face shading.
    _prepare_scene_for_smooth_export(scene)

    # Export GLB with vertex normals included.
    glb_bytes = scene.export(
        file_type="glb",
        include_normals=True,
    )

    with open(output_path_obj, "wb") as f:
        f.write(glb_bytes)

    print(f"[DONE] Exported: {output_path_obj}")

    result = {
        "organ": organ,
        "model_path": str(model_path_obj),
        "sketch_path": str(sketch_path_obj),
        "output_path": str(output_path_obj),
        "debug_path": str(debug_path_obj) if debug_path_obj else None,
        "sketch_debug_path": str(sketch_debug_path_obj) if sketch_debug_path_obj else None,
        "cage_before_path": str(cage_before_path_obj) if cage_before_path_obj else None,
        "cage_after_path": str(cage_after_path_obj) if cage_after_path_obj else None,
        "rms_before": rms_before,
        "rms_after": rms_after,
        "rms_improvement_percent": fit_improvement,

        "similarity_before_percent": similarity_before,
        "similarity_after_percent": similarity_after,

        "mean_deformation_percent": deformation_summary["mean_deformation_percent"],
        "max_deformation_percent": deformation_summary["max_deformation_percent"],

        "mean_depth_deformation_percent": deformation_summary["mean_depth_deformation_percent"],
        "max_depth_deformation_percent": deformation_summary["max_depth_deformation_percent"],

        "silhouette_constraints": int(len(source_corr_2d)),
        "extra_part_constraints": int(len(extra_pilots)),
        "total_constraints": int(len(pilot_indices)),

        "resolution": tuple(resolution),
        "alpha": float(alpha),
        "beta": float(beta),
        "max_displacement": float(max_displacement),
    }

    if log_result:
        append_result(result)

    return result


# ============================================================
# Optional local test
# ============================================================

if __name__ == "__main__":
    result = deform_organ(
        sketch_path=str(ASSETS_DIR / "sketches" / "test_brain.jpg"),
        organ="brain",
        model_path=str(MODELS_DIR / "brain.glb"),
        save_debug=True,
        apply_debug_colors=False,
    )

    print("\nPipeline result:")
    for key, value in result.items():
        print(f"{key}: {value}")