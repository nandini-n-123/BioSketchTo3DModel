import os
import sys
import numpy as np
import trimesh

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.vision.sketch_parser import (
    preprocess_sketch,
    extract_contours,
    optimize_vectors,
    normalize_coordinates,
    get_primary_contour_points,
)

from app.geometry.mesh_prep import load_and_normalize_mesh
from app.deformation.lattice import LatticeFFD, bounds_from_vertices
from app.deformation.cage_weights import calculate_weight_matrix
from app.deformation.solver import projected_mesh_outline

from app.deformation.correspondence import (
    build_landmark_correspondence,
    find_nearest_mesh_vertices_for_outline,
    save_correspondence_debug,
)

from app.deformation.weighted_solver import solve_depth_locked_weighted_cage
from app.config.organ_presets import get_organ_preset

from app.deformation.part_constraints import (
    build_part_constraints,
    print_scene_geometry_names,
)


# ============================================================
# GLOBAL AXIS CONFIG
# ============================================================

# glTF / GLB front-facing convention:
# X/Y = visible sketch plane, Z = depth.
PLANE_AXES = (0, 1)
DEPTH_AXIS = 2

MODELS_DIR = "assets/3d_models"
SKETCHES_DIR = "assets/sketches"

MODEL_EXTENSIONS = (".glb", ".gltf", ".obj")
SKETCH_EXTENSIONS = (".jpg", ".jpeg", ".png")


# ============================================================
# INTERACTIVE MENU HELPERS
# ============================================================

def _list_files(folder, extensions):
    if not os.path.exists(folder):
        return []

    files = []

    for filename in os.listdir(folder):
        if filename.lower().endswith(extensions):
            files.append(filename)

    return sorted(files)


def _choose_from_list(title, options):
    if not options:
        raise RuntimeError(f"No options available for: {title}")

    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

    for i, item in enumerate(options, start=1):
        print(f"[{i}] {item}")

    while True:
        choice = input("\nEnter choice number: ").strip()

        try:
            index = int(choice) - 1

            if 0 <= index < len(options):
                return options[index]

        except ValueError:
            pass

        print("Invalid choice. Please enter a valid number.")


def _guess_organ_from_filename(filename):
    name = filename.lower()

    if "heart" in name or name.startswith("h"):
        return "heart"

    if "brain" in name:
        return "brain"

    if "lung" in name:
        return "lungs"

    return None


def choose_inputs_interactively():
    organs = ["heart", "brain", "lungs"]

    organ = _choose_from_list(
        "Choose organ preset",
        organs,
    )

    model_files = _list_files(MODELS_DIR, MODEL_EXTENSIONS)
    sketch_files = _list_files(SKETCHES_DIR, SKETCH_EXTENSIONS)

    if not model_files:
        raise RuntimeError(f"No 3D models found in {MODELS_DIR}")

    if not sketch_files:
        raise RuntimeError(f"No sketch images found in {SKETCHES_DIR}")

    # Put likely matching models first.
    matching_models = [
        f for f in model_files
        if _guess_organ_from_filename(f) == organ
    ]

    other_models = [
        f for f in model_files
        if f not in matching_models
    ]

    ordered_models = matching_models + other_models

    # Put likely matching sketches first.
    matching_sketches = [
        f for f in sketch_files
        if _guess_organ_from_filename(f) == organ
    ]

    other_sketches = [
        f for f in sketch_files
        if f not in matching_sketches
    ]

    ordered_sketches = matching_sketches + other_sketches

    model_file = _choose_from_list(
        f"Choose 3D model for organ preset '{organ}'",
        ordered_models,
    )

    sketch_file = _choose_from_list(
        f"Choose sketch image for organ preset '{organ}'",
        ordered_sketches,
    )

    model_path = os.path.join(MODELS_DIR, model_file)
    sketch_path = os.path.join(SKETCHES_DIR, sketch_file)

    return organ, model_path, sketch_path


# ============================================================
# PIPELINE HELPERS
# ============================================================

def _stack_scene_vertices(scene):
    return np.vstack([g.vertices.copy() for g in scene.geometry.values()])


def _overwrite_scene_vertices(scene, vertices):
    offset = 0

    for geom in scene.geometry.values():
        n = len(geom.vertices)
        geom.vertices = vertices[offset: offset + n]
        offset += n


def _apply_plane_bbox_alignment(
    vertices,
    target_2d,
    plane_axes=PLANE_AXES,
    depth_axis=DEPTH_AXIS,
):
    """
    Align model bbox to sketch bbox before FFD.

    This prevents the cage from wasting effort on simple scale/translation.
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
        (v[:, plane_axes[0]] - src_center[0]) * scale_2d[0]
        + tgt_center[0]
    )

    v[:, plane_axes[1]] = (
        (v[:, plane_axes[1]] - src_center[1]) * scale_2d[1]
        + tgt_center[1]
    )

    # Preserve depth, but scale it proportionally to the 2D scale.
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


def _set_debug_materials(scene, organ):
    """
    Viewer-only material colors. Does not affect deformation math.
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


def _print_selected_inputs(organ, model_path, sketch_path, preset):
    print("\n=============================================")
    print("BIOSKETCH DEPTH-LOCKED FFD PIPELINE")
    print("=============================================")
    print(f"[ORGAN PRESET] {organ}")
    print(f"[MODEL] {model_path}")
    print(f"[SKETCH] {sketch_path}")
    print(f"[PRESET] {preset['description']}")
    print(
        "[PRESET] "
        f"resolution={preset['resolution']}, "
        f"alpha={preset['alpha']}, "
        f"beta={preset['beta']}, "
        f"max_disp={preset['max_displacement']}"
    )
    print(f"[PRESET] samples={preset['total_samples']}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_full_pipeline(organ, model_path, sketch_path):
    preset = get_organ_preset(organ)

    resolution = preset["resolution"]
    alpha = preset["alpha"]
    beta = preset["beta"]
    max_displacement = preset["max_displacement"]
    total_samples = preset["total_samples"]

    _print_selected_inputs(organ, model_path, sketch_path, preset)

    if not os.path.exists(sketch_path):
        print(f"[ERROR] Sketch does not exist: {sketch_path}")
        return

    if not os.path.exists(model_path):
        print(f"[ERROR] Model does not exist: {model_path}")
        return

    os.makedirs("debug", exist_ok=True)

    # --------------------------------------------------------
    # 1. Read sketch contour
    # --------------------------------------------------------
    print("\n[1] Reading sketch contour")

    processed = preprocess_sketch(sketch_path)
    h, w = processed.shape

    contours = extract_contours(processed)
    optimized = optimize_vectors(contours, initial_epsilon_factor=0.002)
    normalized = normalize_coordinates(optimized, w, h)

    sketch_points = get_primary_contour_points(normalized)

    print(f"[VISION] Sketch anchors: {len(sketch_points)}")

    if len(sketch_points) < 10:
        print("[ERROR] Too few sketch contour points found.")
        return

    # --------------------------------------------------------
    # 2. Load and pre-align mesh
    # --------------------------------------------------------
    print("\n[2] Loading and pre-aligning mesh")

    scene = load_and_normalize_mesh(model_path)

    working_vertices = _stack_scene_vertices(scene)

    working_vertices = _apply_plane_bbox_alignment(
        working_vertices,
        sketch_points,
        plane_axes=PLANE_AXES,
        depth_axis=DEPTH_AXIS,
    )

    temp_scene = scene.copy()
    _overwrite_scene_vertices(temp_scene, working_vertices)
    temp_mesh = trimesh.util.concatenate(tuple(temp_scene.geometry.values()))

    # --------------------------------------------------------
    # 3. Projected source silhouette
    # --------------------------------------------------------
    print("\n[3] Extracting projected source silhouette")

    source_outline = projected_mesh_outline(
        temp_mesh,
        plane_axes=PLANE_AXES,
    )

    print(f"[MESH] Source outline samples before resampling: {len(source_outline)}")

    if len(source_outline) < 10:
        print("[ERROR] Too few source silhouette points found.")
        return

    # --------------------------------------------------------
    # 4. Dynamic cage
    # --------------------------------------------------------
    print("\n[4] Building dynamic cage around the aligned mesh")

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

    reconstructed = W @ cage_points
    recon_error = np.max(np.abs(reconstructed - working_vertices))

    print(f"[CHECK] FFD identity reconstruction max error: {recon_error:.8f}")

    if recon_error > 1e-4:
        print("[WARNING] FFD reconstruction error is high.")
        print("[WARNING] Some vertices may be outside the cage.")

    # --------------------------------------------------------
    # 5. Correspondence
    # --------------------------------------------------------
    print("\n[5] Building organ-aware correspondence")

    # Use the same ordered closed-contour correspondence for all organs,
    # including lungs. Lungs-specific behavior is handled through weights
    # in correspondence.py and moderate part constraints.
    source_corr_2d, target_corr_2d, corr_weights, corr_debug = build_landmark_correspondence(
        source_contour_2d=source_outline,
        sketch_contour_2d=sketch_points,
        organ=organ,
        total_samples=total_samples,
    )

    sketch_base = os.path.splitext(os.path.basename(sketch_path))[0]
    model_base = os.path.splitext(os.path.basename(model_path))[0]

    debug_path = f"debug/{organ}_{model_base}_{sketch_base}_correspondence.png"

    save_correspondence_debug(
        source_corr_2d,
        target_corr_2d,
        debug_path=debug_path,
        source_landmarks=corr_debug["source_landmarks"],
        target_landmarks=corr_debug["target_landmarks"],
        line_stride=max(4, total_samples // 45),
    )

    print(f"[DEBUG] Saved correspondence debug image: {debug_path}")

    # --------------------------------------------------------
    # 6. Pilot vertices and extra part/region constraints
    # --------------------------------------------------------
    print("\n[6] Finding nearest mesh pilot vertices")

    pilot_indices = find_nearest_mesh_vertices_for_outline(
        vertices_3d=working_vertices,
        source_outline_2d=source_corr_2d,
        plane_axes=PLANE_AXES,
    )

    if organ == "lungs":
    # For lungs, the extra region constraints over-anchor the lobes.
    # Use only global closed-contour correspondence with a coarse 4x4x4 cage.
        extra_pilots = np.asarray([], dtype=np.int64)
        extra_targets = np.zeros((0, 2), dtype=np.float32)
        extra_weights = np.asarray([], dtype=np.float32)
    else:
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
    print(f"[SOLVER] Total pilot vertices : {len(pilot_indices)}")
    print(f"[SOLVER] Total target points  : {len(target_corr_2d)}")

    if len(extra_pilots) == 0:
        print("[WARNING] No extra region constraints were added.")
        print_scene_geometry_names(temp_scene)

    # --------------------------------------------------------
    # 7. Weighted depth-locked solve
    # --------------------------------------------------------
    print("\n[7] Solving depth-locked weighted cage")

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

    # --------------------------------------------------------
    # 8. Apply deformation
    # --------------------------------------------------------
    print("\n[8] Applying deformation")

    deformed_vertices = W @ C_new

    _overwrite_scene_vertices(scene, deformed_vertices)

    _set_debug_materials(scene, organ)

    cage_cloud = trimesh.points.PointCloud(
        C_new,
        colors=[255, 0, 0, 255],
    )

    out_scene = trimesh.Scene([scene, cage_cloud])

    print("[DONE] Close the viewer to finish.")
    out_scene.show(background=[40, 40, 40, 255])


def main():
    try:
        organ, model_path, sketch_path = choose_inputs_interactively()
        run_full_pipeline(organ, model_path, sketch_path)

    except KeyboardInterrupt:
        print("\n[EXIT] Cancelled by user.")

    except Exception as e:
        print(f"\n[ERROR] {e}")


if __name__ == "__main__":
    main()