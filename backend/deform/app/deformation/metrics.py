import numpy as np
from scipy.spatial import KDTree


def rms_improvement_percent(rms_before, rms_after):
    """
    Percentage reduction in RMS error after deformation.
    """
    rms_before = float(rms_before)
    rms_after = float(rms_after)

    if rms_before <= 1e-8:
        return 0.0

    improvement = (rms_before - rms_after) / rms_before
    return float(np.clip(improvement * 100.0, 0.0, 100.0))


def _bbox_diagonal_2d(points_2d):
    points = np.asarray(points_2d, dtype=np.float64)

    if len(points) == 0:
        return 1.0

    mn = points.min(axis=0)
    mx = points.max(axis=0)

    diag = np.linalg.norm(mx - mn)

    return float(max(diag, 1e-8))


def bidirectional_chamfer_distance(points_a, points_b):
    """
    Symmetric nearest-neighbor contour distance.

    Lower value means contours are more similar.
    """
    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)

    if len(a) == 0 or len(b) == 0:
        return float("inf")

    tree_b = KDTree(b)
    dist_a_to_b, _ = tree_b.query(a)

    tree_a = KDTree(a)
    dist_b_to_a, _ = tree_a.query(b)

    return float((np.mean(dist_a_to_b) + np.mean(dist_b_to_a)) / 2.0)


def contour_similarity_percent(model_outline_2d, sketch_outline_2d):
    """
    Convert contour distance into a presentation-friendly similarity percentage.

    The score is normalized by the sketch bounding-box diagonal.
    """
    model_outline = np.asarray(model_outline_2d, dtype=np.float64)
    sketch_outline = np.asarray(sketch_outline_2d, dtype=np.float64)

    chamfer = bidirectional_chamfer_distance(model_outline, sketch_outline)
    sketch_diag = _bbox_diagonal_2d(sketch_outline)

    similarity = 1.0 - (chamfer / sketch_diag)

    return float(np.clip(similarity * 100.0, 0.0, 100.0))


def deformation_stats(
    original_vertices,
    deformed_vertices,
    plane_axes=(0, 1),
    depth_axis=2,
):
    """
    Measure how much the 3D model changed.

    Values are normalized by the original model bounding-box diagonal.
    """
    original = np.asarray(original_vertices, dtype=np.float64)
    deformed = np.asarray(deformed_vertices, dtype=np.float64)

    if original.shape != deformed.shape:
        raise ValueError(
            f"original_vertices and deformed_vertices shape mismatch: "
            f"{original.shape} vs {deformed.shape}"
        )

    displacement = deformed - original

    total_disp = np.linalg.norm(displacement, axis=1)
    plane_disp = np.linalg.norm(displacement[:, plane_axes], axis=1)
    depth_disp = np.abs(displacement[:, depth_axis])

    bbox_min = original.min(axis=0)
    bbox_max = original.max(axis=0)
    bbox_diag = np.linalg.norm(bbox_max - bbox_min)
    bbox_diag = max(float(bbox_diag), 1e-8)

    return {
        "mean_vertex_displacement": float(np.mean(total_disp)),
        "max_vertex_displacement": float(np.max(total_disp)),

        "mean_plane_displacement": float(np.mean(plane_disp)),
        "max_plane_displacement": float(np.max(plane_disp)),

        "mean_depth_displacement": float(np.mean(depth_disp)),
        "max_depth_displacement": float(np.max(depth_disp)),

        "mean_deformation_percent": float((np.mean(total_disp) / bbox_diag) * 100.0),
        "max_deformation_percent": float((np.max(total_disp) / bbox_diag) * 100.0),

        "mean_plane_deformation_percent": float((np.mean(plane_disp) / bbox_diag) * 100.0),
        "max_plane_deformation_percent": float((np.max(plane_disp) / bbox_diag) * 100.0),

        "mean_depth_deformation_percent": float((np.mean(depth_disp) / bbox_diag) * 100.0),
        "max_depth_deformation_percent": float((np.max(depth_disp) / bbox_diag) * 100.0),
    }