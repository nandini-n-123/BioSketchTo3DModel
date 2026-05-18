import os
import cv2
import numpy as np
from scipy.spatial import KDTree


# ============================================================
# Basic contour utilities
# ============================================================

def _as_points(contour):
    """
    Convert OpenCV-style contour or normal Nx2 array into clean Nx2 float32 points.
    """
    pts = np.asarray(contour, dtype=np.float32)

    if pts.ndim == 3:
        pts = pts.reshape(-1, 2)

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected Nx2 contour points, got shape {pts.shape}")

    pts = pts[np.isfinite(pts).all(axis=1)]

    # Remove consecutive duplicates
    if len(pts) > 1:
        keep = [0]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-7:
                keep.append(i)
        pts = pts[keep]

    return pts.astype(np.float32)


def _signed_area(points):
    """
    Signed polygon area. Positive/negative tells contour orientation.
    """
    p = _as_points(points)
    x = p[:, 0]
    y = p[:, 1]

    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _ensure_same_orientation(source, target):
    """
    Make target contour orientation match source contour orientation.
    """
    s_area = _signed_area(source)
    t_area = _signed_area(target)

    if s_area * t_area < 0:
        return target[::-1].copy()

    return target.copy()


def _arc_lengths_closed(points):
    """
    Cumulative arc length for a closed contour.
    """
    p = _as_points(points)

    if len(p) < 3:
        raise ValueError("Need at least 3 points for closed contour resampling.")

    closed = np.vstack([p, p[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.hstack([[0.0], np.cumsum(seg)])

    return closed, cumulative


def resample_closed_contour(points, n_samples):
    """
    Resample a closed contour uniformly by perimeter length.
    """
    p = _as_points(points)

    if len(p) < 3:
        raise ValueError("Need at least 3 points for closed contour resampling.")

    closed, cumulative = _arc_lengths_closed(p)
    total = cumulative[-1]

    if total < 1e-8:
        raise ValueError("Contour perimeter is too small.")

    sample_d = np.linspace(0.0, total, n_samples, endpoint=False)

    xs = np.interp(sample_d, cumulative, closed[:, 0])
    ys = np.interp(sample_d, cumulative, closed[:, 1])

    return np.stack([xs, ys], axis=1).astype(np.float32)


def smooth_closed_contour(points, iterations=2, strength=0.35):
    """
    Light smoothing to remove jagged silhouette noise.
    Keeps the contour closed and ordered.
    """
    p = _as_points(points).copy()

    for _ in range(iterations):
        prev_p = np.roll(p, 1, axis=0)
        next_p = np.roll(p, -1, axis=0)
        p = (1.0 - strength) * p + (strength * 0.5) * (prev_p + next_p)

    return p.astype(np.float32)


# ============================================================
# Alignment / correspondence
# ============================================================

def _normalize_for_matching(points):
    """
    Normalize contour only for comparing cyclic shifts.
    Does not affect final output coordinates.
    """
    p = _as_points(points)
    center = p.mean(axis=0)
    q = p - center

    scale = np.max(np.linalg.norm(q, axis=1))
    if scale < 1e-8:
        scale = 1.0

    return q / scale


def _best_cyclic_shift(source, target):
    """
    Find the best cyclic shift of target to match source.
    Both must already have same number of points.
    """
    s = _normalize_for_matching(source)
    t = _normalize_for_matching(target)

    n = len(s)
    best_shift = 0
    best_error = np.inf

    for shift in range(n):
        rolled = np.roll(t, shift, axis=0)
        error = np.mean(np.sum((s - rolled) ** 2, axis=1))

        if error < best_error:
            best_error = error
            best_shift = shift

    return best_shift, best_error


def _align_target_order_to_source(source, target):
    """
    Try normal and reversed target order.
    Choose whichever gives lower cyclic matching error.
    """
    source = _as_points(source)
    target = _as_points(target)

    target_same = _ensure_same_orientation(source, target)

    shift_a, err_a = _best_cyclic_shift(source, target_same)
    aligned_a = np.roll(target_same, shift_a, axis=0)

    target_rev = target_same[::-1].copy()
    shift_b, err_b = _best_cyclic_shift(source, target_rev)
    aligned_b = np.roll(target_rev, shift_b, axis=0)

    if err_b < err_a:
        return aligned_b.astype(np.float32), {
            "reversed": True,
            "shift": int(shift_b),
            "cyclic_error": float(err_b),
        }

    return aligned_a.astype(np.float32), {
        "reversed": False,
        "shift": int(shift_a),
        "cyclic_error": float(err_a),
    }


def _compute_weights(points, organ):
    """
    Organ-specific confidence weights.

    These weights do NOT decide correspondence.
    They only control how strongly each matched contour pair pulls the cage.
    """
    p = _as_points(points)
    x = p[:, 0]
    y = p[:, 1]

    weights = np.ones(len(p), dtype=np.float32)

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    y_range = max(y_max - y_min, 1e-8)
    x_range = max(x_max - x_min, 1e-8)

    y_norm = (y - y_min) / y_range
    x_norm = (x - x_min) / x_range

    organ = organ.lower().strip()

    if organ == "heart":
        # Heart:
        # - main body should guide deformation
        # - top vessels should not dominate
        # - lower-left IVC region should get stronger influence
        weights[:] *= 1.0

        inferior_vena_cava_zone = (
            (x_norm < 0.28)
            & (y_norm < 0.62)
        )
        weights[inferior_vena_cava_zone] *= 1.45

        apex_body = (
            (y_norm < 0.18)
            & (x_norm >= 0.28)
        )
        weights[apex_body] *= 0.55

        top_vessels = y_norm > 0.78
        weights[top_vessels] *= 0.42

        right_detail = (
            (x_norm > 0.62)
            & (y_norm > 0.35)
        )
        weights[right_detail] *= 0.70

        body_middle = (
            (y_norm >= 0.22)
            & (y_norm <= 0.72)
            & (x_norm >= 0.18)
            & (x_norm <= 0.88)
        )
        weights[body_middle] *= 1.22

        left_vertical = (
            (x_norm < 0.30)
            & (y_norm > 0.18)
            & (y_norm < 0.86)
        )
        weights[left_vertical] *= 1.20

    elif organ == "brain":
        # Brain:
        # already matches well, so keep mostly stable.
        weights[:] *= 1.0

        dome = y_norm > 0.72
        weights[dome] *= 1.05

        very_bottom = y_norm < 0.12
        weights[very_bottom] *= 0.75

        lower_middle = (
            (y_norm >= 0.12)
            & (y_norm < 0.35)
            & (x_norm > 0.35)
            & (x_norm < 0.65)
        )
        weights[lower_middle] *= 1.18

        lower_left = (
            (y_norm < 0.35)
            & (x_norm < 0.30)
        )
        weights[lower_left] *= 0.85

    elif organ == "lungs":
        # Lungs:
        # using global closed-contour correspondence again.
        # Encourage broad lobe deformation, reduce central trachea dominance.
        weights[:] *= 0.95

        outer_lobes = (
            (x_norm < 0.20)
            | (x_norm > 0.80)
        )
        weights[outer_lobes] *= 1.15

        top_center = (
            (y_norm > 0.65)
            & (x_norm > 0.35)
            & (x_norm < 0.65)
        )
        weights[top_center] *= 0.55

        center_branch = (
            (y_norm > 0.25)
            & (y_norm < 0.70)
            & (x_norm > 0.35)
            & (x_norm < 0.65)
        )
        weights[center_branch] *= 0.65

        lower_center = (
            (y_norm < 0.30)
            & (x_norm > 0.30)
            & (x_norm < 0.70)
        )
        weights[lower_center] *= 0.75

        bottom_outer = (
            (y_norm < 0.25)
            & ((x_norm < 0.35) | (x_norm > 0.65))
        )
        weights[bottom_outer] *= 1.00

    else:
        print(f"[WARNING] Unknown organ '{organ}', using default weights.")

    weights = np.clip(weights, 0.30, 1.50)
    return weights.astype(np.float32)


def _simple_landmarks(points, organ):
    """
    Landmarks for debug drawing only.
    These are not used to force correspondence.
    """
    p = _as_points(points)
    x = p[:, 0]
    y = p[:, 1]

    landmarks = {
        "top": p[int(np.argmax(y))],
        "bottom": p[int(np.argmin(y))],
        "left": p[int(np.argmin(x))],
        "right": p[int(np.argmax(x))],
    }

    organ = organ.lower().strip()

    if organ == "heart":
        upper = np.where(y > np.percentile(y, 75))[0]
        lower = np.where(y < np.percentile(y, 25))[0]

        if len(upper) > 0:
            landmarks["upper_left"] = p[upper[np.argmin(x[upper])]]
            landmarks["upper_right"] = p[upper[np.argmax(x[upper])]]

        if len(lower) > 0:
            landmarks["apex_region"] = p[lower[np.argmin(y[lower])]]

    elif organ == "brain":
        lower = np.where(y < np.percentile(y, 25))[0]

        if len(lower) > 0:
            landmarks["brainstem"] = p[lower[np.argmin(x[lower])]]

    elif organ == "lungs":
        middle = np.where(
            (x > np.percentile(x, 40))
            & (x < np.percentile(x, 60))
        )[0]

        if len(middle) > 0:
            landmarks["middle_top"] = p[middle[np.argmax(y[middle])]]
            landmarks["middle_bottom"] = p[middle[np.argmin(y[middle])]]

    return landmarks


def build_landmark_correspondence(
    source_contour_2d,
    sketch_contour_2d,
    organ="heart",
    total_samples=140,
):
    """
    Build stable ordered correspondence between:
      source projected mesh silhouette and sketch contour.

    Output:
      source_corr_2d : Nx2
      target_corr_2d : Nx2
      corr_weights   : N
      corr_debug     : dict
    """
    source_raw = _as_points(source_contour_2d)
    target_raw = _as_points(sketch_contour_2d)

    source = resample_closed_contour(source_raw, total_samples)
    target = resample_closed_contour(target_raw, total_samples)

    source = smooth_closed_contour(source, iterations=2, strength=0.25)
    target = smooth_closed_contour(target, iterations=1, strength=0.15)

    target_aligned, align_info = _align_target_order_to_source(source, target)

    weights = _compute_weights(target_aligned, organ)

    debug = {
        "source_landmarks": _simple_landmarks(source, organ),
        "target_landmarks": _simple_landmarks(target_aligned, organ),
        "alignment": align_info,
    }

    print(
        "[CORR] ordered closed-contour correspondence "
        f"organ={organ}, samples={total_samples}, "
        f"reversed={align_info['reversed']}, "
        f"shift={align_info['shift']}, "
        f"cyclic_error={align_info['cyclic_error']:.6f}"
    )

    return (
        source.astype(np.float32),
        target_aligned.astype(np.float32),
        weights.astype(np.float32),
        debug,
    )


# ============================================================
# Mesh pilot lookup
# ============================================================

def find_nearest_mesh_vertices_for_outline(
    vertices_3d,
    source_outline_2d,
    plane_axes=(0, 1),
):
    """
    Find nearest 3D mesh vertices to each 2D source outline point.

    This keeps the actual 3D vertex as the pilot constraint source.
    """
    vertices = np.asarray(vertices_3d, dtype=np.float32)
    outline = _as_points(source_outline_2d)

    projected = vertices[:, plane_axes]

    tree = KDTree(projected)
    _, indices = tree.query(outline)

    return indices.astype(np.int64)


# ============================================================
# Debug visualization
# ============================================================

def _map_to_canvas(points, canvas_size=900, padding=70, bounds=None):
    p = _as_points(points)

    if bounds is None:
        mn = p.min(axis=0)
        mx = p.max(axis=0)
    else:
        mn, mx = bounds

    size = np.maximum(mx - mn, 1e-8)
    scale = (canvas_size - 2 * padding) / max(size[0], size[1])

    q = (p - mn) * scale + padding

    # Convert math Y-up to image Y-down.
    q[:, 1] = canvas_size - q[:, 1]

    return q.astype(np.int32)


def _draw_polyline(img, pts, color, thickness=2):
    pts = np.asarray(pts, dtype=np.int32)

    if len(pts) < 2:
        return

    cv2.polylines(
        img,
        [pts.reshape(-1, 1, 2)],
        isClosed=True,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def save_correspondence_debug(
    source_corr_2d,
    target_corr_2d,
    debug_path,
    source_landmarks=None,
    target_landmarks=None,
    line_stride=5,
):
    """
    Save debug image:
      green = source mesh silhouette
      cyan  = sketch contour
      red   = matched pairs
    """
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)

    src = _as_points(source_corr_2d)
    tgt = _as_points(target_corr_2d)

    all_pts = np.vstack([src, tgt])
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)

    canvas_size = 900
    img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    src_px = _map_to_canvas(src, canvas_size=canvas_size, bounds=(mn, mx))
    tgt_px = _map_to_canvas(tgt, canvas_size=canvas_size, bounds=(mn, mx))

    # Draw matched pairs first.
    for i in range(0, min(len(src_px), len(tgt_px)), max(1, line_stride)):
        cv2.line(
            img,
            tuple(src_px[i]),
            tuple(tgt_px[i]),
            (0, 0, 255),
            1,
            lineType=cv2.LINE_AA,
        )

    _draw_polyline(img, src_px, (0, 255, 0), thickness=2)
    _draw_polyline(img, tgt_px, (255, 255, 0), thickness=2)

    for i in range(0, len(src_px), 10):
        cv2.circle(img, tuple(src_px[i]), 3, (0, 255, 0), -1)

    for i in range(0, len(tgt_px), 10):
        cv2.circle(img, tuple(tgt_px[i]), 3, (255, 255, 0), -1)

    if source_landmarks:
        for name, pt in source_landmarks.items():
            px = _map_to_canvas(
                np.asarray([pt]),
                canvas_size=canvas_size,
                bounds=(mn, mx),
            )[0]

            cv2.circle(img, tuple(px), 6, (0, 180, 0), -1)
            cv2.putText(
                img,
                str(name),
                tuple(px + np.array([5, -5])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    if target_landmarks:
        for name, pt in target_landmarks.items():
            px = _map_to_canvas(
                np.asarray([pt]),
                canvas_size=canvas_size,
                bounds=(mn, mx),
            )[0]

            cv2.circle(img, tuple(px), 6, (255, 255, 0), -1)
            cv2.putText(
                img,
                str(name),
                tuple(px + np.array([5, 12])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        img,
        "green=mesh silhouette, cyan=sketch, red=matched pairs",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite(debug_path, img)