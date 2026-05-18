import numpy as np
import cv2
import trimesh

from app.deformation.cage_weights import calculate_weight_matrix_2d


# Blender Front Orthographic is X horizontal, Z vertical, Y depth.
# If your exported assets are different, change these in test_deformation.py.
DEFAULT_PLANE_AXES = (0, 2)
DEFAULT_DEPTH_AXIS = 1


def _signed_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def resample_closed_curve(points, n=128):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        raise ValueError("A closed curve needs at least 3 points")

    # Remove duplicate terminal point if present.
    if np.linalg.norm(points[0] - points[-1]) < 1e-9:
        points = points[:-1]

    closed = np.vstack([points, points[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    keep = seg > 1e-10
    closed = np.vstack([closed[:-1][keep], closed[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)

    cumulative = np.insert(np.cumsum(seg), 0, 0.0)
    total = cumulative[-1]
    if total <= 1e-10:
        raise ValueError("Curve length is zero")

    samples = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(samples, cumulative, closed[:, 0])
    y = np.interp(samples, cumulative, closed[:, 1])
    return np.column_stack([x, y])


def align_closed_curve_order(source, target):
    """
    Match target's orientation and cyclic start index to source.
    This prevents the solver from pulling one source location toward an unrelated
    sketch location because OpenCV selected a different contour starting point.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) != len(target):
        raise ValueError("source and target must be resampled to the same count first")

    if np.sign(_signed_area(source)) != np.sign(_signed_area(target)):
        target = target[::-1].copy()

    best_shift = 0
    best_error = np.inf
    for shift in range(len(target)):
        rolled = np.roll(target, shift, axis=0)
        err = np.mean(np.sum((source - rolled) ** 2, axis=1))
        if err < best_error:
            best_error = err
            best_shift = shift
    return np.roll(target, best_shift, axis=0)


def projected_mesh_outline(mesh, plane_axes=DEFAULT_PLANE_AXES, image_size=1024, pad_px=64):
    """
    Rasterize the mesh from an orthographic view and extract the external contour.
    This is a real 2D silhouette/projection, unlike selecting vertices from a
    middle-depth slice.
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("projected_mesh_outline expects a Trimesh or Scene")

    pts = np.asarray(mesh.vertices[:, plane_axes], dtype=np.float64)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    span = max(max_xy - min_xy)
    if span <= 1e-12:
        raise ValueError("Projected mesh has zero size")

    scale = (image_size - 2 * pad_px) / span
    pix = np.empty_like(pts)
    pix[:, 0] = (pts[:, 0] - min_xy[0]) * scale + pad_px
    pix[:, 1] = (max_xy[1] - pts[:, 1]) * scale + pad_px  # invert for image coordinates
    pix = np.round(pix).astype(np.int32)

    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    for tri in faces:
        poly = pix[tri]
        cv2.fillConvexPoly(mask, poly, 255)

    # Close tiny holes caused by rasterization gaps and pull a single external outline.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Could not extract projected mesh outline")

    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    out = np.empty_like(cnt)
    out[:, 0] = (cnt[:, 0] - pad_px) / scale + min_xy[0]
    out[:, 1] = max_xy[1] - ((cnt[:, 1] - pad_px) / scale)
    return out


def _grid_laplacian_2d(rx, ry):
    """Small smoothness matrix for the 2D cage columns."""
    rows = []
    for i in range(rx):
        for j in range(ry):
            center = i * ry + j
            neighbours = []
            if i > 0:
                neighbours.append((i - 1) * ry + j)
            if i < rx - 1:
                neighbours.append((i + 1) * ry + j)
            if j > 0:
                neighbours.append(i * ry + (j - 1))
            if j < ry - 1:
                neighbours.append(i * ry + (j + 1))
            row = np.zeros(rx * ry, dtype=np.float64)
            row[center] = 1.0
            for nb in neighbours:
                row[nb] -= 1.0 / len(neighbours)
            rows.append(row)
    return np.vstack(rows)


def solve_depth_locked_cage(
    source_outline_2d,
    target_outline_2d,
    original_cage,
    lattice_min,
    lattice_max,
    resolution=(4, 4, 4),
    plane_axes=DEFAULT_PLANE_AXES,
    depth_axis=DEFAULT_DEPTH_AXIS,
    samples=160,
    alpha=1e-3,
    smooth_lambda=2e-2,
):
    """
    Solve only the 2D cage displacement in the sketch plane, then copy that
    displacement through every depth layer. This preserves the model's depth
    thickness while allowing the silhouette to fit the sketch.
    """
    rx, ry, rz = map(int, resolution)
    source = resample_closed_curve(source_outline_2d, samples)
    target = resample_closed_curve(target_outline_2d, samples)
    target = align_closed_curve_order(source, target)

    plane_min = np.asarray(lattice_min, dtype=np.float64)[list(plane_axes)]
    plane_max = np.asarray(lattice_max, dtype=np.float64)[list(plane_axes)]

    B = calculate_weight_matrix_2d(source, plane_min, plane_max, resolution_xy=(rx, ry), clip=False)
    delta = target - source

    L = _grid_laplacian_2d(rx, ry)
    A = (B.T @ B) + alpha * np.eye(rx * ry) + smooth_lambda * (L.T @ L)
    rhs = B.T @ delta
    disp_2d = np.linalg.solve(A, rhs)  # shape: (rx*ry, 2)

    cage = np.asarray(original_cage, dtype=np.float64).copy()
    for i in range(rx):
        for j in range(ry):
            d = disp_2d[i * ry + j]
            for k in range(rz):
                idx = np.ravel_multi_index((i, j, k), (rx, ry, rz))
                cage[idx, plane_axes[0]] += d[0]
                cage[idx, plane_axes[1]] += d[1]
                cage[idx, depth_axis] += 0.0

    fitted = B @ disp_2d + source
    rms_before = float(np.sqrt(np.mean(np.sum((source - target) ** 2, axis=1))))
    rms_after = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))
    print(f"[SOLVER] Outline RMS before: {rms_before:.5f}")
    print(f"[SOLVER] Outline RMS after : {rms_after:.5f}")
    print("[SOLVER] Depth-locked cage solved; no depth-axis displacement was added.")
    return cage, {"rms_before": rms_before, "rms_after": rms_after}
