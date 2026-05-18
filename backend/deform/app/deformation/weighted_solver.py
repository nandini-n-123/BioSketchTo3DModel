"""
Depth-locked weighted FFD solver with cage smoothness regularization.

For glTF/GLB front-view deformation:
    PLANE_AXES = (0, 1)
    DEPTH_AXIS = 2

The solver only moves cage control columns in the sketch plane. The same X/Y
movement is copied through all depth layers, and Z displacement is forced to 0.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def build_grid_laplacian_edges(nx: int, ny: int) -> np.ndarray:
    """Return first-difference smoothness rows for a 2D nx-by-ny cage grid."""
    rows = []

    def col(i, j):
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            c = col(i, j)
            if i + 1 < nx:
                row = np.zeros(nx * ny, dtype=np.float64)
                row[c] = 1.0
                row[col(i + 1, j)] = -1.0
                rows.append(row)
            if j + 1 < ny:
                row = np.zeros(nx * ny, dtype=np.float64)
                row[c] = 1.0
                row[col(i, j + 1)] = -1.0
                rows.append(row)
    return np.vstack(rows) if rows else np.zeros((0, nx * ny), dtype=np.float64)


def collapse_3d_weights_to_depth_locked_2d(
    W_p: np.ndarray,
    resolution: tuple[int, int, int],
) -> np.ndarray:
    """
    Convert pilot-point FFD weights over nx*ny*nz controls into effective weights
    over nx*ny depth-locked columns by summing over the depth layers.
    """
    nx, ny, nz = resolution
    W_grid = W_p.reshape(W_p.shape[0], nx, ny, nz)
    A = W_grid.sum(axis=3).reshape(W_p.shape[0], nx * ny)
    return A


def clamp_displacements(d: np.ndarray, max_displacement: float) -> np.ndarray:
    norm = np.linalg.norm(d, axis=1)
    mask = norm > max_displacement
    if np.any(mask):
        d[mask] *= (max_displacement / np.maximum(norm[mask], EPS))[:, None]
    return d


def solve_depth_locked_weighted_cage(
    pilot_indices: np.ndarray,
    target_points_2d: np.ndarray,
    vertices_3d: np.ndarray,
    weight_matrix: np.ndarray,
    original_cage: np.ndarray,
    resolution: tuple[int, int, int] = (5, 5, 4),
    plane_axes: tuple[int, int] = (0, 1),
    depth_axis: int = 2,
    point_weights: np.ndarray | None = None,
    alpha: float = 0.01,
    beta: float = 0.10,
    max_displacement: float = 0.35,
) -> np.ndarray:
    """
    Solve for a smooth, depth-locked lattice deformation.

    Objective for each sketch-plane component:
        weighted contour matching
        + alpha * displacement magnitude
        + beta * neighbor smoothness
    """
    nx, ny, nz = resolution
    n_cols = nx * ny

    pilot_indices = np.asarray(pilot_indices, dtype=np.int64)
    target_points_2d = np.asarray(target_points_2d, dtype=np.float64)

    source_points_2d = vertices_3d[pilot_indices][:, list(plane_axes)]
    delta_2d = target_points_2d - source_points_2d

    W_p = weight_matrix[pilot_indices, :].astype(np.float64)
    A = collapse_3d_weights_to_depth_locked_2d(W_p, resolution)

    if point_weights is None:
        point_weights = np.ones(len(pilot_indices), dtype=np.float64)
    point_weights = np.asarray(point_weights, dtype=np.float64)
    point_weights = np.clip(point_weights, 0.05, 10.0)

    sqrt_w = np.sqrt(point_weights)[:, None]
    A_data = A * sqrt_w

    I = np.eye(n_cols, dtype=np.float64)
    L = build_grid_laplacian_edges(nx, ny)

    lhs = np.vstack([
        A_data,
        np.sqrt(alpha) * I,
        np.sqrt(beta) * L,
    ])

    solved_components = []
    for comp in range(2):
        rhs = np.hstack([
            delta_2d[:, comp] * sqrt_w[:, 0],
            np.zeros(n_cols, dtype=np.float64),
            np.zeros(L.shape[0], dtype=np.float64),
        ])
        d, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        solved_components.append(d)

    # Shape: nx*ny by 2
    delta_columns_2d = np.column_stack(solved_components)
    delta_columns_2d = clamp_displacements(delta_columns_2d, max_displacement=max_displacement)

    # Expand from nx*ny columns to nx*ny*nz cage points.
    grid_delta = np.zeros((nx, ny, nz, 3), dtype=np.float64)
    col_delta = delta_columns_2d.reshape(nx, ny, 2)

    ax0, ax1 = plane_axes
    for k in range(nz):
        grid_delta[:, :, k, ax0] = col_delta[:, :, 0]
        grid_delta[:, :, k, ax1] = col_delta[:, :, 1]
        grid_delta[:, :, k, depth_axis] = 0.0

    delta_cage = grid_delta.reshape(nx * ny * nz, 3)
    new_cage = original_cage.astype(np.float64) + delta_cage

    before = float(np.sqrt(np.mean(np.sum(delta_2d ** 2, axis=1))))
    predicted_delta = A @ delta_columns_2d
    residual = delta_2d - predicted_delta
    after = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))

    print(f"[SOLVER] Weighted smooth solve")
    print(f"[SOLVER] Outline RMS before: {before:.5f}")
    print(f"[SOLVER] Outline RMS after : {after:.5f}")
    print(f"[SOLVER] alpha={alpha}, beta={beta}, max_displacement={max_displacement}")
    print("[SOLVER] Depth axis displacement locked to zero.")

    return new_cage
