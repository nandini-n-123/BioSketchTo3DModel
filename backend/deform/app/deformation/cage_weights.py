import numpy as np
from scipy.special import comb


def bernstein_poly(i, n, t):
    t = np.asarray(t, dtype=np.float64)
    return comb(n, i) * (t ** i) * ((1.0 - t) ** (n - i))


def _localize(points, min_bounds, max_bounds, clip=False):
    points = np.asarray(points, dtype=np.float64)
    min_bounds = np.asarray(min_bounds, dtype=np.float64)
    max_bounds = np.asarray(max_bounds, dtype=np.float64)
    denom = max_bounds - min_bounds
    if np.any(denom <= 0):
        raise ValueError("lattice_max must be greater than lattice_min on every axis")
    local = (points - min_bounds) / denom
    if clip:
        return np.clip(local, 0.0, 1.0)
    outside = np.logical_or(local < -1e-6, local > 1.0 + 1e-6)
    if np.any(outside):
        bad = np.where(np.any(outside, axis=1))[0][:5]
        raise ValueError(
            "Some points are outside the lattice. Build the cage from the scaled mesh bounds "
            f"or pass clip=True only for diagnostics. Example bad indices: {bad.tolist()}"
        )
    return np.clip(local, 0.0, 1.0)


def calculate_weight_matrix(vertices, lattice_min, lattice_max, resolution=(4, 4, 4), clip=False):
    """
    Tensor-product Bernstein FFD binding matrix.

    This is not Mean Value Coordinates. For an undeformed regular lattice,
    vertices ~= W @ control_points because Bernstein bases have linear precision.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    rx, ry, rz = map(int, resolution)
    local = _localize(vertices, lattice_min, lattice_max, clip=clip)
    U, V, W = local[:, 0], local[:, 1], local[:, 2]

    weight_matrix = np.zeros((len(vertices), rx * ry * rz), dtype=np.float64)
    col = 0
    for i in range(rx):
        bi = bernstein_poly(i, rx - 1, U)
        for j in range(ry):
            bj = bernstein_poly(j, ry - 1, V)
            for k in range(rz):
                bk = bernstein_poly(k, rz - 1, W)
                weight_matrix[:, col] = bi * bj * bk
                col += 1
    return weight_matrix


def calculate_weight_matrix_2d(points_2d, lattice_min_2d, lattice_max_2d, resolution_xy=(4, 4), clip=False):
    """
    Reduced 2D Bernstein basis for a depth-locked FFD.

    If every depth column of the 3D cage receives the same x/z displacement,
    the 3D tensor-product basis collapses to B_i(u) B_j(v), because the depth
    basis sums to one. This gives 16 control columns for a 4x4x4 cage instead
    of solving 64 independent columns and averaging them afterwards.
    """
    points_2d = np.asarray(points_2d, dtype=np.float64)
    rx, ry = map(int, resolution_xy)
    local = _localize(points_2d, lattice_min_2d, lattice_max_2d, clip=clip)
    U, V = local[:, 0], local[:, 1]

    weight_matrix = np.zeros((len(points_2d), rx * ry), dtype=np.float64)
    col = 0
    for i in range(rx):
        bi = bernstein_poly(i, rx - 1, U)
        for j in range(ry):
            bj = bernstein_poly(j, ry - 1, V)
            weight_matrix[:, col] = bi * bj
            col += 1
    return weight_matrix
