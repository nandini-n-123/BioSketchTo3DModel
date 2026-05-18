import numpy as np


def bounds_from_vertices(vertices, padding=0.08, min_thickness=1e-4):
    """
    Build a lattice box around the actual working vertices.

    Do this *after* any sketch/model pre-alignment. A fixed [-1, 1] cage is
    dangerous once the model has been anisotropically scaled, because vertices
    outside the cage get clamped during weight calculation.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    min_b = vertices.min(axis=0)
    max_b = vertices.max(axis=0)
    span = np.maximum(max_b - min_b, min_thickness)
    pad = span * padding
    return min_b - pad, max_b + pad


class LatticeFFD:
    def __init__(self, resolution=(4, 4, 4), bounds=None):
        self.resolution = np.asarray(resolution, dtype=int)

        if bounds is None:
            self.min_bounds = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
            self.max_bounds = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        else:
            self.min_bounds = np.asarray(bounds[0], dtype=np.float64)
            self.max_bounds = np.asarray(bounds[1], dtype=np.float64)

        if np.any(self.max_bounds <= self.min_bounds):
            raise ValueError("Invalid lattice bounds: max_bounds must be > min_bounds on every axis")

        self.control_points = self._generate_grid()
        self.original_control_points = self.control_points.copy()

    def _generate_grid(self):
        rx, ry, rz = self.resolution
        x_steps = np.linspace(self.min_bounds[0], self.max_bounds[0], rx)
        y_steps = np.linspace(self.min_bounds[1], self.max_bounds[1], ry)
        z_steps = np.linspace(self.min_bounds[2], self.max_bounds[2], rz)
        gx, gy, gz = np.meshgrid(x_steps, y_steps, z_steps, indexing="ij")
        return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(np.float64)

    def get_points(self):
        return self.control_points.copy()
