from __future__ import annotations

import numpy as np
import trimesh


def _as_scene(model: trimesh.Scene | trimesh.Trimesh | str) -> trimesh.Scene:
    """Load a file path or mesh object into a trimesh.Scene."""
    if isinstance(model, trimesh.Scene):
        return model.copy()

    if isinstance(model, trimesh.Trimesh):
        return trimesh.Scene(model.copy())

    loaded = trimesh.load(model, force="scene")
    if isinstance(loaded, trimesh.Scene):
        return loaded
    return trimesh.Scene(loaded)


def load_and_normalize_mesh(model_path: str) -> trimesh.Scene:
    """Load a mesh/scene and normalize it into a centered unit cube.

    The longest axis is scaled to exactly 2.0 units, so the resulting
    bounds are approximately [-1, 1] in the dominant dimension.
    """
    scene = _as_scene(model_path)

    if not scene.geometry:
        raise ValueError(f"No geometry found in model: {model_path}")

    # Concatenate to compute a stable global bounding box.
    mesh = trimesh.util.concatenate(tuple(geom.copy() for geom in scene.geometry.values()))
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    extents = bounds[1] - bounds[0]
    max_extent = float(np.max(extents))
    if max_extent <= 1e-12:
        raise ValueError("Mesh has near-zero extent and cannot be normalized.")

    scale = 2.0 / max_extent

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= scale
    transform[:3, 3] = -center * scale

    # Apply the same normalization to every geometry in the scene.
    normalized = scene.copy()
    for name, geom in normalized.geometry.items():
        geom.apply_transform(transform)

    return normalized
