from pathlib import Path
from typing import Tuple

import numpy as np
import trimesh


def _make_cylinder_between_points(
    p1,
    p2,
    radius,
    color,
):
    """
    Create a small cylinder between two 3D points.
    Used to draw lattice cage edges.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)

    length = np.linalg.norm(p2 - p1)

    if length < 1e-8:
        return None

    cylinder = trimesh.creation.cylinder(
        radius=radius,
        segment=[p1, p2],
        sections=8,
    )

    cylinder.visual.face_colors = color

    return cylinder


def _make_point_sphere(
    point,
    radius,
    color,
):
    """
    Create a small sphere at a cage control point.
    """
    sphere = trimesh.creation.uv_sphere(
        radius=radius,
        count=[8, 8],
    )

    sphere.apply_translation(point)
    sphere.visual.face_colors = color

    return sphere


def create_cage_wireframe_meshes(
    cage_points,
    resolution,
    line_color=(255, 0, 0, 255),
    point_color=(0, 255, 255, 255),
):
    """
    Convert cage control points into visible cage edges and control-point spheres.

    cage_points shape:
        (resolution[0] * resolution[1] * resolution[2], 3)

    resolution example:
        (5, 6, 4)
    """
    cage_points = np.asarray(cage_points, dtype=np.float64)
    resolution = tuple(resolution)

    nx, ny, nz = resolution

    grid = cage_points.reshape(nx, ny, nz, 3)

    bbox_min = cage_points.min(axis=0)
    bbox_max = cage_points.max(axis=0)
    bbox_diag = np.linalg.norm(bbox_max - bbox_min)
    bbox_diag = max(float(bbox_diag), 1e-8)

    line_radius = bbox_diag * 0.0025
    point_radius = bbox_diag * 0.009

    meshes = []

    # Edges along X direction
    for i in range(nx - 1):
        for j in range(ny):
            for k in range(nz):
                cyl = _make_cylinder_between_points(
                    grid[i, j, k],
                    grid[i + 1, j, k],
                    line_radius,
                    line_color,
                )
                if cyl is not None:
                    meshes.append(cyl)

    # Edges along Y direction
    for i in range(nx):
        for j in range(ny - 1):
            for k in range(nz):
                cyl = _make_cylinder_between_points(
                    grid[i, j, k],
                    grid[i, j + 1, k],
                    line_radius,
                    line_color,
                )
                if cyl is not None:
                    meshes.append(cyl)

    # Edges along Z direction
    for i in range(nx):
        for j in range(ny):
            for k in range(nz - 1):
                cyl = _make_cylinder_between_points(
                    grid[i, j, k],
                    grid[i, j, k + 1],
                    line_radius,
                    line_color,
                )
                if cyl is not None:
                    meshes.append(cyl)

    # Control point spheres
    for point in cage_points:
        sphere = _make_point_sphere(
            point,
            point_radius,
            point_color,
        )
        meshes.append(sphere)

    return meshes


def export_scene_with_cage(
    scene,
    cage_points,
    resolution,
    output_path,
    line_color=(255, 0, 0, 255),
    point_color=(0, 255, 255, 255),
    make_model_transparent=False,
):
    """
    Export a visualization GLB containing:
        - model
        - visible lattice cage edges
        - visible cage control points

    This is only for report/presentation images.
    It is not used by the frontend deformation output.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vis_scene = scene.copy()

    if make_model_transparent:
        for geom in vis_scene.geometry.values():
            try:
                geom.visual.face_colors = [180, 180, 180, 90]
            except Exception:
                pass

    cage_meshes = create_cage_wireframe_meshes(
        cage_points=cage_points,
        resolution=resolution,
        line_color=line_color,
        point_color=point_color,
    )

    for index, cage_mesh in enumerate(cage_meshes):
        vis_scene.add_geometry(
            cage_mesh,
            node_name=f"Lattice_Cage_{index}",
        )

    vis_scene.export(str(output_path))

    print(f"[CAGE VIEW] Saved cage visualization: {output_path}")

    return str(output_path)