import numpy as np


def _normalize_plane(points_2d):
    pts = np.asarray(points_2d, dtype=np.float32)

    x = pts[:, 0]
    y = pts[:, 1]

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    xr = max(x_max - x_min, 1e-8)
    yr = max(y_max - y_min, 1e-8)

    xn = (x - x_min) / xr
    yn = (y - y_min) / yr

    return xn, yn


def _choose_vertex_from_region(vertices_3d, plane_axes, region_mask_fn, extreme):
    """
    Pick one global vertex index from a spatial region of the mesh.

    extreme:
        min_x, max_x, min_y, max_y
    """
    projected = vertices_3d[:, plane_axes]
    x = projected[:, 0]
    y = projected[:, 1]

    xn, yn = _normalize_plane(projected)

    mask = region_mask_fn(xn, yn)

    if np.count_nonzero(mask) < 5:
        return None

    candidate_indices = np.where(mask)[0]
    candidate_pts = projected[candidate_indices]

    if extreme == "min_x":
        local = int(np.argmin(candidate_pts[:, 0]))
    elif extreme == "max_x":
        local = int(np.argmax(candidate_pts[:, 0]))
    elif extreme == "min_y":
        local = int(np.argmin(candidate_pts[:, 1]))
    elif extreme == "max_y":
        local = int(np.argmax(candidate_pts[:, 1]))
    else:
        raise ValueError(f"Unknown extreme: {extreme}")

    return int(candidate_indices[local])


def _choose_sketch_target(sketch_points_2d, region_mask_fn, extreme):
    """
    Pick one 2D target point from a spatial region of the sketch contour.
    """
    pts = np.asarray(sketch_points_2d, dtype=np.float32)

    x = pts[:, 0]
    y = pts[:, 1]

    xn, yn = _normalize_plane(pts)

    mask = region_mask_fn(xn, yn)

    if np.count_nonzero(mask) < 2:
        return None

    candidate_indices = np.where(mask)[0]
    candidate_pts = pts[candidate_indices]

    if extreme == "min_x":
        local = int(np.argmin(candidate_pts[:, 0]))
    elif extreme == "max_x":
        local = int(np.argmax(candidate_pts[:, 0]))
    elif extreme == "min_y":
        local = int(np.argmin(candidate_pts[:, 1]))
    elif extreme == "max_y":
        local = int(np.argmax(candidate_pts[:, 1]))
    else:
        raise ValueError(f"Unknown extreme: {extreme}")

    return candidate_pts[local].astype(np.float32)


def _add_constraint(
    specs,
    label,
    mesh_region,
    mesh_extreme,
    sketch_region,
    sketch_extreme,
    weight,
):
    specs.append(
        {
            "label": label,
            "mesh_region": mesh_region,
            "mesh_extreme": mesh_extreme,
            "sketch_region": sketch_region,
            "sketch_extreme": sketch_extreme,
            "weight": float(weight),
        }
    )


def _heart_specs():
    specs = []

    # Inferior vena cava: lower-left vertical vessel.
    # This is the important one you are struggling with.
    _add_constraint(
        specs,
        label="heart_ivc_bottom",
        mesh_region=lambda x, y: (x < 0.34) & (y < 0.62),
        mesh_extreme="min_y",
        sketch_region=lambda x, y: (x < 0.34) & (y < 0.68),
        sketch_extreme="min_y",
        weight=6.0,
    )

    # IVC left wall / side alignment.
    _add_constraint(
        specs,
        label="heart_ivc_left_side",
        mesh_region=lambda x, y: (x < 0.30) & (y > 0.20) & (y < 0.65),
        mesh_extreme="min_x",
        sketch_region=lambda x, y: (x < 0.30) & (y > 0.20) & (y < 0.70),
        sketch_extreme="min_x",
        weight=2.0,
    )

    # Aorta / top middle.
    _add_constraint(
        specs,
        label="heart_aorta_top",
        mesh_region=lambda x, y: (x > 0.38) & (x < 0.62) & (y > 0.70),
        mesh_extreme="max_y",
        sketch_region=lambda x, y: (x > 0.38) & (x < 0.65) & (y > 0.72),
        sketch_extreme="max_y",
        weight=1.6,
    )

    # Upper-left vessel / vena cava top.
    _add_constraint(
        specs,
        label="heart_upper_left_vessel",
        mesh_region=lambda x, y: (x < 0.32) & (y > 0.65),
        mesh_extreme="max_y",
        sketch_region=lambda x, y: (x < 0.34) & (y > 0.65),
        sketch_extreme="max_y",
        weight=1.8,
    )

    # Pulmonary artery / rightward extension.
    _add_constraint(
        specs,
        label="heart_pulmonary_right",
        mesh_region=lambda x, y: (x > 0.58) & (y > 0.42) & (y < 0.78),
        mesh_extreme="max_x",
        sketch_region=lambda x, y: (x > 0.58) & (y > 0.42) & (y < 0.80),
        sketch_extreme="max_x",
        weight=1.8,
    )

    # Heart body bottom/apex.
    _add_constraint(
        specs,
        label="heart_body_bottom",
        mesh_region=lambda x, y: (x > 0.30) & (x < 0.75) & (y < 0.25),
        mesh_extreme="min_y",
        sketch_region=lambda x, y: (x > 0.30) & (x < 0.75) & (y < 0.25),
        sketch_extreme="min_y",
        weight=1.5,
    )

    return specs


def _brain_specs():
    specs = []

    # Top brain dome.
    _add_constraint(
        specs,
        label="brain_top_dome",
        mesh_region=lambda x, y: (x > 0.25) & (x < 0.75) & (y > 0.70),
        mesh_extreme="max_y",
        sketch_region=lambda x, y: (x > 0.25) & (x < 0.75) & (y > 0.70),
        sketch_extreme="max_y",
        weight=1.4,
    )

    # Back/left outer brain.
    _add_constraint(
        specs,
        label="brain_back_left",
        mesh_region=lambda x, y: (x < 0.25) & (y > 0.35),
        mesh_extreme="min_x",
        sketch_region=lambda x, y: (x < 0.25) & (y > 0.35),
        sketch_extreme="min_x",
        weight=1.3,
    )

    # Front/right outer brain.
    _add_constraint(
        specs,
        label="brain_front_right",
        mesh_region=lambda x, y: (x > 0.70) & (y > 0.30),
        mesh_extreme="max_x",
        sketch_region=lambda x, y: (x > 0.70) & (y > 0.30),
        sketch_extreme="max_x",
        weight=1.3,
    )

    # Brainstem bottom.
    _add_constraint(
        specs,
        label="brainstem_bottom",
        mesh_region=lambda x, y: (x > 0.35) & (x < 0.65) & (y < 0.30),
        mesh_extreme="min_y",
        sketch_region=lambda x, y: (x > 0.35) & (x < 0.65) & (y < 0.30),
        sketch_extreme="min_y",
        weight=1.7,
    )

    

    return specs


def _lungs_specs():
    specs = []

    # Left lobe outer side.
    _add_constraint(
        specs,
        label="left_lung_outer",
        mesh_region=lambda x, y: (x < 0.35) & (y > 0.20) & (y < 0.85),
        mesh_extreme="min_x",
        sketch_region=lambda x, y: (x < 0.35) & (y > 0.20) & (y < 0.85),
        sketch_extreme="min_x",
        weight=1.4,
    )

    # Left lobe bottom.
    _add_constraint(
        specs,
        label="left_lung_bottom",
        mesh_region=lambda x, y: (x < 0.45) & (y < 0.35),
        mesh_extreme="min_y",
        sketch_region=lambda x, y: (x < 0.45) & (y < 0.35),
        sketch_extreme="min_y",
        weight=1.1,
    )

    # Right lobe outer side.
    _add_constraint(
        specs,
        label="right_lung_outer",
        mesh_region=lambda x, y: (x > 0.65) & (y > 0.20) & (y < 0.85),
        mesh_extreme="max_x",
        sketch_region=lambda x, y: (x > 0.65) & (y > 0.20) & (y < 0.85),
        sketch_extreme="max_x",
        weight=1.4,
    )

    # Right lobe bottom.
    _add_constraint(
        specs,
        label="right_lung_bottom",
        mesh_region=lambda x, y: (x > 0.55) & (y < 0.35),
        mesh_extreme="min_y",
        sketch_region=lambda x, y: (x > 0.55) & (y < 0.35),
        sketch_extreme="min_y",
        weight=1.1,
    )

    # Trachea top.
    _add_constraint(
        specs,
        label="trachea_top",
        mesh_region=lambda x, y: (x > 0.40) & (x < 0.60) & (y > 0.65),
        mesh_extreme="max_y",
        sketch_region=lambda x, y: (x > 0.38) & (x < 0.62) & (y > 0.65),
        sketch_extreme="max_y",
        weight=1,
    )

    # Central branch / bifurcation.
    _add_constraint(
        specs,
        label="trachea_branch_center",
        mesh_region=lambda x, y: (x > 0.38) & (x < 0.62) & (y > 0.38) & (y < 0.65),
        mesh_extreme="min_y",
        sketch_region=lambda x, y: (x > 0.35) & (x < 0.65) & (y > 0.35) & (y < 0.65),
        sketch_extreme="min_y",
        weight=0.8,
    )

    return specs


def build_part_constraints(
    scene,
    vertices_3d,
    sketch_points_2d,
    organ,
    plane_axes=(0, 1),
):
    """
    Build extra region-aware constraints.

    This does NOT require separate mesh objects or material IDs.
    It works even when Blender/GLB exports everything as one combined mesh.

    Returns:
      pilot_indices_extra : M
      target_points_extra : Mx2
      weights_extra       : M
    """
    organ = organ.lower().strip()

    if organ == "heart":
        specs = _heart_specs()
    elif organ == "brain":
        specs = _brain_specs()
    elif organ == "lungs":
        specs = _lungs_specs()
    else:
        return (
            np.asarray([], dtype=np.int64),
            np.zeros((0, 2), dtype=np.float32),
            np.asarray([], dtype=np.float32),
        )

    pilot_indices = []
    target_points = []
    weights = []

    for spec in specs:
        pilot = _choose_vertex_from_region(
            vertices_3d=vertices_3d,
            plane_axes=plane_axes,
            region_mask_fn=spec["mesh_region"],
            extreme=spec["mesh_extreme"],
        )

        target = _choose_sketch_target(
            sketch_points_2d=sketch_points_2d,
            region_mask_fn=spec["sketch_region"],
            extreme=spec["sketch_extreme"],
        )

        if pilot is None or target is None:
            print(f"[PART] Skipped region constraint: {spec['label']}")
            continue

        pilot_indices.append(pilot)
        target_points.append(target)
        weights.append(spec["weight"])

        print(
            f"[PART] Added region constraint: {spec['label']} | "
            f"pilot={pilot} | "
            f"target=({target[0]:.3f}, {target[1]:.3f}) | "
            f"weight={spec['weight']:.2f}"
        )

    if len(pilot_indices) == 0:
        return (
            np.asarray([], dtype=np.int64),
            np.zeros((0, 2), dtype=np.float32),
            np.asarray([], dtype=np.float32),
        )

    return (
        np.asarray(pilot_indices, dtype=np.int64),
        np.asarray(target_points, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
    )


def print_scene_geometry_names(scene):
    """
    Debug helper. Kept for compatibility with your current test_deformation.py.
    """
    print("\n[DEBUG] Scene geometry names:")
    for name in scene.geometry.keys():
        print(f"  - {name}")