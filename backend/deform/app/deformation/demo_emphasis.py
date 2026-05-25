import numpy as np


def _safe_norm01(values):
    values = np.asarray(values, dtype=np.float64)
    v_min = float(values.min())
    v_max = float(values.max())
    denom = max(v_max - v_min, 1e-8)
    return (values - v_min) / denom


def emphasize_brainstem_from_sketch(
    vertices_3d,
    sketch_points_2d,
    organ,
    plane_axes=(0, 1),
    depth_axis=2,
    strength=0.45,
):
    """
    Mild presentation-time emphasis for brainstem visibility.

    This does NOT create new anatomy.
    It only elongates the existing lower-center brainstem region smoothly.

    Important:
    - only for brain
    - preserves depth axis
    - uses tapered falloff so it does not create a rectangular extrusion
    """
    if organ.lower().strip() != "brain":
        return vertices_3d

    v = np.asarray(vertices_3d, dtype=np.float64).copy()
    sketch = np.asarray(sketch_points_2d, dtype=np.float64)

    ax_x, ax_y = plane_axes

    x = v[:, ax_x]
    y = v[:, ax_y]

    x_norm = _safe_norm01(x)
    y_norm = _safe_norm01(y)

    sx = sketch[:, 0]
    sy = sketch[:, 1]

    sx_norm = _safe_norm01(sx)
    sy_norm = _safe_norm01(sy)

    # ------------------------------------------------------------
    # 1. Estimate sketch brainstem center and bottom
    # ------------------------------------------------------------
    sketch_stem_mask = (
        (sx_norm > 0.40)
        & (sx_norm < 0.60)
        & (sy_norm < 0.35)
    )

    if np.count_nonzero(sketch_stem_mask) >= 2:
        sketch_stem_center_x_norm = float(np.mean(sx_norm[sketch_stem_mask]))
        sketch_stem_bottom_y = float(np.min(sy[sketch_stem_mask]))
    else:
        sketch_stem_center_x_norm = 0.50
        sketch_stem_bottom_y = float(np.min(sy))

    # ------------------------------------------------------------
    # 2. Narrow brainstem-only selection
    # ------------------------------------------------------------
    center = sketch_stem_center_x_norm

    stem_mask = (
        (x_norm > center - 0.06)
        & (x_norm < center + 0.06)
        & (y_norm < 0.30)
    )

    # small fallback if too few vertices selected
    if np.count_nonzero(stem_mask) < 10:
        stem_mask = (
            (x_norm > center - 0.08)
            & (x_norm < center + 0.08)
            & (y_norm < 0.34)
        )

    if np.count_nonzero(stem_mask) < 10:
        print("[DEMO EMPHASIS] Brainstem region too small. Skipping.")
        return v

    stem_indices = np.where(stem_mask)[0]

    stem_y = y[stem_indices]
    stem_top_y = float(np.max(stem_y))
    stem_bottom_y = float(np.min(stem_y))
    current_length = max(stem_top_y - stem_bottom_y, 1e-8)

    # ------------------------------------------------------------
    # 3. Controlled downward extension
    # ------------------------------------------------------------
    # how much lower the sketch stem is compared to current stem
    target_extra = max(0.0, stem_bottom_y - sketch_stem_bottom_y)

    # clamp for realism
    target_extra = min(target_extra, 0.35 * current_length)

    # if sketch difference is tiny, still give a small visible demo stretch
    min_demo_extra = 0.12 * current_length
    extra = max(target_extra, min_demo_extra)

    # strength factor should stay small (0.35 to 0.55 is good)
    extra *= float(strength)

    # ------------------------------------------------------------
    # 4. Smooth tapered falloff
    # ------------------------------------------------------------
    # bottom points move more, top points move less
    rel = (stem_top_y - stem_y) / current_length
    rel = np.clip(rel, 0.0, 1.0)

    # smoother, more natural falloff
    falloff = rel ** 1.8

    new_y = stem_y - extra * falloff

    v[stem_indices, ax_y] = new_y

    # ------------------------------------------------------------
    # 5. Very slight center pull to avoid blockiness
    # ------------------------------------------------------------
    stem_x = v[stem_indices, ax_x]
    stem_center_x = float(np.mean(stem_x))

    # slightly tighten the stem instead of widening it
    x_rel = stem_x - stem_center_x
    v[stem_indices, ax_x] = stem_center_x + x_rel * 0.96

    print(
        "[DEMO EMPHASIS] Mild brainstem stretch applied | "
        f"vertices={len(stem_indices)}, "
        f"extra={extra:.4f}, "
        f"old_bottom={stem_bottom_y:.4f}, "
        f"new_bottom={float(np.min(v[stem_indices, ax_y])):.4f}"
    )

    return v