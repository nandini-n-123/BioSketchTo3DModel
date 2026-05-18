ORGAN_PRESETS = {
    "heart": {
        "resolution": (5, 6, 4),
        "alpha": 0.018,
        "beta": 0.18,
        "max_displacement": 0.32,
        "total_samples": 240,
        "description": "Heart: stronger inferior vena cava extension, conservative top-vessel deformation.",
    },

    "brain": {
        "resolution": (5, 5, 4),
        "alpha": 0.02,
        "beta": 0.20,
        "max_displacement": 0.25,
        "total_samples": 200,
        "description": "Brain: stable closed contour with slightly improved brainstem fitting.",
    },

    "lungs": {
        "resolution": (4, 4, 4),
        "alpha": 0.015,
        "beta": 0.08,
        "max_displacement": 0.42,
        "total_samples": 180,
        "description": "Lungs: coarse global FFD with smooth lobe deformation.",
    },
}


def get_organ_preset(organ):
    organ = organ.lower().strip()

    if organ not in ORGAN_PRESETS:
        raise ValueError(
            f"Unknown organ '{organ}'. Expected one of: {list(ORGAN_PRESETS.keys())}"
        )

    return ORGAN_PRESETS[organ]