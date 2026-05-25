ORGAN_PRESETS = {
    "heart": {
    "resolution": (5, 6, 4),
    "alpha": 0.018,
    "beta": 0.24,
    "max_displacement": 0.26,
    "total_samples": 220,
    "description": "Heart: conservative global FFD to avoid sideways atrium stretching.",
},

    "brain": {
    "description": "Brain: visible brainstem deformation with controlled body preservation.",
    "resolution": (5, 7, 4),
    "alpha": 0.012,
    "beta": 0.08,
    "max_displacement": 0.42,
    "total_samples": 220,
    "use_part_constraints": True,
    },

    "lungs": {
        "resolution": (4, 4, 4),
        "alpha": 0.015,
        "beta": 0.06,
        "max_displacement": 0.50,
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