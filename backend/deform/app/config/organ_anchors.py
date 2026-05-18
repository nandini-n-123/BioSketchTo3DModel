"""
Organ-specific configuration for BioSketch deformation.

Keep this as ONE file for now. It is easier to maintain while the project has only
heart, lungs, and brain. Later, if each organ needs many custom rules, split this
into heart_anchors.py, lung_anchors.py, brain_anchors.py.
"""

from __future__ import annotations

ORGAN_CONFIGS = {
    "heart": {
        "materials": [
            "Veins",
            "Heart_Base",
            "Ventricle",
            "Pulmonary_Artery",
            "Right_atrium",
            "Left_Atrium",
            "Left_Auricle",
            "Aorta",
            "Inferior_vena_cava",
            "PulmonaryArtery_Invisible",
        ],
        # Heuristic contour landmarks in clockwise/counter-clockwise order.
        # These names are used only to split the contour into stable sections.
        "landmarks": [
            "apex_bottom",
            "left_body",
            "top_left_vessel",
            "aorta_top",
            "top_right_vessel",
            "right_body",
        ],
        "weights": {
            "apex_bottom": 0.55,
            "left_body": 1.00,
            "right_body": 1.00,
            "top_left_vessel": 0.80,
            "aorta_top": 0.80,
            "top_right_vessel": 0.80,
            "default": 1.00,
        },
    },
    "lungs": {
        "materials": [
            "trachea",
            "left_lung",
            "right_lung",
        ],
        "landmarks": [
            "trachea_top",
            "left_outer",
            "left_bottom",
            "middle_bottom",
            "right_bottom",
            "right_outer",
        ],
        "weights": {
            "trachea_top": 0.75,
            "left_outer": 1.00,
            "right_outer": 1.00,
            "left_bottom": 0.85,
            "right_bottom": 0.85,
            "middle_bottom": 0.70,
            "default": 1.00,
        },
    },
    "brain": {
        "materials": [
            "Brain_Base",
            "Cerebellum",
            "Main_Lobe",
            "Brain_Stem",
        ],
        "landmarks": [
            "brain_top",
            "front_outer",
            "brainstem_bottom",
            "back_outer",
        ],
        "weights": {
            "brain_top": 1.00,
            "front_outer": 1.00,
            "back_outer": 1.00,
            "brainstem_bottom": 0.75,
            "default": 1.00,
        },
    },
}


def get_organ_config(organ: str) -> dict:
    organ_key = organ.lower().strip()
    if organ_key not in ORGAN_CONFIGS:
        raise ValueError(f"Unknown organ '{organ}'. Expected one of: {list(ORGAN_CONFIGS)}")
    return ORGAN_CONFIGS[organ_key]
