import os
import sys
from pathlib import Path

# Make sure Python can import app/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.pipeline import deform_organ


def test_one_organ(organ, sketch, model):
    result = deform_organ(
        sketch_path=sketch,
        organ=organ,
        model_path=model,
        save_debug=True,
        apply_debug_colors=False,
    )

    print("\n========== RESULT ==========")
    for key, value in result.items():
        print(f"{key}: {value}")

    output_path = result["output_path"]

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Output GLB was not created: {output_path}")

    if os.path.getsize(output_path) == 0:
        raise RuntimeError(f"Output GLB is empty: {output_path}")

    print(f"\n[SUCCESS] {organ} pipeline worked.")
    print(f"[OUTPUT] {output_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]

    assets = root / "assets"
    models = assets / "3d_models"
    sketches = assets / "sketches"

    test_one_organ(
        organ="heart",
        sketch=str(sketches / "h1.jpeg"),
        model=str(models / "heart.glb"),
    )

    test_one_organ(
        organ="brain",
        sketch=str(sketches / "test_brain.jpg"),
        model=str(models / "brain.glb"),
    )

    test_one_organ(
        organ="lungs",
        sketch=str(sketches / "lungs3.jpg"),
        model=str(models / "lungs.glb"),
    )