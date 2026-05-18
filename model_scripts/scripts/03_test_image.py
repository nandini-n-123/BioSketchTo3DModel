from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from biosketch_classifier.models import create_model
from biosketch_classifier.preprocessing import (
    adobe_scan_like_preprocess,
    make_transforms,
    pil_open_rgb,
    square_pad_pil,
)


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify hand-drawn organ sketch")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/organ_classifier/best_model.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional direct path to one sketch image",
    )
    parser.add_argument(
        "--image-folder",
        default="test_sketches",
        help="Folder containing test sketch images",
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument(
        "--no-scan",
        action="store_true",
        help="Disable Adobe-Scan-like preprocessing",
    )
    parser.add_argument(
        "--save-processed",
        default="reports/processed_test.png",
        help="Optional path to save black-on-white processed image",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.55,
        help="Warn when top probability is below this",
    )
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(
            f"Image folder not found: {folder.resolve()}\n"
            f"Create this folder and place your sketches inside it."
        )

    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

    return sorted(images)


def choose_image_interactively(image_folder: str) -> Path:
    folder = Path(image_folder)
    images = list_images(folder)

    if not images:
        raise FileNotFoundError(
            f"No images found in {folder.resolve()}.\n"
            f"Supported formats: {', '.join(sorted(VALID_EXTENSIONS))}"
        )

    print("\nAvailable test sketches:\n")
    for i, image_path in enumerate(images, start=1):
        print(f"{i}. {image_path.name}")

    while True:
        choice = input("\nEnter image number to test: ").strip()

        if not choice.isdigit():
            print("Please enter a valid number.")
            continue

        index = int(choice)

        if 1 <= index <= len(images):
            selected = images[index - 1]
            print(f"\nSelected image: {selected}")
            return selected

        print(f"Please enter a number between 1 and {len(images)}.")


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_to_idx = checkpoint["class_to_idx"]
    raw_idx_to_class = checkpoint["idx_to_class"]

    # Make idx_to_class robust for both possible formats:
    # Format A: {0: "brain", 1: "heart", 2: "lungs"}
    # Format B: {"0": "brain", "1": "heart", "2": "lungs"}
    # Format C: {"brain": 0, "heart": 1, "lungs": 2}
    if isinstance(raw_idx_to_class, dict):
        first_key = next(iter(raw_idx_to_class.keys()))
        first_value = raw_idx_to_class[first_key]

        if isinstance(first_value, str):
            # {0: "brain"} or {"0": "brain"}
            idx_to_class = {
                int(k): v for k, v in raw_idx_to_class.items()
            }
        else:
            # {"brain": 0}
            idx_to_class = {
                int(v): k for k, v in raw_idx_to_class.items()
            }
    else:
        # list format: ["brain", "heart", "lungs"]
        idx_to_class = {
            i: label for i, label in enumerate(raw_idx_to_class)
        }

    num_classes = len(class_to_idx)

    model = create_model(
        checkpoint["model_name"],
        num_classes,
        pretrained=False,
        freeze_backbone=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, checkpoint, idx_to_class, num_classes


def classify_image(
    image_path: Path,
    model,
    checkpoint,
    idx_to_class,
    num_classes: int,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    if args.no_scan:
        img = square_pad_pil(pil_open_rgb(image_path))
    else:
        img = adobe_scan_like_preprocess(image_path)

    if args.save_processed:
        out_path = Path(args.save_processed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        print(f"\nSaved processed image -> {out_path.resolve()}")

    transform = make_transforms(
        train=False,
        image_size=int(checkpoint.get("image_size", 224)),
    )

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu()

    topk = min(args.topk, num_classes)
    values, indices = torch.topk(probs, k=topk)

    print("\nPrediction probabilities:")
    for prob, idx in zip(values.tolist(), indices.tolist()):
        label = idx_to_class[int(idx)]
        print(f"  {label}: {prob:.4f}")

    best_prob = float(values[0])
    best_label = idx_to_class[int(indices[0])]

    print(f"\nFinal predicted organ: {best_label}")
    print(f"Confidence: {best_prob:.4f}")
    print(f"Route to 3D asset suggestion: assets/3d_models/{best_label}.glb")

    if best_prob < args.confidence_threshold:
        print(
            f"\nWARNING: confidence {best_prob:.3f} is below threshold "
            f"{args.confidence_threshold:.3f}.\n"
            "Use manual fallback: ask the student to select Heart, Brain, or Lungs."
        )


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, checkpoint, idx_to_class, num_classes = load_model(
        args.checkpoint,
        device,
    )

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path.resolve()}")
    else:
        image_path = choose_image_interactively(args.image_folder)

    classify_image(
        image_path=image_path,
        model=model,
        checkpoint=checkpoint,
        idx_to_class=idx_to_class,
        num_classes=num_classes,
        device=device,
        args=args,
    )


if __name__ == "__main__":
    main()