from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from biosketch_classifier.models import create_model
from biosketch_classifier.preprocessing import (
    adobe_scan_like_preprocess,
    make_transforms,
    pil_open_rgb,
    square_pad_pil,
)


class OrganClassifier:
    """
    Lightweight wrapper used by the backend.

    Example:
        classifier = OrganClassifier(
            checkpoint_path="model_scripts/checkpoints/organ_classifier/best_model.pt"
        )

        result = classifier.predict("uploads/student_sketch.jpg")
        print(result["organ"], result["confidence"])
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        confidence_threshold: float = 0.65,
        use_scan_preprocessing: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.confidence_threshold = confidence_threshold
        self.use_scan_preprocessing = use_scan_preprocessing

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path.resolve()}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        self.class_to_idx = self.checkpoint["class_to_idx"]
        self.idx_to_class = self._normalize_idx_to_class(self.checkpoint["idx_to_class"])

        self.image_size = int(self.checkpoint.get("image_size", 224))
        self.model_name = self.checkpoint.get("model_name", "efficientnet_b0")
        self.num_classes = len(self.class_to_idx)

        self.model = create_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            pretrained=False,
            freeze_backbone=False,
        ).to(self.device)

        self.model.load_state_dict(self.checkpoint["model_state"])
        self.model.eval()

        self.transform = make_transforms(train=False, image_size=self.image_size)

    def _normalize_idx_to_class(self, raw_idx_to_class: Any) -> dict[int, str]:
        """
        Supports different saved checkpoint formats:

        1. {0: "brain", 1: "heart", 2: "lungs"}
        2. {"0": "brain", "1": "heart", "2": "lungs"}
        3. {"brain": 0, "heart": 1, "lungs": 2}
        4. ["brain", "heart", "lungs"]
        """

        if isinstance(raw_idx_to_class, dict):
            first_key = next(iter(raw_idx_to_class.keys()))
            first_value = raw_idx_to_class[first_key]

            if isinstance(first_value, str):
                return {int(k): str(v) for k, v in raw_idx_to_class.items()}

            return {int(v): str(k) for k, v in raw_idx_to_class.items()}

        if isinstance(raw_idx_to_class, list):
            return {i: str(label) for i, label in enumerate(raw_idx_to_class)}

        raise ValueError(f"Unsupported idx_to_class format: {type(raw_idx_to_class)}")

    def preprocess_image(self, image_path: str | Path):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path.resolve()}")

        if self.use_scan_preprocessing:
            img = adobe_scan_like_preprocess(image_path)
        else:
            img = square_pad_pil(pil_open_rgb(image_path))

        return img

    def predict(
        self,
        image_path: str | Path,
        topk: int = 3,
        save_processed_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Predict organ class from one image.

        Returns:
            {
                "organ": "brain",
                "confidence": 0.94,
                "is_confident": True,
                "route_to_asset": "assets/3d_models/brain.glb",
                "probabilities": {
                    "brain": 0.94,
                    "heart": 0.04,
                    "lungs": 0.02
                },
                "topk": [...]
            }
        """

        img = self.preprocess_image(image_path)

        if save_processed_path is not None:
            save_processed_path = Path(save_processed_path)
            save_processed_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(save_processed_path)

        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu()

        topk = min(topk, self.num_classes)
        values, indices = torch.topk(probs, k=topk)

        probabilities = {
            self.idx_to_class[i]: float(probs[i])
            for i in range(self.num_classes)
        }

        topk_results = []
        for prob, idx in zip(values.tolist(), indices.tolist()):
            label = self.idx_to_class[int(idx)]
            topk_results.append(
                {
                    "organ": label,
                    "confidence": float(prob),
                }
            )

        best_label = topk_results[0]["organ"]
        best_confidence = float(topk_results[0]["confidence"])
        is_confident = best_confidence >= self.confidence_threshold

        return {
            "organ": best_label,
            "confidence": best_confidence,
            "is_confident": is_confident,
            "confidence_threshold": self.confidence_threshold,
            "route_to_asset": f"assets/3d_models/{best_label}.glb",
            "probabilities": probabilities,
            "topk": topk_results,
            "device": str(self.device),
            "model_name": self.model_name,
        }


if __name__ == "__main__":
    classifier = OrganClassifier(
        checkpoint_path="model_scripts/checkpoints/organ_classifier/best_model.pt",
        confidence_threshold=0.65,
    )

    result = classifier.predict(
        image_path="test_sketches/test_brain.jpg",
        save_processed_path="reports/processed_backend_test.png",
    )

    print(result)