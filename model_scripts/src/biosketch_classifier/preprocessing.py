from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from .constants import IMAGENET_MEAN, IMAGENET_STD


def pil_open_rgb(path: str | Path) -> Image.Image:
    """Open an image as RGB, compositing transparent PNGs over white."""
    img = Image.open(path)
    if img.mode in {"RGBA", "LA"} or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba).convert("RGB")
    else:
        img = img.convert("RGB")
    return img


def square_pad_pil(img: Image.Image, fill: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Pad image to square without stretching anatomy."""
    w, h = img.size
    side = max(w, h)
    out = Image.new("RGB", (side, side), fill)
    out.paste(img, ((side - w) // 2, (side - h) // 2))
    return out


def _read_cv2_with_alpha_on_white(path: str | Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        bgr = arr[:, :, :3].astype(np.float32)
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        white = np.full_like(bgr, 255, dtype=np.float32)
        bgr = bgr * alpha + white * (1.0 - alpha)
        return bgr.astype(np.uint8)
    return arr[:, :, :3]


def adobe_scan_like_preprocess(
    path: str | Path,
    *,
    crop_to_content: bool = True,
    pad_ratio: float = 0.12,
    min_component_area: int = 20,
    block_size: int = 35,
    c_value: int = 11,
) -> Image.Image:
    """
    Convert a hand-drawn/photo sketch to black strokes on a white background.

    This is designed for inference on webcam/phone images. It avoids flipping or
    rotating the organ. It only improves paper background and stroke contrast.
    """
    bgr = _read_cv2_with_alpha_on_white(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Smooth tiny noise, then normalize uneven lighting using a large background blur.
    gray = cv2.medianBlur(gray, 3)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
    normalized = cv2.divide(gray, background, scale=255)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    # Local threshold keeps pencil strokes visible under non-uniform lighting.
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)
    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_value,
    )

    # Remove very small black specks while retaining thin lines.
    black = (binary == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(black, 8)
    cleaned_black = np.zeros_like(black)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            cleaned_black[labels == label] = 1
    cleaned = np.where(cleaned_black > 0, 0, 255).astype(np.uint8)

    # Slightly strengthen pencil strokes; use a tiny kernel to avoid changing organ shape.
    stroke_mask = (cleaned == 0).astype(np.uint8) * 255
    stroke_mask = cv2.dilate(stroke_mask, np.ones((2, 2), np.uint8), iterations=1)
    cleaned = np.where(stroke_mask > 0, 0, 255).astype(np.uint8)

    if crop_to_content:
        ys, xs = np.where(cleaned < 250)
        if len(xs) > 0 and len(ys) > 0:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            h, w = cleaned.shape
            pad = int(max(x2 - x1 + 1, y2 - y1 + 1) * pad_ratio)
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w - 1, x2 + pad)
            y2 = min(h - 1, y2 + pad)
            cleaned = cleaned[y1 : y2 + 1, x1 : x2 + 1]

    rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
    return square_pad_pil(Image.fromarray(rgb), fill=(255, 255, 255))


def make_transforms(train: bool, image_size: int = 224):
    """Create torchvision transforms for pretrained CNNs."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    if train:
        return transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
                # Safe sketch augmentation: no horizontal flip, no vertical flip, no upside-down rotations.
                transforms.RandomAffine(
                    degrees=8,
                    translate=(0.04, 0.04),
                    scale=(0.90, 1.08),
                    shear=(-3, 3, -2, 2),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=(255, 255, 255),
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
