"""
Stable sketch preprocessing for BioSketch-3D.

Drop-in replacement for:
    backend/deform/app/vision/sketch_parser.py

This version is intentionally simpler and safer than the previous experimental
multi-scale black-hat pipeline. It removes:
    - soft shadows and uneven paper lighting
    - most scattered dots
    - disconnected shadow edges outside the main organ
    - faint erased marks that are not connected to the intended sketch

The public API remains compatible with the existing pipeline:
    preprocess_sketch(image_path)
    extract_contours(binary_img, min_area=2000)
    optimize_vectors(contours_dict, initial_epsilon_factor=0.002)
    normalize_coordinates(contours_dict, img_width=None, img_height=None)
    get_primary_contour_points(normalized_dict)

Optional:
    preprocess_sketch(
        image_path,
        cleanup_strength="balanced",
        manual_ignore_mask_path=None,
        debug_dir=None,
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ============================================================
# Profiles
# ============================================================

_CLEANUP_PROFILES = {
    "gentle": {
        "adaptive_c": 8,
        "first_min_area": 7,
        "final_min_area": 28,
        "roi_padding_ratio": 0.08,
        "edge_margin_ratio": 0.010,
    },
    "balanced": {
        "adaptive_c": 10,
        "first_min_area": 10,
        "final_min_area": 45,
        "roi_padding_ratio": 0.06,
        "edge_margin_ratio": 0.015,
    },
    "strict": {
        "adaptive_c": 12,
        "first_min_area": 14,
        "final_min_area": 70,
        "roi_padding_ratio": 0.045,
        "edge_margin_ratio": 0.020,
    },
}


# ============================================================
# Basic helpers
# ============================================================

def _save_debug(debug_dir: Optional[str], name: str, image: np.ndarray) -> None:
    if debug_dir is None:
        return

    out_dir = Path(debug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{name}.png"), image)


def _load_ignore_mask(mask_path: Optional[str], shape) -> Optional[np.ndarray]:
    """
    Optional manual cleanup mask.

    White pixels indicate areas that must be removed.
    Black pixels are preserved.
    """
    if mask_path is None:
        return None

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Ignore mask not found: {mask_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"OpenCV could not read ignore mask: {mask_path}")

    mask = cv2.resize(
        mask,
        (shape[1], shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


# ============================================================
# Shadow removal
# ============================================================

def _remove_soft_shadows(gray: np.ndarray) -> np.ndarray:
    """
    Remove gradual shadows and uneven lighting.

    The blurred image estimates the paper illumination. Dividing the original
    image by this background suppresses broad shadows while preserving pencil
    lines.
    """
    gray = np.asarray(gray, dtype=np.uint8)

    height, width = gray.shape[:2]
    min_side = min(height, width)

    sigma = max(18.0, min(85.0, min_side / 14.0))

    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    background = np.maximum(background, 1).astype(np.uint8)

    corrected = cv2.divide(
        gray,
        background,
        scale=255,
    )

    # Mild percentile normalization.
    low = float(np.percentile(corrected, 1.0))
    high = float(np.percentile(corrected, 99.7))

    if high - low > 1e-6:
        corrected = np.clip(
            (corrected.astype(np.float32) - low)
            * (255.0 / (high - low)),
            0,
            255,
        ).astype(np.uint8)

    return corrected


# ============================================================
# Connected-component helpers
# ============================================================

def _touches_image_edge(x, y, w, h, image_width, image_height, margin) -> bool:
    return (
        x <= margin
        or y <= margin
        or (x + w) >= (image_width - margin)
        or (y + h) >= (image_height - margin)
    )


def _component_score(stats_row, image_shape, edge_margin: int) -> float:
    """
    Score a connected component as a possible main organ sketch.

    Prefer large line-like components. Penalize components that touch the
    photograph border because they are often shadow boundaries or frame edges.
    """
    height, width = image_shape[:2]

    x = int(stats_row[cv2.CC_STAT_LEFT])
    y = int(stats_row[cv2.CC_STAT_TOP])
    w = int(stats_row[cv2.CC_STAT_WIDTH])
    h = int(stats_row[cv2.CC_STAT_HEIGHT])
    area = int(stats_row[cv2.CC_STAT_AREA])

    bbox_area = max(w * h, 1)
    fill_ratio = area / bbox_area

    score = float(area)

    if _touches_image_edge(
        x, y, w, h,
        image_width=width,
        image_height=height,
        margin=edge_margin,
    ):
        score *= 0.10

    # Broad dense components are more likely to be shadows than pencil lines.
    if fill_ratio > 0.72:
        score *= 0.25

    return score


def _find_main_component(
    binary: np.ndarray,
    edge_margin_ratio: float,
):
    """
    Find the connected component most likely to contain the organ outline.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if num_labels <= 1:
        return None, labels, stats

    height, width = binary.shape[:2]
    edge_margin = max(2, int(round(max(height, width) * edge_margin_ratio)))

    best_label = None
    best_score = -1.0

    for label in range(1, num_labels):
        score = _component_score(
            stats[label],
            binary.shape,
            edge_margin=edge_margin,
        )

        if score > best_score:
            best_score = score
            best_label = label

    return best_label, labels, stats


def _remove_small_components(
    binary: np.ndarray,
    min_area: int,
    keep_main: bool = True,
    edge_margin_ratio: float = 0.015,
) -> np.ndarray:
    """
    Remove tiny components and reject secondary border-touching artifacts.
    """
    binary = np.asarray(binary, dtype=np.uint8)

    main_label, labels, stats = _find_main_component(
        binary,
        edge_margin_ratio=edge_margin_ratio,
    )

    if main_label is None:
        return np.zeros_like(binary)

    height, width = binary.shape[:2]
    edge_margin = max(2, int(round(max(height, width) * edge_margin_ratio)))

    output = np.zeros_like(binary)

    for label in range(1, len(stats)):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        is_main = label == main_label

        if keep_main and is_main:
            output[labels == label] = 255
            continue

        if area < int(min_area):
            continue

        # This directly removes the curved shadow fragments visible near the
        # lower border in photographed sketches.
        if _touches_image_edge(
            x, y, w, h,
            image_width=width,
            image_height=height,
            margin=edge_margin,
        ):
            continue

        output[labels == label] = 255

    return output


def _retain_main_sketch_roi(
    binary: np.ndarray,
    padding_ratio: float,
    edge_margin_ratio: float,
) -> np.ndarray:
    """
    Keep the main sketch and nearby internal strokes, but remove disconnected
    artifacts outside the main organ's bounding region.

    This is the key fix for sharp shadow boundaries that remain line-like
    after thresholding.
    """
    binary = np.asarray(binary, dtype=np.uint8)

    main_label, labels, stats = _find_main_component(
        binary,
        edge_margin_ratio=edge_margin_ratio,
    )

    if main_label is None:
        return np.zeros_like(binary)

    height, width = binary.shape[:2]

    x = int(stats[main_label, cv2.CC_STAT_LEFT])
    y = int(stats[main_label, cv2.CC_STAT_TOP])
    w = int(stats[main_label, cv2.CC_STAT_WIDTH])
    h = int(stats[main_label, cv2.CC_STAT_HEIGHT])

    padding = int(round(max(width, height) * float(padding_ratio)))

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)

    roi_mask = np.zeros_like(binary)
    roi_mask[y1:y2, x1:x2] = 255

    inside_roi = cv2.bitwise_and(binary, roi_mask)

    # Remove remaining tiny fragments while retaining internal organ lines.
    return _remove_small_components(
        inside_roi,
        min_area=18,
        keep_main=True,
        edge_margin_ratio=edge_margin_ratio,
    )



def _looks_like_external_horizontal_shadow_strip(
    x: int,
    y: int,
    w: int,
    h: int,
    image_width: int,
    image_height: int,
    main_bbox=None,
) -> bool:
    """
    Detect a broad, thin, nearly horizontal component close to the lower
    portion of the photograph.

    Why this is needed:
        A hard shadow boundary can survive illumination normalization because
        it is a genuine high-contrast edge. Small-component filtering also
        keeps it when it is long enough. Such a strip is not part of the organ
        sketch and should be removed only when it is a secondary component.

    The rule is intentionally conservative so that valid organ outlines are
    not deleted.
    """
    if w <= 0 or h <= 0:
        return False

    aspect_ratio = float(w) / float(max(h, 1))

    broad = w >= max(45, int(round(image_width * 0.20)))
    thin = h <= max(18, int(round(image_height * 0.10)))
    horizontal = aspect_ratio >= 3.5

    # Shadow edges in photographed notebook images often appear in the lower
    # band. A component does not have to touch the exact image border.
    in_lower_band = (
        y >= int(round(image_height * 0.66))
        or (y + h) >= int(round(image_height * 0.84))
    )

    if not (broad and thin and horizontal and in_lower_band):
        return False

    if main_bbox is None:
        return True

    main_x, main_y, main_w, main_h = main_bbox
    main_bottom = main_y + main_h

    overlap_x = max(
        0,
        min(x + w, main_x + main_w) - max(x, main_x),
    )

    overlap_y = max(
        0,
        min(y + h, main_y + main_h) - max(y, main_y),
    )

    overlap_area = overlap_x * overlap_y
    component_bbox_area = max(w * h, 1)
    overlap_ratio = overlap_area / component_bbox_area

    mostly_outside_main = overlap_ratio < 0.12
    near_or_below_main_bottom = y >= (main_bottom - max(4, int(round(main_h * 0.08))))

    return mostly_outside_main and near_or_below_main_bottom


def _remove_external_horizontal_shadow_strips(
    binary: np.ndarray,
    edge_margin_ratio: float = 0.015,
) -> np.ndarray:
    """
    Remove disconnected lower horizontal shadow boundaries while keeping the
    main organ sketch unchanged.

    This targets the failure case visible in photographed sketches where a
    strong shadow creates a long white strip below the organ after adaptive
    thresholding.
    """
    binary = np.asarray(binary, dtype=np.uint8)

    main_label, labels, stats = _find_main_component(
        binary,
        edge_margin_ratio=edge_margin_ratio,
    )

    if main_label is None:
        return binary.copy()

    height, width = binary.shape[:2]

    main_x = int(stats[main_label, cv2.CC_STAT_LEFT])
    main_y = int(stats[main_label, cv2.CC_STAT_TOP])
    main_w = int(stats[main_label, cv2.CC_STAT_WIDTH])
    main_h = int(stats[main_label, cv2.CC_STAT_HEIGHT])

    main_bbox = (main_x, main_y, main_w, main_h)

    output = binary.copy()

    for label in range(1, len(stats)):
        if label == main_label:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        if _looks_like_external_horizontal_shadow_strip(
            x=x,
            y=y,
            w=w,
            h=h,
            image_width=width,
            image_height=height,
            main_bbox=main_bbox,
        ):
            output[labels == label] = 0

    return output



# ============================================================
# Main preprocessing entry point
# ============================================================

def preprocess_sketch(
    image_path: str,
    cleanup_strength: str = "balanced",
    manual_ignore_mask_path: Optional[str] = None,
    debug_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Convert a photographed pencil sketch into a clean binary line image.

    Recommended:
        cleanup_strength="balanced"

    Use:
        cleanup_strength="strict"
    when shadow fragments or erased marks remain.

    Use:
        cleanup_strength="gentle"
    if valid faint lines disappear.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    if cleanup_strength not in _CLEANUP_PROFILES:
        raise ValueError(
            f"Unknown cleanup_strength='{cleanup_strength}'. "
            f"Expected one of: {list(_CLEANUP_PROFILES.keys())}"
        )

    cfg = _CLEANUP_PROFILES[cleanup_strength]

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    # --------------------------------------------------------
    # 1. Remove broad shadows / uneven lighting
    # --------------------------------------------------------
    corrected = _remove_soft_shadows(gray)
    _save_debug(debug_dir, "01_shadow_corrected", corrected)

    # Mild local contrast enhancement.
    clahe = cv2.createCLAHE(
        clipLimit=1.45,
        tileGridSize=(8, 8),
    )

    contrast = clahe.apply(corrected)

    # Median blur removes isolated camera noise while preserving line edges.
    contrast = cv2.medianBlur(contrast, 3)

    # Gaussian blur stabilizes adaptive thresholding.
    blur = cv2.GaussianBlur(contrast, (5, 5), 0)

    _save_debug(debug_dir, "02_contrast", contrast)
    _save_debug(debug_dir, "03_blur", blur)

    # --------------------------------------------------------
    # 2. Extract pencil lines
    # --------------------------------------------------------
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        float(cfg["adaptive_c"]),
    )

    _save_debug(debug_dir, "04_threshold", binary)

    # --------------------------------------------------------
    # 3. Remove tiny dots BEFORE dilation enlarges them
    # --------------------------------------------------------
    binary = _remove_small_components(
        binary,
        min_area=int(cfg["first_min_area"]),
        keep_main=True,
        edge_margin_ratio=float(cfg["edge_margin_ratio"]),
    )

    _save_debug(debug_dir, "05_early_cleanup", binary)

    # --------------------------------------------------------
    # 4. Repair small gaps and slightly strengthen outlines
    # --------------------------------------------------------
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    # One opening pass removes isolated speckle noise.
    opened = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    # Two closing passes reconnect small breaks in pencil contours.
    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    strengthened = cv2.dilate(
        closed,
        kernel,
        iterations=1,
    )

    _save_debug(debug_dir, "06_strengthened", strengthened)

    # --------------------------------------------------------
    # 5. Remove disconnected shadow edges outside the organ ROI
    # --------------------------------------------------------
    cleaned = _retain_main_sketch_roi(
        strengthened,
        padding_ratio=float(cfg["roi_padding_ratio"]),
        edge_margin_ratio=float(cfg["edge_margin_ratio"]),
    )

    # Remove hard lower shadow boundaries that remain after soft-shadow
    # normalization. They are often too large for ordinary dot filtering.
    cleaned = _remove_external_horizontal_shadow_strips(
        cleaned,
        edge_margin_ratio=float(cfg["edge_margin_ratio"]),
    )

    cleaned = _remove_small_components(
        cleaned,
        min_area=int(cfg["final_min_area"]),
        keep_main=True,
        edge_margin_ratio=float(cfg["edge_margin_ratio"]),
    )

    # --------------------------------------------------------
    # 6. Optional manual fallback for truly ambiguous artifacts
    # --------------------------------------------------------
    ignore_mask = _load_ignore_mask(
        manual_ignore_mask_path,
        cleaned.shape,
    )

    if ignore_mask is not None:
        cleaned[ignore_mask > 0] = 0

    _save_debug(debug_dir, "07_final_binary", cleaned)

    return cleaned


# ============================================================
# Existing pipeline-compatible contour helpers
# ============================================================

def extract_contours(binary_img, min_area=2000):
    """
    Extract meaningful external contours.

    In addition to small-area and image-edge rejection, this function excludes
    broad lower horizontal shadow strips. This acts as a final safety layer for
    deformation and for the extracted-border debug panel.
    """
    binary_img = np.asarray(binary_img, dtype=np.uint8)

    contours, _ = cv2.findContours(
        binary_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    height, width = binary_img.shape[:2]
    margin = max(2, int(round(max(height, width) * 0.01)))

    area_candidates = [
        contour
        for contour in contours
        if cv2.contourArea(contour) > min_area
    ]

    # Select a plausible organ reference contour while ignoring obvious
    # horizontal shadow strips.
    reference_contour = None
    reference_area = -1.0

    for contour in area_candidates:
        x, y, w, h = cv2.boundingRect(contour)

        if _looks_like_external_horizontal_shadow_strip(
            x=x,
            y=y,
            w=w,
            h=h,
            image_width=width,
            image_height=height,
            main_bbox=None,
        ):
            continue

        area = float(cv2.contourArea(contour))

        if area > reference_area:
            reference_contour = contour
            reference_area = area

    reference_bbox = (
        cv2.boundingRect(reference_contour)
        if reference_contour is not None
        else None
    )

    valid = []

    for contour in area_candidates:
        x, y, w, h = cv2.boundingRect(contour)

        if _looks_like_external_horizontal_shadow_strip(
            x=x,
            y=y,
            w=w,
            h=h,
            image_width=width,
            image_height=height,
            main_bbox=reference_bbox,
        ):
            continue

        touches_edge = _touches_image_edge(
            x,
            y,
            w,
            h,
            image_width=width,
            image_height=height,
            margin=margin,
        )

        if touches_edge:
            continue

        valid.append(contour)

    # Conservative fallback for tightly cropped valid uploads.
    if not valid:
        valid = [
            contour
            for contour in area_candidates
            if not _looks_like_external_horizontal_shadow_strip(
                x=cv2.boundingRect(contour)[0],
                y=cv2.boundingRect(contour)[1],
                w=cv2.boundingRect(contour)[2],
                h=cv2.boundingRect(contour)[3],
                image_width=width,
                image_height=height,
                main_bbox=reference_bbox,
            )
        ]

    valid.sort(
        key=cv2.contourArea,
        reverse=True,
    )

    return {"outer": valid}

def optimize_vectors(
    contours_dict,
    initial_epsilon_factor=0.002,
):
    """
    Reduce contour point count while preserving overall organ shape.
    """
    optimized = {"outer": []}

    for cnt in contours_dict.get("outer", []):
        perimeter = cv2.arcLength(cnt, closed=True)

        epsilon = initial_epsilon_factor * perimeter

        approx = cv2.approxPolyDP(
            cnt,
            epsilon,
            closed=True,
        )

        if len(approx) >= 3:
            optimized["outer"].append(approx)
        else:
            optimized["outer"].append(cnt)

    return optimized


def normalize_coordinates(
    contours_dict,
    img_width=None,
    img_height=None,
):
    """
    Normalize sketch contour coordinates to approximately [-1, 1].
    """
    normalized = {"outer": []}

    contours = contours_dict.get("outer", [])

    if not contours:
        return normalized

    all_points = np.vstack(contours)

    x, y, w, h = cv2.boundingRect(all_points)

    cx = x + w / 2.0
    cy = y + h / 2.0
    scale = max(w, h) / 2.0

    if scale <= 1e-8:
        raise ValueError("Sketch contour has an invalid bounding box.")

    for cnt in contours:
        pts = cnt.reshape(-1, 2).astype(np.float64)

        out = np.empty_like(
            pts,
            dtype=np.float64,
        )

        out[:, 0] = (pts[:, 0] - cx) / scale
        out[:, 1] = -((pts[:, 1] - cy) / scale)

        normalized["outer"].append(
            out.reshape(-1, 1, 2).astype(np.float32)
        )

    return normalized


def get_primary_contour_points(normalized_dict):
    """
    Return the largest external organ contour for deformation.
    """
    if not normalized_dict.get("outer"):
        raise ValueError(
            "No outer sketch contour was found. "
            "Try a clearer image or cleanup_strength='gentle'."
        )

    cnt = max(
        normalized_dict["outer"],
        key=lambda contour: abs(
            cv2.contourArea(
                contour.astype(np.float32)
            )
        ),
    )

    return cnt.reshape(-1, 2).astype(np.float64)
