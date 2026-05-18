import cv2
import numpy as np
import os


def preprocess_sketch(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    # Normalize uneven paper lighting.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    blur = cv2.GaussianBlur(contrast, (5, 5), 0)

    # Pencil lines become foreground.
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 8
    )

    # Pencil outlines are thin and often have gaps. Dilate/close/fill so the
    # external contour describes the organ silhouette, not both sides of a stroke.
    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.dilate(clean, kernel, iterations=1)

# Remove tiny noise components.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)

    filtered = np.zeros_like(clean)

    min_component_area = 80

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_component_area:
            filtered[labels == label] = 255

    return filtered


def extract_contours(binary_img, min_area=2000):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    contours.sort(key=cv2.contourArea, reverse=True)
    return {"outer": contours}


def optimize_vectors(contours_dict, initial_epsilon_factor=0.002):
    optimized = {"outer": []}
    for cnt in contours_dict.get("outer", []):
        perimeter = cv2.arcLength(cnt, closed=True)
        epsilon = initial_epsilon_factor * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        if len(approx) >= 3:
            optimized["outer"].append(approx)
        else:
            optimized["outer"].append(cnt)
    return optimized


def normalize_coordinates(contours_dict, img_width=None, img_height=None):
    normalized = {"outer": []}
    contours = contours_dict.get("outer", [])
    if not contours:
        return normalized

    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    cx = x + w / 2.0
    cy = y + h / 2.0
    scale = max(w, h) / 2.0

    for cnt in contours:
        pts = cnt.reshape(-1, 2).astype(np.float64)
        out = np.empty_like(pts, dtype=np.float64)
        out[:, 0] = (pts[:, 0] - cx) / scale
        out[:, 1] = -((pts[:, 1] - cy) / scale)
        normalized["outer"].append(out.reshape(-1, 1, 2).astype(np.float32))
    return normalized


def get_primary_contour_points(normalized_dict):
    if not normalized_dict.get("outer"):
        raise ValueError("No outer sketch contour was found")
    cnt = max(normalized_dict["outer"], key=lambda c: abs(cv2.contourArea(c.astype(np.float32))))
    return cnt.reshape(-1, 2).astype(np.float64)
