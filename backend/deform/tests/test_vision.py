from __future__ import annotations

import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.vision.sketch_parser import preprocess_sketch, extract_contours, optimize_vectors, normalize_coordinates


def run_test():
    sketches_dir = "assets/sketches"
    if not os.path.exists(sketches_dir):
        print(f"ERROR: Directory {sketches_dir} does not exist.")
        return

    available_files = [f for f in os.listdir(sketches_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    if not available_files:
        return

    print("\n--- BIOSKETCH DEFORM TEST MENU ---")
    for i, filename in enumerate(available_files):
        print(f"[{i + 1}] {filename}")

    try:
        choice = int(input("\nEnter the number of the image to test: ")) - 1
        selected_filename = available_files[choice]
    except Exception:
        return

    test_image_path = os.path.join(sketches_dir, selected_filename)

    print("\nRunning Cleaned Phase 1 Pipeline (Silhouette Only).")

    try:
        processed_img = preprocess_sketch(test_image_path)
        img_height, img_width = processed_img.shape
        contours_dict = extract_contours(processed_img)

        raw_pts = sum(len(c) for c in contours_dict["outer"])
        print(f"\n[DATA] Original Raw Coordinates : {raw_pts} points")

        optimized_dict = optimize_vectors(contours_dict, initial_epsilon_factor=0.002)
        opt_pts = sum(len(c) for c in optimized_dict["outer"])
        print(f"[DATA] Optimized Anchor Points  : {opt_pts} points")

        normalized_dict = normalize_coordinates(optimized_dict, img_width, img_height)

        if normalized_dict["outer"]:
            sample_pt = normalized_dict["outer"][0][0][0]
            print(f"[DATA] Sample 2D Coordinate     : X={sample_pt[0]:.3f}, Y={sample_pt[1]:.3f}")

        original_img = cv2.imread(test_image_path)
        canvas = np.zeros_like(original_img)

        cv2.drawContours(canvas, optimized_dict["outer"], -1, (0, 255, 0), 2)

        for cnt in optimized_dict["outer"]:
            for pt in cnt:
                cv2.circle(canvas, (int(pt[0][0]), int(pt[0][1])), 6, (255, 255, 0), -1)

        display_height = 600
        scale = display_height / original_img.shape[0]
        dim = (int(original_img.shape[1] * scale), display_height)

        cv2.imshow("1. Original Image", cv2.resize(original_img, dim))
        cv2.imshow("3. Optimized Vector (Outer Only)", cv2.resize(canvas, dim))

        print("\nSuccess! Press any key to close.")
        while True:
            if cv2.waitKey(1) & 0xFF != 255:
                break
            if cv2.getWindowProperty("1. Original Image", cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    run_test()
