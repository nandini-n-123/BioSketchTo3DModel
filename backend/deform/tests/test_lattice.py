from __future__ import annotations

import os
import sys
import numpy as np
import trimesh

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.geometry.mesh_prep import load_and_normalize_mesh
from app.deformation.lattice import LatticeFFD
from app.deformation.cage_weights import calculate_weight_matrix


def run_test():
    models_dir = "assets/3d_models"
    if not os.path.exists(models_dir):
        print(f"ERROR: Directory {models_dir} does not exist.")
        return

    available_files = [f for f in os.listdir(models_dir) if f.endswith((".glb", ".gltf", ".obj"))]
    if not available_files:
        print("No 3D models found.")
        return

    print("\n--- BIOSKETCH CAGE GENERATOR TEST ---")
    for i, filename in enumerate(available_files):
        print(f"[{i + 1}] {filename}")

    try:
        choice = int(input("\nEnter the number of the model to load: ")) - 1
        selected_filename = available_files[choice]
    except Exception:
        return

    model_path = os.path.join(models_dir, selected_filename)

    print("\n1. Loading and Normalizing Geometry (Phase 2)...")
    clean_scene = load_and_normalize_mesh(model_path)
    original_vertices = np.vstack([geom.vertices for geom in clean_scene.geometry.values()])

    print("2. Generating adaptive lattice cage (Phase 3)...")
    cage = LatticeFFD.from_vertices(original_vertices, resolution=(4, 4, 4), padding_ratio=0.18)
    cage_points = cage.get_points()

    print("\n3. Executing Phase 4 binding math.")
    W_matrix = calculate_weight_matrix(original_vertices, cage.min_bounds, cage.max_bounds, cage.resolution)
    reconstructed_vertices = W_matrix @ cage_points

    max_error = np.max(np.abs(original_vertices - reconstructed_vertices))
    print(f"[DATA] Math Reconstruction Error: {max_error:.8f}")

    if max_error < 0.0001:
        print("[DATA] STATUS: MATRICES LOCKED. BINDING 100% SUCCESSFUL.")
    else:
        print("[DATA] STATUS: FAILURE. MATRIX CORRUPTED.")

    print(f"\n[DATA] Cage Generated    : {len(cage_points)} Control Points")
    print(f"[DATA] Lattice Min Bounds: X={np.min(cage_points[:,0]):.3f}, Y={np.min(cage_points[:,1]):.3f}, Z={np.min(cage_points[:,2]):.3f}")
    print(f"[DATA] Lattice Max Bounds: X={np.max(cage_points[:,0]):.3f}, Y={np.max(cage_points[:,1]):.3f}, Z={np.max(cage_points[:,2]):.3f}")

    for geometry in clean_scene.geometry.values():
        geometry.visual.face_colors = [150, 150, 150, 100]

    point_cloud = trimesh.points.PointCloud(cage_points, colors=[255, 0, 0, 255])
    test_scene = trimesh.Scene([clean_scene, point_cloud])
    test_scene.show(background=[40, 40, 40, 255])


if __name__ == "__main__":
    run_test()
