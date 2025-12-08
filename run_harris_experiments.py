"""
run_harris_experiments.py

Run Harris corner detection experiments varying:
- threshold_ratio
- k parameter
- Gaussian sigma of the structure tensor

This script is meant to show only the final results (figures),
without step-by-step explanations.
"""

from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from harris_core import (
    load_and_preprocess_image,
    compute_image_gradients,
    compute_structure_tensor,
    compute_harris_response,
    detect_corners,
    overlay_corners_on_image,
)


def ensure_output_dir(path: str = "outputs") -> str:
    """Create an output directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
    return path


def run_threshold_experiments(
    image_rgb: np.ndarray,
    gray_blur: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    s_ix2: np.ndarray,
    s_iy2: np.ndarray,
    s_ixy: np.ndarray,
    output_dir: str,
) -> None:
    """Experiment: effect of different threshold ratios."""
    threshold_values: List[float] = [0.05, 0.1, 0.14, 0.20]

    r = compute_harris_response(s_ix2, s_iy2, s_ixy, k=0.02)

    plt.figure(figsize=(12, 8))
    for idx, thr in enumerate(threshold_values, start=1):
        corner_mask = detect_corners(r, threshold_ratio=thr)
        overlay = overlay_corners_on_image(
            image_rgb, corner_mask, marker_color=(255, 0, 0), marker_size=3
        )

        plt.subplot(2, 2, idx)
        plt.imshow(overlay)
        plt.title(f"threshold_ratio = {thr}")
        plt.axis("off")

    plt.suptitle("Effect of threshold_ratio on Harris corners", fontsize=14)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "experiment_thresholds.png")
    plt.savefig(filepath, dpi=200)
    plt.show()


def run_k_experiments(
    image_rgb: np.ndarray,
    gray_blur: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    s_ix2: np.ndarray,
    s_iy2: np.ndarray,
    s_ixy: np.ndarray,
    output_dir: str,
) -> None:
    """Experiment: effect of different k values."""
    k_values: List[float] = [0.01, 0.02, 0.04, 0.08]

    plt.figure(figsize=(12, 8))
    for idx, k_val in enumerate(k_values, start=1):
        r_k = compute_harris_response(s_ix2, s_iy2, s_ixy, k=k_val)
        corner_mask = detect_corners(r_k, threshold_ratio=0.14)
        overlay = overlay_corners_on_image(
            image_rgb, corner_mask, marker_color=(255, 0, 0), marker_size=3
        )

        plt.subplot(2, 2, idx)
        plt.imshow(overlay)
        plt.title(f"k = {k_val}")
        plt.axis("off")

    plt.suptitle("Effect of k on Harris corners", fontsize=14)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "experiment_k_values.png")
    plt.savefig(filepath, dpi=200)
    plt.show()


def run_sigma_experiments(
    image_rgb: np.ndarray,
    gray_blur: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    output_dir: str,
) -> None:
    """Experiment: effect of different Gaussian sigmas."""
    sigma_values: List[float] = [0.5, 1.0, 2.0, 3.0]

    plt.figure(figsize=(12, 8))
    for idx, sigma_val in enumerate(sigma_values, start=1):
        s_ix2, s_iy2, s_ixy = compute_structure_tensor(
            ix, iy, window_sigma=sigma_val
        )
        r_sigma = compute_harris_response(s_ix2, s_iy2, s_ixy, k=0.02)
        corner_mask = detect_corners(r_sigma, threshold_ratio=0.14)
        overlay = overlay_corners_on_image(
            image_rgb, corner_mask, marker_color=(255, 0, 0), marker_size=1
        )

        plt.subplot(2, 2, idx)
        plt.imshow(overlay)
        plt.title(f"window_sigma = {sigma_val}")
        plt.axis("off")

    plt.suptitle(
        "Effect of Gaussian sigma (structure tensor) on Harris corners",
        fontsize=14,
    )
    plt.tight_layout()

    filepath = os.path.join(output_dir, "experiment_sigmas.png")
    plt.savefig(filepath, dpi=200)
    plt.show()


def main(img_name: str) -> None:
    """Run all Harris experiments on a single test image."""
    output_dir = ensure_output_dir(os.path.join("outputs", img_name))

    image_path = os.path.join("images", img_name)
    image_rgb, gray_blur = load_and_preprocess_image(image_path, sigma=0.50)

    ix, iy = compute_image_gradients(gray_blur)
    s_ix2, s_iy2, s_ixy = compute_structure_tensor(ix, iy, window_sigma=1)

    run_threshold_experiments(
        image_rgb, gray_blur, ix, iy, s_ix2, s_iy2, s_ixy, output_dir
    )
    run_k_experiments(
        image_rgb, gray_blur, ix, iy, s_ix2, s_iy2, s_ixy, output_dir
    )
    run_sigma_experiments(image_rgb, gray_blur, ix, iy, output_dir)


if __name__ == "__main__":
    main(img_name="img_validation.png")
