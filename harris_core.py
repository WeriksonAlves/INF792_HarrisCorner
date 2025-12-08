"""
harris_core.py

Core functions and helpers for Harris corner detection demo (INF792).

This module provides:
- image loading and preprocessing
- gradient computation
- structure tensor components
- Harris response computation
- corner detection
- overlay and visualization helpers
"""

from __future__ import annotations

from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


# =========================
# Visualization helpers
# =========================
def show_image(title: str, image: np.ndarray, cmap: str | None = None) -> None:
    """Show a single image with a given title."""
    plt.figure()
    if cmap is not None:
        plt.imshow(image, cmap=cmap)
    else:
        if image.ndim == 2:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_side_by_side(
    title_left: str,
    img_left: np.ndarray,
    title_right: str,
    img_right: np.ndarray,
    cmap_left: str | None = None,
    cmap_right: str | None = None,
) -> None:
    """Show two images side by side for comparison."""
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img_left, cmap=cmap_left)
    plt.title(title_left)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_right, cmap=cmap_right)
    plt.title(title_right)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# =========================
# Core Harris pipeline
# =========================
def load_and_preprocess_image(
    path: str,
    sigma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a BGR image with OpenCV, convert to grayscale and apply Gaussian blur.

    Args:
        path: path to the input image.
        sigma: standard deviation of the Gaussian blur.

    Returns:
        image_rgb: original image in RGB format (for visualization).
        image_blur: grayscale blurred image in [0, 1].
    """
    image_bgr = cv2.imread(path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_gray = image_gray.astype(np.float32) / 255.0

    image_blur = cv2.GaussianBlur(
        image_gray,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    return image_rgb, image_blur


def compute_image_gradients(
    image: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute image gradients Ix and Iy using Sobel filters.

    Args:
        image: grayscale image in [0, 1].

    Returns:
        ix: horizontal gradient.
        iy: vertical gradient.
    """
    ix = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return ix, iy


def compute_structure_tensor(
    ix: np.ndarray,
    iy: np.ndarray,
    window_sigma: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute smoothed structure tensor components from image gradients.

    Args:
        ix: horizontal gradient.
        iy: vertical gradient.
        window_sigma: standard deviation for Gaussian smoothing.

    Returns:
        s_ix2: smoothed Ix^2.
        s_iy2: smoothed Iy^2.
        s_ixy: smoothed Ix * Iy.
    """
    ix2 = ix * ix
    iy2 = iy * iy
    ixy = ix * iy

    s_ix2 = cv2.GaussianBlur(
        ix2, (0, 0), sigmaX=window_sigma, sigmaY=window_sigma
    )
    s_iy2 = cv2.GaussianBlur(
        iy2, (0, 0), sigmaX=window_sigma, sigmaY=window_sigma
    )
    s_ixy = cv2.GaussianBlur(
        ixy, (0, 0), sigmaX=window_sigma, sigmaY=window_sigma
    )

    return s_ix2, s_iy2, s_ixy


def compute_harris_response(
    s_ix2: np.ndarray,
    s_iy2: np.ndarray,
    s_ixy: np.ndarray,
    k: float = 0.04,
) -> np.ndarray:
    """
    Compute Harris response R for each pixel.

    Args:
        s_ix2: smoothed Ix^2.
        s_iy2: smoothed Iy^2.
        s_ixy: smoothed Ix * Iy.
        k: Harris parameter (typically in [0.04, 0.06]).

    Returns:
        r: Harris response map.
    """
    det_m = s_ix2 * s_iy2 - s_ixy * s_ixy
    trace_m = s_ix2 + s_iy2
    r = det_m - k * (trace_m**2)
    return r


def detect_corners(
    r: np.ndarray,
    threshold_ratio: float = 0.01,
) -> np.ndarray:
    """
    Detect corners using thresholding and non-maximum suppression.

    Args:
        r: Harris response map.
        threshold_ratio: ratio of max(R_normalized) used as threshold.

    Returns:
        final_mask: binary mask where 1 indicates a detected corner.
    """
    r_norm = cv2.normalize(r, None, 0, 1, cv2.NORM_MINMAX)
    threshold = threshold_ratio * r_norm.max()

    corner_mask = (r_norm > threshold).astype(np.uint8)

    r_dilated = cv2.dilate(r_norm, np.ones((3, 3), np.uint8))
    nms_mask = (r_norm == r_dilated).astype(np.uint8)

    final_mask = corner_mask * nms_mask
    return final_mask


def overlay_corners_on_image(
    image_rgb: np.ndarray,
    corner_mask: np.ndarray,
    marker_color: tuple[int, int, int] = (255, 0, 0),
    marker_size: int = 3,
) -> np.ndarray:
    """
    Overlay detected corners on top of the original RGB image.

    Args:
        image_rgb: original RGB image.
        corner_mask: binary mask with 1 at corner locations.
        marker_color: color of the markers (RGB).
        marker_size: radius of the marker.

    Returns:
        overlay: image copy with corner markers drawn.
    """
    overlay = image_rgb.copy()
    ys, xs = np.where(corner_mask > 0)

    for x, y in zip(xs, ys):
        cv2.circle(
            overlay, (int(x), int(y)), marker_size, marker_color, thickness=3
        )

    return overlay


# =========================
# Visualization of internals
# =========================
def show_gradients(
    gray: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
) -> None:
    """Show grayscale image, Ix, Iy and gradient magnitude."""
    magnitude = np.sqrt(ix**2 + iy**2)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale (blurred)")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(ix, cmap="gray")
    plt.title("Ix (Sobel x)")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(iy, cmap="gray")
    plt.title("Iy (Sobel y)")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(magnitude, cmap="gray")
    plt.title("|∇I| (magnitude)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_structure_tensor_components(
    s_ix2: np.ndarray,
    s_iy2: np.ndarray,
    s_ixy: np.ndarray,
) -> None:
    """Visualize the smoothed structure tensor components."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(s_ix2, cmap="gray")
    plt.title("Smoothed Ix^2")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(s_iy2, cmap="gray")
    plt.title("Smoothed Iy^2")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(s_ixy, cmap="gray")
    plt.title("Smoothed Ix * Iy")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_harris_response(r: np.ndarray) -> None:
    """Visualize the Harris response as a heatmap."""
    r_norm = cv2.normalize(r, None, 0, 1, cv2.NORM_MINMAX)

    plt.figure(figsize=(6, 5))
    plt.imshow(r_norm, cmap="jet")
    plt.title("Harris response R (normalized)")
    plt.colorbar(fraction=0.046)
    plt.axis("off")
    plt.show()
