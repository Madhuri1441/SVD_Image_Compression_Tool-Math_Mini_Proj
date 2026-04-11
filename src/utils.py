"""
src/utils.py

Utility functions for:
- Loading images
- Converting color spaces
- Saving images
"""

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk and convert BGR -> RGB

    Args:
        path: Path to image file

    Returns:
        RGB image
    """
    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError(f"Could not load image at {path}")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale

    Args:
        image: RGB image

    Returns:
        Grayscale image
    """
    if image.ndim != 3:
        raise ValueError("Expected RGB image")

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def save_image(path: str, image: np.ndarray):
    """
    Save image to disk (RGB -> BGR for OpenCV)

    Args:
        path: Output path
        image: RGB image
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(path, image)


def resize_image(image: np.ndarray, max_size: int = 512) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio (for faster SVD)

    Args:
        image: Input image
        max_size: Max dimension

    Returns:
        Resized image
    """
    h, w = image.shape[:2]

    if max(h, w) <= max_size:
        return image

    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h))


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0,1]

    Useful for error calculations

    Args:
        image: Input image

    Returns:
        Normalized image
    """
    return image.astype(np.float64) / 255.0