"""
src/visualize.py

Visualization functions:
- Original vs Compressed comparison
- Error vs k graph
"""

import numpy as np
import matplotlib.pyplot as plt

from src.compress import compress_grayscale, reconstruction_error


def show_comparison(original: np.ndarray, compressed: np.ndarray, k: int):
    """
    Display original vs compressed image

    Args:
        original: Original image
        compressed: Compressed image
        k: compression level
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original, cmap='gray' if original.ndim == 2 else None)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Compressed (k={k})")
    plt.imshow(compressed, cmap='gray' if compressed.ndim == 2 else None)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_error_vs_k(gray_image: np.ndarray, ks: list[int]):
    """
    Plot reconstruction error vs k

    Args:
        gray_image: Grayscale image
        ks: list of k values
    """
    errors = []

    for k in ks:
        compressed, _, _, _ = compress_grayscale(gray_image, k)
        error = reconstruction_error(gray_image, compressed)
        errors.append(error)

    plt.figure(figsize=(6, 4))
    plt.plot(ks, errors, marker='o')
    plt.xlabel("k (Number of Singular Values)")
    plt.ylabel("Reconstruction Error")
    plt.title("Error vs k")
    plt.grid(True)
    plt.show()


def plot_multiple_compressions(gray_image: np.ndarray, ks: list[int]):
    """
    Show multiple compressed outputs for different k values

    Args:
        gray_image: Grayscale image
        ks: list of k values
    """
    n = len(ks)

    plt.figure(figsize=(15, 3))

    for i, k in enumerate(ks):
        compressed, _, _, _ = compress_grayscale(gray_image, k)

        plt.subplot(1, n, i + 1)
        plt.title(f"k={k}")
        plt.imshow(compressed, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()