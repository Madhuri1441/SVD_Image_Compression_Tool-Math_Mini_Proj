"""
src/compress.py

Core SVD compression logic for the SVD Image Compression Tool.
Includes:
- SVD decomposition
- Grayscale image compression
- RGB image compression
- Compression ratio calculation
"""

from __future__ import annotations

import numpy as np


def compute_svd(matrix: np.ndarray):
    """
    Compute SVD of a 2D matrix.

    Args:
        matrix: 2D numpy array

    Returns:
        U, S, Vt
    """
    if matrix.ndim != 2:
        raise ValueError("compute_svd expects a 2D matrix.")

    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U, S, Vt


def reconstruct_from_svd(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    """
    Reconstruct matrix using top-k singular values.

    Args:
        U: Left singular vectors
        S: Singular values
        Vt: Right singular vectors
        k: Number of singular values to keep

    Returns:
        Reconstructed matrix
    """
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    max_rank = min(U.shape[1], len(S), Vt.shape[0])
    k = min(k, max_rank)

    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]

    return U_k @ S_k @ Vt_k


def compress_grayscale(gray_image: np.ndarray, k: int):
    """
    Compress a grayscale image using SVD.

    Args:
        gray_image: 2D grayscale image
        k: Number of singular values to retain

    Returns:
        compressed_image, U, S, Vt
    """
    if gray_image.ndim != 2:
        raise ValueError("compress_grayscale expects a 2D grayscale image.")

    gray_float = gray_image.astype(np.float64)
    U, S, Vt = compute_svd(gray_float)
    compressed = reconstruct_from_svd(U, S, Vt, k)
    compressed = np.clip(compressed, 0, 255).astype(np.uint8)

    return compressed, U, S, Vt


def compress_channel(channel: np.ndarray, k: int) -> np.ndarray:
    """
    Compress a single image channel using SVD.

    Args:
        channel: 2D channel matrix
        k: Number of singular values to retain

    Returns:
        Compressed channel
    """
    channel_float = channel.astype(np.float64)
    U, S, Vt = compute_svd(channel_float)
    compressed = reconstruct_from_svd(U, S, Vt, k)
    return np.clip(compressed, 0, 255)


def compress_rgb(image: np.ndarray, k: int) -> np.ndarray:
    """
    Compress an RGB image by applying SVD to each channel separately.

    Args:
        image: 3D RGB image (H, W, 3)
        k: Number of singular values to retain per channel

    Returns:
        Compressed RGB image
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("compress_rgb expects an RGB image with shape (H, W, 3).")

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    r_comp = compress_channel(r, k)
    g_comp = compress_channel(g, k)
    b_comp = compress_channel(b, k)

    compressed = np.stack([r_comp, g_comp, b_comp], axis=2)
    return np.clip(compressed, 0, 255).astype(np.uint8)


def calculate_compression_ratio_grayscale(shape: tuple[int, int], k: int) -> float:
    """
    Calculate compression ratio for grayscale image storage.

    Original grayscale image size:
        m * n

    Compressed SVD storage:
        k * (m + n + 1)

    Args:
        shape: (rows, cols)
        k: Number of singular values retained

    Returns:
        Compression ratio
    """
    if len(shape) != 2:
        raise ValueError("shape must be (rows, cols) for grayscale images.")

    m, n = shape
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    original_size = m * n
    compressed_size = k * (m + n + 1)

    if compressed_size == 0:
        raise ZeroDivisionError("Compressed size became zero.")

    return original_size / compressed_size


def calculate_compression_ratio_rgb(shape: tuple[int, int, int], k: int) -> float:
    """
    Calculate compression ratio for RGB image storage.

    Original RGB image size:
        3 * m * n

    Compressed RGB storage:
        3 * k * (m + n + 1)

    Args:
        shape: (rows, cols, 3)
        k: Number of singular values retained per channel

    Returns:
        Compression ratio
    """
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError("shape must be (rows, cols, 3) for RGB images.")

    m, n, c = shape
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    original_size = m * n * c
    compressed_size = c * k * (m + n + 1)

    if compressed_size == 0:
        raise ZeroDivisionError("Compressed size became zero.")

    return original_size / compressed_size


def reconstruction_error(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Compute reconstruction error using Frobenius norm.

    Args:
        original: Original image/matrix
        compressed: Compressed/reconstructed image/matrix

    Returns:
        Reconstruction error
    """
    if original.shape != compressed.shape:
        raise ValueError("original and compressed must have the same shape.")

    return float(np.linalg.norm(original.astype(np.float64) - compressed.astype(np.float64)))


if __name__ == "__main__":
    # Simple self-test with random data
    gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    gray_comp, U, S, Vt = compress_grayscale(gray, k=20)
    ratio_gray = calculate_compression_ratio_grayscale(gray.shape, k=20)
    error_gray = reconstruction_error(gray, gray_comp)

    print("Grayscale compression test successful")
    print(f"Compression Ratio: {ratio_gray:.4f}")
    print(f"Reconstruction Error: {error_gray:.4f}")

    rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    rgb_comp = compress_rgb(rgb, k=20)
    ratio_rgb = calculate_compression_ratio_rgb(rgb.shape, k=20)
    error_rgb = reconstruction_error(rgb, rgb_comp)

    print("\nRGB compression test successful")
    print(f"Compression Ratio: {ratio_rgb:.4f}")
    print(f"Reconstruction Error: {error_rgb:.4f}") 