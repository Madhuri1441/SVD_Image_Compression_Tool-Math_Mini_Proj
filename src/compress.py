# src/compress.py

import cv2
import numpy as np


def load_image(image_path: str):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not load image from path: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(image_path: str, img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(image_path, img)


def rgb_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def apply_svd(matrix):
    return np.linalg.svd(matrix, full_matrices=False)


def compress_channel(U, S, Vt, k):
    k = max(1, min(k, len(S)))

    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]

    return U_k @ S_k @ Vt_k


def compress_grayscale(gray, k):
    U, S, Vt = apply_svd(gray)
    comp = compress_channel(U, S, Vt, k)
    return np.clip(comp, 0, 255).astype(np.uint8)


def compress_rgb_image(img, k):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    Ur, Sr, Vtr = apply_svd(r)
    Ug, Sg, Vtg = apply_svd(g)
    Ub, Sb, Vtb = apply_svd(b)

    r_comp = compress_channel(Ur, Sr, Vtr, k)
    g_comp = compress_channel(Ug, Sg, Vtg, k)
    b_comp = compress_channel(Ub, Sb, Vtb, k)

    final = np.stack([r_comp, g_comp, b_comp], axis=2)
    return np.clip(final, 0, 255).astype(np.uint8)


def reconstruction_error(original, compressed):
    return np.linalg.norm(original.astype(np.float64) - compressed.astype(np.float64))


def compression_ratio_grayscale(gray, k):
    m, n = gray.shape
    return (m * n) / (k * (m + n + 1))


def compression_ratio_rgb(img, k):
    m, n, c = img.shape
    return (m * n * c) / (c * k * (m + n + 1))