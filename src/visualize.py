# src/visualize.py

import matplotlib.pyplot as plt
import numpy as np

from compress import apply_svd, compress_channel, reconstruction_error


def plot_comparison(original, compressed, k, mode="RGB"):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    if len(original.shape) == 2:
        plt.imshow(original, cmap="gray")
    else:
        plt.imshow(original)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Compressed ({mode}, k={k})")
    if len(compressed.shape) == 2:
        plt.imshow(compressed, cmap="gray")
    else:
        plt.imshow(compressed)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_error_vs_k(gray, ks):
    U, S, Vt = apply_svd(gray)
    errors = []

    for k in ks:
        comp = compress_channel(U, S, Vt, k)
        errors.append(reconstruction_error(gray, comp))

    plt.figure(figsize=(7, 5))
    plt.plot(ks, errors, marker="o")
    plt.xlabel("k")
    plt.ylabel("Reconstruction Error")
    plt.title("Error vs k")
    plt.grid(True)
    plt.show()