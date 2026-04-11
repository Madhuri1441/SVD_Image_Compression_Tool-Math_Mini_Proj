# src/main.py

import os

from compress import (
    load_image,
    save_image,
    rgb_to_grayscale,
    compress_grayscale,
    compress_rgb_image,
    reconstruction_error,
    compression_ratio_grayscale,
    compression_ratio_rgb,
)

from visualize import (
    plot_comparison,
    plot_error_vs_k,
)


def process_image(image_path: str, k: int = 50) -> None:
    """
    Process one image using both grayscale and RGB compression.
    Saves outputs and displays plots.
    """
    print(f"\nProcessing image: {image_path}")

    img = load_image(image_path)
    print(f"Loaded image with shape: {img.shape}")

    gray = rgb_to_grayscale(img)
    print(f"Converted to grayscale with shape: {gray.shape}")

    # Grayscale compression
    gray_compressed = compress_grayscale(gray, k)
    gray_error = reconstruction_error(gray, gray_compressed)
    gray_ratio = compression_ratio_grayscale(gray, k)

    print("\n--- Grayscale Compression ---")
    print(f"k = {k}")
    print(f"Reconstruction Error: {gray_error:.2f}")
    print(f"Compression Ratio: {gray_ratio:.4f}")

    # RGB compression
    rgb_compressed = compress_rgb_image(img, k)
    rgb_ratio = compression_ratio_rgb(img, k)

    r_error = reconstruction_error(img[:, :, 0], rgb_compressed[:, :, 0])
    g_error = reconstruction_error(img[:, :, 1], rgb_compressed[:, :, 1])
    b_error = reconstruction_error(img[:, :, 2], rgb_compressed[:, :, 2])
    avg_rgb_error = (r_error + g_error + b_error) / 3

    print("\n--- RGB Compression ---")
    print(f"k = {k}")
    print(f"Average RGB Reconstruction Error: {avg_rgb_error:.2f}")
    print(f"Compression Ratio: {rgb_ratio:.4f}")

    # Save outputs
    os.makedirs("outputs", exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    gray_output_path = f"outputs/{base_name}_grayscale_k_{k}.jpg"
    rgb_output_path = f"outputs/{base_name}_rgb_k_{k}.jpg"

    save_image(gray_output_path, gray_compressed)
    save_image(rgb_output_path, rgb_compressed)

    print(f"\nSaved grayscale output to: {gray_output_path}")
    print(f"Saved RGB output to: {rgb_output_path}")

    # Visualizations
    plot_comparison(gray, gray_compressed, k, mode="Grayscale")
    plot_comparison(img, rgb_compressed, k, mode="RGB")

    ks = [10, 20, 30, 50, 75, 100, 150]
    valid_ks = [value for value in ks if value <= min(gray.shape)]
    plot_error_vs_k(gray, valid_ks)


def main():
    """
    Main runner for testing multiple images from the data folder.
    """
    image_paths = [
        "data/image1.jpg",
        "data/image2.jpg",
    ]

    k = 50

    for image_path in image_paths:
        try:
            process_image(image_path, k)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


if __name__ == "__main__":
    main()