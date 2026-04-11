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

from visualize import plot_comparison, plot_error_vs_k


def process_image(image_path, k=50):
    print(f"\nProcessing: {image_path}")

    img = load_image(image_path)
    gray = rgb_to_grayscale(img)

    # Grayscale
    gray_comp = compress_grayscale(gray, k)
    gray_error = reconstruction_error(gray, gray_comp)
    gray_ratio = compression_ratio_grayscale(gray, k)

    print("\nGrayscale:")
    print("Error:", gray_error)
    print("Ratio:", gray_ratio)

    # RGB
    rgb_comp = compress_rgb_image(img, k)
    rgb_ratio = compression_ratio_rgb(img, k)

    print("\nRGB:")
    print("Ratio:", rgb_ratio)

    # Save
    os.makedirs("outputs", exist_ok=True)
    name = os.path.basename(image_path).split(".")[0]

    save_image(f"outputs/{name}_gray.jpg", gray_comp)
    save_image(f"outputs/{name}_rgb.jpg", rgb_comp)

    # Plots
    plot_comparison(gray, gray_comp, k, "Grayscale")
    plot_comparison(img, rgb_comp, k, "RGB")

    ks = [10, 20, 50, 100]
    plot_error_vs_k(gray, ks)


def main():
    images = [
        "data/image1.jpg",
        "data/image2.jpg"
    ]

    for img_path in images:
        try:
            process_image(img_path, 50)
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()