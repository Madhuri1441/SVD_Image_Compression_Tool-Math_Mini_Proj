# app/app.py

import os
import sys
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Allow import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from compress import (
    load_image,
    apply_svd,
    compress_channel,
    compress_rgb_image,
    compress_grayscale,
    rgb_to_grayscale,
    reconstruction_error,
    compression_ratio_grayscale,
    compression_ratio_rgb,
)


def load_uploaded_image(uploaded_file) -> np.ndarray:
    """
    Load uploaded image and convert it to RGB.
    """
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not read uploaded image.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def build_error_plot(gray: np.ndarray, max_k: int):
    """
    Create error vs k plot.
    """
    ks = sorted(set([5, 10, 20, 30, 50, 75, 100, 150, max_k]))
    ks = [k for k in ks if 1 <= k <= min(gray.shape)]

    U, S, Vt = apply_svd(gray)
    errors = []

    for k in ks:
        comp = compress_channel(U, S, Vt, k)
        err = reconstruction_error(gray, comp)
        errors.append(err)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ks, errors, marker="o", linewidth=2)
    ax.set_title("Reconstruction Error vs k", fontsize=14, fontweight="bold")
    ax.set_xlabel("k")
    ax.set_ylabel("Reconstruction Error")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    return fig


def convert_rgb_image_to_bytes(img: np.ndarray) -> bytes:
    """
    Convert RGB image to JPG bytes for download.
    """
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".jpg", img_bgr)

    if not success:
        raise ValueError("Failed to encode image.")

    return buffer.tobytes()


def convert_gray_image_to_bytes(img: np.ndarray) -> bytes:
    """
    Convert grayscale image to JPG bytes for download.
    """
    success, buffer = cv2.imencode(".jpg", img)

    if not success:
        raise ValueError("Failed to encode image.")

    return buffer.tobytes()


def get_sample_images():
    """
    Get all valid images from data/ folder.
    """
    data_folder = "data"

    if not os.path.exists(data_folder):
        return []

    valid_extensions = (".jpg", ".jpeg", ".png")
    files = [
        file for file in os.listdir(data_folder)
        if file.lower().endswith(valid_extensions)
    ]

    return files


def inject_custom_css():
    st.markdown(
        """
        <style>
        .main {
            background-color: #0f172a;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        h1, h2, h3 {
            color: white;
        }

        p, label, div, span {
            color: #e2e8f0;
        }

        [data-testid="stMetricValue"] {
            font-size: 28px;
            font-weight: 700;
            color: #ffffff;
        }

        [data-testid="stMetricLabel"] {
            color: #cbd5e1;
        }

        .hero-box {
            padding: 1.5rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1.2rem;
        }

        .section-box {
            padding: 1.2rem;
            border-radius: 16px;
            background-color: #111827;
            border: 1px solid rgba(255,255,255,0.06);
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .small-note {
            color: #94a3b8;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="SVD Image Compression Tool",
        page_icon="🖼️",
        layout="wide",
    )

    inject_custom_css()

    st.markdown(
        """
        <div class="hero-box">
            <h1>🖼️ SVD Image Compression Tool</h1>
            <p>
                Compress images using <b>Singular Value Decomposition (SVD)</b> and compare
                quality, reconstruction error, and compression ratio interactively.
            </p>
            <p class="small-note">
                Supports grayscale and RGB compression, adjustable k-value, sample images, uploads, and downloadable output.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.title("Controls")
    st.sidebar.markdown("Adjust settings and choose an image source.")

    sample_images = get_sample_images()
    source_option = st.sidebar.radio(
        "Choose image source",
        ["Sample Image", "Upload Image"]
    )

    uploaded_file = None
    selected_sample = None

    if source_option == "Sample Image":
        if sample_images:
            selected_sample = st.sidebar.selectbox(
                "Select a sample image",
                sample_images
            )
        else:
            st.sidebar.warning("No images found in data/ folder.")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"]
        )

    mode = st.sidebar.radio(
        "Compression mode",
        ["RGB", "Grayscale"]
    )

    k = st.sidebar.slider(
        "Choose k value",
        min_value=1,
        max_value=200,
        value=50
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Smaller k gives higher compression but lower image quality. "
        "Larger k preserves more detail."
    )

    # Load image
    img = None
    image_name = "uploaded_image"

    try:
        if source_option == "Upload Image":
            if uploaded_file is not None:
                img = load_uploaded_image(uploaded_file)
                image_name = os.path.splitext(uploaded_file.name)[0]
            else:
                st.info("Upload an image from the sidebar to begin.")
                return
        else:
            if selected_sample:
                image_path = os.path.join("data", selected_sample)
                img = load_image(image_path)
                image_name = os.path.splitext(selected_sample)[0]
            else:
                st.info("Select a sample image from the sidebar to begin.")
                return
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        return

    gray = rgb_to_grayscale(img)

    # Restrict k to actual image dimensions
    effective_k = min(k, min(gray.shape))
    if effective_k != k:
        st.warning(f"k adjusted to {effective_k} because image dimensions are smaller than selected k.")
        k = effective_k

    # Compression
    if mode == "RGB":
        compressed_img = compress_rgb_image(img, k)
        ratio = compression_ratio_rgb(img, k)

        r_error = reconstruction_error(img[:, :, 0], compressed_img[:, :, 0])
        g_error = reconstruction_error(img[:, :, 1], compressed_img[:, :, 1])
        b_error = reconstruction_error(img[:, :, 2], compressed_img[:, :, 2])
        error_value = (r_error + g_error + b_error) / 3

        downloadable_bytes = convert_rgb_image_to_bytes(compressed_img)
        original_display = img
        compressed_display = compressed_img

    else:
        compressed_img = compress_grayscale(gray, k)
        ratio = compression_ratio_grayscale(gray, k)
        error_value = reconstruction_error(gray, compressed_img)

        downloadable_bytes = convert_gray_image_to_bytes(compressed_img)
        original_display = gray
        compressed_display = compressed_img

    # Metrics
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Compression Metrics")

    m1, m2, m3 = st.columns(3)
    m1.metric("Mode", mode)
    m2.metric("k Value", k)
    m3.metric("Compression Ratio", f"{ratio:.4f}")

    m4, _ = st.columns(2)
    m4.metric("Reconstruction Error", f"{error_value:.2f}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Image comparison
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Original vs Compressed")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            original_display,
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        st.image(
            compressed_display,
            caption=f"Compressed Image ({mode}, k={k})",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Error graph for grayscale
    if mode == "Grayscale":
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("Error Analysis")
        fig = build_error_plot(gray, k)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Download section
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Download Output")

    st.download_button(
        label="Download Compressed Image",
        data=downloadable_bytes,
        file_name=f"{image_name}_{mode.lower()}_k_{k}.jpg",
        mime="image/jpeg"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown(
        """
        <hr style="margin-top: 2rem; margin-bottom: 1rem; border: 1px solid rgba(255,255,255,0.08);">
        <p style="text-align:center; color:#94a3b8;">
            Built with Python, NumPy, OpenCV, Matplotlib, and Streamlit
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()