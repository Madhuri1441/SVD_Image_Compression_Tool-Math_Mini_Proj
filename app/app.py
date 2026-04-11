# app/app.py

import os
import sys
import cv2
import numpy as np
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from compress import (
    load_image,
    compress_rgb_image,
    compress_grayscale,
    rgb_to_grayscale,
    compression_ratio_rgb,
    compression_ratio_grayscale,
    reconstruction_error,
)


def load_uploaded_image(file):
    bytes_data = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


st.title("SVD Image Compression Tool")

mode = st.sidebar.selectbox("Mode", ["RGB", "Grayscale"])
k = st.sidebar.slider("k value", 1, 200, 50)

choice = st.sidebar.selectbox("Select image", ["Upload", "image1.jpg", "image2.jpg"])

if choice == "Upload":
    uploaded = st.file_uploader("Upload Image")
    if uploaded:
        img = load_uploaded_image(uploaded)
    else:
        st.stop()
else:
    img = load_image(f"data/{choice}")

gray = rgb_to_grayscale(img)

if mode == "RGB":
    comp = compress_rgb_image(img, k)
    ratio = compression_ratio_rgb(img, k)
    error = reconstruction_error(img[:, :, 0], comp[:, :, 0])
else:
    comp = compress_grayscale(gray, k)
    ratio = compression_ratio_grayscale(gray, k)
    error = reconstruction_error(gray, comp)

col1, col2 = st.columns(2)

col1.image(img, caption="Original", use_container_width=True)
col2.image(comp, caption=f"Compressed k={k}", use_container_width=True)

st.write("Compression Ratio:", ratio)
st.write("Error:", error)