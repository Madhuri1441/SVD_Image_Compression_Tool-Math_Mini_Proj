# SVD Image Compression Tool

This project implements an image compression system using Singular Value Decomposition (SVD). It demonstrates how linear algebra techniques can be applied to reduce image size while preserving essential visual information. The project includes both a command-line pipeline and an interactive web interface built with Streamlit.

---

## Overview

Singular Value Decomposition factorizes an image matrix into three components:

A = U Σ Vᵀ

By retaining only the top k singular values, the image can be approximated with significantly fewer parameters. This results in compression with a controllable trade-off between image quality and storage size.

---

## Features

- Image compression using Singular Value Decomposition
- Adjustable compression level using k-value
- Support for both grayscale and RGB image compression
- Reconstruction error calculation
- Compression ratio estimation
- Visualization of original vs compressed images
- Error vs k analysis graph (grayscale)
- Interactive web application using Streamlit
- Support for image upload and dataset-based inputs
- Download option for compressed images

---

## Project Structure

```bash
SVD_IMAGE_COMPRESSION_TOOL/
│
├── app/
│   └── app.py                  # Streamlit web application
│
├── data/
│   ├── image1.jpg              # Sample input image 1
│   └── image2.jpg              # Sample input image 2
│
├── output/
│   ├── compressed/             # Additional compressed image outputs
│   └── plots/                  # Saved plots/graphs
│
├── outputs/
│   ├── grayscale_compressed.jpg
│   ├── image1_gray.jpg
│   ├── image1_rgb.jpg
│   ├── image2_gray.jpg
│   ├── image2_rgb.jpg
│   └── rgb_compressed.jpg      # Generated compressed image files
│
├── src/
│   ├── __pycache__/            # Python cache files
│   ├── compress.py             # Core SVD compression logic
│   ├── main.py                 # Main execution script
│   ├── utils.py                # Helper utility functions
│   └── visualize.py            # Plotting and visualization functions
│
├── .gitignore
└── README.md
