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
