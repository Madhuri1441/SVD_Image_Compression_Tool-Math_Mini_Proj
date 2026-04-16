# SVD Image Compression Tool

This project implements an image compression system using **Singular Value Decomposition (SVD)**. It demonstrates how linear algebra techniques can be applied to reduce image size while preserving essential visual information.

The project includes both:
- a **command-line pipeline** (for core logic)
- an **interactive web interface** using Streamlit

---

## Overview

Singular Value Decomposition factorizes an image matrix into three components:

A = U Σ Vᵀ

By retaining only the top **k singular values**, the image can be approximated using fewer parameters. This results in compression with a controllable trade-off between **image quality** and **storage size**.

---

## Features

- Image compression using Singular Value Decomposition
- Adjustable compression level using **k-value**
- Support for both **grayscale** and **RGB image compression**
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
```


## Installation (Dependencies)

Install required libraries using pip:
```bash
    pip install numpy opencv-python matplotlib streamlit
```

# How to Run:


## Run Core Logic (Command Line)

This runs the full compression pipeline:

    python src/main.py

- Loads images from data/
- Compresses using SVD
- Saves outputs in outputs/
- Displays comparison and graphs


## Run Streamlit Web App

Launch the interactive UI:

    streamlit run app/app.py

- Upload your own image or select sample images
- Adjust k-value using slider
- View original vs compressed image
- See compression ratio and error in real-time



## How It Works:
    - Load image as a matrix
    - Convert to grayscale (optional)
    - Apply SVD:
        A = U Σ Vᵀ
        Keep only top k singular values
        Reconstruct compressed image
    - Compute:
        Reconstruction error
        Compression ratio
        Display and save results
        RGB Compression
        Split image into R, G, B channels
        Apply SVD separately to each channel
        Recombine channels into final compressed image
        Compression Ratio
    - For a grayscale image of size m × n:
        Compression Ratio = (m × n) / [k × (m + n + 1)]
    - Higher ratio = better compression
    - Measured using matrix norm:
        ‖A - A_k‖
    - Lower error = better image quality

# Output:

## Compressed images are saved in:
    outputs/
    output/compressed/
    output/plots/

## Limitations
    SVD is computationally expensive for large images
    Very small k reduces image quality significantly
    Not optimized for real-world formats like JPEG/PNG
    Future Improvements
    Multiple image upload support
    Improved RGB error calculation
    Performance optimization
    Add PSNR / SSIM metrics
    Auto-save plots

## Team Contribution

- **Manasvi (Person 1)**  
  - Implemented core SVD logic in `compress.py`  
  - Developed grayscale and RGB image compression  
  - Implemented compression ratio calculations  

- **Neha (Person 2)**  
  - Built helper functions in `utils.py`  
  - Implemented visualization features in `visualize.py`  
  - Created comparison plots and error vs k graphs  

- **Madhuri (Person 3)**  
  - Integrated full pipeline in `main.py`  
  - Developed Streamlit UI in `app/app.py`  
  - Connected all modules and handled user interaction
