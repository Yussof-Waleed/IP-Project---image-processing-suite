# 📷 Image Processing Mini-Suite

A complete image processing application with a modern GUI built with Python and PySide6.

## 🚀 Quick Start

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m app.main
```

## 📁 Project Structure

```
IP Project/
├── app/
│   ├── main.py              # Application entry point
│   ├── gui/                  # GUI components
│   │   ├── __init__.py
│   │   ├── main_window.py    # Main window
│   │   └── widgets/          # Custom widgets
│   └── core/                 # Image processing logic
│       ├── __init__.py
│       ├── io.py             # Image load/save
│       ├── primitives.py     # Low-level helpers
│       ├── transforms.py     # Affine transforms
│       ├── interpolation.py  # Resize algorithms
│       ├── filters.py        # Spatial filters
│       ├── histogram.py      # Histogram processing
│       └── compression/      # Compression algorithms
├── tests/                    # Unit tests
├── assets/
│   └── sample_images/        # Sample images for testing
├── docs/                     # Documentation
├── requirements.txt
└── README.md
```

## ✨ Features

- **Image Loading & Display**: Upload images, view metadata (resolution, size, type)
- **Format Conversions**: Grayscale, Binary (thresholding)
- **Affine Transformations**: Translation, Scaling, Rotation, Shear X/Y
- **Interpolation**: Nearest Neighbor, Bilinear, Bicubic
- **Cropping**: Interactive region selection
- **Histogram Processing**: Analysis and equalization
- **Spatial Filtering**: Gaussian, Median, Laplacian, Sobel
- **Compression**: Huffman, LZW, RLE, DCT, Wavelet, and more

## 🛠️ Development Principles

This project follows:
- **DRY** - Don't Repeat Yourself
- **SOLID** - Single Responsibility, Open/Closed, Liskov, Interface Segregation, Dependency Inversion
- **CQS** - Command-Query Separation
- **YAGNI** - You Aren't Gonna Need It
- **KISS** - Keep It Simple, Stupid

## 📝 License

Educational project for Image Processing course.
