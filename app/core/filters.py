"""
Spatial filtering processors.

Low-pass: Gaussian, Median
High-pass: Laplacian, Sobel, Gradient
"""

import numpy as np

from app.core.interfaces import ImageProcessor
from app.core.primitives import (
    convolve2d,
    apply_median_filter,
    build_gaussian_kernel,
    build_laplacian_kernel,
    build_sobel_kernels,
    rgb_to_grayscale,
    normalize_to_uint8,
    clip_to_uint8,
)


class GaussianFilter(ImageProcessor):
    """Gaussian smoothing filter (19x19, σ=3 by default)."""

    @property
    def name(self) -> str:
        return "Gaussian Smoothing"

    @property
    def category(self) -> str:
        return "Spatial Filters"

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        size = params.get("size", 19)
        sigma = params.get("sigma", 3.0)
        if size % 2 == 0:
            size += 1
        kernel = build_gaussian_kernel(size, sigma)
        return clip_to_uint8(convolve2d(image, kernel))


class MedianFilter(ImageProcessor):
    """Median filter (7x7 by default)."""

    @property
    def name(self) -> str:
        return "Median Filter"

    @property
    def category(self) -> str:
        return "Spatial Filters"

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        size = params.get("size", 7)
        if size % 2 == 0:
            size += 1
        return apply_median_filter(image, size)


# ─────────────────────────────────────────────────────────────────────────────
# High-Pass Filters (Edge Detection)
# ─────────────────────────────────────────────────────────────────────────────

class LaplacianFilter(ImageProcessor):
    """Laplacian edge detection filter."""

    @property
    def name(self) -> str:
        return "Laplacian"

    @property
    def category(self) -> str:
        return "Spatial Filters"

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        gray = rgb_to_grayscale(image) if image.ndim == 3 else image.astype(np.float64)
        kernel = build_laplacian_kernel(params.get("variant", "standard"))
        result = convolve2d(gray, kernel)
        return normalize_to_uint8(np.abs(result))


class SobelFilter(ImageProcessor):
    """Sobel edge detection filter."""

    @property
    def name(self) -> str:
        return "Sobel"

    @property
    def category(self) -> str:
        return "Spatial Filters"

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        direction = params.get("direction", "both")
        gray = rgb_to_grayscale(image) if image.ndim == 3 else image.astype(np.float64)
        sobel_x, sobel_y = build_sobel_kernels()
        
        if direction == "x":
            return normalize_to_uint8(np.abs(convolve2d(gray, sobel_x)))
        elif direction == "y":
            return normalize_to_uint8(np.abs(convolve2d(gray, sobel_y)))
        else:
            gx = convolve2d(gray, sobel_x)
            gy = convolve2d(gray, sobel_y)
            return normalize_to_uint8(np.sqrt(gx**2 + gy**2))


class GradientFilter(ImageProcessor):
    """Simple gradient-based edge detection."""

    @property
    def name(self) -> str:
        return "Gradient Magnitude"

    @property
    def category(self) -> str:
        return "Spatial Filters"

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        gray = rgb_to_grayscale(image) if image.ndim == 3 else image.astype(np.float64)
        gx = convolve2d(gray, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64))
        gy = convolve2d(gray, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64))
        return normalize_to_uint8(np.sqrt(gx**2 + gy**2))


class SharpeningFilter(ImageProcessor):
    """Unsharp masking for image sharpening."""

    @property
    def name(self) -> str:
        return "Sharpening"

    @property
    def category(self) -> str:
        return "Spatial Filters"

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        amount = params.get("amount", 1.5)
        radius = params.get("radius", 3)
        if radius % 2 == 0:
            radius += 1
        
        kernel = build_gaussian_kernel(radius * 2 + 1, radius / 3)
        blurred = convolve2d(image.astype(np.float64), kernel)
        result = image.astype(np.float64) + amount * (image.astype(np.float64) - blurred)
        return clip_to_uint8(result)
