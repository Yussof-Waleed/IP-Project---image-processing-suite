"""
Spatial filtering processors.

Implements:
- Low-pass filters: Gaussian (19x19, σ=3), Median (7x7)
- High-pass filters: Laplacian, Sobel, Gradient magnitude
"""

import numpy as np

from app.core.interfaces import ConvolutionProcessor, ImageProcessor, ParamInfo
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


# ─────────────────────────────────────────────────────────────────────────────
# Low-Pass Filters
# ─────────────────────────────────────────────────────────────────────────────

class GaussianFilter(ConvolutionProcessor):
    """Gaussian smoothing filter (19x19, σ=3 by default)."""

    @property
    def name(self) -> str:
        return "Gaussian Smoothing"

    @property
    def category(self) -> str:
        return "Spatial Filters"

    def get_default_params(self) -> dict:
        return {"size": 19, "sigma": 3.0}

    def get_param_info(self) -> dict[str, ParamInfo]:
        return {
            "size": ParamInfo("Kernel Size", "int", 19, 3, 31,
                             tooltip="Must be odd (3, 5, 7, ...)"),
            "sigma": ParamInfo("Sigma (σ)", "float", 3.0, 0.1, 10.0,
                              tooltip="Standard deviation"),
        }

    def get_kernel(self, **params) -> np.ndarray:
        size = params.get("size", 19)
        sigma = params.get("sigma", 3.0)
        # Ensure odd size
        if size % 2 == 0:
            size += 1
        return build_gaussian_kernel(size, sigma)

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        kernel = self.get_kernel(**params)
        result = convolve2d(image, kernel)
        return clip_to_uint8(result)


class MedianFilter(ImageProcessor):
    """Median filter (7x7 by default)."""

    @property
    def name(self) -> str:
        return "Median Filter"

    @property
    def category(self) -> str:
        return "Spatial Filters"

    def get_default_params(self) -> dict:
        return {"size": 7}

    def get_param_info(self) -> dict[str, ParamInfo]:
        return {
            "size": ParamInfo("Kernel Size", "int", 7, 3, 15,
                             tooltip="Must be odd"),
        }

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        size = params.get("size", 7)
        if size % 2 == 0:
            size += 1
        return apply_median_filter(image, size)


# ─────────────────────────────────────────────────────────────────────────────
# High-Pass Filters (Edge Detection)
# ─────────────────────────────────────────────────────────────────────────────

class LaplacianFilter(ConvolutionProcessor):
    """Laplacian edge detection filter."""

    @property
    def name(self) -> str:
        return "Laplacian"

    @property
    def category(self) -> str:
        return "Spatial Filters"

    def get_default_params(self) -> dict:
        return {"variant": "standard"}

    def get_kernel(self, **params) -> np.ndarray:
        variant = params.get("variant", "standard")
        return build_laplacian_kernel(variant)

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        # Convert to grayscale if color
        if image.ndim == 3:
            gray = rgb_to_grayscale(image)
        else:
            gray = image.astype(np.float64)
        
        kernel = self.get_kernel(**params)
        result = convolve2d(gray, kernel)
        
        # Normalize to show edges clearly
        return normalize_to_uint8(np.abs(result))


class SobelFilter(ImageProcessor):
    """Sobel edge detection filter."""

    @property
    def name(self) -> str:
        return "Sobel"

    @property
    def category(self) -> str:
        return "Spatial Filters"

    def get_default_params(self) -> dict:
        return {"direction": "both"}

    def get_param_info(self) -> dict[str, ParamInfo]:
        return {
            "direction": ParamInfo("Direction", "choice", "both",
                                  choices=("x", "y", "both"),
                                  tooltip="Gradient direction"),
        }

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        direction = params.get("direction", "both")
        
        # Convert to grayscale if color
        if image.ndim == 3:
            gray = rgb_to_grayscale(image)
        else:
            gray = image.astype(np.float64)
        
        sobel_x, sobel_y = build_sobel_kernels()
        
        if direction == "x":
            result = convolve2d(gray, sobel_x)
            return normalize_to_uint8(np.abs(result))
        elif direction == "y":
            result = convolve2d(gray, sobel_y)
            return normalize_to_uint8(np.abs(result))
        else:  # both - compute magnitude
            gx = convolve2d(gray, sobel_x)
            gy = convolve2d(gray, sobel_y)
            magnitude = np.sqrt(gx**2 + gy**2)
            return normalize_to_uint8(magnitude)


class GradientFilter(ImageProcessor):
    """Simple gradient-based edge detection."""

    @property
    def name(self) -> str:
        return "Gradient Magnitude"

    @property
    def category(self) -> str:
        return "Spatial Filters"

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        # Convert to grayscale if color
        if image.ndim == 3:
            gray = rgb_to_grayscale(image)
        else:
            gray = image.astype(np.float64)
        
        h, w = gray.shape
        
        # Simple gradient kernels
        kernel_x = np.array([[-1, 0, 1]], dtype=np.float64)
        kernel_y = np.array([[-1], [0], [1]], dtype=np.float64)
        
        # Pad for convolution
        gx = convolve2d(gray, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64))
        gy = convolve2d(gray, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64))
        
        magnitude = np.sqrt(gx**2 + gy**2)
        return normalize_to_uint8(magnitude)


class SharpeningFilter(ImageProcessor):
    """Unsharp masking for image sharpening."""

    @property
    def name(self) -> str:
        return "Sharpening"

    @property
    def category(self) -> str:
        return "Spatial Filters"

    def get_default_params(self) -> dict:
        return {"amount": 1.5, "radius": 3}

    def get_param_info(self) -> dict[str, ParamInfo]:
        return {
            "amount": ParamInfo("Amount", "float", 1.5, 0.1, 5.0,
                               tooltip="Sharpening strength"),
            "radius": ParamInfo("Radius", "int", 3, 1, 11,
                               tooltip="Blur radius for unsharp mask"),
        }

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        amount = params.get("amount", 1.5)
        radius = params.get("radius", 3)
        
        # Ensure odd radius
        if radius % 2 == 0:
            radius += 1
        
        # Create Gaussian blur
        kernel = build_gaussian_kernel(radius * 2 + 1, radius / 3)
        blurred = convolve2d(image.astype(np.float64), kernel)
        
        # Unsharp mask: result = original + amount * (original - blurred)
        result = image.astype(np.float64) + amount * (image.astype(np.float64) - blurred)
        
        return clip_to_uint8(result)
