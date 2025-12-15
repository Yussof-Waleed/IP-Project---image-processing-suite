"""
Core image processing primitives.

Building blocks: convolution, kernels, color conversion, utilities.
"""

import numpy as np
import math


def pad_image(image: np.ndarray, pad_y: int, pad_x: int) -> np.ndarray:
    """Pad image by replicating edge pixels."""
    if image.ndim == 3:
        return np.pad(image, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode='edge')
    return np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode='edge')


# ─────────────────────────────────────────────────────────────────────────────
# Convolution Engine
# ─────────────────────────────────────────────────────────────────────────────

def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Perform 2D convolution (no cv2/scipy)."""
    if kernel.ndim != 2 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Kernel must be 2D with odd dimensions")

    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2
    h, w = image.shape[:2]
    img_float = image.astype(np.float64)
    
    if image.ndim == 3:
        output = np.zeros((h, w, image.shape[2]), dtype=np.float64)
        for c in range(image.shape[2]):
            output[:, :, c] = _convolve_channel(img_float[:, :, c], kernel, pad_y, pad_x)
    else:
        output = _convolve_channel(img_float, kernel, pad_y, pad_x)
    return output


def _convolve_channel(image: np.ndarray, kernel: np.ndarray, pad_y: int, pad_x: int) -> np.ndarray:
    """Convolve single channel."""
    h, w = image.shape
    padded = pad_image(image, pad_y, pad_x)
    output = np.zeros((h, w), dtype=np.float64)
    kh, kw = kernel.shape
    
    for y in range(h):
        for x in range(w):
            output[y, x] = np.sum(padded[y:y+kh, x:x+kw] * kernel)
    return output


def apply_median_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply median filter."""
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    pad = kernel_size // 2
    
    if image.ndim == 3:
        output = np.zeros_like(image)
        for c in range(image.shape[2]):
            output[:, :, c] = _median_channel(image[:, :, c], kernel_size, pad)
        return output
    return _median_channel(image, kernel_size, pad)


def _median_channel(image: np.ndarray, kernel_size: int, pad: int) -> np.ndarray:
    """Median filter single channel."""
    h, w = image.shape
    padded = pad_image(image, pad, pad)
    output = np.zeros((h, w), dtype=image.dtype)
    
    for y in range(h):
        for x in range(w):
            output[y, x] = np.median(padded[y:y+kernel_size, x:x+kernel_size])
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Kernel Builders
# ─────────────────────────────────────────────────────────────────────────────

def build_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Build normalized Gaussian kernel."""
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    center = size // 2
    y, x = np.ogrid[:size, :size]
    kernel = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def build_box_kernel(size: int) -> np.ndarray:
    """Build a box (averaging) kernel."""
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    return np.ones((size, size), dtype=np.float64) / (size * size)


def build_laplacian_kernel(variant: str = "standard") -> np.ndarray:
    """Build Laplacian kernel. variant: 'standard' or 'diagonal'."""
    if variant == "diagonal":
        return np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64)
    return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def build_sobel_kernels() -> tuple[np.ndarray, np.ndarray]:
    """
    Build Sobel kernels for gradient detection.

    Returns:
        Tuple of (kernel_x, kernel_y) for horizontal and vertical gradients.
    """
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float64)
    
    return sobel_x, sobel_y


def build_prewitt_kernels() -> tuple[np.ndarray, np.ndarray]:
    """Build Prewitt kernels for gradient detection."""
    prewitt_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    prewitt_y = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ], dtype=np.float64)
    
    return prewitt_x, prewitt_y


# ─────────────────────────────────────────────────────────────────────────────
# Color Conversion Helpers
# ─────────────────────────────────────────────────────────────────────────────

def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale using luminance formula.
    
    Formula: Y = 0.299*R + 0.587*G + 0.114*B (ITU-R BT.601)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be RGB image with shape (H, W, 3)")

    weights = np.array([0.299, 0.587, 0.114])
    return np.dot(image.astype(np.float64), weights)


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert grayscale image to RGB by replicating channels."""
    if image.ndim != 2:
        raise ValueError("Input must be grayscale image with shape (H, W)")
    return np.stack([image, image, image], axis=2)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize float image to uint8 range [0, 255]."""
    img_min = image.min()
    img_max = image.max()
    
    if img_max - img_min == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    
    normalized = (image - img_min) / (img_max - img_min) * 255
    return normalized.astype(np.uint8)


def clip_to_uint8(image: np.ndarray) -> np.ndarray:
    """Clip values to [0, 255] and convert to uint8."""
    return np.clip(image, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Image Statistics (Queries)
# ─────────────────────────────────────────────────────────────────────────────

def compute_histogram(image: np.ndarray) -> np.ndarray:
    """
    Compute histogram of grayscale image.
    Returns array of 256 bins.
    """
    if image.ndim != 2:
        raise ValueError("Input must be grayscale image")

    # Clip to valid range and count occurrences
    clipped = np.clip(image, 0, 255).astype(np.uint8)
    histogram = np.bincount(clipped.ravel(), minlength=256)
    return histogram.astype(np.int64)


def compute_mean_intensity(image: np.ndarray) -> float:
    """Compute mean intensity of image."""
    return float(np.mean(image))
