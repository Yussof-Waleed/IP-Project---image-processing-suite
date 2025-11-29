"""
Core image processing primitives and utilities.

This module provides low-level building blocks used by all processors:
- Pixel access with boundary handling
- Manual convolution engine
- Kernel builders
- Color space helpers

All functions are pure (no side effects) following CQS.
"""

import numpy as np
from enum import Enum, auto
from typing import Callable
import math


# ─────────────────────────────────────────────────────────────────────────────
# Boundary Handling
# ─────────────────────────────────────────────────────────────────────────────

class BorderMode(Enum):
    """Boundary handling modes for convolution and sampling."""
    CONSTANT = auto()   # Pad with constant value (default: 0)
    REPLICATE = auto()  # Replicate edge pixels
    REFLECT = auto()    # Reflect at boundary
    WRAP = auto()       # Wrap around (periodic)


def get_pixel_safe(
    image: np.ndarray,
    y: int,
    x: int,
    border_mode: BorderMode = BorderMode.REPLICATE,
    constant_value: float = 0.0
) -> np.ndarray | float:
    """
    Safely get pixel value with boundary handling.

    Args:
        image: Input image array (H, W) or (H, W, C).
        y: Row index (can be out of bounds).
        x: Column index (can be out of bounds).
        border_mode: How to handle out-of-bounds access.
        constant_value: Value to use for CONSTANT mode.

    Returns:
        Pixel value (scalar for grayscale, array for color).
    """
    h, w = image.shape[:2]

    if border_mode == BorderMode.CONSTANT:
        if y < 0 or y >= h or x < 0 or x >= w:
            if image.ndim == 3:
                return np.full(image.shape[2], constant_value, dtype=image.dtype)
            return constant_value
        return image[y, x]

    elif border_mode == BorderMode.REPLICATE:
        y = max(0, min(y, h - 1))
        x = max(0, min(x, w - 1))
        return image[y, x]

    elif border_mode == BorderMode.REFLECT:
        # Reflect at boundaries
        if y < 0:
            y = -y - 1
        elif y >= h:
            y = 2 * h - y - 1
        if x < 0:
            x = -x - 1
        elif x >= w:
            x = 2 * w - x - 1
        # Clamp in case of multiple reflections needed
        y = max(0, min(y, h - 1))
        x = max(0, min(x, w - 1))
        return image[y, x]

    elif border_mode == BorderMode.WRAP:
        y = y % h
        x = x % w
        return image[y, x]

    return image[max(0, min(y, h-1)), max(0, min(x, w-1))]


def pad_image(
    image: np.ndarray,
    pad_y: int,
    pad_x: int,
    border_mode: BorderMode = BorderMode.REPLICATE,
    constant_value: float = 0.0
) -> np.ndarray:
    """
    Pad image with specified boundary handling.

    Args:
        image: Input image (H, W) or (H, W, C).
        pad_y: Padding size in y direction (top and bottom).
        pad_x: Padding size in x direction (left and right).
        border_mode: Boundary handling mode.
        constant_value: Value for CONSTANT mode.

    Returns:
        Padded image array.
    """
    h, w = image.shape[:2]
    has_channels = image.ndim == 3
    
    if has_channels:
        new_shape = (h + 2 * pad_y, w + 2 * pad_x, image.shape[2])
    else:
        new_shape = (h + 2 * pad_y, w + 2 * pad_x)

    padded = np.zeros(new_shape, dtype=image.dtype)
    
    for ny in range(new_shape[0]):
        for nx in range(new_shape[1]):
            oy = ny - pad_y
            ox = nx - pad_x
            padded[ny, nx] = get_pixel_safe(image, oy, ox, border_mode, constant_value)

    return padded


# ─────────────────────────────────────────────────────────────────────────────
# Convolution Engine
# ─────────────────────────────────────────────────────────────────────────────

def convolve2d(
    image: np.ndarray,
    kernel: np.ndarray,
    border_mode: BorderMode = BorderMode.REPLICATE
) -> np.ndarray:
    """
    Perform 2D convolution manually (no cv2/scipy).

    Args:
        image: Input image (H, W) for grayscale or (H, W, C) for color.
        kernel: 2D convolution kernel (must be odd-sized).
        border_mode: Boundary handling mode.

    Returns:
        Convolved image of same shape as input.
    """
    if kernel.ndim != 2:
        raise ValueError("Kernel must be 2D")
    
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Kernel dimensions must be odd")

    pad_y = kh // 2
    pad_x = kw // 2

    h, w = image.shape[:2]
    is_color = image.ndim == 3

    # Work in float for precision
    img_float = image.astype(np.float64)
    
    if is_color:
        output = np.zeros((h, w, image.shape[2]), dtype=np.float64)
        for c in range(image.shape[2]):
            output[:, :, c] = _convolve2d_single_channel(
                img_float[:, :, c], kernel, pad_y, pad_x, border_mode
            )
    else:
        output = _convolve2d_single_channel(img_float, kernel, pad_y, pad_x, border_mode)

    return output


def _convolve2d_single_channel(
    image: np.ndarray,
    kernel: np.ndarray,
    pad_y: int,
    pad_x: int,
    border_mode: BorderMode
) -> np.ndarray:
    """Convolve single channel image with kernel."""
    h, w = image.shape
    kh, kw = kernel.shape
    output = np.zeros((h, w), dtype=np.float64)

    # Pad the image
    padded = pad_image(image, pad_y, pad_x, border_mode)

    # Perform convolution
    for y in range(h):
        for x in range(w):
            region = padded[y:y + kh, x:x + kw]
            output[y, x] = np.sum(region * kernel)

    return output


def apply_median_filter(
    image: np.ndarray,
    kernel_size: int,
    border_mode: BorderMode = BorderMode.REPLICATE
) -> np.ndarray:
    """
    Apply median filter manually.

    Args:
        image: Input image (H, W) or (H, W, C).
        kernel_size: Size of the median filter (must be odd).
        border_mode: Boundary handling mode.

    Returns:
        Filtered image.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    h, w = image.shape[:2]
    is_color = image.ndim == 3
    pad = kernel_size // 2

    if is_color:
        output = np.zeros_like(image)
        for c in range(image.shape[2]):
            output[:, :, c] = _median_filter_single_channel(
                image[:, :, c], kernel_size, pad, border_mode
            )
    else:
        output = _median_filter_single_channel(image, kernel_size, pad, border_mode)

    return output


def _median_filter_single_channel(
    image: np.ndarray,
    kernel_size: int,
    pad: int,
    border_mode: BorderMode
) -> np.ndarray:
    """Apply median filter to single channel."""
    h, w = image.shape
    output = np.zeros((h, w), dtype=image.dtype)
    padded = pad_image(image, pad, pad, border_mode)

    for y in range(h):
        for x in range(w):
            region = padded[y:y + kernel_size, x:x + kernel_size]
            output[y, x] = np.median(region)

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Kernel Builders
# ─────────────────────────────────────────────────────────────────────────────

def build_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Build a Gaussian kernel manually.

    Args:
        size: Kernel size (must be odd).
        sigma: Standard deviation.

    Returns:
        Normalized 2D Gaussian kernel.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    kernel = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    
    sum_val = 0.0
    for y in range(size):
        for x in range(size):
            dy = y - center
            dx = x - center
            val = math.exp(-(dx*dx + dy*dy) / (2 * sigma * sigma))
            kernel[y, x] = val
            sum_val += val

    # Normalize
    kernel /= sum_val
    return kernel


def build_box_kernel(size: int) -> np.ndarray:
    """Build a box (averaging) kernel."""
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    return np.ones((size, size), dtype=np.float64) / (size * size)


def build_laplacian_kernel(variant: str = "standard") -> np.ndarray:
    """
    Build a Laplacian kernel for edge detection.

    Args:
        variant: "standard" (4-connected) or "diagonal" (8-connected).

    Returns:
        Laplacian kernel.
    """
    if variant == "standard":
        return np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float64)
    elif variant == "diagonal":
        return np.array([
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ], dtype=np.float64)
    else:
        raise ValueError(f"Unknown Laplacian variant: {variant}")


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
    Convert RGB image to grayscale manually.
    
    Uses luminance formula: Y = 0.299*R + 0.587*G + 0.114*B

    Args:
        image: RGB image (H, W, 3).

    Returns:
        Grayscale image (H, W).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be RGB image with shape (H, W, 3)")

    # Luminance weights (ITU-R BT.601)
    weights = np.array([0.299, 0.587, 0.114])
    
    grayscale = np.zeros(image.shape[:2], dtype=np.float64)
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            grayscale[y, x] = (
                weights[0] * image[y, x, 0] +
                weights[1] * image[y, x, 1] +
                weights[2] * image[y, x, 2]
            )

    return grayscale


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
    Compute histogram of grayscale image manually.

    Args:
        image: Grayscale image (H, W), values 0-255.

    Returns:
        Histogram array of 256 bins.
    """
    if image.ndim != 2:
        raise ValueError("Input must be grayscale image")

    histogram = np.zeros(256, dtype=np.int64)
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            val = int(image[y, x])
            val = max(0, min(255, val))
            histogram[val] += 1

    return histogram


def compute_mean_intensity(image: np.ndarray) -> float:
    """Compute mean intensity of image."""
    total = 0.0
    count = 0
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image.ndim == 3:
                total += np.mean(image[y, x])
            else:
                total += image[y, x]
            count += 1

    return total / count if count > 0 else 0.0
