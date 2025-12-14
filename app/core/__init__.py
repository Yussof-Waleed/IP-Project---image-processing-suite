"""
Core image processing package.
"""

from app.core.io import load_image, save_image, get_image_metadata, ImageMetadata
from app.core.interfaces import ImageProcessor, ParamInfo, HistogramResult
from app.core.primitives import (
    convolve2d,
    apply_median_filter,
    build_gaussian_kernel,
    build_laplacian_kernel,
    build_sobel_kernels,
    rgb_to_grayscale,
    normalize_to_uint8,
    clip_to_uint8,
    compute_histogram,
)
from app.core.conversions import GrayscaleProcessor, BinaryThresholdProcessor

__all__ = [
    # I/O
    "load_image", "save_image", "get_image_metadata", "ImageMetadata",
    # Interfaces
    "ImageProcessor", "ParamInfo", "HistogramResult",
    # Primitives
    "convolve2d", "apply_median_filter",
    "build_gaussian_kernel", "build_laplacian_kernel", "build_sobel_kernels",
    "rgb_to_grayscale", "normalize_to_uint8", "clip_to_uint8", "compute_histogram",
    # Conversions
    "GrayscaleProcessor", "BinaryThresholdProcessor",
]
