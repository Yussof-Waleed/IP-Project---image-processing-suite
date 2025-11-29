"""
Core image processing package.

Contains all processing logic, separated from GUI concerns.
"""

from app.core.io import load_image, save_image, get_image_metadata, ImageMetadata
from app.core.interfaces import (
    ImageProcessor,
    ConvolutionProcessor,
    TransformProcessor,
    ResizeProcessor,
    CompressionProcessor,
    InterpolationMethod,
    ParamInfo,
    HistogramResult,
    CompressionResult,
    ProcessorRegistry,
    default_registry,
)
from app.core.primitives import (
    BorderMode,
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
    "load_image",
    "save_image",
    "get_image_metadata",
    "ImageMetadata",
    # Interfaces
    "ImageProcessor",
    "ConvolutionProcessor",
    "TransformProcessor",
    "ResizeProcessor",
    "CompressionProcessor",
    "InterpolationMethod",
    "ParamInfo",
    "HistogramResult",
    "CompressionResult",
    "ProcessorRegistry",
    "default_registry",
    # Primitives
    "BorderMode",
    "convolve2d",
    "apply_median_filter",
    "build_gaussian_kernel",
    "build_laplacian_kernel",
    "build_sobel_kernels",
    "rgb_to_grayscale",
    "normalize_to_uint8",
    "clip_to_uint8",
    "compute_histogram",
    # Conversions
    "GrayscaleProcessor",
    "BinaryThresholdProcessor",
]
