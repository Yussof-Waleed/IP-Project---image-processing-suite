"""
Histogram processing and analysis.

Implements:
- Manual histogram computation
- Contrast analysis
- Histogram equalization
"""

import numpy as np

from app.core.interfaces import ImageProcessor, HistogramResult, ParamInfo
from app.core.primitives import rgb_to_grayscale, compute_histogram


def analyze_histogram(histogram: np.ndarray) -> HistogramResult:
    """
    Analyze histogram to determine contrast quality.

    Args:
        histogram: 256-bin histogram array.

    Returns:
        HistogramResult with analysis.
    """
    # Compute statistics
    total_pixels = histogram.sum()
    
    # Weighted mean
    mean = 0.0
    for i in range(256):
        mean += i * histogram[i]
    mean /= total_pixels if total_pixels > 0 else 1
    
    # Standard deviation
    variance = 0.0
    for i in range(256):
        variance += histogram[i] * (i - mean) ** 2
    variance /= total_pixels if total_pixels > 0 else 1
    std = np.sqrt(variance)
    
    # Find min/max non-zero values
    min_val = 0
    max_val = 255
    for i in range(256):
        if histogram[i] > 0:
            min_val = i
            break
    for i in range(255, -1, -1):
        if histogram[i] > 0:
            max_val = i
            break
    
    # Analyze contrast
    dynamic_range = max_val - min_val
    
    # Low contrast indicators:
    # 1. Small dynamic range (< 100)
    # 2. Low standard deviation (< 40)
    # 3. Most pixels concentrated in small region
    
    is_low_contrast = dynamic_range < 100 or std < 40
    
    if is_low_contrast:
        if dynamic_range < 50:
            analysis = (
                f"Very low contrast. Dynamic range only {dynamic_range} levels "
                f"({min_val}-{max_val}). Histogram equalization strongly recommended."
            )
        else:
            analysis = (
                f"Low contrast detected. Standard deviation is {std:.1f} "
                f"(good contrast typically > 50). Consider histogram equalization."
            )
    else:
        if std > 70:
            analysis = (
                f"Good contrast. Wide distribution with σ={std:.1f}. "
                f"Dynamic range: {min_val}-{max_val} ({dynamic_range} levels)."
            )
        else:
            analysis = (
                f"Moderate contrast. σ={std:.1f}, range {min_val}-{max_val}. "
                f"Equalization may slightly improve appearance."
            )
    
    return HistogramResult(
        histogram=histogram,
        mean=mean,
        std=std,
        min_val=min_val,
        max_val=max_val,
        is_low_contrast=is_low_contrast,
        contrast_analysis=analysis,
    )


def compute_cdf(histogram: np.ndarray) -> np.ndarray:
    """Compute cumulative distribution function from histogram."""
    cdf = np.zeros(256, dtype=np.float64)
    cumsum = 0
    total = histogram.sum()
    
    for i in range(256):
        cumsum += histogram[i]
        cdf[i] = cumsum / total if total > 0 else 0
    
    return cdf


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization manually.

    Args:
        image: Grayscale image (H, W) with values 0-255.

    Returns:
        Equalized image.
    """
    if image.ndim != 2:
        raise ValueError("Image must be grayscale for histogram equalization")
    
    h, w = image.shape
    
    # Compute histogram
    histogram = compute_histogram(image)
    
    # Compute CDF
    cdf = compute_cdf(histogram)
    
    # Find minimum non-zero CDF value
    cdf_min = 0
    for i in range(256):
        if cdf[i] > 0:
            cdf_min = cdf[i]
            break
    
    # Build lookup table for equalization
    # Formula: new_value = round((cdf(v) - cdf_min) / (1 - cdf_min) * 255)
    lut = np.zeros(256, dtype=np.uint8)
    denominator = 1 - cdf_min
    
    for i in range(256):
        if denominator > 0:
            lut[i] = int(round((cdf[i] - cdf_min) / denominator * 255))
        else:
            lut[i] = i
        lut[i] = max(0, min(255, lut[i]))
    
    # Apply lookup table
    output = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            output[y, x] = lut[image[y, x]]
    
    return output


class HistogramProcessor(ImageProcessor):
    """Compute and display histogram."""

    @property
    def name(self) -> str:
        return "Histogram"

    @property
    def category(self) -> str:
        return "Histogram"

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        """Return grayscale version for histogram computation."""
        if image.ndim == 3:
            return rgb_to_grayscale(image).astype(np.uint8)
        return image

    def compute_histogram(self, image: np.ndarray) -> HistogramResult:
        """Compute histogram and analyze contrast."""
        if image.ndim == 3:
            gray = rgb_to_grayscale(image).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        
        histogram = compute_histogram(gray)
        return analyze_histogram(histogram)


class HistogramEqualizationProcessor(ImageProcessor):
    """Apply histogram equalization."""

    @property
    def name(self) -> str:
        return "Histogram Equalization"

    @property
    def category(self) -> str:
        return "Histogram"

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        """Apply histogram equalization."""
        if image.ndim == 3:
            # Convert to grayscale first
            gray = rgb_to_grayscale(image).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        
        return equalize_histogram(gray)
