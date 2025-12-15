"""
Histogram processing: analysis and equalization.
"""

import numpy as np

from app.core.interfaces import ImageProcessor, HistogramResult
from app.core.primitives import rgb_to_grayscale, compute_histogram


def analyze_histogram(histogram: np.ndarray) -> HistogramResult:
    """Analyze histogram to determine contrast quality."""
    total = histogram.sum()
    bins = np.arange(256)
    
    mean = np.sum(bins * histogram) / total if total > 0 else 0
    std = np.sqrt(np.sum(histogram * (bins - mean) ** 2) / total) if total > 0 else 0
    
    nonzero = np.where(histogram > 0)[0]
    min_val = int(nonzero[0]) if len(nonzero) > 0 else 0
    max_val = int(nonzero[-1]) if len(nonzero) > 0 else 255
    
    dynamic_range = max_val - min_val
    is_low_contrast = dynamic_range < 100 or std < 40
    
    if is_low_contrast:
        analysis = f"Low contrast (range={dynamic_range}, σ={std:.1f}). Equalization recommended."
    else:
        analysis = f"Good contrast (range={dynamic_range}, σ={std:.1f})."
    
    return HistogramResult(
        histogram=histogram, mean=mean, std=std,
        min_val=min_val, max_val=max_val,
        is_low_contrast=is_low_contrast, contrast_analysis=analysis
    )


def compute_cdf(histogram: np.ndarray) -> np.ndarray:
    """Compute cumulative distribution function from histogram."""
    cdf = np.cumsum(histogram).astype(np.float64)
    return cdf / cdf[-1] if cdf[-1] > 0 else cdf


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """Apply histogram equalization."""
    if image.ndim != 2:
        raise ValueError("Image must be grayscale")
    
    histogram = compute_histogram(image)
    cdf = compute_cdf(histogram)
    
    # Find min non-zero CDF
    cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
    
    # Build lookup table
    denom = 1 - cdf_min
    if denom > 0:
        lut = np.round((cdf - cdf_min) / denom * 255).astype(np.uint8)
    else:
        lut = np.arange(256, dtype=np.uint8)
    
    # Apply LUT using indexing
    return lut[image.astype(np.uint8)]


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
