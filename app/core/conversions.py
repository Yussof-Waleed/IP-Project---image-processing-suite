"""
Format conversion processors: Grayscale and Binary thresholding.
"""

import numpy as np

from app.core.interfaces import ImageProcessor
from app.core.primitives import rgb_to_grayscale, clip_to_uint8


def compute_otsu_threshold(gray: np.ndarray) -> float:
    """Compute optimal threshold using Otsu's method.
    
    Finds the threshold that minimizes intra-class variance
    (or equivalently maximizes inter-class variance).
    """
    # Compute histogram
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = gray.size
    
    # Precompute cumulative sums
    sum_all = np.sum(np.arange(256) * hist)
    sum_bg = 0.0
    weight_bg = 0.0
    
    max_variance = 0.0
    best_threshold = 0
    
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        
        sum_bg += t * hist[t]
        
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        
        # Inter-class variance
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        
        if variance > max_variance:
            max_variance = variance
            best_threshold = t
    
    return float(best_threshold)


class GrayscaleProcessor(ImageProcessor):
    """Convert RGB to grayscale using luminance formula."""

    @property
    def name(self) -> str:
        return "Grayscale"

    @property
    def category(self) -> str:
        return "Format Conversions"

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        if image.ndim == 2:
            return image.copy()
        return clip_to_uint8(rgb_to_grayscale(image))


class BinaryThresholdProcessor(ImageProcessor):
    """Convert image to binary using thresholding."""

    @property
    def name(self) -> str:
        return "Binary (Threshold)"

    @property
    def category(self) -> str:
        return "Format Conversions"

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        threshold = params.get("threshold")
        invert = params.get("invert", False)
        
        gray = rgb_to_grayscale(image) if image.ndim == 3 else image.astype(np.float64)
        
        if threshold is None:
            threshold = np.mean(gray)
        
        # Vectorized thresholding
        binary = np.where(gray >= threshold, 255, 0).astype(np.uint8)
        if invert:
            binary = 255 - binary
        return binary

    def evaluate_threshold(self, image: np.ndarray, threshold: float | None = None) -> dict:
        """Evaluate if the mean-based threshold is optimal.
        
        Per requirement: Calculate threshold using average pixel intensity,
        then evaluate whether this threshold is optimal or not.
        
        Compares the mean threshold against Otsu's optimal threshold.
        """
        gray = rgb_to_grayscale(image) if image.ndim == 3 else image.astype(np.float64)
        gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
        
        # Check if image is already binary (only 2 unique values)
        unique_values = np.unique(gray_uint8)
        is_binary = len(unique_values) <= 2
        
        if is_binary:
            # Image is already binary - thresholding not applicable
            fg_ratio = float(np.mean(gray > 127))
            return {
                "is_optimal": True,
                "is_binary": True,
                "analysis": "ℹ️ Image is already binary. Threshold evaluation not applicable.",
                "mean_threshold": float(np.mean(gray)),
                "otsu_threshold": 0.0,
                "mean_quality_percent": 100.0,
                "foreground_ratio": fg_ratio,
                "background_ratio": 1 - fg_ratio,
            }
        
        # Calculate threshold using average pixel intensity (as per requirement)
        mean_threshold = float(np.mean(gray))
        
        # Calculate Otsu's optimal threshold for comparison
        otsu_threshold = compute_otsu_threshold(gray_uint8)
        
        # Helper to compute inter-class variance for a given threshold
        def compute_interclass_variance(t: float) -> tuple[float, float, float, float, float]:
            fg_mask = gray > t
            fg_count = np.sum(fg_mask)
            total = gray.size
            
            if fg_count == 0 or fg_count == total:
                return 0.0, fg_count / total, 1 - fg_count / total, 0.0, 0.0
            
            fg_ratio = fg_count / total
            bg_ratio = 1 - fg_ratio
            fg_mean = float(np.mean(gray[fg_mask]))
            bg_mean = float(np.mean(gray[~fg_mask]))
            variance = fg_ratio * bg_ratio * (fg_mean - bg_mean) ** 2
            return variance, fg_ratio, bg_ratio, fg_mean, bg_mean
        
        # Compute stats for mean threshold
        mean_variance, fg_ratio, bg_ratio, fg_mean, bg_mean = compute_interclass_variance(mean_threshold)
        
        # Compute Otsu variance for comparison
        otsu_variance = compute_interclass_variance(otsu_threshold)[0]
        
        # Calculate quality ratio (how close to optimal)
        if otsu_variance > 0:
            quality_ratio = min(mean_variance / otsu_variance, 1.0)
        else:
            quality_ratio = 1.0
        
        # Calculate threshold difference 
        threshold_diff = abs(mean_threshold - otsu_threshold)
        
        # Optimal if: quality >= 95% AND difference <= 12.75 levels
        is_optimal = quality_ratio >= 0.95 and threshold_diff <= 12.75
        
        # Build analysis message
        if is_optimal:
            analysis = f"✅ Mean threshold ({mean_threshold:.0f}) is optimal. Quality: {quality_ratio*100:.0f}%"
        else:
            analysis = f"❌ Mean threshold ({mean_threshold:.0f}) is NOT optimal. Use Otsu ({otsu_threshold:.0f}) instead."
        
        return {
            "is_optimal": is_optimal,
            "analysis": analysis,
            "mean_threshold": mean_threshold,
            "otsu_threshold": otsu_threshold,
            "mean_quality_percent": quality_ratio * 100,
            "foreground_ratio": fg_ratio,
            "background_ratio": bg_ratio,
            "foreground_mean": fg_mean,
            "background_mean": bg_mean,
            "class_separation": fg_mean - bg_mean,
        }
