"""
Image format conversion processors.

Implements:
- Grayscale conversion (manual luminance formula)
- Binary thresholding with automatic threshold evaluation
"""

import numpy as np

from app.core.interfaces import ImageProcessor, ParamInfo
from app.core.primitives import rgb_to_grayscale, clip_to_uint8


class GrayscaleProcessor(ImageProcessor):
    """Convert RGB image to grayscale using luminance formula."""

    @property
    def name(self) -> str:
        return "Grayscale"

    @property
    def category(self) -> str:
        return "Format Conversions"

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Convert image to grayscale.

        Args:
            image: RGB image (H, W, 3) or grayscale (H, W).

        Returns:
            Grayscale image (H, W) as uint8.
        """
        if image.ndim == 2:
            # Already grayscale
            return image.copy()

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input must be RGB (H, W, 3) or grayscale (H, W)")

        grayscale = rgb_to_grayscale(image)
        return clip_to_uint8(grayscale)


class BinaryThresholdProcessor(ImageProcessor):
    """
    Convert image to binary using thresholding.
    
    Supports automatic threshold (mean intensity) with optimality evaluation.
    """

    @property
    def name(self) -> str:
        return "Binary (Threshold)"

    @property
    def category(self) -> str:
        return "Format Conversions"

    def get_default_params(self) -> dict:
        return {
            "threshold": None,  # None = auto (mean intensity)
            "invert": False,
        }

    def get_param_info(self) -> dict[str, ParamInfo]:
        return {
            "threshold": ParamInfo(
                label="Threshold",
                param_type="int",
                default=None,
                min_val=0,
                max_val=255,
                tooltip="Leave empty for automatic (mean intensity)"
            ),
            "invert": ParamInfo(
                label="Invert",
                param_type="bool",
                default=False,
                tooltip="Invert binary output"
            ),
        }

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Convert image to binary.

        Args:
            image: Input image (grayscale or RGB).
            threshold: Threshold value (0-255). None = auto (mean).
            invert: If True, invert the result.

        Returns:
            Binary image (H, W) with values 0 or 255.
        """
        threshold = params.get("threshold")
        invert = params.get("invert", False)

        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = rgb_to_grayscale(image)
        else:
            gray = image.astype(np.float64)

        # Auto threshold using mean intensity
        if threshold is None:
            threshold = self._compute_mean_threshold(gray)

        # Apply thresholding manually
        h, w = gray.shape
        binary = np.zeros((h, w), dtype=np.uint8)

        for y in range(h):
            for x in range(w):
                if gray[y, x] >= threshold:
                    binary[y, x] = 255 if not invert else 0
                else:
                    binary[y, x] = 0 if not invert else 255

        return binary

    def _compute_mean_threshold(self, gray: np.ndarray) -> float:
        """Compute mean intensity as threshold."""
        total = 0.0
        count = gray.shape[0] * gray.shape[1]

        for y in range(gray.shape[0]):
            for x in range(gray.shape[1]):
                total += gray[y, x]

        return total / count if count > 0 else 128.0

    def evaluate_threshold(self, image: np.ndarray, threshold: float | None = None) -> dict:
        """
        Evaluate if the threshold is optimal for the image.

        Returns analysis dict with:
        - threshold_used: The threshold value
        - is_optimal: Boolean indicating if threshold seems good
        - analysis: Text explanation
        - foreground_ratio: Ratio of foreground pixels
        - background_ratio: Ratio of background pixels
        """
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = rgb_to_grayscale(image)
        else:
            gray = image.astype(np.float64)

        if threshold is None:
            threshold = self._compute_mean_threshold(gray)

        # Count pixels above/below threshold
        h, w = gray.shape
        total = h * w
        foreground = 0

        for y in range(h):
            for x in range(w):
                if gray[y, x] >= threshold:
                    foreground += 1

        background = total - foreground
        fg_ratio = foreground / total
        bg_ratio = background / total

        # Evaluate optimality
        # Good threshold typically separates image into meaningful regions
        # Very unbalanced ratios (< 5% or > 95%) often indicate poor threshold
        is_balanced = 0.05 <= fg_ratio <= 0.95

        # Check if threshold is near histogram peaks (simple check)
        # Compare with Otsu-like variance analysis
        variance_quality = self._compute_inter_class_variance(gray, threshold)

        is_optimal = is_balanced and variance_quality > 0.1

        if is_optimal:
            analysis = (
                f"Threshold {threshold:.1f} appears optimal. "
                f"Good separation with {fg_ratio*100:.1f}% foreground."
            )
        elif not is_balanced:
            dominant = "foreground" if fg_ratio > 0.5 else "background"
            analysis = (
                f"Threshold {threshold:.1f} may not be optimal. "
                f"Image is {fg_ratio*100:.1f}% foreground - heavily {dominant} dominant. "
                f"Consider adjusting threshold manually."
            )
        else:
            analysis = (
                f"Threshold {threshold:.1f} provides moderate separation "
                f"({fg_ratio*100:.1f}% foreground) but inter-class variance is low. "
                f"Try Otsu's method or manual adjustment for better results."
            )

        return {
            "threshold_used": threshold,
            "is_optimal": is_optimal,
            "analysis": analysis,
            "foreground_ratio": fg_ratio,
            "background_ratio": bg_ratio,
            "variance_quality": variance_quality,
        }

    def _compute_inter_class_variance(self, gray: np.ndarray, threshold: float) -> float:
        """
        Compute normalized inter-class variance (Otsu-like metric).
        Higher values indicate better class separation.
        """
        h, w = gray.shape

        # Compute class means
        sum_bg, count_bg = 0.0, 0
        sum_fg, count_fg = 0.0, 0

        for y in range(h):
            for x in range(w):
                val = gray[y, x]
                if val < threshold:
                    sum_bg += val
                    count_bg += 1
                else:
                    sum_fg += val
                    count_fg += 1

        if count_bg == 0 or count_fg == 0:
            return 0.0

        mean_bg = sum_bg / count_bg
        mean_fg = sum_fg / count_fg

        total = h * w
        weight_bg = count_bg / total
        weight_fg = count_fg / total

        # Inter-class variance
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        # Normalize by max possible variance (when means are 0 and 255)
        max_variance = 0.25 * 255 * 255  # max when weights = 0.5 each
        return variance / max_variance if max_variance > 0 else 0.0
