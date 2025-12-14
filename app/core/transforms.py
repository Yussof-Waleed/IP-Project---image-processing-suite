"""
Affine transformation processors.

Translation, Scaling, Rotation, Shear X/Y
All use 3x3 matrices with inverse mapping.
"""

import math
import numpy as np

from app.core.interfaces import ImageProcessor


def _bilinear_sample(image: np.ndarray, y: float, x: float) -> np.ndarray | float:
    """Sample image at (y, x) using bilinear interpolation."""
    h, w = image.shape[:2]
    
    # Get the four neighboring pixels
    y0, x0 = int(math.floor(y)), int(math.floor(x))
    y1, x1 = y0 + 1, x0 + 1
    
    # Compute weights
    wy = y - y0
    wx = x - x0
    
    # Check bounds and get pixel values
    def get_pixel(yi, xi):
        if 0 <= yi < h and 0 <= xi < w:
            return image[yi, xi].astype(np.float64)
        if image.ndim == 3:
            return np.zeros(image.shape[2], dtype=np.float64)
        return 0.0
    
    # Bilinear interpolation
    p00 = get_pixel(y0, x0)
    p01 = get_pixel(y0, x1)
    p10 = get_pixel(y1, x0)
    p11 = get_pixel(y1, x1)
    
    result = (
        p00 * (1 - wy) * (1 - wx) +
        p01 * (1 - wy) * wx +
        p10 * wy * (1 - wx) +
        p11 * wy * wx
    )
    
    return result


def apply_transform(
    image: np.ndarray,
    matrix: np.ndarray,
    output_shape: tuple[int, int] | None = None
) -> np.ndarray:
    """Apply affine transformation using bilinear interpolation."""
    h, w = image.shape[:2]
    out_h, out_w = output_shape if output_shape else (h, w)
    
    if image.ndim == 3:
        output = np.zeros((out_h, out_w, image.shape[2]), dtype=image.dtype)
    else:
        output = np.zeros((out_h, out_w), dtype=image.dtype)
    
    try:
        inv_matrix = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return output
    
    for out_y in range(out_h):
        for out_x in range(out_w):
            src = inv_matrix @ np.array([out_x, out_y, 1])
            value = _bilinear_sample(image, src[1], src[0])
            
            if image.ndim == 3:
                output[out_y, out_x] = np.clip(value, 0, 255).astype(image.dtype)
            else:
                output[out_y, out_x] = int(np.clip(value, 0, 255))
    
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Transformation Matrix Builders
# ─────────────────────────────────────────────────────────────────────────────

def translation_matrix(tx: float, ty: float) -> np.ndarray:
    """Create translation matrix."""
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float64)


def scaling_matrix(sx: float, sy: float, center: tuple[float, float] = (0, 0)) -> np.ndarray:
    """Create scaling matrix around a center point."""
    cx, cy = center
    # Translate to origin, scale, translate back
    to_origin = translation_matrix(-cx, -cy)
    scale = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    from_origin = translation_matrix(cx, cy)
    return from_origin @ scale @ to_origin


def rotation_matrix(angle_degrees: float, center: tuple[float, float] = (0, 0)) -> np.ndarray:
    """Create rotation matrix around a center point."""
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    cx, cy = center
    
    # Translate to origin, rotate, translate back
    to_origin = translation_matrix(-cx, -cy)
    rotate = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    from_origin = translation_matrix(cx, cy)
    return from_origin @ rotate @ to_origin


def shear_x_matrix(shear: float, center: tuple[float, float] = (0, 0)) -> np.ndarray:
    """Create horizontal shear matrix."""
    cx, cy = center
    to_origin = translation_matrix(-cx, -cy)
    shear_mat = np.array([
        [1, shear, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    from_origin = translation_matrix(cx, cy)
    return from_origin @ shear_mat @ to_origin


def shear_y_matrix(shear: float, center: tuple[float, float] = (0, 0)) -> np.ndarray:
    """Create vertical shear matrix."""
    cx, cy = center
    to_origin = translation_matrix(-cx, -cy)
    shear_mat = np.array([
        [1, 0, 0],
        [shear, 1, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    from_origin = translation_matrix(cx, cy)
    return from_origin @ shear_mat @ to_origin


# ─────────────────────────────────────────────────────────────────────────────
# Processor Classes
# ─────────────────────────────────────────────────────────────────────────────

class TranslationProcessor(ImageProcessor):
    """Translate image by (tx, ty) pixels."""

    @property
    def name(self) -> str:
        return "Translation"

    @property
    def category(self) -> str:
        return "Affine Transformations"

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        tx, ty = params.get("tx", 0), params.get("ty", 0)
        return apply_transform(image, translation_matrix(tx, ty))


class ScalingProcessor(ImageProcessor):
    """Scale image by (sx, sy) factors."""

    @property
    def name(self) -> str:
        return "Scaling"

    @property
    def category(self) -> str:
        return "Affine Transformations"

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        sx, sy = params.get("sx", 1.0), params.get("sy", 1.0)
        h, w = image.shape[:2]
        center = (w / 2, h / 2) if params.get("center", True) else (0, 0)
        matrix = scaling_matrix(sx, sy, center)
        return apply_transform(image, matrix, (int(h * sy), int(w * sx)))


class RotationProcessor(ImageProcessor):
    """Rotate image by angle degrees around center."""

    @property
    def name(self) -> str:
        return "Rotation"

    @property
    def category(self) -> str:
        return "Affine Transformations"

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        angle = params.get("angle", 0.0)
        expand = params.get("expand", True)
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        
        if expand:
            angle_rad = math.radians(abs(angle))
            cos_a, sin_a = abs(math.cos(angle_rad)), abs(math.sin(angle_rad))
            new_w, new_h = int(w * cos_a + h * sin_a), int(h * cos_a + w * sin_a)
            new_center = (new_w / 2, new_h / 2)
            
            angle_rad = math.radians(angle)
            to_origin = translation_matrix(-center[0], -center[1])
            rotate = np.array([
                [math.cos(angle_rad), -math.sin(angle_rad), 0],
                [math.sin(angle_rad), math.cos(angle_rad), 0],
                [0, 0, 1]
            ], dtype=np.float64)
            to_new_center = translation_matrix(new_center[0], new_center[1])
            matrix = to_new_center @ rotate @ to_origin
            return apply_transform(image, matrix, (new_h, new_w))
        else:
            return apply_transform(image, rotation_matrix(angle, center))


class ShearXProcessor(ImageProcessor):
    """Shear image horizontally."""

    @property
    def name(self) -> str:
        return "Shear X"

    @property
    def category(self) -> str:
        return "Affine Transformations"

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        shear = params.get("shear", 0.0)
        h, w = image.shape[:2]
        extra_w = int(abs(shear) * h)
        matrix = shear_x_matrix(shear, (w / 2, h / 2))
        matrix = translation_matrix(extra_w / 2, 0) @ matrix
        return apply_transform(image, matrix, (h, w + extra_w))


class ShearYProcessor(ImageProcessor):
    """Shear image vertically."""

    @property
    def name(self) -> str:
        return "Shear Y"

    @property
    def category(self) -> str:
        return "Affine Transformations"

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        shear = params.get("shear", 0.0)
        h, w = image.shape[:2]
        extra_h = int(abs(shear) * w)
        matrix = shear_y_matrix(shear, (w / 2, h / 2))
        matrix = translation_matrix(0, extra_h / 2) @ matrix
        return apply_transform(image, matrix, (h + extra_h, w))
