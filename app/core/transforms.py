"""
Affine transformation processors.

Implements manual affine transformations:
- Translation
- Scaling
- Rotation
- Shear X / Shear Y

All transformations use homogeneous coordinates (3x3 matrices).
Interpolation is pluggable (defaults to nearest neighbor).
"""

import math
import numpy as np

from app.core.interfaces import TransformProcessor, ParamInfo, InterpolationMethod


def _nearest_neighbor_sample(image: np.ndarray, y: float, x: float) -> np.ndarray | float:
    """Sample image at (y, x) using nearest neighbor interpolation."""
    h, w = image.shape[:2]
    yi, xi = int(round(y)), int(round(x))
    
    if 0 <= yi < h and 0 <= xi < w:
        return image[yi, xi]
    
    # Out of bounds - return black
    if image.ndim == 3:
        return np.zeros(image.shape[2], dtype=image.dtype)
    return 0


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
    output_shape: tuple[int, int] | None = None,
    interpolation: InterpolationMethod = InterpolationMethod.BILINEAR
) -> np.ndarray:
    """
    Apply affine transformation to image using inverse mapping.

    Args:
        image: Input image (H, W) or (H, W, C).
        matrix: 3x3 affine transformation matrix.
        output_shape: (height, width) of output. None = same as input.
        interpolation: Interpolation method to use.

    Returns:
        Transformed image.
    """
    h, w = image.shape[:2]
    out_h, out_w = output_shape if output_shape else (h, w)
    
    # Create output array
    if image.ndim == 3:
        output = np.zeros((out_h, out_w, image.shape[2]), dtype=image.dtype)
    else:
        output = np.zeros((out_h, out_w), dtype=image.dtype)
    
    # Compute inverse matrix for reverse mapping
    try:
        inv_matrix = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return output  # Singular matrix, return empty
    
    # Select interpolation function
    if interpolation == InterpolationMethod.NEAREST:
        sample_fn = _nearest_neighbor_sample
    else:
        sample_fn = _bilinear_sample
    
    # Apply inverse mapping
    for out_y in range(out_h):
        for out_x in range(out_w):
            # Transform output coordinates to input coordinates
            src = inv_matrix @ np.array([out_x, out_y, 1])
            src_x, src_y = src[0], src[1]
            
            # Sample from source image
            value = sample_fn(image, src_y, src_x)
            
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

class TranslationProcessor(TransformProcessor):
    """Translate image by (tx, ty) pixels."""

    @property
    def name(self) -> str:
        return "Translation"

    @property
    def category(self) -> str:
        return "Affine Transformations"

    def get_default_params(self) -> dict:
        return {"tx": 0, "ty": 0}

    def get_param_info(self) -> dict[str, ParamInfo]:
        return {
            "tx": ParamInfo("X Offset", "int", 0, -1000, 1000, tooltip="Horizontal shift"),
            "ty": ParamInfo("Y Offset", "int", 0, -1000, 1000, tooltip="Vertical shift"),
        }

    def get_transform_matrix(self, image_shape: tuple, **params) -> np.ndarray:
        tx = params.get("tx", 0)
        ty = params.get("ty", 0)
        return translation_matrix(tx, ty)

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        matrix = self.get_transform_matrix(image.shape, **params)
        return apply_transform(image, matrix)


class ScalingProcessor(TransformProcessor):
    """Scale image by (sx, sy) factors."""

    @property
    def name(self) -> str:
        return "Scaling"

    @property
    def category(self) -> str:
        return "Affine Transformations"

    def get_default_params(self) -> dict:
        return {"sx": 1.0, "sy": 1.0, "center": True}

    def get_param_info(self) -> dict[str, ParamInfo]:
        return {
            "sx": ParamInfo("Scale X", "float", 1.0, 0.1, 5.0, tooltip="Horizontal scale factor"),
            "sy": ParamInfo("Scale Y", "float", 1.0, 0.1, 5.0, tooltip="Vertical scale factor"),
            "center": ParamInfo("From Center", "bool", True, tooltip="Scale from image center"),
        }

    def get_transform_matrix(self, image_shape: tuple, **params) -> np.ndarray:
        sx = params.get("sx", 1.0)
        sy = params.get("sy", 1.0)
        center_flag = params.get("center", True)
        
        h, w = image_shape[:2]
        center = (w / 2, h / 2) if center_flag else (0, 0)
        return scaling_matrix(sx, sy, center)

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        sx = params.get("sx", 1.0)
        sy = params.get("sy", 1.0)
        
        h, w = image.shape[:2]
        new_h = int(h * sy)
        new_w = int(w * sx)
        
        matrix = self.get_transform_matrix(image.shape, **params)
        return apply_transform(image, matrix, (new_h, new_w))


class RotationProcessor(TransformProcessor):
    """Rotate image by angle degrees around center."""

    @property
    def name(self) -> str:
        return "Rotation"

    @property
    def category(self) -> str:
        return "Affine Transformations"

    def get_default_params(self) -> dict:
        return {"angle": 0.0, "expand": True}

    def get_param_info(self) -> dict[str, ParamInfo]:
        return {
            "angle": ParamInfo("Angle (°)", "float", 0.0, -360, 360, tooltip="Rotation angle in degrees"),
            "expand": ParamInfo("Expand Canvas", "bool", True, tooltip="Expand output to fit rotated image"),
        }

    def get_transform_matrix(self, image_shape: tuple, **params) -> np.ndarray:
        angle = params.get("angle", 0.0)
        h, w = image_shape[:2]
        center = (w / 2, h / 2)
        return rotation_matrix(angle, center)

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        angle = params.get("angle", 0.0)
        expand = params.get("expand", True)
        
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        
        if expand:
            # Calculate new dimensions to fit rotated image
            angle_rad = math.radians(abs(angle))
            cos_a = abs(math.cos(angle_rad))
            sin_a = abs(math.sin(angle_rad))
            new_w = int(w * cos_a + h * sin_a)
            new_h = int(h * cos_a + w * sin_a)
            
            # Adjust matrix for new dimensions
            # Translate to new center
            new_center = (new_w / 2, new_h / 2)
            
            # Build composite matrix
            to_origin = translation_matrix(-center[0], -center[1])
            angle_rad = math.radians(angle)
            rotate = np.array([
                [math.cos(angle_rad), -math.sin(angle_rad), 0],
                [math.sin(angle_rad), math.cos(angle_rad), 0],
                [0, 0, 1]
            ], dtype=np.float64)
            to_new_center = translation_matrix(new_center[0], new_center[1])
            
            matrix = to_new_center @ rotate @ to_origin
            return apply_transform(image, matrix, (new_h, new_w))
        else:
            matrix = self.get_transform_matrix(image.shape, **params)
            return apply_transform(image, matrix)


class ShearXProcessor(TransformProcessor):
    """Shear image horizontally."""

    @property
    def name(self) -> str:
        return "Shear X"

    @property
    def category(self) -> str:
        return "Affine Transformations"

    def get_default_params(self) -> dict:
        return {"shear": 0.0}

    def get_param_info(self) -> dict[str, ParamInfo]:
        return {
            "shear": ParamInfo("Shear Factor", "float", 0.0, -2.0, 2.0, tooltip="Horizontal shear factor"),
        }

    def get_transform_matrix(self, image_shape: tuple, **params) -> np.ndarray:
        shear = params.get("shear", 0.0)
        h, w = image_shape[:2]
        center = (w / 2, h / 2)
        return shear_x_matrix(shear, center)

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        shear = params.get("shear", 0.0)
        h, w = image.shape[:2]
        
        # Expand width to fit sheared image
        extra_w = int(abs(shear) * h)
        new_w = w + extra_w
        
        # Adjust matrix to center result
        center = (w / 2, h / 2)
        matrix = shear_x_matrix(shear, center)
        
        # Shift to keep image centered in new canvas
        if shear > 0:
            shift = translation_matrix(extra_w / 2, 0)
        else:
            shift = translation_matrix(extra_w / 2, 0)
        
        matrix = shift @ matrix
        return apply_transform(image, matrix, (h, new_w))


class ShearYProcessor(TransformProcessor):
    """Shear image vertically."""

    @property
    def name(self) -> str:
        return "Shear Y"

    @property
    def category(self) -> str:
        return "Affine Transformations"

    def get_default_params(self) -> dict:
        return {"shear": 0.0}

    def get_param_info(self) -> dict[str, ParamInfo]:
        return {
            "shear": ParamInfo("Shear Factor", "float", 0.0, -2.0, 2.0, tooltip="Vertical shear factor"),
        }

    def get_transform_matrix(self, image_shape: tuple, **params) -> np.ndarray:
        shear = params.get("shear", 0.0)
        h, w = image_shape[:2]
        center = (w / 2, h / 2)
        return shear_y_matrix(shear, center)

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        shear = params.get("shear", 0.0)
        h, w = image.shape[:2]
        
        # Expand height to fit sheared image
        extra_h = int(abs(shear) * w)
        new_h = h + extra_h
        
        # Adjust matrix to center result
        center = (w / 2, h / 2)
        matrix = shear_y_matrix(shear, center)
        
        # Shift to keep image centered in new canvas
        shift = translation_matrix(0, extra_h / 2)
        matrix = shift @ matrix
        
        return apply_transform(image, matrix, (new_h, w))
