"""
Image interpolation/resizing algorithms.

Implements three manual resizing methods:
- Nearest Neighbor (fastest, blocky)
- Bilinear (smooth, 4 neighbors)
- Bicubic (smoothest, 16 neighbors)

All implementations are manual without using cv2.resize or similar.
"""

import math
import numpy as np

from app.core.interfaces import ResizeProcessor, InterpolationMethod, ParamInfo


def _cubic_weight(t: float) -> float:
    """
    Compute cubic interpolation weight (Catmull-Rom spline).
    
    Args:
        t: Distance from sample point (0 to 2).
    
    Returns:
        Weight value.
    """
    t = abs(t)
    if t <= 1:
        return 1.5 * t**3 - 2.5 * t**2 + 1
    elif t <= 2:
        return -0.5 * t**3 + 2.5 * t**2 - 4 * t + 2
    return 0.0


def resize_nearest_neighbor(image: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    """
    Resize image using nearest neighbor interpolation.
    
    Fastest method, produces blocky/pixelated results.
    Good for pixel art or when speed is critical.

    Args:
        image: Input image (H, W) or (H, W, C).
        new_width: Target width.
        new_height: Target height.

    Returns:
        Resized image.
    """
    h, w = image.shape[:2]
    is_color = image.ndim == 3
    
    if is_color:
        output = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    else:
        output = np.zeros((new_height, new_width), dtype=image.dtype)
    
    # Scale factors
    scale_y = h / new_height
    scale_x = w / new_width
    
    for out_y in range(new_height):
        for out_x in range(new_width):
            # Map to source coordinates
            src_y = int(out_y * scale_y)
            src_x = int(out_x * scale_x)
            
            # Clamp to valid range
            src_y = min(src_y, h - 1)
            src_x = min(src_x, w - 1)
            
            output[out_y, out_x] = image[src_y, src_x]
    
    return output


def resize_bilinear(image: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    """
    Resize image using bilinear interpolation.
    
    Uses 4 nearest neighbors with linear weights.
    Good balance of quality and speed.

    Args:
        image: Input image (H, W) or (H, W, C).
        new_width: Target width.
        new_height: Target height.

    Returns:
        Resized image.
    """
    h, w = image.shape[:2]
    is_color = image.ndim == 3
    
    if is_color:
        output = np.zeros((new_height, new_width, image.shape[2]), dtype=np.float64)
    else:
        output = np.zeros((new_height, new_width), dtype=np.float64)
    
    # Scale factors
    scale_y = h / new_height
    scale_x = w / new_width
    
    for out_y in range(new_height):
        for out_x in range(new_width):
            # Map to source coordinates (center of pixel)
            src_y = (out_y + 0.5) * scale_y - 0.5
            src_x = (out_x + 0.5) * scale_x - 0.5
            
            # Get integer and fractional parts
            y0 = int(math.floor(src_y))
            x0 = int(math.floor(src_x))
            y1 = y0 + 1
            x1 = x0 + 1
            
            # Fractional weights
            fy = src_y - y0
            fx = src_x - x0
            
            # Clamp coordinates
            y0 = max(0, min(y0, h - 1))
            y1 = max(0, min(y1, h - 1))
            x0 = max(0, min(x0, w - 1))
            x1 = max(0, min(x1, w - 1))
            
            # Get 4 neighbors
            p00 = image[y0, x0].astype(np.float64)
            p01 = image[y0, x1].astype(np.float64)
            p10 = image[y1, x0].astype(np.float64)
            p11 = image[y1, x1].astype(np.float64)
            
            # Bilinear interpolation
            value = (
                p00 * (1 - fy) * (1 - fx) +
                p01 * (1 - fy) * fx +
                p10 * fy * (1 - fx) +
                p11 * fy * fx
            )
            
            output[out_y, out_x] = value
    
    return np.clip(output, 0, 255).astype(image.dtype)


def resize_bicubic(image: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    """
    Resize image using bicubic interpolation.
    
    Uses 16 nearest neighbors (4x4 grid) with cubic weights.
    Highest quality, but slowest.

    Args:
        image: Input image (H, W) or (H, W, C).
        new_width: Target width.
        new_height: Target height.

    Returns:
        Resized image.
    """
    h, w = image.shape[:2]
    is_color = image.ndim == 3
    channels = image.shape[2] if is_color else 1
    
    if is_color:
        output = np.zeros((new_height, new_width, channels), dtype=np.float64)
    else:
        output = np.zeros((new_height, new_width), dtype=np.float64)
    
    # Scale factors
    scale_y = h / new_height
    scale_x = w / new_width
    
    def get_pixel(y: int, x: int) -> np.ndarray:
        """Get pixel with clamped coordinates."""
        y = max(0, min(y, h - 1))
        x = max(0, min(x, w - 1))
        return image[y, x].astype(np.float64)
    
    for out_y in range(new_height):
        for out_x in range(new_width):
            # Map to source coordinates
            src_y = (out_y + 0.5) * scale_y - 0.5
            src_x = (out_x + 0.5) * scale_x - 0.5
            
            # Integer part (center of 4x4 grid)
            y_int = int(math.floor(src_y))
            x_int = int(math.floor(src_x))
            
            # Fractional part
            fy = src_y - y_int
            fx = src_x - x_int
            
            # Accumulate weighted sum over 4x4 neighborhood
            if is_color:
                value = np.zeros(channels, dtype=np.float64)
            else:
                value = 0.0
            
            for j in range(-1, 3):  # -1, 0, 1, 2
                wy = _cubic_weight(fy - j)
                for i in range(-1, 3):
                    wx = _cubic_weight(fx - i)
                    weight = wy * wx
                    
                    pixel = get_pixel(y_int + j, x_int + i)
                    value = value + pixel * weight
            
            output[out_y, out_x] = value
    
    return np.clip(output, 0, 255).astype(image.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Processor Classes
# ─────────────────────────────────────────────────────────────────────────────

class NearestNeighborResizer(ResizeProcessor):
    """Resize using nearest neighbor interpolation."""

    @property
    def name(self) -> str:
        return "Nearest Neighbor"

    @property
    def category(self) -> str:
        return "Interpolation"

    @property
    def method(self) -> InterpolationMethod:
        return InterpolationMethod.NEAREST

    def get_default_params(self) -> dict:
        return {"scale": 2.0, "width": None, "height": None}

    def get_param_info(self) -> dict[str, ParamInfo]:
        return {
            "scale": ParamInfo("Scale Factor", "float", 2.0, 0.1, 10.0,
                              tooltip="Scale factor (overrides width/height)"),
            "width": ParamInfo("Target Width", "int", None, 1, 10000,
                              tooltip="Target width in pixels"),
            "height": ParamInfo("Target Height", "int", None, 1, 10000,
                               tooltip="Target height in pixels"),
        }

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        h, w = image.shape[:2]
        scale = params.get("scale", 2.0)
        new_width = params.get("width") or int(w * scale)
        new_height = params.get("height") or int(h * scale)
        return resize_nearest_neighbor(image, new_width, new_height)


class BilinearResizer(ResizeProcessor):
    """Resize using bilinear interpolation."""

    @property
    def name(self) -> str:
        return "Bilinear"

    @property
    def category(self) -> str:
        return "Interpolation"

    @property
    def method(self) -> InterpolationMethod:
        return InterpolationMethod.BILINEAR

    def get_default_params(self) -> dict:
        return {"scale": 2.0, "width": None, "height": None}

    def get_param_info(self) -> dict[str, ParamInfo]:
        return {
            "scale": ParamInfo("Scale Factor", "float", 2.0, 0.1, 10.0,
                              tooltip="Scale factor"),
            "width": ParamInfo("Target Width", "int", None, 1, 10000),
            "height": ParamInfo("Target Height", "int", None, 1, 10000),
        }

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        h, w = image.shape[:2]
        scale = params.get("scale", 2.0)
        new_width = params.get("width") or int(w * scale)
        new_height = params.get("height") or int(h * scale)
        return resize_bilinear(image, new_width, new_height)


class BicubicResizer(ResizeProcessor):
    """Resize using bicubic interpolation."""

    @property
    def name(self) -> str:
        return "Bicubic"

    @property
    def category(self) -> str:
        return "Interpolation"

    @property
    def method(self) -> InterpolationMethod:
        return InterpolationMethod.BICUBIC

    def get_default_params(self) -> dict:
        return {"scale": 2.0, "width": None, "height": None}

    def get_param_info(self) -> dict[str, ParamInfo]:
        return {
            "scale": ParamInfo("Scale Factor", "float", 2.0, 0.1, 10.0,
                              tooltip="Scale factor"),
            "width": ParamInfo("Target Width", "int", None, 1, 10000),
            "height": ParamInfo("Target Height", "int", None, 1, 10000),
        }

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        h, w = image.shape[:2]
        scale = params.get("scale", 2.0)
        new_width = params.get("width") or int(w * scale)
        new_height = params.get("height") or int(h * scale)
        return resize_bicubic(image, new_width, new_height)


def compare_interpolation_methods(
    image: np.ndarray,
    scale: float = 2.0
) -> dict[str, dict]:
    """
    Compare all interpolation methods on the same image.

    Args:
        image: Input image.
        scale: Scale factor for resizing.

    Returns:
        Dict with method names as keys, containing:
        - 'result': Resized image
        - 'time_ms': Processing time (if measured)
        - 'description': Method characteristics
    """
    import time
    
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    results = {}
    
    # Nearest Neighbor
    start = time.perf_counter()
    nn_result = resize_nearest_neighbor(image, new_w, new_h)
    nn_time = (time.perf_counter() - start) * 1000
    results["Nearest Neighbor"] = {
        "result": nn_result,
        "time_ms": nn_time,
        "description": "Fastest. Blocky/pixelated artifacts. Good for pixel art."
    }
    
    # Bilinear
    start = time.perf_counter()
    bl_result = resize_bilinear(image, new_w, new_h)
    bl_time = (time.perf_counter() - start) * 1000
    results["Bilinear"] = {
        "result": bl_result,
        "time_ms": bl_time,
        "description": "Good balance. Smooth but may blur edges slightly."
    }
    
    # Bicubic
    start = time.perf_counter()
    bc_result = resize_bicubic(image, new_w, new_h)
    bc_time = (time.perf_counter() - start) * 1000
    results["Bicubic"] = {
        "result": bc_result,
        "time_ms": bc_time,
        "description": "Highest quality. Sharpest edges. Slowest."
    }
    
    return results
