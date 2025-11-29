"""
Image I/O operations.

Responsibilities:
- Load images from disk into numpy arrays
- Save numpy arrays to disk as images
- Extract image metadata

Follows CQS: load/get functions are queries, save is a command.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageMetadata:
    """Immutable container for image metadata (query result)."""
    width: int
    height: int
    channels: int
    file_size_bytes: int
    format: str

    @property
    def file_size_str(self) -> str:
        """Human-readable file size."""
        size = self.file_size_bytes
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"


def load_image(path: Path | str) -> np.ndarray:
    """
    Load an image from disk as a numpy array (RGB, uint8).

    Args:
        path: Path to the image file.

    Returns:
        numpy array of shape (H, W, 3) for color or (H, W) for grayscale.
        Values are uint8 in range [0, 255].

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file is not a valid image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        with Image.open(path) as img:
            # Convert to RGB if necessary (handles RGBA, palette, etc.)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            elif img.mode == "L":
                # Keep grayscale as-is
                return np.array(img, dtype=np.uint8)
            elif img.mode != "RGB":
                img = img.convert("RGB")
            
            return np.array(img, dtype=np.uint8)
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}") from e


def save_image(image: np.ndarray, path: Path | str) -> None:
    """
    Save a numpy array as an image file (command).

    Args:
        image: numpy array (H, W) for grayscale or (H, W, 3) for RGB.
        path: Destination path (format inferred from extension).

    Raises:
        ValueError: If array shape is invalid.
    """
    path = Path(path)
    
    if image.ndim not in (2, 3):
        raise ValueError(f"Invalid image dimensions: {image.ndim}")
    
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"Invalid number of channels: {image.shape[2]}")

    # Ensure uint8
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Handle single-channel 3D arrays
    if image.ndim == 3 and image.shape[2] == 1:
        image = image.squeeze(axis=2)

    img = Image.fromarray(image)
    
    # Create parent directories if needed
    path.parent.mkdir(parents=True, exist_ok=True)
    
    img.save(path)


def get_image_metadata(path: Path | str) -> ImageMetadata:
    """
    Extract metadata from an image file without loading full data (query).

    Args:
        path: Path to the image file.

    Returns:
        ImageMetadata with resolution, channels, file size, and format.
    """
    path = Path(path)
    
    file_size = path.stat().st_size
    
    with Image.open(path) as img:
        width, height = img.size
        
        # Determine channel count
        mode_channels = {
            "1": 1, "L": 1, "P": 1,
            "RGB": 3, "RGBA": 4,
            "CMYK": 4, "YCbCr": 3,
            "LAB": 3, "HSV": 3,
            "I": 1, "F": 1,
        }
        channels = mode_channels.get(img.mode, 3)
        
        img_format = img.format or path.suffix.upper().lstrip(".")

    return ImageMetadata(
        width=width,
        height=height,
        channels=channels,
        file_size_bytes=file_size,
        format=img_format,
    )
