"""
Tests for core.io module.
"""

import numpy as np
import pytest
from pathlib import Path

from app.core.io import load_image, save_image, get_image_metadata, ImageMetadata


class TestImageMetadata:
    """Tests for ImageMetadata dataclass."""

    def test_file_size_str_bytes(self):
        meta = ImageMetadata(100, 100, 3, 512, "PNG")
        assert meta.file_size_str == "512.0 B"

    def test_file_size_str_kilobytes(self):
        meta = ImageMetadata(100, 100, 3, 2048, "PNG")
        assert meta.file_size_str == "2.0 KB"

    def test_file_size_str_megabytes(self):
        meta = ImageMetadata(100, 100, 3, 1024 * 1024 * 5, "PNG")
        assert meta.file_size_str == "5.0 MB"


class TestLoadImage:
    """Tests for load_image function."""

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image(Path("nonexistent_image.png"))

    def test_load_invalid_file_raises(self, tmp_path):
        # Create a text file, not an image
        fake = tmp_path / "fake.png"
        fake.write_text("not an image")
        with pytest.raises(ValueError):
            load_image(fake)


class TestSaveImage:
    """Tests for save_image function."""

    def test_save_and_reload_rgb(self, tmp_path):
        # Create a simple RGB image
        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        arr[10:40, 10:40] = [255, 0, 0]  # Red square
        
        out_path = tmp_path / "test_rgb.png"
        save_image(arr, out_path)
        
        assert out_path.exists()
        
        # Reload and verify
        loaded = load_image(out_path)
        assert loaded.shape == (50, 50, 3)
        assert np.array_equal(loaded[25, 25], [255, 0, 0])

    def test_save_grayscale(self, tmp_path):
        arr = np.zeros((30, 30), dtype=np.uint8)
        arr[5:25, 5:25] = 128
        
        out_path = tmp_path / "test_gray.png"
        save_image(arr, out_path)
        
        assert out_path.exists()

    def test_save_creates_directories(self, tmp_path):
        arr = np.zeros((10, 10), dtype=np.uint8)
        nested_path = tmp_path / "a" / "b" / "c" / "img.png"
        
        save_image(arr, nested_path)
        assert nested_path.exists()

    def test_save_invalid_dims_raises(self, tmp_path):
        arr = np.zeros((10,), dtype=np.uint8)  # 1D array
        with pytest.raises(ValueError):
            save_image(arr, tmp_path / "bad.png")
