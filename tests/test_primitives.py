"""
Tests for core.primitives module.
"""

import numpy as np
import pytest

from app.core.primitives import (
    BorderMode,
    get_pixel_safe,
    pad_image,
    convolve2d,
    apply_median_filter,
    build_gaussian_kernel,
    build_box_kernel,
    build_laplacian_kernel,
    build_sobel_kernels,
    rgb_to_grayscale,
    normalize_to_uint8,
    clip_to_uint8,
    compute_histogram,
    compute_mean_intensity,
)


class TestBoundaryHandling:
    """Tests for pixel access and boundary handling."""

    def test_get_pixel_safe_in_bounds(self):
        img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        assert get_pixel_safe(img, 0, 0) == 1
        assert get_pixel_safe(img, 1, 1) == 4

    def test_get_pixel_safe_replicate(self):
        img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        # Out of bounds should replicate edge
        assert get_pixel_safe(img, -1, 0, BorderMode.REPLICATE) == 1
        assert get_pixel_safe(img, 2, 1, BorderMode.REPLICATE) == 4

    def test_get_pixel_safe_constant(self):
        img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        assert get_pixel_safe(img, -1, 0, BorderMode.CONSTANT, 0) == 0
        assert get_pixel_safe(img, -1, 0, BorderMode.CONSTANT, 255) == 255

    def test_pad_image_shape(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        padded = pad_image(img, 2, 3)
        assert padded.shape == (14, 16)


class TestConvolution:
    """Tests for convolution engine."""

    def test_identity_kernel(self):
        """Identity kernel should return same image."""
        img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
        result = convolve2d(img, identity)
        np.testing.assert_array_almost_equal(result, img)

    def test_box_kernel_averaging(self):
        """Box kernel should average values."""
        img = np.ones((5, 5), dtype=np.float64) * 100
        box = build_box_kernel(3)
        result = convolve2d(img, box)
        # Center should still be 100 (average of all 100s)
        assert abs(result[2, 2] - 100) < 0.01

    def test_kernel_must_be_odd(self):
        img = np.zeros((5, 5), dtype=np.float64)
        even_kernel = np.ones((4, 4), dtype=np.float64)
        with pytest.raises(ValueError):
            convolve2d(img, even_kernel)


class TestKernelBuilders:
    """Tests for kernel builder functions."""

    def test_gaussian_kernel_shape(self):
        kernel = build_gaussian_kernel(5, 1.0)
        assert kernel.shape == (5, 5)

    def test_gaussian_kernel_normalized(self):
        kernel = build_gaussian_kernel(7, 1.5)
        assert abs(kernel.sum() - 1.0) < 1e-10

    def test_gaussian_kernel_center_max(self):
        kernel = build_gaussian_kernel(5, 1.0)
        center = 2
        # Center should be maximum
        assert kernel[center, center] == kernel.max()

    def test_laplacian_standard(self):
        kernel = build_laplacian_kernel("standard")
        assert kernel.shape == (3, 3)
        assert kernel[1, 1] == -4

    def test_sobel_kernels_shape(self):
        kx, ky = build_sobel_kernels()
        assert kx.shape == (3, 3)
        assert ky.shape == (3, 3)


class TestColorConversion:
    """Tests for color space conversion."""

    def test_rgb_to_grayscale_shape(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        gray = rgb_to_grayscale(rgb)
        assert gray.shape == (10, 10)

    def test_rgb_to_grayscale_white(self):
        # Pure white should give ~255
        rgb = np.ones((5, 5, 3), dtype=np.uint8) * 255
        gray = rgb_to_grayscale(rgb)
        assert abs(gray[0, 0] - 255) < 1

    def test_rgb_to_grayscale_luminance(self):
        # Test specific color
        rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        rgb[0, 0] = [100, 150, 50]  # R, G, B
        gray = rgb_to_grayscale(rgb)
        expected = 0.299 * 100 + 0.587 * 150 + 0.114 * 50
        assert abs(gray[0, 0] - expected) < 0.1

    def test_rgb_to_grayscale_rejects_non_rgb(self):
        gray_img = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError):
            rgb_to_grayscale(gray_img)


class TestNormalization:
    """Tests for normalization functions."""

    def test_normalize_to_uint8_range(self):
        img = np.array([[-100, 0], [100, 200]], dtype=np.float64)
        result = normalize_to_uint8(img)
        assert result.min() == 0
        assert result.max() == 255
        assert result.dtype == np.uint8

    def test_clip_to_uint8(self):
        img = np.array([[-50, 128], [300, 255]], dtype=np.float64)
        result = clip_to_uint8(img)
        np.testing.assert_array_equal(result, [[0, 128], [255, 255]])


class TestHistogram:
    """Tests for histogram computation."""

    def test_histogram_shape(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        hist = compute_histogram(img)
        assert hist.shape == (256,)

    def test_histogram_single_value(self):
        img = np.ones((10, 10), dtype=np.uint8) * 128
        hist = compute_histogram(img)
        assert hist[128] == 100
        assert hist.sum() == 100

    def test_histogram_two_values(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        img[:5, :] = 50
        img[5:, :] = 200
        hist = compute_histogram(img)
        assert hist[50] == 50
        assert hist[200] == 50


class TestMedianFilter:
    """Tests for median filter."""

    def test_median_removes_salt_pepper(self):
        # Create image with salt-and-pepper noise
        img = np.ones((5, 5), dtype=np.uint8) * 128
        img[2, 2] = 255  # Salt
        
        result = apply_median_filter(img, 3)
        # Median should remove the outlier at center
        assert result[2, 2] == 128

    def test_median_preserves_edges(self):
        # Step edge
        img = np.zeros((5, 5), dtype=np.uint8)
        img[:, 3:] = 255
        
        result = apply_median_filter(img, 3)
        # Edge should be somewhat preserved
        assert result[2, 0] == 0
        assert result[2, 4] == 255
