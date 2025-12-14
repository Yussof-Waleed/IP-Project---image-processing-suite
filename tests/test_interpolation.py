"""
Tests for core.interpolation module.
"""

import numpy as np
import pytest
from app.core.interpolation import (
    resize_nearest_neighbor,
    resize_bilinear,
    resize_bicubic,
)


class TestInterpolation:
    def test_nearest_neighbor_solid_color(self):
        img = np.ones((10, 10), dtype=np.uint8) * 100
        res = resize_nearest_neighbor(img, 20, 20)
        assert res.shape == (20, 20)
        assert np.all(res == 100)

    def test_nearest_neighbor_pattern(self):
        img = np.zeros((2, 2), dtype=np.uint8)
        img[0, 0] = 255
        res = resize_nearest_neighbor(img, 4, 4)
        # Top-left quadrant should be 255
        assert res[0, 0] == 255
        assert res[1, 1] == 255
        assert res[0, 2] == 0

    def test_bilinear_solid_color(self):
        img = np.ones((10, 10), dtype=np.uint8) * 100
        res = resize_bilinear(img, 20, 20)
        assert res.shape == (20, 20)
        # Allow small float error margin
        assert np.all(np.abs(res - 100) < 1)

    def test_bilinear_gradient(self):
        # 2x2 gradient: 0 -> 100
        img = np.array([[0, 100], [0, 100]], dtype=np.uint8)
        res = resize_bilinear(img, 4, 2)
        # Middle column should be around 25 (calculated manually: 0*0.75 + 100*0.25)
        # The previous assertion (40 < res < 60) was based on wrong math.
        assert 20 < res[0, 1] < 30

    def test_bicubic_solid_color(self):
        img = np.ones((10, 10), dtype=np.uint8) * 100
        res = resize_bicubic(img, 20, 20)
        assert res.shape == (20, 20)
        assert np.all(np.abs(res - 100) < 1)

    def test_bicubic_quality(self):
        # Bicubic should handle edges smoother or sharper than bilinear, 
        # but for a simple test, we just ensure it runs and produces valid output
        img = np.zeros((10, 10), dtype=np.uint8)
        img[4:6, 4:6] = 255
        
        res = resize_bicubic(img, 20, 20)
        assert res.shape == (20, 20)
        assert res.max() > 200
        assert res.min() == 0

    def test_color_image_support(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:] = [255, 0, 0]  # Red
        
        res_nn = resize_nearest_neighbor(img, 20, 20)
        assert res_nn.shape == (20, 20, 3)
        assert np.all(res_nn[:, :, 0] == 255)
        
        res_bl = resize_bilinear(img, 20, 20)
        assert res_bl.shape == (20, 20, 3)
        
        res_bc = resize_bicubic(img, 20, 20)
        assert res_bc.shape == (20, 20, 3)
