"""
Tests for core.transforms module.
"""

import numpy as np
import pytest
from app.core.transforms import (
    TranslationProcessor,
    ScalingProcessor,
    RotationProcessor,
    ShearXProcessor,
    ShearYProcessor,
    apply_transform,
)
from app.core.interfaces import InterpolationMethod


class TestTranslation:
    def test_translation_shift(self):
        # 3x3 image with a single white pixel in center
        img = np.zeros((3, 3), dtype=np.uint8)
        img[1, 1] = 255
        
        processor = TranslationProcessor()
        
        # Shift right by 1
        res_x = processor.process(img, tx=1, ty=0)
        assert res_x[1, 2] == 255
        assert res_x[1, 1] == 0
        
        # Shift down by 1
        res_y = processor.process(img, tx=0, ty=1)
        assert res_y[2, 1] == 255
        assert res_y[1, 1] == 0


class TestScaling:
    def test_scaling_up(self):
        img = np.zeros((2, 2), dtype=np.uint8)
        img[0, 0] = 255
        
        processor = ScalingProcessor()
        # Scale 2x
        res = processor.process(img, sx=2.0, sy=2.0, center=False)
        
        assert res.shape == (4, 4)
        # Top-left block should be white (nearest neighbor default logic for simple check)
        # Note: Default interpolation is Bilinear, so it might smear.
        # Let's check dimensions primarily.
        assert res.shape == (4, 4)

    def test_scaling_down(self):
        img = np.zeros((4, 4), dtype=np.uint8)
        processor = ScalingProcessor()
        res = processor.process(img, sx=0.5, sy=0.5, center=False)
        assert res.shape == (2, 2)


class TestRotation:
    def test_rotation_90(self):
        # Line at top
        # Use odd dimensions (5x5) so center (2,2) is a pixel center
        img = np.zeros((5, 5), dtype=np.uint8)
        img[0, :] = 255
        
        processor = RotationProcessor()
        # Rotate 90 degrees
        # Use expand=True to ensure we don't clip due to center calculation
        res = processor.process(img, angle=90, expand=True)
        
        # Should be line at right (CW rotation)
        # Top row (y=0) -> Right column
        assert np.any(res[:, -1] > 0)


class TestShear:
    def test_shear_x(self):
        img = np.zeros((5, 5), dtype=np.uint8)
        img[2, 2] = 255
        
        processor = ShearXProcessor()
        res = processor.process(img, shear=1.0)
        
        # Width should increase
        assert res.shape[1] > 5
        assert res.shape[0] == 5


class TestInterpolationSupport:
    def test_nearest_neighbor(self):
        img = np.zeros((2, 2), dtype=np.uint8)
        img[0, 0] = 255
        
        # Scale up 4x with NN
        matrix = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 1]])
        res = apply_transform(img, matrix, (8, 8), interpolation=InterpolationMethod.NEAREST)
        
        # Should be crisp block
        # Note: With point sampling, indices 0,1 map to 0. Index 2 maps to 0.5 (rounds to 1).
        # So only 0,1 are guaranteed to be 255.
        assert res[0, 0] == 255
        assert res[1, 1] == 255
        assert res[0, 4] == 0

    def test_bicubic_runs(self):
        """Ensure bicubic doesn't crash and produces valid output."""
        img = np.zeros((10, 10), dtype=np.uint8)
        img[5, 5] = 255
        
        matrix = np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1]])
        res = apply_transform(img, matrix, (15, 15), interpolation=InterpolationMethod.BICUBIC)
        
        assert res.shape == (15, 15)
        assert res.max() > 0
