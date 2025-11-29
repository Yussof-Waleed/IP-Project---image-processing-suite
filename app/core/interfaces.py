"""
Abstract base classes and interfaces for image processors.

This module defines the contracts that all processors must follow,
enabling dependency inversion and consistent API across the application.

Design Principles:
- Interface Segregation: Small, focused interfaces
- Liskov Substitution: All implementations respect base contracts
- Dependency Inversion: GUI depends on these abstractions, not concrete classes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Base Processor Interface
# ─────────────────────────────────────────────────────────────────────────────

class ImageProcessor(ABC):
    """
    Base interface for all image processing operations.
    
    Follows CQS: process() is a query that returns a new image,
    never modifying the input.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for UI display."""
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """Category for grouping in UI (e.g., 'Filters', 'Transforms')."""
        pass

    @abstractmethod
    def process(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply the processing operation to an image.

        Args:
            image: Input image as numpy array (H, W) or (H, W, C).
            **params: Operation-specific parameters.

        Returns:
            New processed image array (never modifies input).
        """
        pass

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for this processor."""
        return {}

    def get_param_info(self) -> dict[str, "ParamInfo"]:
        """Return metadata about parameters for UI generation."""
        return {}


@dataclass(frozen=True)
class ParamInfo:
    """Metadata for a processor parameter (for dynamic UI generation)."""
    label: str
    param_type: str  # 'int', 'float', 'bool', 'choice'
    default: Any
    min_val: Any = None
    max_val: Any = None
    choices: tuple = ()
    tooltip: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Specialized Processor Interfaces
# ─────────────────────────────────────────────────────────────────────────────

class ConvolutionProcessor(ImageProcessor):
    """Interface for processors that use convolution kernels."""

    @abstractmethod
    def get_kernel(self, **params) -> np.ndarray:
        """Return the convolution kernel."""
        pass


class TransformProcessor(ImageProcessor):
    """Interface for geometric transformation processors."""

    @abstractmethod
    def get_transform_matrix(self, image_shape: tuple, **params) -> np.ndarray:
        """
        Return the 3x3 affine transformation matrix.

        Args:
            image_shape: (H, W) or (H, W, C) shape of input image.
            **params: Transform-specific parameters.

        Returns:
            3x3 numpy array representing the affine transform.
        """
        pass


class InterpolationMethod(Enum):
    """Available interpolation methods for geometric transforms."""
    NEAREST = auto()
    BILINEAR = auto()
    BICUBIC = auto()


class ResizeProcessor(ImageProcessor):
    """Interface for image resizing/interpolation processors."""

    @property
    @abstractmethod
    def method(self) -> InterpolationMethod:
        """The interpolation method used."""
        pass


class CompressionProcessor(ABC):
    """
    Interface for compression algorithms.
    
    Separate from ImageProcessor since compression doesn't return an image,
    but encoded data and metrics.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name."""
        pass

    @abstractmethod
    def encode(self, data: np.ndarray) -> bytes:
        """
        Encode image data.

        Args:
            data: Input array (typically grayscale flattened or 2D).

        Returns:
            Encoded bytes.
        """
        pass

    @abstractmethod
    def decode(self, encoded: bytes, shape: tuple) -> np.ndarray:
        """
        Decode back to array.

        Args:
            encoded: Encoded bytes from encode().
            shape: Original array shape for reconstruction.

        Returns:
            Reconstructed numpy array.
        """
        pass

    def get_compression_ratio(self, original: np.ndarray, encoded: bytes) -> float:
        """Calculate compression ratio (original_size / compressed_size)."""
        original_bytes = original.nbytes
        compressed_bytes = len(encoded)
        if compressed_bytes == 0:
            return float('inf')
        return original_bytes / compressed_bytes


# ─────────────────────────────────────────────────────────────────────────────
# Result Types (for CQS - queries return immutable results)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HistogramResult:
    """Immutable result from histogram analysis."""
    histogram: np.ndarray  # 256 bins
    mean: float
    std: float
    min_val: int
    max_val: int
    is_low_contrast: bool
    contrast_analysis: str


@dataclass(frozen=True)
class CompressionResult:
    """Immutable result from compression operation."""
    encoded_data: bytes
    original_size: int
    compressed_size: int
    compression_ratio: float
    algorithm_name: str

    @property
    def space_saving_percent(self) -> float:
        """Percentage of space saved."""
        return (1 - self.compressed_size / self.original_size) * 100


# ─────────────────────────────────────────────────────────────────────────────
# Processor Registry (for dependency injection)
# ─────────────────────────────────────────────────────────────────────────────

class ProcessorRegistry:
    """
    Registry for all available processors.
    
    Enables Open/Closed principle: new processors can be registered
    without modifying existing code.
    """

    def __init__(self):
        self._processors: dict[str, ImageProcessor] = {}
        self._compression: dict[str, CompressionProcessor] = {}

    def register(self, processor: ImageProcessor) -> None:
        """Register an image processor."""
        key = f"{processor.category}.{processor.name}"
        self._processors[key] = processor

    def register_compression(self, processor: CompressionProcessor) -> None:
        """Register a compression processor."""
        self._compression[processor.name] = processor

    def get_processor(self, category: str, name: str) -> ImageProcessor | None:
        """Retrieve a processor by category and name."""
        return self._processors.get(f"{category}.{name}")

    def get_compression(self, name: str) -> CompressionProcessor | None:
        """Retrieve a compression processor by name."""
        return self._compression.get(name)

    def list_by_category(self) -> dict[str, list[ImageProcessor]]:
        """Return all processors grouped by category."""
        result: dict[str, list[ImageProcessor]] = {}
        for proc in self._processors.values():
            if proc.category not in result:
                result[proc.category] = []
            result[proc.category].append(proc)
        return result

    def list_compression(self) -> list[CompressionProcessor]:
        """Return all compression processors."""
        return list(self._compression.values())


# Global registry instance (can be replaced for testing)
default_registry = ProcessorRegistry()
