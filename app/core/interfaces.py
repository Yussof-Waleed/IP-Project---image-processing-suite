"""
Base classes for image processors.

Simple, minimal interfaces - no over-abstraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


class ImageProcessor(ABC):
    """Base class for image processors. Override name, category, and process()."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name for UI display."""
        pass

    @property
    def category(self) -> str:
        """Category for grouping. Override if needed."""
        return "General"

    @abstractmethod
    def process(self, image: np.ndarray, **params) -> np.ndarray:
        """Process image and return new array (never modify input)."""
        pass


@dataclass
class ParamInfo:
    """Metadata for a processor parameter (for UI generation)."""
    label: str
    param_type: str  # 'int', 'float', 'bool', 'choice'
    default: Any
    min_val: Any = None
    max_val: Any = None
    choices: tuple = ()
    tooltip: str = ""


@dataclass
class HistogramResult:
    """Result from histogram analysis."""
    histogram: np.ndarray  # 256 bins
    mean: float
    std: float
    min_val: int
    max_val: int
    is_low_contrast: bool
    contrast_analysis: str
