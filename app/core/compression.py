"""
Compression algorithms for image data analysis.

These compressors analyze compression ratios without full encode/decode.
Used by the GUI to compare algorithm performance.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np
from collections import Counter
import heapq


@dataclass
class CompressionResult:
    """Result from compression analysis."""
    algorithm: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# -----------------------------------------------------------------------------
# Huffman Coding
# -----------------------------------------------------------------------------

class HuffmanCompressor:
    """Huffman coding compression analysis."""
    
    @property
    def name(self) -> str:
        return "Huffman"
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        flat = image.flatten().astype(np.uint8)
        original_size = flat.nbytes
        
        # Count symbol frequencies
        freq = Counter(flat)
        n_symbols = len(freq)
        
        if n_symbols <= 1:
            # Single symbol = 1 bit per pixel minimum
            compressed_size = (len(flat) + 7) // 8
        else:
            # Estimate bits using entropy (accurate for Huffman)
            total = len(flat)
            entropy = -sum((c/total) * np.log2(c/total) for c in freq.values())
            total_bits = int(np.ceil(entropy * total))
            compressed_size = max((total_bits + 7) // 8, 1)
        
        return CompressionResult(
            algorithm=self.name,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / max(compressed_size, 1),
            metadata={"unique_symbols": len(freq)}
        )


# -----------------------------------------------------------------------------
# Golomb-Rice Coding
# -----------------------------------------------------------------------------

class GolombRiceCompressor:
    """Golomb-Rice coding for prediction residuals."""
    
    def __init__(self, k: int = 4):
        self._k = k
        self._m = 2 ** k
    
    @property
    def name(self) -> str:
        return "Golomb-Rice"
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        gray = image if image.ndim == 2 else np.mean(image, axis=2)
        flat = gray.flatten().astype(np.int32)
        original_size = flat.nbytes
        
        # Prediction residuals
        residuals = np.diff(flat, prepend=flat[0])
        
        # Count bits for Golomb-Rice encoding
        total_bits = 0
        for r in residuals:
            mapped = 2 * r if r >= 0 else -2 * r - 1
            q = abs(mapped) // self._m
            total_bits += q + 1 + self._k  # unary + binary
        
        compressed_size = (total_bits + 7) // 8
        
        return CompressionResult(
            algorithm=self.name,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / max(compressed_size, 1),
            metadata={"k_parameter": self._k}
        )


# -----------------------------------------------------------------------------
# Arithmetic Coding
# -----------------------------------------------------------------------------

class ArithmeticCompressor:
    """Arithmetic coding compression analysis."""
    
    @property
    def name(self) -> str:
        return "Arithmetic"
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        flat = image.flatten().astype(np.uint8)
        original_size = flat.nbytes
        
        # Estimate using entropy
        freq = Counter(flat)
        total = len(flat)
        entropy = -sum((c/total) * np.log2(c/total) for c in freq.values() if c > 0)
        
        compressed_bits = int(entropy * total)
        compressed_size = max((compressed_bits + 7) // 8, 1)
        
        return CompressionResult(
            algorithm=self.name,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / compressed_size,
            metadata={"entropy": entropy}
        )


# -----------------------------------------------------------------------------
# LZW Compression
# -----------------------------------------------------------------------------

class LZWCompressor:
    """LZW compression analysis."""
    
    @property
    def name(self) -> str:
        return "LZW"
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        flat = image.flatten().astype(np.uint8)
        original_size = flat.nbytes
        
        # Simulate LZW
        dictionary = {bytes([i]): i for i in range(256)}
        next_code = 256
        result_count = 0
        current = bytes()
        
        for byte in flat:
            test = current + bytes([byte])
            if test in dictionary:
                current = test
            else:
                result_count += 1
                if next_code < 4096:
                    dictionary[test] = next_code
                    next_code += 1
                current = bytes([byte])
        
        if current:
            result_count += 1
        
        compressed_size = (result_count * 12 + 7) // 8  # 12-bit codes
        
        return CompressionResult(
            algorithm=self.name,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / max(compressed_size, 1),
            metadata={"dictionary_size": next_code}
        )


# -----------------------------------------------------------------------------
# Run-Length Encoding
# -----------------------------------------------------------------------------

class RLECompressor:
    """Run-Length Encoding compression analysis."""
    
    @property
    def name(self) -> str:
        return "RLE"
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        flat = image.flatten().astype(np.uint8)
        original_size = flat.nbytes
        
        # Count runs
        run_count = 1
        for i in range(1, len(flat)):
            if flat[i] != flat[i-1]:
                run_count += 1
        
        compressed_size = run_count * 2  # value + count per run
        
        return CompressionResult(
            algorithm=self.name,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / max(compressed_size, 1),
            metadata={"run_count": run_count}
        )


# -----------------------------------------------------------------------------
# Symbol-Based Coding
# -----------------------------------------------------------------------------

class SymbolBasedCompressor:
    """Symbol-based coding using fixed-length codes."""
    
    @property
    def name(self) -> str:
        return "Symbol-Based"
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        flat = image.flatten().astype(np.uint8)
        original_size = flat.nbytes
        
        unique = len(np.unique(flat))
        bits_per_symbol = max(1, int(np.ceil(np.log2(max(unique, 2)))))
        
        compressed_bits = len(flat) * bits_per_symbol
        compressed_size = (compressed_bits + 7) // 8
        
        return CompressionResult(
            algorithm=self.name,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / max(compressed_size, 1),
            metadata={"unique_symbols": unique, "bits_per_symbol": bits_per_symbol}
        )


# -----------------------------------------------------------------------------
# Bit-Plane Compression
# -----------------------------------------------------------------------------

class BitPlaneCompressor:
    """Bit-plane compression analysis."""
    
    @property
    def name(self) -> str:
        return "Bit-Plane"
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        gray = image if image.ndim == 2 else np.mean(image, axis=2).astype(np.uint8)
        original_size = gray.size
        
        # RLE each bit plane
        total_runs = 0
        for bit in range(8):
            plane = (gray >> bit) & 1
            flat = plane.flatten()
            runs = 1 + np.sum(flat[1:] != flat[:-1])
            total_runs += runs
        
        compressed_size = total_runs * 2  # Each run needs position info
        
        return CompressionResult(
            algorithm=self.name,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / max(compressed_size, 1),
            metadata={"total_runs": total_runs}
        )


# -----------------------------------------------------------------------------
# DCT-Based Compression
# -----------------------------------------------------------------------------

class DCTCompressor:
    """DCT-based compression (JPEG-like) analysis."""
    
    def __init__(self, quality: int = 50):
        self._quality = quality
    
    @property
    def name(self) -> str:
        return "DCT"
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        gray = image if image.ndim == 2 else np.mean(image, axis=2)
        original_size = gray.size
        
        h, w = gray.shape
        block_size = 8
        
        # Estimate based on quality and image variance
        scale = (100 - self._quality) / 100
        kept_coeffs = max(1, int(64 * (1 - scale * 0.9)))  # Coeffs kept per block
        
        num_blocks = ((h + 7) // 8) * ((w + 7) // 8)
        compressed_size = num_blocks * kept_coeffs  # ~1 byte per kept coeff
        
        return CompressionResult(
            algorithm=self.name,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / max(compressed_size, 1),
            metadata={"quality": self._quality, "blocks": num_blocks}
        )


# -----------------------------------------------------------------------------
# Predictive Coding
# -----------------------------------------------------------------------------

class PredictiveCompressor:
    """Predictive coding compression analysis."""
    
    def __init__(self, predictor: str = "average"):
        self._predictor = predictor
    
    @property
    def name(self) -> str:
        return "Predictive"
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        gray = image if image.ndim == 2 else np.mean(image, axis=2)
        original_size = gray.size
        
        # Calculate prediction residuals
        flat = gray.flatten().astype(np.float64)
        predicted = np.roll(flat, 1)
        predicted[0] = 128
        residuals = flat - predicted
        
        # Entropy of residuals
        residuals_int = np.clip(residuals + 128, 0, 255).astype(np.uint8)
        freq = Counter(residuals_int)
        total = len(residuals_int)
        entropy = -sum((c/total) * np.log2(c/total) for c in freq.values() if c > 0)
        
        compressed_bits = int(entropy * total)
        compressed_size = max((compressed_bits + 7) // 8, 1)
        
        return CompressionResult(
            algorithm=self.name,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / compressed_size,
            metadata={"residual_entropy": entropy}
        )


# -----------------------------------------------------------------------------
# Wavelet Compression
# -----------------------------------------------------------------------------

class WaveletCompressor:
    """Haar wavelet compression analysis."""
    
    def __init__(self, threshold: float = 10.0, levels: int = 3):
        self._threshold = threshold
        self._levels = levels
    
    @property
    def name(self) -> str:
        return "Wavelet"
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        gray = image if image.ndim == 2 else np.mean(image, axis=2).astype(np.float64)
        h, w = gray.shape
        original_size = h * w
        
        # Simple Haar transform simulation
        coeffs = gray.copy()
        for level in range(self._levels):
            step = 2 ** level
            if h // step < 2 or w // step < 2:
                break
            sub_h, sub_w = h // step, w // step
            for i in range(0, sub_h, 2):
                for j in range(0, sub_w, 2):
                    if i+1 < sub_h and j+1 < sub_w:
                        block = coeffs[i:i+2, j:j+2]
                        avg = np.mean(block)
                        coeffs[i:i+2, j:j+2] -= avg
        
        # Count coefficients above threshold
        nonzero = np.sum(np.abs(coeffs) >= self._threshold)
        bits_per_coeff = 16  # position + value
        compressed_size = max((nonzero * bits_per_coeff + 7) // 8, 1)
        
        return CompressionResult(
            algorithm=self.name,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / compressed_size,
            metadata={"threshold": self._threshold, "kept_ratio": nonzero / max(original_size, 1)}
        )


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def get_all_compressors():
    """Get list of all available compressors."""
    return [
        HuffmanCompressor(),
        GolombRiceCompressor(),
        ArithmeticCompressor(),
        LZWCompressor(),
        RLECompressor(),
        SymbolBasedCompressor(),
        BitPlaneCompressor(),
        DCTCompressor(quality=50),
        PredictiveCompressor(),
        WaveletCompressor(),
    ]


def compare_compression(image: np.ndarray) -> list:
    """Compare all compression algorithms on an image."""
    results = []
    for compressor in get_all_compressors():
        try:
            result = compressor.compress(image)
            results.append(result)
        except Exception as e:
            results.append(CompressionResult(
                algorithm=compressor.name,
                original_size=image.nbytes,
                compressed_size=image.nbytes,
                compression_ratio=1.0,
                metadata={"error": str(e)}
            ))
    
    results.sort(key=lambda r: r.compression_ratio, reverse=True)
    return results
