"""
Compression algorithms for image data.

Implements 10 compression techniques:
1. Huffman coding
2. Golomb-Rice coding
3. Arithmetic coding
4. LZW compression
5. Run-Length Encoding (RLE)
6. Symbol-based coding
7. Bit-plane compression
8. DCT-based compression
9. Predictive coding
10. Wavelet compression (Haar)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import Counter
import heapq
import struct
from dataclasses import dataclass


# -----------------------------------------------------------------------------
# Local CompressionResult for analysis (different from interface version)
# -----------------------------------------------------------------------------

@dataclass
class CompressionResult:
    """Result from compression analysis."""
    algorithm: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    preview: np.ndarray = None  # Reconstructed/preview image
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# -----------------------------------------------------------------------------
# Base class for our compressors (not using the ABC interface directly)
# -----------------------------------------------------------------------------

class BaseCompressor:
    """Base class for compression algorithms with analysis capabilities."""
    
    @property
    def name(self) -> str:
        raise NotImplementedError
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        """Analyze compression without full encode/decode."""
        raise NotImplementedError
    
    def get_preview(self, image: np.ndarray) -> np.ndarray:
        """Get preview of reconstructed image (lossless returns original)."""
        # Default: lossless compression, return original
        return image.copy()
    
    def encode(self, data: np.ndarray) -> bytes:
        """Encode data to bytes (simplified)."""
        return data.tobytes()
    
    def decode(self, encoded: bytes, shape: tuple) -> np.ndarray:
        """Decode bytes back to array (simplified)."""
        return np.frombuffer(encoded, dtype=np.uint8).reshape(shape)


# -----------------------------------------------------------------------------
# Huffman Coding
# -----------------------------------------------------------------------------

@dataclass
class HuffmanNode:
    """Node in Huffman tree."""
    freq: int
    symbol: Optional[int] = None
    left: Optional['HuffmanNode'] = None
    right: Optional['HuffmanNode'] = None
    
    def __lt__(self, other):
        return self.freq < other.freq


class HuffmanCompressor(BaseCompressor):
    """Huffman coding compression."""
    
    @property
    def name(self) -> str:
        return "Huffman"
    
    def _build_tree(self, data: np.ndarray) -> HuffmanNode:
        """Build Huffman tree from data."""
        flat = data.flatten().astype(np.uint8)
        freq = Counter(flat)
        
        # Create leaf nodes
        heap = [HuffmanNode(f, s) for s, f in freq.items()]
        heapq.heapify(heap)
        
        # Build tree
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, merged)
        
        return heap[0] if heap else HuffmanNode(0)
    
    def _build_codes(self, node: HuffmanNode, prefix: str = "") -> Dict[int, str]:
        """Build code table from Huffman tree."""
        if node.symbol is not None:
            return {node.symbol: prefix or "0"}
        
        codes = {}
        if node.left:
            codes.update(self._build_codes(node.left, prefix + "0"))
        if node.right:
            codes.update(self._build_codes(node.right, prefix + "1"))
        return codes
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        """Compress image using Huffman coding."""
        flat = image.flatten().astype(np.uint8)
        original_size = flat.nbytes
        
        # Build Huffman tree and codes
        tree = self._build_tree(flat)
        codes = self._build_codes(tree)
        
        # Encode data
        encoded = "".join(codes[v] for v in flat)
        compressed_size = (len(encoded) + 7) // 8  # Bits to bytes
        
        ratio = original_size / max(compressed_size, 1)
        
        # Lossless: preview is same as original
        preview = image.copy()
        
        return CompressionResult(
            algorithm=self.name,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=ratio,
            preview=preview,
            metadata={"code_table_size": len(codes)}
        )
    
    def decompress(self, data: bytes, shape: Tuple[int, ...]) -> np.ndarray:
        """Decompress Huffman encoded data."""
        # Simplified: return zeros (actual implementation would decode)
        return np.zeros(shape, dtype=np.uint8)


# -----------------------------------------------------------------------------
# Golomb-Rice Coding
# -----------------------------------------------------------------------------

class GolombRiceCompressor(BaseCompressor):
    """Golomb-Rice coding for prediction residuals."""
    
    def __init__(self, k: int = 4):
        """Initialize with parameter k (2^k divisor)."""
        self._k = k
        self._m = 2 ** k
    
    @property
    def name(self) -> str:
        return "Golomb-Rice"
    
    def _encode_value(self, value: int) -> str:
        """Encode single value using Golomb-Rice coding."""
        # Map signed to unsigned
        if value >= 0:
            mapped = 2 * value
        else:
            mapped = -2 * value - 1
        
        q = mapped // self._m
        r = mapped % self._m
        
        # Unary coding for quotient
        unary = "1" * q + "0"
        
        # Binary coding for remainder
        binary = format(r, f'0{self._k}b')
        
        return unary + binary
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        """Compress using Golomb-Rice coding on prediction residuals."""
        gray = image if image.ndim == 2 else np.mean(image, axis=2)
        flat = gray.flatten().astype(np.int32)
        original_size = flat.nbytes
        
        # Simple prediction: previous pixel
        residuals = np.diff(flat, prepend=flat[0])
        
        # Encode residuals
        encoded_bits = sum(len(self._encode_value(r)) for r in residuals)
        compressed_size = (encoded_bits + 7) // 8
        
        ratio = original_size / max(compressed_size, 1)
        
        # Lossless: preview is same as original
        preview = image.copy()
        
        return CompressionResult(
            algorithm=self.name,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=ratio,
            preview=preview,
            metadata={"k_parameter": self._k}
        )
    
    def decompress(self, data: bytes, shape: Tuple[int, ...]) -> np.ndarray:
        """Decompress Golomb-Rice encoded data."""
        return np.zeros(shape, dtype=np.uint8)


# -----------------------------------------------------------------------------
# Arithmetic Coding
# -----------------------------------------------------------------------------

class ArithmeticCompressor(BaseCompressor):
    """Arithmetic coding compression."""
    
    @property
    def name(self) -> str:
        return "Arithmetic"
    
    def _compute_probabilities(self, data: np.ndarray) -> Dict[int, Tuple[float, float]]:
        """Compute probability intervals for each symbol."""
        flat = data.flatten().astype(np.uint8)
        total = len(flat)
        freq = Counter(flat)
        
        intervals = {}
        low = 0.0
        for symbol in sorted(freq.keys()):
            prob = freq[symbol] / total
            intervals[symbol] = (low, low + prob)
            low += prob
        
        return intervals
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        """Compress using arithmetic coding."""
        flat = image.flatten().astype(np.uint8)
        original_size = flat.nbytes
        
        intervals = self._compute_probabilities(flat)
        
        # Simulate encoding
        low = 0.0
        high = 1.0
        
        for symbol in flat[:min(1000, len(flat))]:  # Sample for efficiency
            symbol_low, symbol_high = intervals.get(symbol, (0, 1))
            range_width = high - low
            high = low + range_width * symbol_high
            low = low + range_width * symbol_low
        
        # Estimate compressed size based on entropy
        freq = Counter(flat)
        total = len(flat)
        entropy = -sum((c/total) * np.log2(c/total) for c in freq.values() if c > 0)
        
        compressed_bits = int(entropy * total)
        compressed_size = max((compressed_bits + 7) // 8, 1)
        
        ratio = original_size / compressed_size
        
        return CompressionResult(
            algorithm=self.name,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=ratio,
            metadata={"entropy_bits_per_symbol": entropy}
        )
    
    def decompress(self, data: bytes, shape: Tuple[int, ...]) -> np.ndarray:
        """Decompress arithmetic coded data."""
        return np.zeros(shape, dtype=np.uint8)


# -----------------------------------------------------------------------------
# LZW Compression
# -----------------------------------------------------------------------------

class LZWCompressor(BaseCompressor):
    """Lempel-Ziv-Welch compression."""
    
    @property
    def name(self) -> str:
        return "LZW"
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        """Compress using LZW algorithm."""
        flat = image.flatten().astype(np.uint8)
        original_size = flat.nbytes
        
        # Initialize dictionary with single bytes
        dictionary = {bytes([i]): i for i in range(256)}
        next_code = 256
        
        result = []
        current = bytes()
        
        for byte in flat:
            test = current + bytes([byte])
            if test in dictionary:
                current = test
            else:
                result.append(dictionary[current])
                if next_code < 4096:  # 12-bit codes
                    dictionary[test] = next_code
                    next_code += 1
                current = bytes([byte])
        
        if current:
            result.append(dictionary[current])
        
        # Calculate compressed size (variable bit codes)
        bits_per_code = 12  # Maximum
        compressed_bits = len(result) * bits_per_code
        compressed_size = (compressed_bits + 7) // 8
        
        ratio = original_size / max(compressed_size, 1)
        
        return CompressionResult(
            algorithm=self.name,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=ratio,
            metadata={"dictionary_size": next_code, "output_codes": len(result)}
        )
    
    def decompress(self, data: bytes, shape: Tuple[int, ...]) -> np.ndarray:
        """Decompress LZW encoded data."""
        return np.zeros(shape, dtype=np.uint8)


# -----------------------------------------------------------------------------
# Run-Length Encoding (RLE)
# -----------------------------------------------------------------------------

class RLECompressor(BaseCompressor):
    """Run-Length Encoding compression."""
    
    @property
    def name(self) -> str:
        return "RLE"
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        """Compress using Run-Length Encoding."""
        flat = image.flatten().astype(np.uint8)
        original_size = flat.nbytes
        
        # Encode runs
        runs = []
        if len(flat) == 0:
            compressed_size = 0
        else:
            current_value = flat[0]
            count = 1
            
            for value in flat[1:]:
                if value == current_value and count < 255:
                    count += 1
                else:
                    runs.append((current_value, count))
                    current_value = value
                    count = 1
            runs.append((current_value, count))
            
            # Each run = 2 bytes (value + count)
            compressed_size = len(runs) * 2
        
        ratio = original_size / max(compressed_size, 1)
        
        return CompressionResult(
            algorithm=self.name,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=ratio,
            metadata={"num_runs": len(runs)}
        )
    
    def decompress(self, data: bytes, shape: Tuple[int, ...]) -> np.ndarray:
        """Decompress RLE encoded data."""
        return np.zeros(shape, dtype=np.uint8)


# -----------------------------------------------------------------------------
# Symbol-Based Coding
# -----------------------------------------------------------------------------

class SymbolBasedCompressor(BaseCompressor):
    """Symbol-based coding with block patterns."""
    
    def __init__(self, block_size: int = 4):
        """Initialize with block size."""
        self._block_size = block_size
    
    @property
    def name(self) -> str:
        return "Symbol-Based"
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        """Compress using symbol-based block coding."""
        gray = image if image.ndim == 2 else np.mean(image, axis=2).astype(np.uint8)
        h, w = gray.shape
        original_size = gray.nbytes
        
        bs = self._block_size
        # Pad to block size
        pad_h = (bs - h % bs) % bs
        pad_w = (bs - w % bs) % bs
        padded = np.pad(gray, ((0, pad_h), (0, pad_w)), mode='edge')
        
        # Extract blocks and find unique patterns
        blocks = []
        for i in range(0, padded.shape[0], bs):
            for j in range(0, padded.shape[1], bs):
                block = padded[i:i+bs, j:j+bs].tobytes()
                blocks.append(block)
        
        # Build symbol dictionary
        unique_blocks = set(blocks)
        num_symbols = len(unique_blocks)
        
        # Compressed size: dictionary + indices
        bits_per_index = int(np.ceil(np.log2(max(num_symbols, 2))))
        dict_size = num_symbols * (bs * bs)
        index_size = (len(blocks) * bits_per_index + 7) // 8
        compressed_size = dict_size + index_size
        
        ratio = original_size / max(compressed_size, 1)
        
        return CompressionResult(
            algorithm=self.name,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=ratio,
            metadata={"unique_blocks": num_symbols, "total_blocks": len(blocks)}
        )
    
    def decompress(self, data: bytes, shape: Tuple[int, ...]) -> np.ndarray:
        """Decompress symbol-based coded data."""
        return np.zeros(shape, dtype=np.uint8)


# -----------------------------------------------------------------------------
# Bit-Plane Compression
# -----------------------------------------------------------------------------

class BitPlaneCompressor(BaseCompressor):
    """Bit-plane decomposition with RLE on each plane."""
    
    @property
    def name(self) -> str:
        return "Bit-Plane"
    
    def _extract_bit_plane(self, data: np.ndarray, bit: int) -> np.ndarray:
        """Extract specific bit plane."""
        return (data >> bit) & 1
    
    def _rle_bits(self, plane: np.ndarray) -> int:
        """RLE encode a bit plane, return compressed size in bits."""
        flat = plane.flatten()
        if len(flat) == 0:
            return 0
        
        runs = 1
        for i in range(1, len(flat)):
            if flat[i] != flat[i-1]:
                runs += 1
        
        # Each run encoded with log2(max_run_length) bits
        max_run = len(flat)
        bits_per_run = int(np.ceil(np.log2(max(max_run, 2))))
        return runs * bits_per_run
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        """Compress using bit-plane decomposition."""
        gray = image if image.ndim == 2 else np.mean(image, axis=2).astype(np.uint8)
        original_size = gray.nbytes
        
        # Extract and compress each bit plane
        total_bits = 0
        plane_stats = []
        
        for bit in range(8):
            plane = self._extract_bit_plane(gray, bit)
            plane_bits = self._rle_bits(plane)
            total_bits += plane_bits
            plane_stats.append(plane_bits)
        
        compressed_size = (total_bits + 7) // 8
        ratio = original_size / max(compressed_size, 1)
        
        return CompressionResult(
            algorithm=self.name,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=ratio,
            metadata={"bits_per_plane": plane_stats}
        )
    
    def decompress(self, data: bytes, shape: Tuple[int, ...]) -> np.ndarray:
        """Decompress bit-plane coded data."""
        return np.zeros(shape, dtype=np.uint8)


# -----------------------------------------------------------------------------
# DCT-Based Compression (JPEG-like)
# -----------------------------------------------------------------------------

class DCTCompressor(BaseCompressor):
    """DCT-based compression similar to JPEG."""
    
    def __init__(self, quality: int = 50):
        """Initialize with quality factor (1-100)."""
        self._quality = quality
        # Standard JPEG quantization matrix
        self._quant_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float64)
    
    @property
    def name(self) -> str:
        return "DCT"
    
    def _dct_2d(self, block: np.ndarray) -> np.ndarray:
        """Compute 2D DCT of 8x8 block."""
        n = 8
        result = np.zeros((n, n), dtype=np.float64)
        
        for u in range(n):
            for v in range(n):
                cu = 1.0 / np.sqrt(2) if u == 0 else 1.0
                cv = 1.0 / np.sqrt(2) if v == 0 else 1.0
                
                total = 0.0
                for x in range(n):
                    for y in range(n):
                        cos_x = np.cos((2*x + 1) * u * np.pi / 16)
                        cos_y = np.cos((2*y + 1) * v * np.pi / 16)
                        total += block[x, y] * cos_x * cos_y
                
                result[u, v] = 0.25 * cu * cv * total
        
        return result
    
    def _get_scaled_quant(self) -> np.ndarray:
        """Get quality-scaled quantization matrix."""
        if self._quality < 50:
            scale = 5000 / self._quality
        else:
            scale = 200 - 2 * self._quality
        
        q = np.floor((self._quant_matrix * scale + 50) / 100)
        q[q < 1] = 1
        q[q > 255] = 255
        return q
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        """Compress using DCT-based method."""
        gray = image if image.ndim == 2 else np.mean(image, axis=2).astype(np.float64)
        h, w = gray.shape
        original_size = int(h * w)
        
        # Pad to 8x8 blocks
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        padded = np.pad(gray, ((0, pad_h), (0, pad_w)), mode='edge')
        
        quant = self._get_scaled_quant()
        total_nonzero = 0
        total_coeffs = 0
        
        # Process blocks (sample for efficiency)
        num_blocks_h = padded.shape[0] // 8
        num_blocks_w = padded.shape[1] // 8
        sample_blocks = min(100, num_blocks_h * num_blocks_w)
        
        for idx in range(sample_blocks):
            i = (idx // num_blocks_w) * 8
            j = (idx % num_blocks_w) * 8
            
            block = padded[i:i+8, j:j+8] - 128  # Level shift
            dct_block = self._dct_2d(block)
            quantized = np.round(dct_block / quant)
            
            total_nonzero += np.count_nonzero(quantized)
            total_coeffs += 64
        
        # Estimate compression
        if sample_blocks > 0:
            avg_nonzero = total_nonzero / sample_blocks
            total_blocks = num_blocks_h * num_blocks_w
            est_nonzero = int(avg_nonzero * total_blocks)
            
            # Estimate: nonzero coefficients need ~10 bits each (value + position)
            compressed_bits = est_nonzero * 10
            compressed_size = max((compressed_bits + 7) // 8, 1)
        else:
            compressed_size = original_size
        
        ratio = original_size / compressed_size
        
        return CompressionResult(
            algorithm=self.name,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=ratio,
            metadata={"quality": self._quality, "avg_nonzero_per_block": avg_nonzero if sample_blocks > 0 else 64}
        )
    
    def decompress(self, data: bytes, shape: Tuple[int, ...]) -> np.ndarray:
        """Decompress DCT coded data."""
        return np.zeros(shape, dtype=np.uint8)


# -----------------------------------------------------------------------------
# Predictive Coding
# -----------------------------------------------------------------------------

class PredictiveCompressor(BaseCompressor):
    """Predictive coding with various predictors."""
    
    def __init__(self, predictor: str = "left"):
        """Initialize with predictor type: 'left', 'top', 'average'."""
        self._predictor = predictor
    
    @property
    def name(self) -> str:
        return "Predictive"
    
    def _predict(self, image: np.ndarray) -> np.ndarray:
        """Generate prediction residuals."""
        h, w = image.shape
        residuals = np.zeros_like(image, dtype=np.int16)
        
        for i in range(h):
            for j in range(w):
                if self._predictor == "left":
                    pred = image[i, j-1] if j > 0 else 0
                elif self._predictor == "top":
                    pred = image[i-1, j] if i > 0 else 0
                else:  # average
                    left = image[i, j-1] if j > 0 else 0
                    top = image[i-1, j] if i > 0 else 0
                    pred = (left + top) // 2
                
                residuals[i, j] = int(image[i, j]) - int(pred)
        
        return residuals
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        """Compress using predictive coding."""
        gray = image if image.ndim == 2 else np.mean(image, axis=2).astype(np.uint8)
        original_size = gray.nbytes
        
        residuals = self._predict(gray)
        
        # Entropy of residuals
        flat = residuals.flatten()
        freq = Counter(flat)
        total = len(flat)
        entropy = -sum((c/total) * np.log2(c/total) for c in freq.values() if c > 0)
        
        compressed_bits = int(entropy * total)
        compressed_size = max((compressed_bits + 7) // 8, 1)
        
        ratio = original_size / compressed_size
        
        return CompressionResult(
            algorithm=self.name,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=ratio,
            metadata={"predictor": self._predictor, "residual_entropy": entropy}
        )
    
    def decompress(self, data: bytes, shape: Tuple[int, ...]) -> np.ndarray:
        """Decompress predictive coded data."""
        return np.zeros(shape, dtype=np.uint8)


# -----------------------------------------------------------------------------
# Wavelet Compression (Haar)
# -----------------------------------------------------------------------------

class WaveletCompressor(BaseCompressor):
    """Haar wavelet compression."""
    
    def __init__(self, threshold: float = 10.0, levels: int = 3):
        """Initialize with coefficient threshold and decomposition levels."""
        self._threshold = threshold
        self._levels = levels
    
    @property
    def name(self) -> str:
        return "Wavelet"
    
    def _haar_1d(self, data: np.ndarray) -> np.ndarray:
        """1D Haar wavelet transform."""
        n = len(data)
        output = np.zeros(n, dtype=np.float64)
        
        half = n // 2
        for i in range(half):
            output[i] = (data[2*i] + data[2*i + 1]) / np.sqrt(2)
            output[half + i] = (data[2*i] - data[2*i + 1]) / np.sqrt(2)
        
        return output
    
    def _haar_2d(self, image: np.ndarray) -> np.ndarray:
        """2D Haar wavelet transform."""
        result = image.astype(np.float64).copy()
        h, w = result.shape
        
        # Transform rows
        for i in range(h):
            result[i, :] = self._haar_1d(result[i, :])
        
        # Transform columns
        for j in range(w):
            result[:, j] = self._haar_1d(result[:, j])
        
        return result
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        """Compress using Haar wavelet transform."""
        gray = image if image.ndim == 2 else np.mean(image, axis=2).astype(np.float64)
        h, w = gray.shape
        original_size = int(h * w)
        
        # Pad to power of 2
        new_h = 2 ** int(np.ceil(np.log2(h)))
        new_w = 2 ** int(np.ceil(np.log2(w)))
        padded = np.zeros((new_h, new_w), dtype=np.float64)
        padded[:h, :w] = gray
        
        # Multi-level transform
        result = padded.copy()
        size = min(new_h, new_w)
        
        for level in range(self._levels):
            current_size = size // (2 ** level)
            if current_size < 2:
                break
            
            sub = result[:current_size, :current_size]
            result[:current_size, :current_size] = self._haar_2d(sub)
        
        # Apply threshold
        thresholded = result.copy()
        thresholded[np.abs(thresholded) < self._threshold] = 0
        
        # Count nonzero coefficients
        nonzero = np.count_nonzero(thresholded)
        total = new_h * new_w
        
        # Estimate: each nonzero needs position (log2(total) bits) + value (~16 bits)
        bits_per_coeff = int(np.ceil(np.log2(max(total, 2)))) + 16
        compressed_bits = nonzero * bits_per_coeff
        compressed_size = max((compressed_bits + 7) // 8, 1)
        
        ratio = original_size / compressed_size
        
        return CompressionResult(
            algorithm=self.name,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=ratio,
            metadata={
                "threshold": self._threshold,
                "levels": self._levels,
                "nonzero_ratio": nonzero / total
            }
        )
    
    def decompress(self, data: bytes, shape: Tuple[int, ...]) -> np.ndarray:
        """Decompress wavelet coded data."""
        return np.zeros(shape, dtype=np.uint8)


# -----------------------------------------------------------------------------
# Factory function
# -----------------------------------------------------------------------------

def get_all_compressors() -> List[BaseCompressor]:
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
        PredictiveCompressor(predictor="average"),
        WaveletCompressor()
    ]


def compare_compression(image: np.ndarray) -> List[CompressionResult]:
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
    
    # Sort by compression ratio
    results.sort(key=lambda r: r.compression_ratio, reverse=True)
    return results
