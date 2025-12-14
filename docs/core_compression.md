# Core Compression (`app.core.compression`)

## Overview
This module implements a comprehensive suite of **10 Compression Algorithms**, ranging from simple Run-Length Encoding to complex Wavelet transforms. It serves as a museum of compression history.

## Ideology: The Art of Squeezing
Compression is about finding **Redundancy**.
-   **Spatial Redundancy**: Neighbors are similar (RLE, Predictive).
-   **Statistical Redundancy**: Some values are more common (Huffman, Arithmetic).
-   **Perceptual Redundancy**: Eyes can't see high-frequency noise (DCT, Wavelet).

## Visualizing the Pipeline

```mermaid
graph LR
    Input[Raw Image] --> Transform[Transform (DCT/Wavelet/Prediction)]
    Transform --> Quant[Quantization (Lossy only)]
    Quant --> Encode[Entropy Encoding (Huffman/RLE)]
    Encode --> Bytes[Compressed Bytes]
```

## Simplification: Packing a Suitcase
-   **RLE**: "5 blue shirts" instead of "blue shirt, blue shirt, blue shirt...".
-   **Huffman**: Giving shorter nicknames to your best friends and longer names to strangers.
-   **DCT (JPEG)**: Folding your clothes. You throw away the wrinkles (high frequency details) because nobody sees them anyway.

## Technical Details

### Lossless Algorithms (Perfect Reconstruction)
1.  **Huffman Coding**: Builds a binary tree based on frequency. Most common pixels get 1-bit codes.
2.  **Golomb-Rice**: Optimized for geometric distributions (like prediction residuals). Uses a tunable parameter $k$.
3.  **Arithmetic Coding**: Encodes the entire message as a single float number between 0.0 and 1.0. More efficient than Huffman but slower.
4.  **LZW**: Dictionary-based. Learns patterns (sequences of pixels) and assigns them new codes. Used in GIF/TIFF.
5.  **RLE**: Stores `(value, count)` pairs. Great for simple graphics, bad for photos.
6.  **Bit-Plane**: Splits image into 8 binary images. Compresses the "most significant" planes (structure) differently from "least significant" (noise).
7.  **Predictive**: Guesses the next pixel based on the previous one ($P_{pred} = P_{left}$). Encodes the error ($P_{actual} - P_{pred}$).

### Lossy Algorithms (Approximation)
8.  **DCT (Discrete Cosine Transform)**:
    -   Breaks image into 8x8 blocks.
    -   Converts to frequency domain (sums of cosine waves).
    -   **Quantization**: Divides high-frequency coefficients by large numbers, turning them to zero. This is where space is saved and quality is lost.
9.  **Wavelet (Haar)**:
    -   Recursive decomposition.
    -   Splits image into "Average" (Low pass) and "Detail" (High pass).
    -   Repeats on the "Average" part.
    -   Thresholding removes small detail coefficients.

### `CompressionResult`
A standardized report returned by all compressors.
-   **`compression_ratio`**: Original Size / Compressed Size.
-   **`preview`**: What the image looks like after decompression (crucial for lossy methods to see artifacts).
-   **`metadata`**: Dictionary of algorithm-specific stats.

## Code Reference

### `BaseCompressor`
Abstract base class for all compressors.
*   `compress(image) -> CompressionResult`: Analyzes compression.
*   `encode(data) -> bytes`: Encodes data to bytes.
*   `decode(encoded, shape) -> np.ndarray`: Decodes bytes to array.

### Compressors
*   `HuffmanCompressor`: Huffman coding.
*   `GolombRiceCompressor(k=4)`: Golomb-Rice coding.
*   `ArithmeticCompressor`: Arithmetic coding.
*   `LZWCompressor`: Lempel-Ziv-Welch compression.
*   `RLECompressor`: Run-Length Encoding.
*   `SymbolBasedCompressor(block_size=4)`: Block-based coding.
*   `BitPlaneCompressor`: Bit-plane decomposition.
*   `DCTCompressor(quality=50)`: DCT-based (JPEG-like) compression.
*   `PredictiveCompressor(predictor="left")`: Predictive coding.
*   `WaveletCompressor(threshold=10.0, levels=3)`: Haar wavelet compression.

### `CompressionResult`
*Dataclass*
*   `algorithm`: Name of the algorithm.
*   `original_size`: Size in bytes.
*   `compressed_size`: Size in bytes.
*   `compression_ratio`: Ratio (Original / Compressed).
*   `preview`: Reconstructed image (for lossy methods).
*   `metadata`: Dictionary of algorithm-specific stats.
