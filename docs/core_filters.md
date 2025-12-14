# Core Filters (`app.core.filters`)

## Overview
This module implements **Spatial Filters**: operations that modify a pixel based on its neighbors. This includes blurring (smoothing), sharpening, and edge detection.

## Ideology: The Kernel is King
All filters in this module (except Median) are based on **Convolution**. They differ only in their **Kernel**—the small matrix that defines how neighbors are weighted.
-   **Gaussian**: Kernel has a bell-curve shape (center is heavy, edges are light).
-   **Laplacian**: Kernel has a positive center and negative edges (or vice versa).
-   **Sobel**: Kernel calculates the difference between left/right or top/bottom neighbors.

## Visualizing Kernels

| Filter | Kernel Shape | Effect |
| :--- | :--- | :--- |
| **Box Blur** | Flat (all 1s) | Averages everything. Rough blur. |
| **Gaussian** | Hill (peak in middle) | Smooth, natural blur. |
| **Laplacian** | Valley (dip in middle) | Highlights rapid changes (edges). |
| **Sobel** | Slope (gradient) | Highlights directional edges. |

## Simplification: The Sunglasses
-   **Gaussian Filter**: Like looking through frosted glass. Details get lost, noise disappears.
-   **Median Filter**: Like a voting system. If one pixel is bright red but all neighbors are black, the "vote" goes to black. Great for removing "salt and pepper" noise (random white/black dots).
-   **Edge Detection**: Like sketching. It ignores flat colors and only draws lines where colors change abruptly.

## Technical Details

### Low-Pass Filters (Blur)
-   **`GaussianFilter`**:
    -   Uses `build_gaussian_kernel(size, sigma)`.
    -   **Sigma ($\sigma$)** controls the "spread". Higher $\sigma$ = blurrier.
    -   **Size** determines how many neighbors are included. Must be large enough to capture the bell curve (usually $6\sigma$).

-   **`MedianFilter`**:
    -   **Not** a convolution. It sorts all pixels in the window and picks the middle value.
    -   **Non-linear**: It preserves edges better than Gaussian blur while removing outliers.

### High-Pass Filters (Edges)
-   **`LaplacianFilter`**:
    -   Calculates the 2nd derivative (rate of change of the rate of change).
    -   Zero-crossing indicates an edge.
    -   Kernel: $\begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$

-   **`SobelFilter`**:
    -   Calculates the 1st derivative (gradient).
    -   **X-Direction**: $\begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$ (Vertical edges).
    -   **Y-Direction**: $\begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$ (Horizontal edges).
    -   **Magnitude**: $\sqrt{G_x^2 + G_y^2}$. Combines both directions.

-   **`SharpeningFilter`**:
    -   Uses **Unsharp Masking**.
    -   Formula: $Original + Amount \times (Original - Blurred)$.
    -   It exaggerates the difference between the image and its blurred version, effectively boosting high frequencies (edges).

## Code Reference

### `GaussianFilter`
Applies Gaussian smoothing.
*   `size`: Kernel size (odd integer).
*   `sigma`: Standard deviation (float).

### `MedianFilter`
Applies median filtering (non-linear).
*   `size`: Window size (odd integer).

### `LaplacianFilter`
Detects edges using the Laplacian operator.
*   `variant`: "standard" or "diagonal".

### `SobelFilter`
Detects edges using Sobel operators.
*   `direction`: "x", "y", or "both".

### `GradientFilter`
Simple gradient magnitude using `[-1, 0, 1]` kernel.

### `SharpeningFilter`
Sharpens image using unsharp masking.
*   `amount`: Strength of sharpening.
*   `radius`: Blur radius for the mask.
