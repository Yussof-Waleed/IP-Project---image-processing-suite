# Core Histogram (`app.core.histogram`)

## Overview
This module deals with **Image Statistics**. It analyzes the distribution of pixel intensities (brightness) to understand contrast, exposure, and dynamic range.

## Ideology: The Fingerprint of an Image
A histogram is a fingerprint.
-   **Dark Image**: Peaks on the left.
-   **Bright Image**: Peaks on the right.
-   **Low Contrast**: Narrow spike in the middle.
-   **High Contrast**: Spread out across the full range.

## Visualizing Equalization

```mermaid
graph TD
    Input[Low Contrast Image] --> Hist[Compute Histogram]
    Hist --> CDF[Compute Cumulative Distribution (CDF)]
    CDF --> Map[Create Lookup Table]
    Map --> Output[Remap Pixels]
    
    subgraph "The Goal"
    Output -- Has --> Flat[Flat/Uniform Histogram]
    end
```

## Simplification: Spreading the Butter
Imagine you have a piece of toast (the image range 0-255) and a lump of butter (pixel values) stuck in the middle.
-   **Histogram Equalization** spreads that butter evenly across the entire slice of toast.
-   Dark areas get darker, bright areas get brighter. Details that were hidden in the gray mud suddenly pop out.

## Technical Details

### `analyze_histogram`
Returns a health report for the image.
-   **Mean**: Average brightness.
-   **Standard Deviation ($\sigma$)**: Measure of contrast. Low $\sigma$ (< 40) means the image looks "flat" or "washed out".
-   **Dynamic Range**: Difference between the brightest and darkest pixel.

### `equalize_histogram`
The algorithm to fix low contrast.
1.  **Histogram**: Count how many times each gray level appears.
2.  **CDF (Cumulative Distribution Function)**: The running total. $CDF(i)$ tells us "what percentage of pixels are darker than or equal to level $i$".
3.  **Remapping**:
    $$ \text{NewValue} = \text{round}\left( \frac{CDF(v) - CDF_{min}}{1 - CDF_{min}} \times 255 \right) $$
    This formula guarantees that the output histogram is as flat (uniform) as possible.

### `HistogramResult`
A data class containing the analysis.
-   **`is_low_contrast`**: Boolean flag triggering a UI warning.
-   **`contrast_analysis`**: Human-readable string explaining *why* the contrast is bad (e.g., "Dynamic range only 30 levels").

## Code Reference

### `analyze_histogram`
`def analyze_histogram(histogram) -> HistogramResult`
Analyzes a histogram to determine contrast quality.

### `equalize_histogram`
`def equalize_histogram(image) -> np.ndarray`
Applies histogram equalization to improve contrast.

### `compute_histogram`
`def compute_histogram(image) -> np.ndarray`
Computes a 256-bin histogram for a grayscale image.

### `HistogramResult`
*Dataclass*
*   `mean`: Average intensity.
*   `std`: Standard deviation (contrast).
*   `min_val`, `max_val`: Dynamic range.
*   `is_low_contrast`: Boolean flag.
*   `contrast_analysis`: Human-readable analysis string.
