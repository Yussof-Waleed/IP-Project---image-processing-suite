# Core Interpolation (`app.core.interpolation`)

## Overview
This module handles **Resizing**: changing the dimensions of an image. Since pixels are discrete dots, making an image larger requires "guessing" what goes in the gaps between the original dots. This guessing is called **Interpolation**.

## Ideology: Quality vs. Speed
There is no "best" method. It's a trade-off.
-   **Nearest Neighbor**: Instant, but ugly (blocky).
-   **Bilinear**: Fast and decent, but slightly blurry.
-   **Bicubic**: Slow and beautiful, sharp and smooth.

We implement all three manually to demonstrate the math behind the magic.

## Visualizing Interpolation

```mermaid
graph TD
    Grid[Grid of Pixels]
    Point[Sample Point (float coordinates)]
    
    Point --> NN[Nearest Neighbor]
    NN --> P1[Pick closest pixel]
    
    Point --> BL[Bilinear]
    BL --> P4[Pick 4 neighbors]
    P4 --> W4[Weighted Average]
    
    Point --> BC[Bicubic]
    BC --> P16[Pick 16 neighbors]
    P16 --> W16[Cubic Spline Weights]
```

## Simplification: Filling the Gaps
Imagine you have a small mosaic and want to make a mural 10 times bigger.
-   **Nearest Neighbor**: You just use bigger tiles. A single red tile becomes a 10x10 block of red tiles. It looks like Lego.
-   **Bilinear**: You blend the colors. If a red tile is next to a blue tile, you paint a gradient (purple) in between.
-   **Bicubic**: You look at the pattern. If the red and blue tiles are part of a curve, you paint a curve in between. It's smarter.

## Technical Details

### `resize_nearest_neighbor`
-   **Algorithm**: Round the float coordinate to the nearest integer.
-   **Math**: $I(x, y) = I(\text{round}(x), \text{round}(y))$
-   **Pros**: Preserves hard edges (good for pixel art/QR codes). Fastest ($O(1)$).
-   **Cons**: Aliasing (jaggies).

### `resize_bilinear`
-   **Algorithm**: Linear interpolation in X, then in Y.
-   **Math**: Uses 4 neighbors ($2 \times 2$).
    $$ I(x, y) = (1-\Delta x)(1-\Delta y)I_{00} + \Delta x(1-\Delta y)I_{10} + \dots $$
-   **Pros**: Smooth. No jaggies.
-   **Cons**: Blurs sharp edges. Low-pass filter effect.

### `resize_bicubic`
-   **Algorithm**: Cubic interpolation using Catmull-Rom splines.
-   **Math**: Uses 16 neighbors ($4 \times 4$).
    -   Weights are calculated using a cubic polynomial $W(x)$.
    -   $W(x)$ allows for negative weights, which creates a "sharpening" effect (undershoot/overshoot) that preserves edges better than Bilinear.
-   **Pros**: Sharpest results. Best for photos.
-   **Cons**: Slowest ($16 \times$ more lookups than Nearest). Can create ringing artifacts (halos) around strong edges.

## Code Reference

### `resize_nearest_neighbor`
`def resize_nearest_neighbor(image, new_shape) -> np.ndarray`
Resizes image using nearest neighbor interpolation. Fast but blocky.

### `resize_bilinear`
`def resize_bilinear(image, new_shape) -> np.ndarray`
Resizes image using bilinear interpolation. Smooth but slightly blurry.

### `resize_bicubic`
`def resize_bicubic(image, new_shape) -> np.ndarray`
Resizes image using bicubic interpolation. Sharp and high quality.

### `InterpolationMethod`
*Enum*
*   `NEAREST`
*   `BILINEAR`
*   `BICUBIC`
