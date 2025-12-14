# Core IO (`app.core.io`)

## Overview
This module handles the **Input/Output** operations: reading images from disk into memory and saving them back. It acts as the bridge between the file system and the application's internal numpy array representation.

## Ideology: Robustness & Safety
Loading files is prone to errors (missing files, corrupt data, wrong formats). This module prioritizes **safety**:
-   **Wrappers**: We don't call `PIL.Image.open` directly in the GUI. We use `load_image` which handles exceptions and ensures the output is always a standard numpy array.
-   **Consistency**: Regardless of the input format (PNG, JPG, BMP), the application always receives a standard `uint8` numpy array.

## Simplification: The Doorway
Think of the application as a clean room.
-   **`load_image`**: This is the decontamination chamber. It takes dirty, messy files from the outside world (disk), cleans them up, removes weird formats (like palettes or alpha channels if not needed), and brings them inside as standard "clean" data.
-   **`save_image`**: This packages the clean data back into a box (file format) to be sent out.

## Technical Details

### `load_image`
Loads an image and standardizes it.
-   **Input**: File path.
-   **Process**:
    1.  Checks if file exists.
    2.  Opens with `Pillow`.
    3.  **Converts to RGB**: If the image is CMYK, Palette-based, or RGBA, it converts it to standard RGB. This prevents crashes later in processing algorithms that expect 3 channels.
    4.  **Converts to Numpy**: Returns a `uint8` array.

### `save_image`
Saves a numpy array to disk.
-   **Input**: Numpy array and destination path.
-   **Process**:
    1.  Validates dimensions (must be 2D or 3D).
    2.  Clips values to 0-255 (safety check).
    3.  Converts to `Pillow` image.
    4.  Saves to disk (format inferred from file extension).

### `ImageMetadata`
A lightweight data class for "peeking" at a file.
-   **Why?** Sometimes we want to know how big an image is without loading the whole thing into RAM (which could be slow for huge files).
-   **`get_image_metadata`**: Opens the file, reads the header (width, height, format), and closes it immediately.

## Code Reference

### `load_image`
`def load_image(path: Path | str) -> np.ndarray`
Loads an image from disk.
*   **Returns**: Numpy array (uint8). RGB (H, W, 3) or Grayscale (H, W).
*   **Raises**: `FileNotFoundError`, `ValueError`.

### `save_image`
`def save_image(image: np.ndarray, path: Path | str) -> None`
Saves a numpy array as an image file.
*   **Input**: (H, W) or (H, W, 3) uint8 array.
*   **Raises**: `ValueError` for invalid shapes.

### `get_image_metadata`
`def get_image_metadata(path: Path | str) -> ImageMetadata`
Extracts metadata without loading the full image data.

### `ImageMetadata`
*Dataclass*
*   `width`: int
*   `height`: int
*   `channels`: int
*   `file_size_bytes`: int
*   `format`: str
