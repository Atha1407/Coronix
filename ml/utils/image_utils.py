"""Image utility functions for loading, conversion, and validation."""

import base64
import io
from pathlib import Path
from typing import Tuple, Union

import numpy as np


def load_image(source: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
    """Load an image from a file path, raw bytes, or return the ndarray if already loaded.

    Args:
        source: File path (str/Path), raw bytes, or an existing numpy array.

    Returns:
        np.ndarray: BGR or Grayscale image array (uint8).

    Raises:
        ValueError: If image source cannot be read or is invalid.
    """
    if isinstance(source, np.ndarray):
        return source.copy()

    try:
        import cv2
    except ImportError as e:
        raise ImportError("opencv-python is required for image loading.") from e

    if isinstance(source, (str, Path)):
        path_str = str(source)
        if not Path(path_str).is_file():
            raise FileNotFoundError(f"Image file does not exist: {path_str}")
        image = cv2.imread(path_str, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image from path: {path_str}")
        return image

    if isinstance(source, bytes):
        np_arr = np.frombuffer(source, dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from provided byte stream.")
        return image

    raise TypeError(f"Unsupported image source type: {type(source)}")


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image array to single-channel 8-bit grayscale.

    Args:
        image: Input image (grayscale 2D or color 3D).

    Returns:
        np.ndarray: 2D uint8 grayscale image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy.ndarray.")

    if image.ndim == 2:
        if image.dtype != np.uint8:
            # Scale or cast to uint8
            if image.max() <= 1.0 and image.min() >= 0.0:
                return (image * 255).astype(np.uint8)
            return np.clip(image, 0, 255).astype(np.uint8)
        return image.copy()

    try:
        import cv2
    except ImportError as e:
        raise ImportError("opencv-python is required for color conversion.") from e

    if image.ndim == 3:
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 1:
            return image[:, :, 0].copy()

    raise ValueError(f"Unsupported image array shape for grayscale conversion: {image.shape}")


def validate_coordinates(
    point: Tuple[int, int],
    image_shape: Tuple[int, int]
) -> Tuple[int, int]:
    """Validate and clamp pixel coordinates to image boundary limits.

    Args:
        point: (x, y) coordinate tuple.
        image_shape: (height, width) of the image.

    Returns:
        Tuple[int, int]: Validated and clamped (x, y) coordinates.
    """
    h, w = image_shape[:2]
    x, y = int(point[0]), int(point[1])
    clamped_x = max(0, min(w - 1, x))
    clamped_y = max(0, min(h - 1, y))
    return clamped_x, clamped_y


def encode_image_to_base64(image: np.ndarray, ext: str = ".png") -> str:
    """Encode an image array into a base64 string for API responses.

    Args:
        image: Numpy image array.
        ext: Image format extension (e.g., '.png', '.jpg').

    Returns:
        str: Base64 data string (without 'data:image/png;base64,' prefix).
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError("opencv-python is required for image encoding.") from e

    success, buffer = cv2.imencode(ext, image)
    if not success:
        raise ValueError("Failed to encode image to base64.")
    return base64.b64encode(buffer).decode("utf-8")
