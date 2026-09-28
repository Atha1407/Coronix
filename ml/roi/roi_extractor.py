"""A/B Guided Region of Interest (ROI) Extractor for StenoTrace.

Responsible for:
- STEP 1: Strict input validation (NumPy array, dimensions, bounds, distinct A/B points)
- STEP 2: Creating the A-B vessel corridor geometry and perpendicular vector
- STEP 3: Unrolling the corridor into an aligned rectangular ROI via perspective/affine transform
- STEP 4: Handling all segment orientations (horizontal, vertical, diagonal, arbitrary angles)
- STEP 5: Boundary handling using reflection/replication to prevent border distortion or crashes
- STEP 6: Returning structured results with full forward and inverse coordinate mapping

Coordinate Convention:
- All external points are specified as (x, y):
    x = horizontal pixel coordinate (column, 0 <= x < image_width)
    y = vertical pixel coordinate (row, 0 <= y < image_height)
- NumPy / OpenCV matrix indexing is image[y, x] (rows first, then columns).
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from ml.config import ROIConfig


@dataclass
class ROIExtractionResult:
    """Structured result returned by the ROI extraction step.

    Attributes:
        roi_image: Rectified 2D (or 3D) image strip oriented along the A-B vector.
                   Dimensions: (corridor_width, length) or (corridor_width, length, channels).
        point_a: Validated (x, y) start coordinate on full image.
        point_b: Validated (x, y) end coordinate on full image.
        corridor_width: Width of the extracted corridor strip (perpendicular to A-B).
        length: Euclidean distance between Point A and Point B (along A-B).
        angle_degrees: Vector orientation angle from Point A to Point B (-180 to 180 deg).
        corridor_polygon: (4, 2) float32 corner coordinates [C1, C2, C3, C4] on original image.
        bounding_box: (x, y, w, h) axis-aligned bounding box enclosing corridor on original image.
        transform_matrix: 3x3 perspective transform matrix mapping (image -> ROI).
        inv_transform_matrix: 3x3 inverse perspective matrix mapping (ROI -> image).
    """
    roi_image: np.ndarray
    point_a: Tuple[int, int]
    point_b: Tuple[int, int]
    corridor_width: int
    length: float
    angle_degrees: float
    corridor_polygon: np.ndarray
    bounding_box: Tuple[int, int, int, int]
    transform_matrix: np.ndarray
    inv_transform_matrix: np.ndarray

    def roi_to_image_coords(self, roi_points: np.ndarray) -> np.ndarray:
        """Project (x, y) coordinates from the rectified ROI back to the original image.

        Args:
            roi_points: (N, 2) array of coordinates in ROI pixel space.

        Returns:
            np.ndarray: (N, 2) array of corresponding coordinates in original image space.
        """
        pts = np.asarray(roi_points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.inv_transform_matrix)
        return transformed.reshape(-1, 2)

    def roi_bbox_to_image_polygon(
        self,
        roi_bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Map a bounding box [x, y, w, h] from ROI space to a 4-point polygon on original image.

        Args:
            roi_bbox: (x, y, w, h) bounding box within the rectified ROI strip.

        Returns:
            np.ndarray: (4, 2) float32 coordinates of the 4 rotated polygon corners on the image.
        """
        rx, ry, rw, rh = roi_bbox
        corners = np.array([
            [rx, ry],
            [rx + rw, ry],
            [rx + rw, ry + rh],
            [rx, ry + rh],
        ], dtype=np.float32)
        return self.roi_to_image_coords(corners)

    def roi_bbox_to_image_bbox(
        self,
        roi_bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """Convert a bounding box from ROI space to an axis-aligned bounding box on the image.

        Args:
            roi_bbox: (x, y, w, h) bounding box within the rectified ROI strip.
            image_shape: (height, width) of the original angiogram image.

        Returns:
            Tuple[int, int, int, int]: (x, y, w, h) axis-aligned bounding box clamped to image.
        """
        polygon = self.roi_bbox_to_image_polygon(roi_bbox)
        x_min = int(math.floor(np.min(polygon[:, 0])))
        y_min = int(math.floor(np.min(polygon[:, 1])))
        x_max = int(math.ceil(np.max(polygon[:, 0])))
        y_max = int(math.ceil(np.max(polygon[:, 1])))

        h, w = image_shape[:2]
        x_min = max(0, min(w - 1, x_min))
        y_min = max(0, min(h - 1, y_min))
        x_max = max(0, min(w, x_max))
        y_max = max(0, min(h, y_max))

        return (x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min))


def validate_input(
    image: np.ndarray,
    point_a: Tuple[Union[int, float], Union[int, float]],
    point_b: Tuple[Union[int, float], Union[int, float]],
    min_distance: float = 12.0
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Validate input image array and Point A / Point B coordinates.

    Validation criteria:
    - image is not None
    - image is a valid NumPy ndarray with valid dimensions (2D or 3D, H > 0, W > 0)
    - Point A and Point B exist and have 2 coordinates (x, y)
    - Point A and Point B are strictly within image bounds [0 <= x < W, 0 <= y < H]
    - Point A and Point B are not identical
    - Separation distance is >= min_distance

    Args:
        image: Angiogram image array.
        point_a: (x, y) coordinates of Point A.
        point_b: (x, y) coordinates of Point B.
        min_distance: Minimum allowed Euclidean distance between A and B in pixels.

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: Validated integer coordinates ((x1, y1), (x2, y2)).

    Raises:
        ValueError: If input violates dimension, bounds, identical points, or distance rules.
        TypeError: If image or points are of invalid types.
    """
    if image is None:
        raise ValueError("Input image cannot be None.")

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Image must be a numpy.ndarray, got {type(image)}.")

    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError(f"Invalid image dimensions: shape={getattr(image, 'shape', None)}.")

    img_h, img_w = image.shape[:2]
    if img_h <= 0 or img_w <= 0:
        raise ValueError(f"Image height and width must be > 0, got ({img_h}, {img_w}).")

    if point_a is None or point_b is None:
        raise ValueError("Point A and Point B must both be provided.")

    try:
        ax, ay = int(round(float(point_a[0]))), int(round(float(point_a[1])))
        bx, by = int(round(float(point_b[0]))), int(round(float(point_b[1])))
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"Points must contain two numerical coordinates (x, y): {exc}") from exc

    # Check bounds strictly
    if not (0 <= ax < img_w and 0 <= ay < img_h):
        raise ValueError(
            f"Point A ({ax}, {ay}) is out of bounds for image with width={img_w}, height={img_h}."
        )

    if not (0 <= bx < img_w and 0 <= by < img_h):
        raise ValueError(
            f"Point B ({bx}, {by}) is out of bounds for image with width={img_w}, height={img_h}."
        )

    # Check identity
    if ax == bx and ay == by:
        raise ValueError(
            f"Point A ({ax}, {ay}) and Point B ({bx}, {by}) are identical. "
            f"Please select two distinct points along the vessel."
        )

    dx = bx - ax
    dy = by - ay
    distance = math.hypot(dx, dy)
    if distance < min_distance:
        raise ValueError(
            f"Point A ({ax}, {ay}) and Point B ({bx}, {by}) are too close "
            f"(distance: {distance:.2f}px < minimum required {min_distance}px). "
            f"Please select two distinct points defining a vessel segment."
        )

    return (ax, ay), (bx, by)


# Backward-compatible alias for existing tests
validate_points = validate_input


def compute_corridor_geometry(
    point_a: Tuple[int, int],
    point_b: Tuple[int, int],
    corridor_width: int
) -> Tuple[float, float, np.ndarray]:
    """Compute direction vectors, segment length, and 4 corner vertices of the corridor.

    Corridor geometry:
    Given A=(x1, y1) and B=(x2, y2):
    - Tangent unit vector: u_parallel = (dx / L, dy / L)
    - Normal unit vector:  u_perp     = (-dy / L, dx / L)
    - Half-width: hw = corridor_width / 2

    Four corners ordered clockwise from top-left (near A):
    - C1: A - hw * u_perp = (x1 + hw * dy / L, y1 - hw * dx / L)
    - C2: B - hw * u_perp = (x2 + hw * dy / L, y2 - hw * dx / L)
    - C3: B + hw * u_perp = (x2 - hw * dy / L, y2 + hw * dx / L)
    - C4: A + hw * u_perp = (x1 - hw * dy / L, y1 + hw * dx / L)

    Args:
        point_a: Start point (x1, y1).
        point_b: End point (x2, y2).
        corridor_width: Perpendicular width of the corridor strip in pixels.

    Returns:
        Tuple[float, float, np.ndarray]: (length, angle_degrees, 4x2 corner polygon).
    """
    ax, ay = point_a
    bx, by = point_b

    dx = float(bx - ax)
    dy = float(by - ay)
    length = math.hypot(dx, dy)
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    hw = float(corridor_width) / 2.0

    # Normal vector perpendicular to A-B segment
    ux = dx / length
    uy = dy / length
    perp_x = -uy
    perp_y = ux

    # Corner 1 (top-left near A)
    c1 = [ax - hw * perp_x, ay - hw * perp_y]
    # Corner 2 (top-right near B)
    c2 = [bx - hw * perp_x, by - hw * perp_y]
    # Corner 3 (bottom-right near B)
    c3 = [bx + hw * perp_x, by + hw * perp_y]
    # Corner 4 (bottom-left near A)
    c4 = [ax + hw * perp_x, ay + hw * perp_y]

    polygon = np.array([c1, c2, c3, c4], dtype=np.float32)
    return length, angle_deg, polygon


def extract_roi(
    image: np.ndarray,
    point_a: Tuple[int, int],
    point_b: Tuple[int, int],
    width: Optional[int] = None,
    config: Optional[ROIConfig] = None
) -> ROIExtractionResult:
    """Extract an aligned, rectangular ROI corridor between Point A and Point B.

    The corridor is extracted using a perspective transform that maps the 4 corridor
    corners [C1, C2, C3, C4] to a rectangular destination canvas of size (length, width).
    Point A is mapped to (0, width/2) and Point B to (length, width/2).

    Boundary Handling:
    When the corridor polygon extends outside image boundaries (e.g. clicks near image
    margins), cv2.warpPerspective handles sampling using `cv2.BORDER_REFLECT_101`.
    This prevents black zero-padding artifacts, avoids gradient spikes at borders,
    and guarantees zero crashes on edge points.

    Args:
        image: Angiogram image (2D grayscale or 3D BGR/RGB).
        point_a: User-selected start point (x, y).
        point_b: User-selected end point (x, y).
        width: Optional custom corridor width in pixels (overrides config).
        config: Optional ROIConfig override.

    Returns:
        ROIExtractionResult: Structured container with the rectified strip, transformation
                             matrices, and geometric projection helpers.

    Raises:
        ValueError: If input image or points are invalid, out-of-bounds, or too close.
    """
    cfg = config or ROIConfig()
    corridor_w = width if width is not None else cfg.default_corridor_width
    corridor_w = max(cfg.min_corridor_width, min(cfg.max_corridor_width, int(corridor_w)))

    # Step 1: Strict input validation
    valid_a, valid_b = validate_input(
        image,
        point_a,
        point_b,
        min_distance=cfg.min_point_distance_px
    )

    # Step 2: Compute corridor geometry and 4 corners
    length, angle_deg, src_polygon = compute_corridor_geometry(
        valid_a,
        valid_b,
        corridor_width=corridor_w
    )

    # Destination dimensions for unrolled rectangular ROI
    target_width = max(1, int(round(length)))
    target_height = max(1, int(round(corridor_w)))

    dst_polygon = np.array([
        [0.0, 0.0],
        [float(target_width), 0.0],
        [float(target_width), float(target_height)],
        [0.0, float(target_height)],
    ], dtype=np.float32)

    # Step 3: Compute forward and inverse perspective transform matrices
    transform_matrix = cv2.getPerspectiveTransform(src_polygon, dst_polygon)
    inv_transform_matrix = cv2.getPerspectiveTransform(dst_polygon, src_polygon)

    # Step 5: Boundary-safe unrolling with BORDER_REFLECT_101
    # Ensures smooth continuity without edge artifacts or crashes near margins
    rectified_roi = cv2.warpPerspective(
        image,
        transform_matrix,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Calculate axis-aligned bounding box on original image
    x_min = int(math.floor(np.min(src_polygon[:, 0])))
    y_min = int(math.floor(np.min(src_polygon[:, 1])))
    x_max = int(math.ceil(np.max(src_polygon[:, 0])))
    y_max = int(math.ceil(np.max(src_polygon[:, 1])))

    img_h, img_w = image.shape[:2]
    clamped_x_min = max(0, min(img_w - 1, x_min))
    clamped_y_min = max(0, min(img_h - 1, y_min))
    clamped_x_max = max(0, min(img_w, x_max))
    clamped_y_max = max(0, min(img_h, y_max))

    full_bbox = (
        clamped_x_min,
        clamped_y_min,
        max(1, clamped_x_max - clamped_x_min),
        max(1, clamped_y_max - clamped_y_min),
    )

    # Step 6: Return structured result
    return ROIExtractionResult(
        roi_image=rectified_roi,
        point_a=valid_a,
        point_b=valid_b,
        corridor_width=target_height,
        length=length,
        angle_degrees=angle_deg,
        corridor_polygon=src_polygon,
        bounding_box=full_bbox,
        transform_matrix=transform_matrix,
        inv_transform_matrix=inv_transform_matrix,
    )
