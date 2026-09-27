import cv2
import numpy as np
from typing import Tuple, Dict, Any, Union, List


def create_roi(image: np.ndarray, point_a: Tuple[int, int], point_b: Tuple[int, int], width: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a corridor-shaped ROI around the line joining A and B.

    image   : input image (numpy array)
    point_a : (x, y) coordinates of segment start
    point_b : (x, y) coordinates of segment end
    width   : corridor width in pixels

    Returns:
        roi_image : image with corridor isolated (rest zeroed out)
        mask      : binary uint8 mask of the corridor
    """
    # Create empty mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Ensure integer coordinates
    pt_a = (int(round(point_a[0])), int(round(point_a[1])))
    pt_b = (int(round(point_b[0])), int(round(point_b[1])))

    # Draw line corridor between A and B
    cv2.line(
        mask,
        pt_a,
        pt_b,
        255,
        thickness=width
    )

    # Extract the ROI
    roi_image = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    return roi_image, mask


def extract_roi_corridor(
    image: np.ndarray,
    point_a: Union[Tuple[int, int], List[int]],
    point_b: Union[Tuple[int, int], List[int]],
    width: int = 40
) -> Dict[str, Any]:
    """
    Extracts corridor-shaped ROI along with its bounding box and cropped patch.

    Returns dictionary containing:
        - roi_image: full-size image with only corridor pixels kept
        - roi_mask: full-size binary mask of corridor
        - roi_bbox: [x, y, w, h] enclosing the corridor
        - cropped_roi: cropped rectangular sub-image bounded by roi_bbox
        - cropped_mask: cropped binary mask bounded by roi_bbox
        - corridor_length: Euclidean distance between A and B
        - point_a: sanitized (x, y)
        - point_b: sanitized (x, y)
    """
    pt_a = (int(round(point_a[0])), int(round(point_a[1])))
    pt_b = (int(round(point_b[0])), int(round(point_b[1])))

    # Generate corridor mask and full-frame isolated ROI
    roi_image, mask = create_roi(image, pt_a, pt_b, width=width)

    # Find bounding box enclosing the corridor
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = image.shape[:2]

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
    else:
        # Fallback bounding box if contour detection fails
        x_min = max(0, min(pt_a[0], pt_b[0]) - width // 2)
        y_min = max(0, min(pt_a[1], pt_b[1]) - width // 2)
        x_max = min(img_w, max(pt_a[0], pt_b[0]) + width // 2)
        y_max = min(img_h, max(pt_a[1], pt_b[1]) + width // 2)
        x, y, w, h = x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min)

    # Clamp bbox within image boundary
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))

    cropped_roi = roi_image[y : y + h, x : x + w].copy()
    cropped_mask = mask[y : y + h, x : x + w].copy()

    length = float(np.hypot(pt_b[0] - pt_a[0], pt_b[1] - pt_a[1]))

    return {
        "roi_image": roi_image,
        "roi_mask": mask,
        "roi_bbox": [int(x), int(y), int(w), int(h)],
        "cropped_roi": cropped_roi,
        "cropped_mask": cropped_mask,
        "corridor_length": length,
        "point_a": pt_a,
        "point_b": pt_b,
        "corridor_width": width
    }