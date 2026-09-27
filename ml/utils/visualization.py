"""Visualization utilities for rendering analysis overlays on angiogram frames."""

from typing import Optional, Tuple

import numpy as np

from ml.config import MEDICAL_DISCLAIMER, SEVERITY_COLORS


def draw_analysis_overlay(
    image: np.ndarray,
    point_a: Tuple[int, int],
    point_b: Tuple[int, int],
    lesion_bbox: Optional[Tuple[int, int, int, int]] = None,
    severity_category: Optional[str] = None,
    cadica_id: Optional[str] = None,
    corridor_width: int = 40,
    corridor_polygon: Optional[np.ndarray] = None,
    confidence: Optional[float] = None,
    lesions: Optional[list] = None,
    include_disclaimer: bool = False,
) -> np.ndarray:
    """Render analysis visualization overlay on top of an angiogram image.

    Renders:
    - Point A marker (distinct green circle)
    - Point B marker (distinct blue circle)
    - Segment corridor axis line between A and B
    - Vessel corridor boundaries or polygon
    - Lesion bounding box(es) if detected [x, y, w, h]
    - Severity badge / label with CADICA category & confidence

    Args:
        image: Original input image (grayscale or color).
        point_a: Pixel coordinate (x, y) for Point A.
        point_b: Pixel coordinate (x, y) for Point B.
        lesion_bbox: Optional bounding box [x, y, w, h] in full-image coordinates.
        severity_category: CADICA severity category string (e.g. '50-70%').
        cadica_id: Optional CADICA classification ID (e.g. 'p50_70').
        corridor_width: Width of analysis corridor in pixels.
        corridor_polygon: Optional (4, 2) corner coordinates of the unrolled corridor.
        confidence: Optional confidence score (0.0 to 1.0).
        lesions: Optional list of detected lesion dicts: [{"bbox": [x,y,w,h], "severity": "...", "confidence": ...}]
        include_disclaimer: Whether to render the research disclaimer text onto the image.

    Returns:
        np.ndarray: BGR image with visualization elements rendered.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError("opencv-python is required for visualization utilities.") from e

    # Ensure color 3-channel for colored overlays
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        canvas = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    ax, ay = int(round(point_a[0])), int(round(point_a[1]))
    bx, by = int(round(point_b[0])), int(round(point_b[1]))

    # 1. Draw Corridor Polygon / Corridor Boundaries if provided
    if corridor_polygon is not None and len(corridor_polygon) >= 4:
        poly_pts = np.int32(np.round(corridor_polygon)).reshape((-1, 1, 2))
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [poly_pts], (255, 230, 100))
        cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)
        cv2.polylines(canvas, [poly_pts], isClosed=True, color=(255, 220, 60), thickness=1, lineType=cv2.LINE_AA)

    # 2. Draw segment axis between Point A and Point B (light cyan line)
    cv2.line(canvas, (ax, ay), (bx, by), (255, 230, 100), 2, cv2.LINE_AA)

    # 3. Draw Point A marker (Vibrant Green circle with 'A' label)
    cv2.circle(canvas, (ax, ay), 7, (0, 220, 0), -1, cv2.LINE_AA)
    cv2.circle(canvas, (ax, ay), 9, (255, 255, 255), 1, cv2.LINE_AA)
    t_size_a = cv2.getTextSize("Point A", cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.rectangle(canvas, (ax + 10, ay - 18), (ax + 14 + t_size_a[0], ay + 2), (0, 160, 0), -1)
    cv2.putText(canvas, "Point A", (ax + 12, ay - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Draw Point B marker (Vibrant Azure/Blue circle with 'B' label)
    cv2.circle(canvas, (bx, by), 7, (230, 100, 0), -1, cv2.LINE_AA)
    cv2.circle(canvas, (bx, by), 9, (255, 255, 255), 1, cv2.LINE_AA)
    t_size_b = cv2.getTextSize("Point B", cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.rectangle(canvas, (bx + 10, by - 18), (bx + 14 + t_size_b[0], by + 2), (180, 80, 0), -1)
    cv2.putText(canvas, "Point B", (bx + 12, by - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 5. Draw Lesion Bounding Boxes
    all_lesions = []
    if lesions:
        all_lesions.extend(lesions)
    elif lesion_bbox is not None:
        all_lesions.append({
            "bbox": lesion_bbox,
            "severity": severity_category or cadica_id or "Lesion",
            "confidence": confidence if confidence is not None else 0.75,
        })

    for l_item in all_lesions:
        box = l_item["bbox"]
        lx, ly, lw, lh = [int(round(v)) for v in box]
        sev = l_item.get("severity", severity_category or "Lesion")
        conf = l_item.get("confidence")

        # Color: Red/Coral for detected lesion
        box_color = (0, 50, 255)
        cv2.rectangle(canvas, (lx, ly), (lx + lw, ly + lh), box_color, 2, cv2.LINE_AA)

        # Label badge
        conf_str = f" ({conf:.1%})" if conf is not None else ""
        label_text = f"Lesion: {sev}{conf_str}"
        t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        badge_y = max(ly - 6, t_size[1] + 6)
        cv2.rectangle(
            canvas,
            (lx, badge_y - t_size[1] - 4),
            (lx + t_size[0] + 6, badge_y + 2),
            (0, 0, 190),
            -1,
        )
        cv2.putText(
            canvas,
            label_text,
            (lx + 3, badge_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 6. Optional disclaimer banner at bottom of image
    if include_disclaimer:
        h, w = canvas.shape[:2]
        disclaimer_short = "AI Research Prototype - Not for Clinical Use"
        cv2.putText(
            canvas,
            disclaimer_short,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
    return canvas
