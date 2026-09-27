"""Visual verification script for A/B guided ROI corridor extraction.

Renders and saves a visualization containing:
1. Original angiogram (or labeled synthetic software test frame)
2. Point A marker
3. Point B marker
4. A-B connecting line (segment centerline)
5. Visible ROI corridor polygon
6. Extracted rectangular ROI strip
7. Back-projected candidate region overlay

Saves output to ml/data/processed/roi_corridor_extraction_demo.png.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

# Ensure repository root is on sys.path for direct script execution
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ml.config import ROIConfig
from ml.preprocessing.image_preprocessing import preprocess_image
from ml.roi.roi_extractor import extract_roi


def run_roi_visualization(
    input_image_path: Optional[str] = None,
    point_a: Optional[Tuple[int, int]] = None,
    point_b: Optional[Tuple[int, int]] = None,
    corridor_width: int = 40,
    output_dir: Optional[str] = None
) -> Path:
    """Generate and save ROI corridor extraction visualization.

    Args:
        input_image_path: Optional path to an actual CADICA frame.
        point_a: Optional (x, y) start coordinate.
        point_b: Optional (x, y) end coordinate.
        corridor_width: Width of the corridor strip in pixels.
        output_dir: Destination directory for saving the figure.

    Returns:
        Path: Path to the generated figure file.
    """
    repo_root = Path(__file__).resolve().parent.parent
    demo_dir = repo_root / "data" / "demo"
    raw_dir = repo_root / "data" / "raw"
    save_dir = Path(output_dir) if output_dir else (repo_root / "data" / "processed")
    save_dir.mkdir(parents=True, exist_ok=True)

    target_path = None
    if input_image_path and Path(input_image_path).is_file():
        target_path = Path(input_image_path)
    else:
        candidates = list(demo_dir.glob("*.png")) + list(demo_dir.glob("*.jpg")) + \
                     list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpg"))
        candidates = [p for p in candidates if p.name != ".gitkeep"]
        if candidates:
            target_path = candidates[0]

    cfg = ROIConfig(default_corridor_width=corridor_width)

    if target_path:
        print(f"[INFO] Using actual angiogram image: {target_path.name}")
        raw_img = cv2.imread(str(target_path))
        # Default test points on 512x512 image if not supplied
        pt_a = point_a or (150, 180)
        pt_b = point_b or (340, 310)
        title_tag = f"Angiogram: {target_path.name}"
    else:
        print(
            "[INFO] No actual angiogram frame found in ml/data/demo/ or ml/data/raw/. "
            "Using synthetic test array strictly for software stage verification."
        )
        # Synthetic test pattern with an oblique diagonal simulated vessel strip
        h, w = 512, 512
        synth = np.full((h, w), 180, dtype=np.uint8)
        # Draw a simulated dark vessel line across the canvas
        cv2.line(synth, (120, 160), (360, 320), 50, 6, cv2.LINE_AA)
        cv2.circle(synth, (240, 240), 2, 40, -1)  # small focal point
        raw_img = synth
        pt_a = point_a or (120, 160)
        pt_b = point_b or (360, 320)
        title_tag = "Synthetic Software Test Frame"

    # Preprocessing integration: raw -> preprocess -> extract_roi
    preprocessed = preprocess_image(raw_img)
    roi_result = extract_roi(preprocessed, point_a=pt_a, point_b=pt_b, width=corridor_width, config=cfg)

    # Prepare visual canvas for original image with corridor overlays
    if raw_img.ndim == 2:
        canvas = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
    else:
        canvas = raw_img.copy()

    # Draw corridor polygon (cyan boundary)
    poly = np.int32([roi_result.corridor_polygon])
    cv2.polylines(canvas, poly, isClosed=True, color=(255, 220, 0), thickness=2, lineType=cv2.LINE_AA)

    # Draw A-B segment centerline (yellow line)
    cv2.line(canvas, roi_result.point_a, roi_result.point_b, (0, 255, 255), 2, cv2.LINE_AA)

    # Draw Point A (distinct green marker)
    cv2.circle(canvas, roi_result.point_a, 6, (0, 220, 0), -1, cv2.LINE_AA)
    cv2.circle(canvas, roi_result.point_a, 8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "A", (roi_result.point_a[0] + 10, roi_result.point_a[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw Point B (distinct blue marker)
    cv2.circle(canvas, roi_result.point_b, 6, (230, 100, 0), -1, cv2.LINE_AA)
    cv2.circle(canvas, roi_result.point_b, 8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "B", (roi_result.point_b[0] + 10, roi_result.point_b[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 100, 0), 2, cv2.LINE_AA)

    # Simulate a candidate lesion box in ROI space and back-project to original image
    roi_h, roi_w = roi_result.roi_image.shape[:2]
    sim_roi_bbox = (int(roi_w * 0.4), int(roi_h * 0.2), int(roi_w * 0.2), int(roi_h * 0.6))
    projected_poly = roi_result.roi_bbox_to_image_polygon(sim_roi_bbox)
    cv2.polylines(canvas, [np.int32(projected_poly)], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    # Prepare figure: Left = original with corridor & markers, Right = unrolled rectangular ROI
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0])

    ax_main = fig.add_subplot(gs[:, 0])
    ax_roi = fig.add_subplot(gs[0, 1])
    ax_annot = fig.add_subplot(gs[1, 1])

    # Left plot: original angiogram with corridor geometry
    ax_main.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax_main.set_title(
        f"Original Angiogram with A/B Corridor\n{title_tag}\n"
        f"Point A={roi_result.point_a}, Point B={roi_result.point_b}, "
        f"Length={roi_result.length:.1f}px, Angle={roi_result.angle_degrees:.1f}°",
        fontsize=11, fontweight="bold"
    )
    ax_main.axis("off")

    # Top right: unrolled rectangular ROI
    ax_roi.imshow(roi_result.roi_image, cmap="gray")
    # Draw centerline and simulated lesion in ROI
    ax_roi.axhline(y=roi_h / 2, color="yellow", linestyle="--", linewidth=1, alpha=0.7, label="Centerline")
    ax_roi.plot(0, roi_h / 2, marker="o", color="green", markersize=8, label="Point A")
    ax_roi.plot(roi_w - 1, roi_h / 2, marker="o", color="blue", markersize=8, label="Point B")
    rect = plt.Rectangle(
        (sim_roi_bbox[0], sim_roi_bbox[1]), sim_roi_bbox[2], sim_roi_bbox[3],
        edgecolor="red", facecolor="none", linewidth=2, label="Sample Lesion Window"
    )
    ax_roi.add_patch(rect)
    ax_roi.set_title(
        f"Extracted Rectangular ROI Strip\nDimensions: ({roi_h}px height × {roi_w}px length)",
        fontsize=11, fontweight="bold"
    )
    ax_roi.set_xlabel("Pixel Index along Vessel Axis (A -> B)")
    ax_roi.set_ylabel("Perpendicular Offset (px)")
    ax_roi.legend(loc="upper right", fontsize=8)

    # Bottom right: Preprocessed frame for comparison
    ax_annot.imshow(preprocessed, cmap="gray")
    ax_annot.set_title("Preprocessed Angiogram (CLAHE + Denoised)", fontsize=11, fontweight="bold")
    ax_annot.axis("off")

    plt.tight_layout()
    output_file = save_dir / "roi_corridor_extraction_demo.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[SUCCESS] ROI visualization saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    run_roi_visualization()
