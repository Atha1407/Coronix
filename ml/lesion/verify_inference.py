"""Verification script for CADICA lesion detector inference pipeline."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch

from ml.lesion.config import PROJECT_ROOT, TEST_CSV
from ml.lesion.device import get_device
from ml.lesion.inference import load_model, predict


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def get_5_representative_test_images(manifest_csv: Path) -> List[Dict[str, Any]]:
    """Retrieve 5 representative test images containing ground-truth lesions."""
    target_frame_ids = [
        "p4_v2_00020",
        "p1_v10_00030",
        "p1_v2_00032",
        "p1_v3_00021",
        "p25_v2_00019",
    ]

    frames_dict: Dict[str, Dict[str, Any]] = {}

    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("is_lesion") != "1":
                continue

            frame_id = row["frame_id"]
            if frame_id not in target_frame_ids:
                continue

            img_rel_path = row["image_path"]
            img_abs_path = PROJECT_ROOT / img_rel_path

            if not img_abs_path.exists():
                continue

            bx = float(row["bbox_x"])
            by = float(row["bbox_y"])
            bw = float(row["bbox_width"])
            bh = float(row["bbox_height"])
            sev = row["severity_category"]

            gt_box_xyxy = [bx, by, bx + bw, by + bh]

            if frame_id not in frames_dict:
                frames_dict[frame_id] = {
                    "frame_id": frame_id,
                    "image_path": img_abs_path,
                    "rel_path": img_rel_path,
                    "patient_id": row["patient_id"],
                    "video_id": row["video_id"],
                    "gt_boxes": [],
                    "severities": [],
                }

            frames_dict[frame_id]["gt_boxes"].append(gt_box_xyxy)
            frames_dict[frame_id]["severities"].append(sev)

    # Preserve exact order of target_frame_ids
    ordered = [frames_dict[fid] for fid in target_frame_ids if fid in frames_dict]
    return ordered


def render_and_save_prediction(
    image_bgr: np.ndarray,
    frame_id: str,
    gt_boxes: List[List[float]],
    gt_severities: List[str],
    detections: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Render original angiogram with GT (green) and predictions (red)."""
    vis = image_bgr.copy()
    h, w = vis.shape[:2]

    # Draw Ground Truth boxes in GREEN
    for box, sev in zip(gt_boxes, gt_severities):
        x1, y1, x2, y2 = [int(round(c)) for c in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label_text = f"GT: {sev}"
        t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        label_y = max(y1 - 6, t_size[1] + 4)
        cv2.rectangle(
            vis,
            (x1, label_y - t_size[1] - 4),
            (x1 + t_size[0] + 4, label_y + 2),
            (0, 160, 0),
            -1,
        )
        cv2.putText(
            vis,
            label_text,
            (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Draw Predictions in RED
    for det in detections:
        x1, y1, x2, y2 = [int(round(c)) for c in det["bbox"]]
        conf = det["confidence"]
        sev = det["severity"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label_text = f"Pred: {sev} ({conf:.2f})"
        t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        label_y = min(y2 + t_size[1] + 6, h - 4)
        cv2.rectangle(
            vis,
            (x1, label_y - t_size[1] - 4),
            (x1 + t_size[0] + 4, label_y + 2),
            (0, 0, 200),
            -1,
        )
        cv2.putText(
            vis,
            label_text,
            (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Top header banner
    banner_height = 36
    banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)

    header_text = (
        f"Frame: {frame_id} | GT: {len(gt_boxes)} (Green) | Preds: {len(detections)} (Red)"
    )
    cv2.putText(
        banner,
        header_text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    combined = np.vstack([banner, vis])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined)


def run_inference_verification() -> Dict[str, Any]:
    """Execute complete inference verification task."""
    checkpoint_path = PROJECT_ROOT / "ml" / "models" / "best_lesion_detector.pth"
    output_dir = PROJECT_ROOT / "ml" / "models" / "predictions" / "inference_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CADICA LESION DETECTOR INFERENCE VERIFICATION")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)

    # 1. Device check
    device = get_device(verbose=True)
    cuda_pass = (device.type == "cuda")

    # 2. Checkpoint load
    print("\n[Step 1] Loading model checkpoint...")
    try:
        model = load_model(checkpoint_path=checkpoint_path, device=device)
        model_device = next(model.parameters()).device
        print(f"Model loaded successfully on: {model_device}")
        checkpoint_load_pass = True
    except Exception as e:
        print(f"FATAL: Failed to load checkpoint: {e}")
        checkpoint_load_pass = False
        raise

    # 3. Select 5 representative test images with GT lesions
    print("\n[Step 2] Selecting 5 representative test images containing ground-truth lesions...")
    selected_frames = get_5_representative_test_images(TEST_CSV)
    for idx, f in enumerate(selected_frames, start=1):
        print(
            f"  {idx}. Frame: {f['frame_id']} | Patient: {f['patient_id']} | "
            f"Video: {f['video_id']} | Severities: {f['severities']} | GT Boxes: {len(f['gt_boxes'])}"
        )

    # 4. Run inference and record table
    print("\n[Step 3] Running inference on selected images...")
    results_table = []
    generated_images_count = 0

    for idx, frame_info in enumerate(selected_frames, start=1):
        img_path = frame_info["image_path"]
        frame_id = frame_info["frame_id"]
        gt_boxes = frame_info["gt_boxes"]
        gt_sevs = frame_info["severities"]

        # Read image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        # Run inference using predict (confidence threshold 0.15 matching config)
        detections = predict(
            image=img_bgr,
            model=model,
            device=device,
            confidence_threshold=0.15,
        )

        # Generate visual prediction image
        out_img_name = f"inference_{idx:02d}_{frame_id}.png"
        out_img_path = output_dir / out_img_name
        render_and_save_prediction(
            image_bgr=img_bgr,
            frame_id=frame_id,
            gt_boxes=gt_boxes,
            gt_severities=gt_sevs,
            detections=detections,
            output_path=out_img_path,
        )
        generated_images_count += 1

        # Match detection to GT for table
        # Pick the detection that best overlaps with GT, or highest confidence
        if detections:
            best_det = None
            best_iou = 0.0
            best_gt_box = gt_boxes[0]

            for d in detections:
                for gt_b in gt_boxes:
                    iou = compute_iou(d["bbox"], gt_b)
                    if iou > best_iou:
                        best_iou = iou
                        best_det = d
                        best_gt_box = gt_b

            # If no detection overlapped GT, pick top confidence detection
            if best_det is None:
                best_det = max(detections, key=lambda d: d["confidence"])
                best_gt_box = gt_boxes[0]
                best_iou = compute_iou(best_det["bbox"], best_gt_box)

            pred_box = best_det["bbox"]
            pred_sev = best_det["severity"]
            pred_conf = best_det["confidence"]

            results_table.append({
                "image": frame_info["rel_path"],
                "frame_id": frame_id,
                "predicted_bbox": f"[{pred_box[0]:.1f}, {pred_box[1]:.1f}, {pred_box[2]:.1f}, {pred_box[3]:.1f}]",
                "predicted_severity": pred_sev,
                "confidence": f"{pred_conf:.4f}",
                "ground_truth_bbox": f"[{best_gt_box[0]:.1f}, {best_gt_box[1]:.1f}, {best_gt_box[2]:.1f}, {best_gt_box[3]:.1f}]",
                "IoU": f"{best_iou:.4f}",
                "visual_file": out_img_name,
            })
        else:
            # No detection above threshold
            first_gt = gt_boxes[0]
            results_table.append({
                "image": frame_info["rel_path"],
                "frame_id": frame_id,
                "predicted_bbox": "None",
                "predicted_severity": "None",
                "confidence": "0.0000",
                "ground_truth_bbox": f"[{first_gt[0]:.1f}, {first_gt[1]:.1f}, {first_gt[2]:.1f}, {first_gt[3]:.1f}]",
                "IoU": "0.0000",
                "visual_file": out_img_name,
            })

    # Print requested table
    print("\n" + "=" * 125)
    print("INFERENCE VERIFICATION RESULTS TABLE")
    print("=" * 125)
    col_fmt = "{:<48} | {:<25} | {:<18} | {:<10} | {:<25} | {:<6}"
    print(col_fmt.format("image", "predicted_bbox", "predicted_severity", "confidence", "ground_truth_bbox", "IoU"))
    print("-" * 125)
    for r in results_table:
        short_img = Path(r["image"]).name
        print(col_fmt.format(
            short_img,
            r["predicted_bbox"],
            r["predicted_severity"],
            r["confidence"],
            r["ground_truth_bbox"],
            r["IoU"],
        ))
    print("=" * 125)

    inference_pass = (generated_images_count == 5)

    print("\nVERIFICATION SUMMARY:")
    print(f"CHECKPOINT LOAD: {'PASS' if checkpoint_load_pass else 'FAIL'}")
    print(f"CUDA INFERENCE: {'PASS' if cuda_pass else 'FAIL'}")
    print(f"5 IMAGE INFERENCE: {'PASS' if inference_pass else 'FAIL'}")
    print(f"PREDICTIONS GENERATED: {generated_images_count}")
    print(f"OUTPUT DIRECTORY: {output_dir}")

    return {
        "checkpoint_load_pass": checkpoint_load_pass,
        "cuda_pass": cuda_pass,
        "inference_pass": inference_pass,
        "generated_count": generated_images_count,
        "output_dir": output_dir,
        "results_table": results_table,
    }


if __name__ == "__main__":
    run_inference_verification()
