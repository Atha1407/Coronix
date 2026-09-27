"""Evaluation metrics for CADICA lesion detection and severity classification."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.lesion.config import (
    CLASS_NAMES,
    ID_TO_CLASS,
    NUM_CLASSES,
)


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute pairwise Intersection over Union (IoU) between two sets of boxes.

    Args:
        boxes1: Shape [N, 4] in [x1, y1, x2, y2].
        boxes2: Shape [M, 4] in [x1, y1, x2, y2].

    Returns:
        np.ndarray: IoU matrix of shape [N, M].
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    # Coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0:1], boxes1[:, 1:2], boxes1[:, 2:3], boxes1[:, 3:4]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y2 = np.minimum(b1_y2, b2_y2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = area1 + area2 - inter_area

    # Avoid divide by zero
    iou = np.where(union_area > 0.0, inter_area / union_area, 0.0)
    return iou


def match_detections(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int, List[float], Dict[int, Dict[str, int]]]:
    """Greedy bipartite matching of detections to ground-truth boxes.

    Args:
        pred_boxes: [N, 4] array of predicted boxes.
        pred_scores: [N] array of confidence scores.
        pred_labels: [N] array of predicted class integers.
        gt_boxes: [M, 4] array of ground-truth boxes.
        gt_labels: [M] array of ground-truth class integers.
        iou_threshold: IoU cutoff for a true positive match (default: 0.5).

    Returns:
        tp_det: True positive detection count.
        fp_det: False positive detection count.
        fn_det: False negative detection count.
        matched_ious: List of IoU values for matched pairs.
        per_class_counts: Dict mapping class_id to {'tp': count, 'fp': count, 'fn': count}.
    """
    per_class = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in range(1, NUM_CLASSES)}
    matched_ious: List[float] = []

    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)

    if num_preds == 0:
        # All ground truth are false negatives
        for gl in gt_labels:
            if gl in per_class:
                per_class[gl]["fn"] += 1
        return 0, 0, num_gts, [], per_class

    if num_gts == 0:
        # All predictions are false positives
        for pl in pred_labels:
            if pl in per_class:
                per_class[pl]["fp"] += 1
        return 0, num_preds, 0, [], per_class

    # Sort predictions by descending confidence score
    order = np.argsort(-pred_scores)
    sorted_p_boxes = pred_boxes[order]
    sorted_p_labels = pred_labels[order]

    iou_mat = compute_iou_matrix(sorted_p_boxes, gt_boxes)  # [num_preds, num_gts]

    matched_gt = set()
    tp_det = 0
    fp_det = 0

    for p_idx in range(num_preds):
        best_gt_idx = -1
        best_iou = -1.0
        for g_idx in range(num_gts):
            if g_idx in matched_gt:
                continue
            cur_iou = iou_mat[p_idx, g_idx]
            if cur_iou >= iou_threshold and cur_iou > best_iou:
                best_iou = cur_iou
                best_gt_idx = g_idx

        p_label = int(sorted_p_labels[p_idx])

        if best_gt_idx >= 0:
            # Matched true positive detection
            tp_det += 1
            matched_gt.add(best_gt_idx)
            matched_ious.append(float(best_iou))

            g_label = int(gt_labels[best_gt_idx])
            if p_label == g_label:
                if p_label in per_class:
                    per_class[p_label]["tp"] += 1
            else:
                if p_label in per_class:
                    per_class[p_label]["fp"] += 1
                if g_label in per_class:
                    per_class[g_label]["fn"] += 1
        else:
            # Unmatched prediction: False positive
            fp_det += 1
            if p_label in per_class:
                per_class[p_label]["fp"] += 1

    # Any unmatched ground-truth boxes are false negatives
    fn_det = num_gts - len(matched_gt)
    for g_idx in range(num_gts):
        if g_idx not in matched_gt:
            g_label = int(gt_labels[g_idx])
            if g_label in per_class:
                per_class[g_label]["fn"] += 1

    return tp_det, fp_det, fn_det, matched_ious, per_class


@torch.no_grad()
def evaluate_detector(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.3,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate detector on a dataset using greedy bipartite matching at an IoU threshold.

    Computes:
    - Number of images, ground-truth boxes, predicted boxes
    - Detection Precision, Recall, F1
    - Mean IoU of matched predictions
    - Per-severity-class classification statistics

    Args:
        model: PyTorch Faster R-CNN model.
        dataloader: DataLoader returning (images, targets).
        device: Compute device (cuda or cpu).
        iou_threshold: Minimum IoU for a match (default: 0.5).
        confidence_threshold: Minimum prediction score (default: 0.3).
        max_batches: Optional limit on number of batches to evaluate.

    Returns:
        Dict[str, Any]: Comprehensive evaluation report.
    """
    model.eval()

    total_images = 0
    total_gt_boxes = 0
    total_pred_boxes = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_matched_ious: List[float] = []

    # Aggregated per-class counts
    per_class_accum = {
        cid: {"tp": 0, "fp": 0, "fn": 0, "support": 0, "preds": 0}
        for cid in range(1, NUM_CLASSES)
    }

    notes: List[str] = [
        f"Evaluated at IoU threshold = {iou_threshold:.2f}",
        f"Confidence score filter = {confidence_threshold:.2f}",
    ]

    for batch_idx, (images, targets) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = [img.to(device) for img in images]
        predictions = model(images)

        for target, pred in zip(targets, predictions):
            total_images += 1

            # Extract ground-truth
            gt_b = target["boxes"]
            gt_l = target["labels"]
            if isinstance(gt_b, torch.Tensor):
                gt_boxes_np = gt_b.detach().cpu().numpy()
                gt_labels_np = gt_l.detach().cpu().numpy()
            else:
                gt_boxes_np = np.asarray(gt_b, dtype=np.float32)
                gt_labels_np = np.asarray(gt_l, dtype=np.int64)

            total_gt_boxes += len(gt_boxes_np)
            for gl in gt_labels_np:
                if int(gl) in per_class_accum:
                    per_class_accum[int(gl)]["support"] += 1

            # Extract predictions and filter by confidence
            p_boxes = pred["boxes"].detach().cpu().numpy()
            p_scores = pred["scores"].detach().cpu().numpy()
            p_labels = pred["labels"].detach().cpu().numpy()

            keep = p_scores >= confidence_threshold
            filtered_boxes = p_boxes[keep]
            filtered_scores = p_scores[keep]
            filtered_labels = p_labels[keep]

            total_pred_boxes += len(filtered_boxes)
            for pl in filtered_labels:
                if int(pl) in per_class_accum:
                    per_class_accum[int(pl)]["preds"] += 1

            tp, fp, fn, matched_ious, per_cls = match_detections(
                pred_boxes=filtered_boxes,
                pred_scores=filtered_scores,
                pred_labels=filtered_labels,
                gt_boxes=gt_boxes_np,
                gt_labels=gt_labels_np,
                iou_threshold=iou_threshold,
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn
            all_matched_ious.extend(matched_ious)

            for cid, counts in per_cls.items():
                per_class_accum[cid]["tp"] += counts["tp"]
                per_class_accum[cid]["fp"] += counts["fp"]
                per_class_accum[cid]["fn"] += counts["fn"]

    # Calculate overall detection metrics
    precision = float(total_tp) / float(total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = float(total_tp) / float(total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        (2.0 * precision * recall) / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )

    if total_pred_boxes == 0:
        notes.append("No bounding box predictions exceeded the confidence threshold; precision is 0.0.")
    if total_gt_boxes == 0:
        notes.append("No ground-truth lesion boxes found in the evaluated dataset split.")

    mean_iou = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0
    if not all_matched_ious:
        notes.append("No predicted boxes matched ground-truth boxes at IoU >= 0.5.")

    # Calculate per-class metrics
    per_class_results = {}
    for cid in range(1, NUM_CLASSES):
        cname = ID_TO_CLASS.get(cid, f"class_{cid}")
        stats = per_class_accum[cid]
        c_tp = stats["tp"]
        c_fp = stats["fp"]
        c_fn = stats["fn"]
        c_prec = float(c_tp) / float(c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        c_rec = float(c_tp) / float(c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        c_f1 = (2.0 * c_prec * c_rec) / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0

        per_class_results[cname] = {
            "class_id": cid,
            "support_gt_boxes": stats["support"],
            "predicted_boxes": stats["preds"],
            "tp": c_tp,
            "fp": c_fp,
            "fn": c_fn,
            "precision": round(c_prec, 4),
            "recall": round(c_rec, 4),
            "f1": round(c_f1, 4),
        }

    return {
        "num_images": total_images,
        "num_gt_boxes": total_gt_boxes,
        "num_pred_boxes": total_pred_boxes,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_matched_iou": round(mean_iou, 4),
        "iou_threshold": iou_threshold,
        "confidence_threshold": confidence_threshold,
        "per_class": per_class_results,
        "notes": notes,
    }
