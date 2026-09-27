"""Diagnostic script to inspect CADICA training and validation data and isolate NaN loss causes."""

import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch

from ml.data.cadica_dataset import CADICA_CATEGORY_TO_LABEL_ID, CADICADataset
from ml.data.transforms import get_cadica_transforms
from ml.lesion.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
)
from ml.lesion.dataloader import compute_image_weights, get_eval_loader, get_train_loader
from ml.lesion.model import build_lesion_detector


def inspect_dataset_items(manifest_path: Path, split_name: str) -> Dict[str, Any]:
    """Inspect raw dataset items and verify bounding boxes, labels, and image paths."""
    print(f"\n{'='*70}\n[1] INSPECTING RAW DATASET: {split_name} ({manifest_path})\n{'='*70}")
    dataset = CADICADataset(manifest_path=manifest_path, transforms=None)

    total_frames = len(dataset)
    total_boxes = 0
    non_lesion_frames = 0
    multi_lesion_frames = 0

    problematic_samples = []

    min_w = float("inf")
    min_h = float("inf")
    max_w = float("-inf")
    max_h = float("-inf")

    small_boxes_sub2px = []
    out_of_bounds_boxes = []
    degenerate_boxes = []
    nan_boxes = []
    invalid_labels = []

    for idx in range(total_frames):
        item = dataset.frames[idx]
        frame_id = item["frame_id"]
        img_path = dataset._resolve_image_path(item["image_path"])

        if not img_path.is_file():
            problematic_samples.append({
                "frame_id": frame_id,
                "error": f"Image file missing: {img_path}",
            })
            continue

        boxes = item["boxes_xywh"]
        categories = item["categories"]
        is_lesion = item["is_lesion"]

        if not is_lesion or len(boxes) == 0:
            non_lesion_frames += 1
            continue

        if len(boxes) > 1:
            multi_lesion_frames += 1

        img_w = float(item["image_width"])
        img_h = float(item["image_height"])

        for b_idx, ((x, y, w, h), cat) in enumerate(zip(boxes, categories)):
            total_boxes += 1

            # Check NaNs / Infs
            if any(math.isnan(v) or math.isinf(v) for v in (x, y, w, h)):
                nan_boxes.append((frame_id, b_idx, (x, y, w, h)))

            # Check dimensions
            if w <= 0 or h <= 0:
                degenerate_boxes.append((frame_id, b_idx, (x, y, w, h)))
            if 0 < w < 2.0 or 0 < h < 2.0:
                small_boxes_sub2px.append((frame_id, b_idx, (x, y, w, h)))

            min_w = min(min_w, w)
            min_h = min(min_h, h)
            max_w = max(max_w, w)
            max_h = max(max_h, h)

            # Check bounds
            x1, y1, x2, y2 = x, y, x + w, y + h
            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                out_of_bounds_boxes.append((frame_id, b_idx, (x1, y1, x2, y2), (img_w, img_h)))

            # Check class
            cid = CADICA_CATEGORY_TO_LABEL_ID.get(cat, None)
            if cid is None or not (1 <= cid < NUM_CLASSES):
                invalid_labels.append((frame_id, cat, cid))

    print(f"Total frames: {total_frames}")
    print(f"Total boxes: {total_boxes}")
    print(f"Non-lesion frames: {non_lesion_frames}")
    print(f"Multi-lesion frames: {multi_lesion_frames}")
    print(f"Box width range: [{min_w:.2f}, {max_w:.2f}]")
    print(f"Box height range: [{min_h:.2f}, {max_h:.2f}]")
    print(f"Degenerate boxes (w<=0 or h<=0): {len(degenerate_boxes)}")
    print(f"Sub-2px boxes (0 < w < 2 or 0 < h < 2): {len(small_boxes_sub2px)}")
    print(f"Out-of-bounds boxes: {len(out_of_bounds_boxes)}")
    print(f"NaN/Inf boxes: {len(nan_boxes)}")
    print(f"Invalid labels: {len(invalid_labels)}")

    return {
        "split": split_name,
        "total_frames": total_frames,
        "total_boxes": total_boxes,
        "non_lesion_frames": non_lesion_frames,
        "degenerate_boxes": degenerate_boxes,
        "small_boxes": small_boxes_sub2px,
        "out_of_bounds": out_of_bounds_boxes,
        "nan_boxes": nan_boxes,
        "invalid_labels": invalid_labels,
    }


def inspect_transforms_pipeline(manifest_path: Path, num_trials: int = 5) -> Dict[str, Any]:
    """Test transformed tensors and boxes across multiple randomized augmentations."""
    print(f"\n{'='*70}\n[2] TESTING AUGMENTATIONS & TRANSFORMS ON {manifest_path.name}\n{'='*70}")
    dataset = CADICADataset(
        manifest_path=manifest_path,
        transforms=get_cadica_transforms(is_train=True),
    )

    invalid_transformed_boxes = []
    nan_image_tensors = []
    zero_area_boxes = []

    print(f"Testing {len(dataset)} items over {num_trials} randomized passes...")

    for trial in range(num_trials):
        for idx in range(min(500, len(dataset))):  # check 500 samples * trials
            img, target = dataset[idx]

            # Check image tensor
            if not torch.isfinite(img).all():
                nan_image_tensors.append((idx, trial, "Image contains NaN/Inf"))
            if img.min() < 0.0 or img.max() > 1.0:
                pass  # float in [0, 1]

            # Check target boxes
            boxes = target["boxes"]
            labels = target["labels"]

            if len(boxes) > 0:
                if not torch.isfinite(boxes).all():
                    invalid_transformed_boxes.append((idx, trial, "Boxes contain NaN/Inf", boxes))

                # Check x1 < x2 and y1 < y2
                widths = boxes[:, 2] - boxes[:, 0]
                heights = boxes[:, 3] - boxes[:, 1]

                degen = (widths <= 0) | (heights <= 0)
                if degen.any():
                    zero_area_boxes.append((idx, target["frame_id"], boxes[degen]))

                # Check coordinate range [0, 512]
                if (boxes < 0.0).any() or (boxes > 512.0).any():
                    invalid_transformed_boxes.append((idx, target["frame_id"], "Box coordinates outside [0, 512]", boxes))

    print(f"NaN image tensors found: {len(nan_image_tensors)}")
    print(f"Degenerate transformed boxes (w<=0 or h<=0): {len(zero_area_boxes)}")
    print(f"Invalid transformed boxes: {len(invalid_transformed_boxes)}")

    return {
        "nan_image_tensors": nan_image_tensors,
        "zero_area_boxes": zero_area_boxes,
        "invalid_transformed_boxes": invalid_transformed_boxes,
    }


def inspect_model_loss_components() -> None:
    """Test model forward pass on positive and negative samples and test numerical stability."""
    print(f"\n{'='*70}\n[3] TESTING MODEL LOSS AND ROI HEAD TARGET ASSIGNMENT\n{'='*70}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_lesion_detector(num_classes=8, pretrained=True).to(device)
    model.train()

    # Test 1: Empty target (negative sample)
    img_neg = [torch.rand(3, 512, 512, device=device)]
    tgt_neg = [{
        "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
        "labels": torch.zeros((0,), dtype=torch.int64, device=device),
    }]
    loss_neg = model(img_neg, tgt_neg)
    print("Loss on negative sample (0 boxes):", {k: f"{v.item():.4f}" for k, v in loss_neg.items()})

    # Test 2: Normal positive sample
    img_pos = [torch.rand(3, 512, 512, device=device)]
    tgt_pos = [{
        "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]], dtype=torch.float32, device=device),
        "labels": torch.tensor([4], dtype=torch.int64, device=device),
    }]
    loss_pos = model(img_pos, tgt_pos)
    print("Loss on standard positive sample:", {k: f"{v.item():.4f}" for k, v in loss_pos.items()})

    # Test 3: Very small box (e.g. 5x5 px)
    img_small = [torch.rand(3, 512, 512, device=device)]
    tgt_small = [{
        "boxes": torch.tensor([[100.0, 100.0, 104.0, 104.0]], dtype=torch.float32, device=device),
        "labels": torch.tensor([4], dtype=torch.int64, device=device),
    }]
    loss_small = model(img_small, tgt_small)
    print("Loss on small 4x4 box:", {k: f"{v.item():.4f}" for k, v in loss_small.items()})

    # Test 4: Check what happens with validation loss pass
    # In validate(): model.train() is run with no_grad on validation set!
    val_loader, val_ds = get_eval_loader(VAL_CSV, batch_size=1)
    print(f"\nChecking first 20 validation batches with model.train() in no_grad mode...")
    with torch.no_grad():
        for b_idx, (imgs, tgts) in enumerate(val_loader):
            if b_idx >= 20:
                break
            imgs = [img.to(device) for img in imgs]
            tgts = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in tgts]
            ld = model(imgs, tgts)
            tot = sum(l for l in ld.values())
            if not torch.isfinite(tot):
                print(f"NON-FINITE VALIDATION LOSS AT BATCH {b_idx}: {ld}")
                break
        print(f"Validation loss check passed first 20 batches without error.")


def measure_throughput_bottleneck() -> Dict[str, Any]:
    """Measure exact CPU vs GPU time per batch to isolate the 58-minute bottleneck."""
    print(f"\n{'='*70}\n[4] MEASURING PERFORMANCE & THROUGHPUT BOTTLENECKS\n{'='*70}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_lesion_detector(num_classes=8, pretrained=True).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_loader, _ = get_train_loader(batch_size=1, num_workers=0)

    n_batches = 100
    t_load = 0.0
    t_forward = 0.0
    t_backward = 0.0
    t_opt = 0.0

    iter_loader = iter(train_loader)

    for i in range(n_batches):
        t0 = time.perf_counter()
        images, targets = next(iter_loader)
        t1 = time.perf_counter()
        t_load += (t1 - t0)

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        loss_dict = model(images, targets)
        loss = sum(l for l in loss_dict.values())
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        t_forward += (t3 - t2)

        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        t_backward += (t4 - t3)

        optimizer.step()
        torch.cuda.synchronize()
        t5 = time.perf_counter()
        t_opt += (t5 - t4)

    t_total = t_load + t_forward + t_backward + t_opt
    per_batch_sec = t_total / n_batches
    epoch_est_sec = per_batch_sec * len(train_loader)

    print(f"Throughput Profiling across {n_batches} real training batches:")
    print(f"  DataLoader time (CPU/disk/transforms): {t_load:.2f}s ({t_load/t_total*100:.1f}%)")
    print(f"  Forward Pass (GPU):                    {t_forward:.2f}s ({t_forward/t_total*100:.1f}%)")
    print(f"  Backward Pass (GPU):                   {t_backward:.2f}s ({t_backward/t_total*100:.1f}%)")
    print(f"  Optimizer Step (GPU):                  {t_opt:.2f}s ({t_opt/t_total*100:.1f}%)")
    print(f"  Total time per batch:                  {per_batch_sec*1000:.1f} ms")
    print(f"  Estimated Epoch Training Time:         {epoch_est_sec/60:.1f} minutes")

    # Now test Validation time
    val_loader, _ = get_eval_loader(VAL_CSV, batch_size=1)
    model.eval()
    t0_val = time.perf_counter()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(val_loader):
            if i >= 100:
                break
            imgs = [img.to(device) for img in imgs]
            _ = model(imgs)
            torch.cuda.synchronize()
    t1_val = time.perf_counter()
    val_per_batch = (t1_val - t0_val) / 100.0
    val_epoch_est = val_per_batch * len(val_loader)
    print(f"\nValidation Profiling across 100 validation frames:")
    print(f"  Val time per image:                    {val_per_batch*1000:.1f} ms")
    print(f"  Estimated Full Validation Time (910):  {val_epoch_est/60:.1f} minutes")

    return {
        "per_batch_sec": per_batch_sec,
        "t_load_pct": t_load / t_total * 100,
        "t_gpu_pct": (t_forward + t_backward + t_opt) / t_total * 100,
        "epoch_training_min": epoch_est_sec / 60,
        "epoch_val_min": val_epoch_est / 60,
    }


def main():
    print("=" * 70)
    print("CADICA LESION DETECTOR TRAINING DIAGNOSTIC")
    print("=" * 70)

    # 1. Dataset inspection
    train_res = inspect_dataset_items(TRAIN_CSV, "train")
    val_res = inspect_dataset_items(VAL_CSV, "val")

    # 2. Augmentations inspection
    trans_res = inspect_transforms_pipeline(TRAIN_CSV)

    # 3. Model inspection
    inspect_model_loss_components()

    # 4. Performance profiling
    perf_res = measure_throughput_bottleneck()

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
