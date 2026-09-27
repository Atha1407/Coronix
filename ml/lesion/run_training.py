"""CADICA Lesion Detector 5-Epoch GPU Training Pipeline.

Orchestrates:
1. Safety pre-flight check (1 real training batch on CUDA).
2. 5-epoch training loop with weighted sampler for CADICA class imbalance.
3. Checkpoint saving after every epoch (epoch checkpoints, latest, best).
4. Validation detection evaluation (Precision, Recall, F1 @ IoU >= 0.5, Mean IoU).
5. Best model selection based on validation F1 (with loss fallback).
6. Final test evaluation strictly on test.csv using the best checkpoint.
7. Serialization of training_results.json and training_report.md.
8. Generation of 10-20 visual predictions from test set.
"""

import json
import os
import sys
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ml.lesion.config import (
    CHECKPOINT_DIR,
    CLASS_NAMES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    NUM_CLASSES,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
    LesionTrainingConfig,
)
from ml.lesion.dataloader import create_dataloaders, get_eval_loader
from ml.lesion.device import get_device, get_device_info
from ml.lesion.evaluate import compute_iou_matrix, evaluate_detector
from ml.lesion.inference import load_model, predict
from ml.lesion.model import build_lesion_detector
from ml.lesion.train import load_checkpoint, save_checkpoint, train_one_epoch, validate


def run_safety_preflight(
    model: nn.Module,
    train_loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    num_preflight_batches: int = 20,
) -> None:
    """Execute a short preflight of exactly 20 training batches under AMP."""
    print("=" * 70, flush=True)
    print(f"[Pre-flight Safety Check] Verifying {num_preflight_batches} real training batches on CUDA with AMP...", flush=True)
    print("=" * 70, flush=True)

    if device.type != "cuda":
        raise RuntimeError("Pre-flight safety check failed: Device is not CUDA!")

    model.train()
    use_amp = (device.type == "cuda")
    iter_loader = iter(train_loader)

    total_loss_accum = 0.0
    cls_loss_accum = 0.0
    box_loss_accum = 0.0
    obj_loss_accum = 0.0
    rpn_loss_accum = 0.0

    t0_pre = time.perf_counter()

    for batch_i in range(1, num_preflight_batches + 1):
        sample_images, sample_targets = next(iter_loader)

        # Move images & targets to CUDA
        images_cuda = [img.to(device) for img in sample_images]
        targets_cuda = []
        for t in sample_targets:
            dt = {}
            for k, v in t.items():
                dt[k] = v.to(device) if isinstance(v, torch.Tensor) else v
            targets_cuda.append(dt)

        # Forward under AMP
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss_dict = model(images_cuda, targets_cuda)
            total_loss = sum(l for l in loss_dict.values())

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Pre-flight safety check failed at batch {batch_i}: Loss is non-finite: {loss_dict}"
            )

        optimizer.zero_grad()

        # Backward with GradScaler
        if scaler is not None and use_amp:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss_accum += total_loss.item()
        cls_loss_accum += loss_dict.get("loss_classifier", torch.tensor(0.0)).item()
        box_loss_accum += loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()
        obj_loss_accum += loss_dict.get("loss_objectness", torch.tensor(0.0)).item()
        rpn_loss_accum += loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item()

    grads_finite = all(
        torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.grad is not None
    )
    if not grads_finite:
        raise FloatingPointError("Pre-flight safety check failed: Final gradients contain NaN/Inf!")

    torch.cuda.synchronize()
    t_elapsed = time.perf_counter() - t0_pre
    n_batches = float(num_preflight_batches)

    alloc_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
    total_train_steps = len(train_loader)
    sec_per_batch = t_elapsed / n_batches
    est_epoch_min = (sec_per_batch * total_train_steps) / 60.0

    print(
        f"[Pre-flight Safety Check] ALL {num_preflight_batches} BATCHES PASSED!\n"
        f"  Average Loss:       {total_loss_accum / n_batches:.4f}\n"
        f"  Classifier Loss:    {cls_loss_accum / n_batches:.4f}\n"
        f"  Box Reg Loss:       {box_loss_accum / n_batches:.4f}\n"
        f"  Objectness Loss:    {obj_loss_accum / n_batches:.4f}\n"
        f"  RPN Box Loss:       {rpn_loss_accum / n_batches:.4f}\n"
        f"  GPU Memory Alloc:   {alloc_mb:.1f} MB\n"
        f"  GPU Memory Reserved:{reserved_mb:.1f} MB\n"
        f"  Elapsed Time (20b): {t_elapsed:.2f}s ({sec_per_batch * 1000:.1f} ms/batch)\n"
        f"  Estimated Epoch:    {est_epoch_min:.1f} minutes ({total_train_steps} batches)\n"
        "  Status: Zero NaN/Inf losses, gradients finite, AMP functional, optimizer step verified.\n"
        "Proceeding immediately to full 5-epoch training.\n",
        flush=True,
    )


def generate_test_visualizations(
    model: nn.Module,
    test_csv: Path,
    output_dir: Path,
    device: torch.device,
    num_samples: int = 15,
) -> List[Dict[str, Any]]:
    """Generate visual prediction comparisons (original, GT, Pred, IoU) on test set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    loader, dataset = get_eval_loader(
        manifest_csv=test_csv,
        batch_size=1,
        num_workers=0,
    )

    model.eval()
    visualized_count = 0
    records = []

    # Iterate through test dataset to find diverse examples (both positive and negative)
    for idx, (images, targets) in enumerate(loader):
        if visualized_count >= num_samples:
            break

        target = targets[0]
        is_lesion = target.get("is_lesion", False)
        frame_id = target.get("frame_id", f"frame_{idx}")
        gt_boxes = target["boxes"].cpu().numpy() if isinstance(target["boxes"], torch.Tensor) else np.array(target["boxes"])
        gt_labels = target["labels"].cpu().numpy() if isinstance(target["labels"], torch.Tensor) else np.array(target["labels"])

        # Prioritize positive lesion frames, but also include negative frames
        if not is_lesion and visualized_count > (num_samples // 3):
            continue

        # Inference
        img_tensor = images[0]
        detections = predict(img_tensor, model, device=device, confidence_threshold=0.15)

        # Convert tensor image back to OpenCV BGR for visualization
        # img_tensor: [3, H, W] float32 in [0, 1]
        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        # Draw Ground Truth boxes in GREEN (0, 255, 0)
        for g_box, g_lbl in zip(gt_boxes, gt_labels):
            gx1, gy1, gx2, gy2 = [int(round(c)) for c in g_box]
            cat_name = ID_TO_CLASS.get(int(g_lbl), f"class_{g_lbl}")
            cv2.rectangle(img_bgr, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            cv2.putText(
                img_bgr,
                f"GT: {cat_name}",
                (gx1, max(15, gy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Draw Predicted boxes in CORAL/RED (0, 69, 255)
        matched_ious = []
        for det in detections:
            px1, py1, px2, py2 = [int(round(c)) for c in det["bbox"]]
            pred_cat = det["severity"]
            conf = det["confidence"]

            # Calculate IoU with GT if GT exists
            cur_iou = 0.0
            if len(gt_boxes) > 0:
                p_box_np = np.array([[px1, py1, px2, py2]], dtype=np.float32)
                iou_mat = compute_iou_matrix(p_box_np, gt_boxes)
                cur_iou = float(np.max(iou_mat))
                matched_ious.append(cur_iou)

            cv2.rectangle(img_bgr, (px1, py1), (px2, py2), (0, 69, 255), 2)
            lbl_str = f"Pred: {pred_cat} ({conf:.2f})"
            if len(gt_boxes) > 0:
                lbl_str += f" [IoU:{cur_iou:.2f}]"

            cv2.putText(
                img_bgr,
                lbl_str,
                (px1, min(h - 5, py2 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 69, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw header banner
        banner = np.zeros((32, w, 3), dtype=np.uint8)
        header_text = (
            f"Frame: {frame_id} | GT: {len(gt_boxes)} boxes | Preds: {len(detections)} boxes"
        )
        cv2.putText(
            banner,
            header_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        combined = np.vstack([banner, img_bgr])

        save_filename = f"pred_{visualized_count + 1:02d}_{frame_id}.png"
        save_path = output_dir / save_filename
        cv2.imwrite(str(save_path), combined)

        records.append({
            "filename": save_filename,
            "frame_id": frame_id,
            "gt_boxes_count": len(gt_boxes),
            "predicted_boxes_count": len(detections),
            "max_iou": round(max(matched_ious), 4) if matched_ious else 0.0,
            "detections": detections,
        })
        visualized_count += 1

    return records


def train_cadica_pipeline() -> None:
    """Main execution entry point for full 5-epoch training."""
    total_start_time = time.time()
    config = LesionTrainingConfig()

    print("=" * 70, flush=True)
    print("STARTING 5-EPOCH CADICA LESION DETECTOR TRAINING", flush=True)
    print(f"Device: {config.device} | Epochs: {config.num_epochs} | Batch Size: {config.batch_size}", flush=True)
    print(f"Learning Rate: {config.learning_rate} | Weight Decay: {config.weight_decay}", flush=True)
    print(f"Weighted Sampler: {config.use_weighted_sampler} | Num Workers: {config.num_workers}", flush=True)
    print("=" * 70, flush=True)

    # 1. Device check
    device = get_device(verbose=True)
    if device.type != "cuda":
        raise RuntimeError("FATAL: CUDA device required for lesion detector training.")

    # 2. Build model with pretrained weights
    print("[1/5] Instantiating Faster R-CNN MobileNetV3-Large 320 FPN with COCO weights...", flush=True)
    model = build_lesion_detector(num_classes=config.num_classes, pretrained=True)
    model.to(device)

    # 3. Optimizer, Scaler & Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=4096.0, enabled=(device.type == "cuda"))
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=1e-6,
    )

    # 4. DataLoaders
    print("[2/5] Initializing CADICA DataLoaders (Train: weighted sampler enabled)...", flush=True)
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(
        f"      Loaded: {len(train_loader.dataset)} train frames, "
        f"{len(val_loader.dataset)} val frames, {len(test_loader.dataset)} test frames.",
        flush=True,
    )

    # 5. Pre-flight safety check (20 batches with AMP)
    try:
        run_safety_preflight(model, train_loader, optimizer, device, scaler=scaler, num_preflight_batches=20)
    except torch.cuda.OutOfMemoryError as oom:
        print(f"\nFATAL: CUDA Out-Of-Memory during pre-flight check: {oom}", file=sys.stderr, flush=True)
        sys.exit(1)

    # Checkpoint paths
    best_chk_path = config.checkpoint_dir / "best_lesion_detector.pth"
    latest_chk_path = config.checkpoint_dir / "latest_lesion_detector.pth"

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 1
    best_val_metrics: Dict[str, Any] = {}

    epoch_records: List[Dict[str, Any]] = []

    print("[3/5] Commencing 5-epoch training loop...", flush=True)

    # 6. Epoch loop
    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()
        print(f"\n>>> Starting Epoch {epoch}/{config.num_epochs}", flush=True)

        try:
            # Training
            train_losses = train_one_epoch(
                model=model,
                optimizer=optimizer,
                dataloader=train_loader,
                device=device,
                epoch=epoch,
                log_interval=config.log_interval,
                scaler=scaler,
            )

            # Validation
            print(f"    Running validation for Epoch {epoch}...", flush=True)
            val_losses, val_metrics = validate(
                model=model,
                dataloader=val_loader,
                device=device,
                iou_threshold=config.iou_threshold,
                confidence_threshold=config.confidence_threshold,
            )

            lr_scheduler.step()

        except torch.cuda.OutOfMemoryError as oom:
            alloc_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
            max_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(
                f"\nFATAL: CUDA Out-Of-Memory at Epoch {epoch}: {oom}\n"
                f"Current VRAM allocated: {alloc_mb:.2f} MB | Peak allocated: {max_alloc_mb:.2f} MB\n"
                "Stopping training without modifying model architecture.",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)

        epoch_duration = time.time() - epoch_start_time
        total_elapsed = time.time() - total_start_time
        epochs_left = config.num_epochs - epoch
        estimated_remaining = epoch_duration * epochs_left

        val_f1 = val_metrics.get("f1", 0.0)
        val_total_loss = val_losses.get("val_total_loss", float("inf"))

        # Save per-epoch checkpoint
        epoch_chk_path = config.checkpoint_dir / f"checkpoint_epoch_{epoch:02d}.pth"
        save_checkpoint(
            checkpoint_path=epoch_chk_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_metric=val_f1,
            config=config,
            val_details=val_metrics,
        )

        # Save latest checkpoint
        save_checkpoint(
            checkpoint_path=latest_chk_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_metric=val_f1,
            config=config,
            val_details=val_metrics,
        )

        # Best model selection logic
        is_best = False
        if val_f1 > best_val_f1:
            is_best = True
        elif val_f1 == best_val_f1 and val_f1 == 0.0 and val_total_loss < best_val_loss:
            is_best = True

        if is_best:
            best_val_f1 = val_f1
            best_val_loss = val_total_loss
            best_epoch = epoch
            best_val_metrics = val_metrics
            save_checkpoint(
                checkpoint_path=best_chk_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_metric=val_f1,
                config=config,
                val_details=val_metrics,
            )
            best_tag = f" --> [NEW BEST SAVED (Val F1: {val_f1:.4f}, Loss: {val_total_loss:.4f})]"
        else:
            best_tag = ""

        # Concise Epoch Summary
        alloc_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        gpu_name = torch.cuda.get_device_name(0)

        print("\n" + "=" * 70, flush=True)
        print(f"Epoch {epoch}/{config.num_epochs} Summary{best_tag}", flush=True)
        print(
            f"Duration: {epoch_duration:.1f}s | Elapsed: {str(timedelta(seconds=int(total_elapsed)))} | "
            f"Est. Remaining: {str(timedelta(seconds=int(estimated_remaining)))}",
            flush=True,
        )
        print("Training Losses:", flush=True)
        print(f"  Total Loss:       {train_losses['total_loss']:.4f}", flush=True)
        print(f"  Classifier Loss:  {train_losses['loss_classifier']:.4f}", flush=True)
        print(f"  Box Reg Loss:     {train_losses['loss_box_reg']:.4f}", flush=True)
        print(f"  Objectness Loss:  {train_losses['loss_objectness']:.4f}", flush=True)
        print(f"  RPN Box Loss:     {train_losses['loss_rpn_box_reg']:.4f}", flush=True)
        print("Validation Metrics (IoU >= 0.5):", flush=True)
        print(f"  Val Total Loss:   {val_total_loss:.4f}", flush=True)
        print(f"  Val F1:           {val_f1:.4f}", flush=True)
        print(f"  Val Precision:    {val_metrics.get('precision', 0.0):.4f}", flush=True)
        print(f"  Val Recall:       {val_metrics.get('recall', 0.0):.4f}", flush=True)
        print(f"  Val Matched IoU:  {val_metrics.get('mean_matched_iou', 0.0):.4f}", flush=True)
        print(f"  Predictions:      {val_metrics.get('num_pred_boxes', 0)}", flush=True)
        print(f"  Ground-Truth:     {val_metrics.get('num_gt_boxes', 0)}", flush=True)
        print("Hardware:", flush=True)
        print(f"  GPU Name:         {gpu_name}", flush=True)
        print(f"  VRAM Allocated:   {alloc_mb:.1f} MB", flush=True)
        print("=" * 70, flush=True)

        epoch_records.append({
            "epoch": epoch,
            "duration_seconds": round(epoch_duration, 2),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_metrics": val_metrics,
        })

    # 7. Final Test Evaluation strictly using best checkpoint
    print(f"\n[4/5] Evaluating Best Model (Epoch {best_epoch}) on TEST SET ONLY...", flush=True)
    best_model = load_model(best_chk_path, device=device, num_classes=config.num_classes)

    test_metrics = evaluate_detector(
        model=best_model,
        dataloader=test_loader,
        device=device,
        iou_threshold=config.iou_threshold,
        confidence_threshold=config.confidence_threshold,
    )

    print("\n" + "=" * 70, flush=True)
    print("FINAL TEST SET EVALUATION REPORT (IoU >= 0.5)", flush=True)
    print("=" * 70, flush=True)
    print(f"Test Images:          {test_metrics['num_images']}", flush=True)
    print(f"Ground-Truth Boxes:   {test_metrics['num_gt_boxes']}", flush=True)
    print(f"Predicted Boxes:      {test_metrics['num_pred_boxes']}", flush=True)
    print(f"True Positives:       {test_metrics['tp']}", flush=True)
    print(f"False Positives:      {test_metrics['fp']}", flush=True)
    print(f"False Negatives:      {test_metrics['fn']}", flush=True)
    print(f"Test Precision:       {test_metrics['precision']:.4f}", flush=True)
    print(f"Test Recall:          {test_metrics['recall']:.4f}", flush=True)
    print(f"Test F1 Score:        {test_metrics['f1']:.4f}", flush=True)
    print(f"Test Mean Matched IoU:{test_metrics['mean_matched_iou']:.4f}", flush=True)
    print("-" * 70, flush=True)
    print("Per-Severity-Class Test Statistics:", flush=True)
    for cname, cstats in test_metrics.get("per_class", {}).items():
        print(
            f"  {cname:<10} | GT: {cstats['support_gt_boxes']:<3} | Pred: {cstats['predicted_boxes']:<3} | "
            f"TP: {cstats['tp']:<3} | Prec: {cstats['precision']:.4f} | Rec: {cstats['recall']:.4f} | F1: {cstats['f1']:.4f}",
            flush=True,
        )
    print("=" * 70, flush=True)

    # 8. Visual predictions on test set
    predictions_dir = config.checkpoint_dir / "predictions"
    print(f"\n[5/5] Generating test set visual predictions in: {predictions_dir}", flush=True)
    vis_records = generate_test_visualizations(
        model=best_model,
        test_csv=config.test_csv,
        output_dir=predictions_dir,
        device=device,
        num_samples=16,
    )
    print(f"      Saved {len(vis_records)} visual prediction examples.", flush=True)

    # 9. Save final training_results.json
    total_training_duration = time.time() - total_start_time
    results_json_path = config.checkpoint_dir / "training_results.json"
    results_payload = {
        "model": "Faster R-CNN MobileNetV3-Large 320 FPN",
        "pretrained_weights": "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT",
        "epochs": config.num_epochs,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "total_training_duration_seconds": round(total_training_duration, 2),
        "total_training_duration_formatted": str(timedelta(seconds=int(total_training_duration))),
        "best_epoch": best_epoch,
        "best_validation_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "class_mapping": {
            "class_names": CLASS_NAMES,
            "class_to_id": CLASS_TO_ID,
            "id_to_class": ID_TO_CLASS,
        },
        "training_configuration": asdict(config),
        "epoch_records": epoch_records,
        "visual_predictions": vis_records,
    }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=4, default=str)
    print(f"      Saved: {results_json_path}", flush=True)

    # 10. Save training_report.md
    report_md_path = config.checkpoint_dir / "training_report.md"
    generate_markdown_report(report_md_path, results_payload)
    print(f"      Saved: {report_md_path}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"CADICA LESION DETECTOR TRAINING COMPLETED IN {str(timedelta(seconds=int(total_training_duration)))}", flush=True)
    print(f"Best Checkpoint: {best_chk_path}", flush=True)
    print(f"Test F1: {test_metrics['f1']:.4f} | Test Precision: {test_metrics['precision']:.4f} | Test Recall: {test_metrics['recall']:.4f}", flush=True)
    print("=" * 70, flush=True)


def generate_markdown_report(report_path: Path, data: Dict[str, Any]) -> None:
    """Generate Markdown training summary report."""
    cfg = data["training_configuration"]
    tm = data["test_metrics"]
    b_val = data["best_validation_metrics"]

    md = f"""# CADICA Lesion Detector Training Report

## 1. Executive Summary & Model Overview
- **Model Architecture**: Faster R-CNN MobileNetV3-Large 320 FPN
- **Pretrained Weights**: Torchvision COCO Default (`FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT`)
- **Total Classes**: 8 (0 = background, 1-7 = CADICA severity categories)
- **Target Device**: {data["gpu_name"]} ({data["device"]})
- **Total Epochs**: {data["epochs"]}
- **Best Epoch**: Epoch {data["best_epoch"]}
- **Total Training Duration**: {data["total_training_duration_formatted"]} ({data["total_training_duration_seconds"]:.1f}s)
- **Best Validation F1**: {b_val.get('f1', 0.0):.4f} (IoU >= 0.5)

---

## 2. Training Hyperparameters
| Parameter | Value |
| :--- | :--- |
| **Batch Size** | {cfg.get('batch_size', 1)} |
| **Learning Rate** | {cfg.get('learning_rate', 1e-4)} |
| **Weight Decay** | {cfg.get('weight_decay', 1e-4)} |
| **Optimizer** | AdamW with Cosine Annealing LR Scheduler |
| **Weighted Sampling** | {cfg.get('use_weighted_sampler', True)} (Rarest present class weighting for p99/p100) |
| **Detection Thresholds** | IoU >= {cfg.get('iou_threshold', 0.5)}, Confidence >= {cfg.get('confidence_threshold', 0.3)} |

---

## 3. Epoch-by-Epoch Progress
| Epoch | Duration | Train Total Loss | Cls Loss | Box Loss | Val Loss | Val F1 | Val Precision | Val Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for rec in data.get("epoch_records", []):
        ep = rec["epoch"]
        dur = f"{rec['duration_seconds']:.1f}s"
        tl = f"{rec['train_losses']['total_loss']:.4f}"
        cls_l = f"{rec['train_losses']['loss_classifier']:.4f}"
        box_l = f"{rec['train_losses']['loss_box_reg']:.4f}"
        vl = f"{rec['val_losses']['val_total_loss']:.4f}"
        vf1 = f"{rec['val_metrics'].get('f1', 0.0):.4f}"
        vp = f"{rec['val_metrics'].get('precision', 0.0):.4f}"
        vr = f"{rec['val_metrics'].get('recall', 0.0):.4f}"
        md += f"| {ep} | {dur} | {tl} | {cls_l} | {box_l} | {vl} | {vf1} | {vp} | {vr} |\n"

    md += f"""
---

## 4. Final Test Set Evaluation
*Evaluated strictly on `ml/data/processed/cadica/test.csv` (Patient-isolated test set, zero train/val patient overlap).*

| Metric | Result (IoU >= 0.5) |
| :--- | :--- |
| **Evaluated Images** | {tm['num_images']} |
| **Ground-Truth Boxes** | {tm['num_gt_boxes']} |
| **Predicted Boxes** | {tm['num_pred_boxes']} |
| **True Positives** | {tm['tp']} |
| **False Positives** | {tm['fp']} |
| **False Negatives** | {tm['fn']} |
| **Detection Precision** | **{tm['precision']:.4f}** |
| **Detection Recall** | **{tm['recall']:.4f}** |
| **Detection F1 Score** | **{tm['f1']:.4f}** |
| **Mean Matched IoU** | **{tm['mean_matched_iou']:.4f}** |

### Per-Severity-Class Breakdown (Test Set)
| Severity Class | Support (GT) | Predicted | TP | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for cname, cstats in tm.get("per_class", {}).items():
        md += f"| {cname} | {cstats['support_gt_boxes']} | {cstats['predicted_boxes']} | {cstats['tp']} | {cstats['precision']:.4f} | {cstats['recall']:.4f} | {cstats['f1']:.4f} |\n"

    md += """
---

## 5. Visual Test Predictions
Test set visual predictions showing ground-truth (green) and predicted bounding boxes (red) are saved in:
`ml/models/predictions/`

---

## 6. Disclaimer & Clinical Limitations
> [!IMPORTANT]
> **Research and Hackathon Prototype Only**: This model is an engineering prototype trained on the CADICA research dataset. It is **NOT** clinically validated, **NOT** certified for medical diagnostics, and must **NOT** be used for clinical decision-making or patient management.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    try:
        train_cadica_pipeline()
    except Exception as e:
        print(f"\nFATAL UNHANDLED EXCEPTION IN TRAINING: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
