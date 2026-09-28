"""GPU Smoke Test for CADICA lesion detection training pipeline.

Performs a controlled 1-pass verification of:
1. CUDA availability and GPU device verification.
2. Dataset loading and target tensor contract compliance.
3. Pretrained Faster R-CNN MobileNetV3 model construction and weight initialization.
4. Device movement to CUDA.
5. Forward and backward pass with gradient computation.
6. Multi-batch training step with finite loss assertion.
7. Validation pass (loss calculation + detection evaluation).
8. Checkpoint saving and reloading.
9. Inference module prediction verification.

FAILS LOUDLY if any assertion or GPU contract is violated.
"""

import sys
import time
from pathlib import Path
import torch
import torch.optim as optim

from ml.lesion.config import (
    CHECKPOINT_DIR,
    CLASS_NAMES,
    NUM_CLASSES,
    TRAIN_CSV,
    VAL_CSV,
    LesionTrainingConfig,
)
from ml.lesion.dataloader import get_eval_loader, get_train_loader
from ml.lesion.device import get_device
from ml.lesion.evaluate import evaluate_detector
from ml.lesion.inference import load_model, predict
from ml.lesion.model import build_lesion_detector
from ml.lesion.train import save_checkpoint, train_one_epoch, validate


def run_gpu_smoke_test(num_train_batches: int = 10, num_val_batches: int = 5) -> dict:
    """Execute end-to-end GPU smoke test.

    Args:
        num_train_batches: Number of training batches to run (default: 10).
        num_val_batches: Number of validation batches to run (default: 5).

    Returns:
        dict: Diagnostics and smoke test results.
    """
    start_time = time.time()
    print("=" * 70)
    print("STARTING CADICA LESION DETECTOR GPU SMOKE TEST")
    print("=" * 70)

    # 1. HARD CUDA CHECK (FAIL LOUDLY IF UNAVAILABLE)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "FATAL: CUDA is not available! The smoke test requires a CUDA-enabled GPU."
        )

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[1/8] GPU Confirmed: {gpu_name} ({total_vram_gb:.2f} GB total VRAM)")

    # 2. DATASET VERIFICATION
    if not TRAIN_CSV.is_file():
        raise FileNotFoundError(f"Training manifest not found: {TRAIN_CSV}")
    if not VAL_CSV.is_file():
        raise FileNotFoundError(f"Validation manifest not found: {VAL_CSV}")

    train_loader, train_ds = get_train_loader(
        train_csv=TRAIN_CSV,
        batch_size=1,
        num_workers=0,
        use_weighted_sampler=True,
    )
    val_loader, val_ds = get_eval_loader(
        manifest_csv=VAL_CSV,
        batch_size=1,
        num_workers=0,
    )

    print(f"[2/8] Datasets loaded: {len(train_ds)} train frames, {len(val_ds)} val frames")

    # Inspect first batch targets
    sample_images, sample_targets = next(iter(train_loader))
    assert len(sample_images) == 1, f"Expected batch size 1, got {len(sample_images)}"
    sample_img = sample_images[0]
    sample_tgt = sample_targets[0]

    assert isinstance(sample_img, torch.Tensor), "Image must be a PyTorch Tensor"
    assert sample_img.ndim == 3 and sample_img.shape[0] == 3, f"Image shape must be [3, H, W], got {sample_img.shape}"
    assert "boxes" in sample_tgt and "labels" in sample_tgt, "Target must contain 'boxes' and 'labels'"

    boxes = sample_tgt["boxes"]
    labels = sample_tgt["labels"]
    assert boxes.ndim == 2 and boxes.shape[1] == 4, f"Boxes shape must be [N, 4], got {boxes.shape}"
    assert labels.ndim == 1, f"Labels shape must be [N], got {labels.shape}"
    print(f"      Sample target verified: {boxes.shape[0]} boxes, tensor shape: {boxes.shape}")

    # 3. BUILD MODEL
    print("[3/8] Building pretrained Faster R-CNN MobileNetV3-Large 320 FPN...")
    model = build_lesion_detector(num_classes=NUM_CLASSES, pretrained=True)
    assert model.num_classes == 8, f"Expected 8 classes, got {model.num_classes}"

    # 4. MOVE TO CUDA
    model.to(device)
    initial_alloc_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    print(f"[4/8] Model moved to CUDA. Initial VRAM allocated: {initial_alloc_mb:.2f} MB")

    # 5. INITIALIZE OPTIMIZER
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 6. RUN TRAINING BATCHES
    print(f"[5/8] Running {num_train_batches} training batches on GPU...")
    train_loss_dict = train_one_epoch(
        model=model,
        optimizer=optimizer,
        dataloader=train_loader,
        device=device,
        epoch=1,
        log_interval=5,
        max_batches=num_train_batches,
    )

    for k, v in train_loss_dict.items():
        assert torch.isfinite(torch.tensor(v)), f"Loss component '{k}' is not finite: {v}"
    print(f"      Training batches completed. Mean total loss: {train_loss_dict['total_loss']:.4f}")

    # 7. RUN VALIDATION
    print(f"[6/8] Running {num_val_batches} validation batches...")
    val_losses, val_metrics = validate(
        model=model,
        dataloader=val_loader,
        device=device,
        max_batches=num_val_batches,
    )
    assert torch.isfinite(torch.tensor(val_losses["val_total_loss"])), "Validation loss is non-finite"
    print(f"      Validation completed. Mean val loss: {val_losses['val_total_loss']:.4f}")
    print(f"      Val detection boxes evaluated: {val_metrics['num_pred_boxes']} preds, {val_metrics['num_gt_boxes']} GT")

    # 8. CHECKPOINT SAVE & LOAD TEST
    smoke_chk_path = CHECKPOINT_DIR / "smoke_test_detector.pth"
    print(f"[7/8] Saving temporary checkpoint to: {smoke_chk_path}")
    save_checkpoint(
        checkpoint_path=smoke_chk_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        val_metric=val_losses["val_total_loss"],
        config={"smoke_test": True, "num_classes": NUM_CLASSES},
        val_details=val_metrics,
    )
    assert smoke_chk_path.is_file(), "Checkpoint file was not created"

    # Reload model
    loaded_model = load_model(smoke_chk_path, device=device, num_classes=NUM_CLASSES)
    assert loaded_model is not None, "Failed to reload model from checkpoint"

    # 9. INFERENCE TEST
    print("[8/8] Testing inference on sample frame...")
    test_img = sample_img.clone()
    detections = predict(test_img, loaded_model, device=device, confidence_threshold=0.0)
    assert isinstance(detections, list), "Inference output must be a list"
    if len(detections) > 0:
        first_det = detections[0]
        assert "bbox" in first_det and len(first_det["bbox"]) == 4, "Detection missing 4-coord bbox"
        assert "class_id" in first_det and 0 <= first_det["class_id"] < NUM_CLASSES, "Invalid class_id"
        assert "severity" in first_det, "Detection missing severity"
        assert "confidence" in first_det and 0.0 <= first_det["confidence"] <= 1.0, "Invalid confidence"
        print(f"      Inference verified: detected {len(detections)} candidate boxes (sample: {first_det})")
    else:
        print("      Inference verified: 0 detections returned (empty)")

    torch.cuda.synchronize()
    final_vram_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    duration_sec = time.time() - start_time

    print("=" * 70)
    print("CADICA GPU SMOKE TEST: PASSED")
    print(f"Duration: {duration_sec:.2f}s | Final VRAM Allocated: {final_vram_mb:.2f} MB")
    print("=" * 70)

    return {
        "status": "PASSED",
        "gpu_name": gpu_name,
        "vram_gb_total": total_vram_gb,
        "final_vram_allocated_mb": round(final_vram_mb, 2),
        "duration_seconds": round(duration_sec, 2),
        "train_loss": train_loss_dict,
        "val_loss": val_losses,
        "val_metrics": val_metrics,
        "checkpoint_path": str(smoke_chk_path),
    }


if __name__ == "__main__":
    try:
        results = run_gpu_smoke_test()
    except Exception as e:
        print(f"\nFATAL ERROR IN GPU SMOKE TEST: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
