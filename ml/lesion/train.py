"""Training engine for CADICA Faster R-CNN lesion detector."""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.lesion.config import (
    CLASS_NAMES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    LesionTrainingConfig,
)
from ml.lesion.device import get_device
from ml.lesion.evaluate import evaluate_detector


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    val_metric: float,
    config: Optional[Union[LesionTrainingConfig, Dict[str, Any]]] = None,
    val_details: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save model checkpoint with training metadata.

    Args:
        checkpoint_path: Destination file path.
        model: PyTorch model.
        optimizer: PyTorch optimizer (optional).
        epoch: Current epoch index.
        val_metric: Validation metric (e.g. loss or F1 score).
        config: Training configuration dataclass or dict.
        val_details: Additional evaluation details.

    Returns:
        Path: Absolute path to the saved checkpoint.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cfg_dict = asdict(config) if isinstance(config, LesionTrainingConfig) else (config or {})

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "validation_metric": val_metric,
        "validation_details": val_details or {},
        "class_mapping": {
            "class_names": CLASS_NAMES,
            "class_to_id": CLASS_TO_ID,
            "id_to_class": ID_TO_CLASS,
        },
        "training_configuration": cfg_dict,
    }

    torch.save(payload, path)
    return path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load model and optional optimizer state from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: Target model instance.
        optimizer: Optional target optimizer.
        device: Device to map tensors onto.

    Returns:
        Dict[str, Any]: Metadata loaded from checkpoint.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    dev = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=dev, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return {
            "epoch": checkpoint.get("epoch", 0),
            "validation_metric": checkpoint.get("validation_metric", 0.0),
            "validation_details": checkpoint.get("validation_details", {}),
            "training_configuration": checkpoint.get("training_configuration", {}),
            "class_mapping": checkpoint.get("class_mapping", {}),
        }
    else:
        # Raw state dict
        model.load_state_dict(checkpoint)
        return {"epoch": 0, "validation_metric": 0.0}


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    log_interval: int = 50,
    max_batches: Optional[int] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, float]:
    """Execute one training epoch with AMP autocast and loss component logging.

    Args:
        model: Faster R-CNN detector.
        optimizer: PyTorch optimizer (e.g. AdamW).
        dataloader: Training DataLoader returning (images, targets).
        device: Compute device.
        epoch: Epoch number for logging.
        log_interval: Step frequency for log output.
        max_batches: Optional batch count cap for smoke testing.
        scaler: Optional PyTorch AMP GradScaler for mixed precision training.

    Returns:
        Dict[str, float]: Mean loss values across the epoch.
    """
    model.train()

    running_total_loss = 0.0
    running_cls_loss = 0.0
    running_box_loss = 0.0
    running_obj_loss = 0.0
    running_rpn_loss = 0.0
    total_steps = 0

    num_total_batches = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    base_lr = optimizer.param_groups[0]["lr"]
    warmup_iters = min(150, num_total_batches) if epoch == 1 else 0
    use_amp = (device.type == "cuda")

    for batch_idx, (images, targets) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Linear warmup during initial iterations of Epoch 1
        if epoch == 1 and batch_idx < warmup_iters:
            warmup_factor = (batch_idx + 1) / float(warmup_iters)
            warmup_lr = base_lr * (0.05 + 0.95 * warmup_factor)
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        # Move tensors to device
        images = [img.to(device) for img in images]
        device_targets = []
        for t in targets:
            dt = {}
            for k, v in t.items():
                if isinstance(v, torch.Tensor):
                    dt[k] = v.to(device)
                else:
                    dt[k] = v
            device_targets.append(dt)

        # Forward pass under AMP autocast
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss_dict = model(images, device_targets)
            losses = sum(loss for loss in loss_dict.values())

        # Validate finiteness of forward loss
        if not torch.isfinite(losses):
            print(
                f"\n[Warning] Non-finite loss at Epoch {epoch}, Batch {batch_idx + 1}: {loss_dict}. Skipping batch.",
                flush=True,
            )
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()

        # Backward pass with AMP GradScaler
        if scaler is not None and use_amp:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # Track losses
        l_total = losses.item()
        l_cls = loss_dict.get("loss_classifier", torch.tensor(0.0)).item()
        l_box = loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()
        l_obj = loss_dict.get("loss_objectness", torch.tensor(0.0)).item()
        l_rpn = loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item()

        running_total_loss += l_total
        running_cls_loss += l_cls
        running_box_loss += l_box
        running_obj_loss += l_obj
        running_rpn_loss += l_rpn
        total_steps += 1

        # Logging
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_total_batches:
            lr = optimizer.param_groups[0]["lr"]
            vram_info = ""
            if device.type == "cuda":
                allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
                vram_info = f" | VRAM: {allocated_mb:.1f} MB"

            print(
                f"Epoch: [{epoch:02d}] Batch: [{batch_idx + 1:04d}/{num_total_batches:04d}] "
                f"Loss: {l_total:.4f} (Cls: {l_cls:.4f}, Box: {l_box:.4f}, Obj: {l_obj:.4f}, RPN: {l_rpn:.4f}) | "
                f"LR: {lr:.6f}{vram_info}",
                flush=True,
            )

    steps = max(1, total_steps)
    return {
        "total_loss": running_total_loss / steps,
        "loss_classifier": running_cls_loss / steps,
        "loss_box_reg": running_box_loss / steps,
        "loss_objectness": running_obj_loss / steps,
        "loss_rpn_box_reg": running_rpn_loss / steps,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.3,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Compute validation loss and detection evaluation metrics.

    Args:
        model: Faster R-CNN model.
        dataloader: Validation DataLoader.
        device: Compute device.
        max_batches: Optional limit on batches.
        iou_threshold: IoU cutoff for detection matching.
        confidence_threshold: Confidence score cutoff.

    Returns:
        val_losses: Dict of mean validation losses.
        val_metrics: Dict of detection metrics (precision, recall, F1, mean IoU).
    """
    # 1. Validation Loss Pass (model.train mode under no_grad to compute loss_dict)
    model.train()
    running_loss = 0.0
    running_cls = 0.0
    running_box = 0.0
    running_obj = 0.0
    running_rpn = 0.0
    steps = 0
    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = [img.to(device) for img in images]
            device_targets = []
            for t in targets:
                dt = {}
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        dt[k] = v.to(device)
                    else:
                        dt[k] = v
                device_targets.append(dt)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss_dict = model(images, device_targets)
                losses = sum(loss for loss in loss_dict.values())

            # Skip non-finite loss from poisoning the overall validation metric
            if not torch.isfinite(losses):
                continue

            running_loss += losses.item()
            running_cls += loss_dict.get("loss_classifier", torch.tensor(0.0)).item()
            running_box += loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()
            running_obj += loss_dict.get("loss_objectness", torch.tensor(0.0)).item()
            running_rpn += loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item()
            steps += 1

    steps = max(1, steps)
    val_losses = {
        "val_total_loss": running_loss / steps,
        "val_loss_classifier": running_cls / steps,
        "val_loss_box_reg": running_box / steps,
        "val_loss_objectness": running_obj / steps,
        "val_loss_rpn_box_reg": running_rpn / steps,
    }

    # 2. Detection Evaluation Pass (model.eval mode)
    val_metrics = evaluate_detector(
        model=model,
        dataloader=dataloader,
        device=device,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
        max_batches=max_batches,
    )

    return val_losses, val_metrics
