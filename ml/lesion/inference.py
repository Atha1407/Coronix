"""Inference module for CADICA lesion detection."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
import torch
import torch.nn as nn

from ml.lesion.config import ID_TO_CLASS, NUM_CLASSES
from ml.lesion.device import get_device
from ml.lesion.model import build_lesion_detector


def load_model(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """Load trained lesion detector model from a checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        device: Device to load model onto. If None, auto-detected.
        num_classes: Number of detector classes (default: 8).

    Returns:
        nn.Module: Loaded model in eval mode.
    """
    chk_path = Path(checkpoint_path)
    if not chk_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {chk_path}")

    dev = device or get_device(verbose=False)

    # Instantiate model structure
    model = build_lesion_detector(num_classes=num_classes, pretrained=False)

    # Load weights
    checkpoint = torch.load(chk_path, map_location=dev, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and any("roi_heads" in k for k in checkpoint.keys()):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unrecognized checkpoint format at {chk_path}")

    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()

    return model


def _prepare_image_tensor(image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Ensure image is a [3, H, W] float32 tensor normalized to [0, 1]."""
    if isinstance(image, torch.Tensor):
        if image.ndim == 2:
            # [H, W] -> [3, H, W]
            t = image.unsqueeze(0).repeat(3, 1, 1).float()
        elif image.ndim == 3:
            if image.shape[0] == 1:
                t = image.repeat(3, 1, 1).float()
            elif image.shape[0] == 3:
                t = image.float()
            elif image.shape[2] in (1, 3):
                # HWC -> CHW
                t = image.permute(2, 0, 1).float()
                if t.shape[0] == 1:
                    t = t.repeat(3, 1, 1)
            else:
                raise ValueError(f"Unsupported tensor shape: {image.shape}")
        else:
            raise ValueError(f"Unsupported tensor dimension: {image.ndim}")

        if t.max() > 1.0:
            t = t / 255.0
        return t

    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                rgb = image
            else:
                raise ValueError(f"Unsupported numpy image channels: {image.shape[2]}")
        else:
            raise ValueError(f"Unsupported numpy image shape: {image.shape}")

        # HWC uint8/float -> CHW float32 [0, 1]
        img_float = rgb.astype(np.float32)
        if img_float.max() > 1.0:
            img_float = img_float / 255.0
        tensor = torch.from_numpy(img_float.transpose((2, 0, 1)))
        return tensor

    else:
        raise TypeError(f"Expected numpy ndarray or torch Tensor, got {type(image)}")


@torch.no_grad()
def predict(
    image: Union[np.ndarray, torch.Tensor],
    model: nn.Module,
    device: Optional[torch.device] = None,
    confidence_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """Run lesion detection inference on a single image.

    Args:
        image: Single image as numpy ndarray or torch Tensor.
        model: Trained Faster R-CNN model.
        device: Device to run on (if None, inferred from model parameters).
        confidence_threshold: Minimum confidence score to return.

    Returns:
        List[Dict[str, Any]]: Detected lesions with structure:
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "class_id": 1,
                    "severity": "p0_20",
                    "confidence": 0.91
                }
            ]
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    tensor_img = _prepare_image_tensor(image).to(device)
    outputs = model([tensor_img])

    detections = []
    output = outputs[0]

    boxes = output["boxes"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        conf = float(score)
        if conf < confidence_threshold:
            continue

        cid = int(label)
        severity_name = ID_TO_CLASS.get(cid, f"class_{cid}")

        detections.append({
            "bbox": [round(float(coord), 2) for coord in box],
            "class_id": cid,
            "severity": severity_name,
            "confidence": round(conf, 4),
        })

    return detections


@torch.no_grad()
def predict_batch(
    images: List[Union[np.ndarray, torch.Tensor]],
    model: nn.Module,
    device: Optional[torch.device] = None,
    confidence_threshold: float = 0.3,
) -> List[List[Dict[str, Any]]]:
    """Run lesion detection inference on a batch of images."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    tensor_images = [_prepare_image_tensor(img).to(device) for img in images]
    outputs = model(tensor_images)

    batch_detections = []
    for output in outputs:
        detections = []
        boxes = output["boxes"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            conf = float(score)
            if conf < confidence_threshold:
                continue

            cid = int(label)
            severity_name = ID_TO_CLASS.get(cid, f"class_{cid}")

            detections.append({
                "bbox": [round(float(coord), 2) for coord in box],
                "class_id": cid,
                "severity": severity_name,
                "confidence": round(conf, 4),
            })
        batch_detections.append(detections)

    return batch_detections
