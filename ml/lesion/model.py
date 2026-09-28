"""Faster R-CNN MobileNetV3-Large 320 FPN model builder for CADICA lesion detection."""

from typing import Optional
import torch
import torch.nn as nn
import torchvision.models.detection as detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights

from ml.lesion.config import NUM_CLASSES


def build_lesion_detector(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    trainable_backbone_layers: Optional[int] = None,
) -> detection.FasterRCNN:
    """Build Faster R-CNN MobileNetV3-Large 320 FPN with custom classification head.

    Args:
        num_classes: Total number of classes including background (default: 8).
                     0: background, 1-7: CADICA severity categories.
        pretrained: If True, initializes backbone with pretrained COCO weights.
        trainable_backbone_layers: Number of trainable backbone layers (None for default).

    Returns:
        torchvision.models.detection.FasterRCNN: Configured detector model.
    """
    if pretrained:
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    else:
        weights = None

    kwargs = {}
    if trainable_backbone_layers is not None:
        kwargs["trainable_backbone_layers"] = trainable_backbone_layers

    model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights,
        **kwargs
    )

    # Get input feature dimension of the original classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace box predictor with new head for CADICA classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Crucial: set RPN score threshold to 0.0 so negative / non-lesion frames
    # retain background anchor proposals, preventing empty RoI target divide-by-zero (NaN)
    model.rpn.score_thresh = 0.0

    # Store num_classes attribute for easy introspection
    model.num_classes = num_classes

    return model
