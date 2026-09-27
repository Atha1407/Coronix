"""CADICA Lesion detection module for StenoTrace."""

from ml.lesion.config import (
    CLASS_NAMES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    NUM_CLASSES,
    LesionTrainingConfig,
)
from ml.lesion.device import get_device, get_device_info
from ml.lesion.lesion_detector import LesionDetectionResult, detect_lesion
from ml.lesion.model import build_lesion_detector
from ml.lesion.dataloader import create_dataloaders, get_train_loader, get_eval_loader
from ml.lesion.evaluate import evaluate_detector
from ml.lesion.inference import load_model, predict, predict_batch
from ml.lesion.train import train_one_epoch, validate, save_checkpoint, load_checkpoint

__all__ = [
    "LesionDetectionResult",
    "detect_lesion",
    "build_lesion_detector",
    "get_device",
    "get_device_info",
    "create_dataloaders",
    "get_train_loader",
    "get_eval_loader",
    "evaluate_detector",
    "load_model",
    "predict",
    "predict_batch",
    "train_one_epoch",
    "validate",
    "save_checkpoint",
    "load_checkpoint",
    "LesionTrainingConfig",
    "CLASS_NAMES",
    "CLASS_TO_ID",
    "ID_TO_CLASS",
    "NUM_CLASSES",
]
