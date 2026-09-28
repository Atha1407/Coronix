"""Tests for CADICA Faster R-CNN lesion detector model, dataloader, and evaluation."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from ml.lesion.config import (
    CLASS_NAMES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    NUM_CLASSES,
    LesionTrainingConfig,
)
from ml.lesion.dataloader import collate_fn, compute_image_weights
from ml.lesion.device import get_device, get_device_info
from ml.lesion.evaluate import compute_iou_matrix, match_detections
from ml.lesion.inference import load_model, predict, predict_batch
from ml.lesion.model import build_lesion_detector
from ml.lesion.train import load_checkpoint, save_checkpoint, train_one_epoch


class TestModelConstruction:
    """Tests for model architecture and class heads."""

    def test_model_construction_cpu(self):
        """Model builds correctly with 8 classes and custom predictor head."""
        model = build_lesion_detector(num_classes=8, pretrained=False)
        assert isinstance(model, nn.Module)
        assert model.num_classes == 8
        assert model.roi_heads.box_predictor.cls_score.out_features == 8
        assert model.roi_heads.box_predictor.bbox_pred.out_features == 8 * 4

    def test_class_mapping_integrity(self):
        """Class mapping contains 8 classes with correct CADICA names."""
        assert NUM_CLASSES == 8
        assert len(CLASS_NAMES) == 8
        assert CLASS_NAMES[0] == "background"
        assert CLASS_NAMES[1] == "p0_20"
        assert CLASS_NAMES[6] == "p99"
        assert CLASS_NAMES[7] == "p100"

        # Bijective mapping
        for idx, name in enumerate(CLASS_NAMES):
            assert CLASS_TO_ID[name] == idx
            assert ID_TO_CLASS[idx] == name

    def test_device_selection(self):
        """Device selection automatically chooses CUDA if available."""
        dev = get_device(verbose=False)
        assert isinstance(dev, torch.device)
        info = get_device_info()
        assert "cuda_available" in info
        assert "device_type" in info
        if torch.cuda.is_available():
            assert dev.type == "cuda"
            assert info["cuda_available"] is True
            assert info["vram_gb"] > 0
        else:
            assert dev.type == "cpu"


class TestForwardPassAndCompatibility:
    """Tests for dataset target compatibility and loss computation."""

    def test_positive_sample_forward_pass_cpu(self):
        """Model computes finite detection losses on a sample with bounding boxes."""
        model = build_lesion_detector(num_classes=8, pretrained=False)
        model.train()

        img = torch.rand(3, 320, 320, dtype=torch.float32)
        target = {
            "boxes": torch.tensor([[50.0, 50.0, 120.0, 100.0]], dtype=torch.float32),
            "labels": torch.tensor([3], dtype=torch.int64),  # p50_70
        }

        loss_dict = model([img], [target])
        assert isinstance(loss_dict, dict)
        for expected_key in ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]:
            assert expected_key in loss_dict
            assert torch.isfinite(loss_dict[expected_key])

        total_loss = sum(loss for loss in loss_dict.values())
        assert torch.isfinite(total_loss)
        assert total_loss.item() > 0

    def test_empty_target_negative_sample_cpu(self):
        """Model natively handles negative frames with 0 boxes without crashing."""
        model = build_lesion_detector(num_classes=8, pretrained=False)
        model.train()

        img = torch.rand(3, 320, 320, dtype=torch.float32)
        empty_target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

        loss_dict = model([img], [empty_target])
        total_loss = sum(loss for loss in loss_dict.values())
        assert torch.isfinite(total_loss)
        assert loss_dict["loss_box_reg"].item() == 0.0

    def test_collate_fn(self):
        """Torchvision collate function groups samples into lists."""
        batch = [
            (torch.rand(3, 64, 64), {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,))}),
            (torch.rand(3, 64, 64), {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,))}),
        ]
        images, targets = collate_fn(batch)
        assert isinstance(images, list)
        assert isinstance(targets, list)
        assert len(images) == 2
        assert len(targets) == 2


class TestInferenceAndCheckpointing:
    """Tests for inference format and checkpoint management."""

    def test_inference_output_format(self):
        """predict() returns structured list with bbox, class_id, severity, confidence."""
        model = build_lesion_detector(num_classes=8, pretrained=False)
        dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)

        detections = predict(dummy_img, model, confidence_threshold=0.0)
        assert isinstance(detections, list)
        for det in detections:
            assert "bbox" in det and len(det["bbox"]) == 4
            assert "class_id" in det and 0 <= det["class_id"] < 8
            assert "severity" in det
            assert "confidence" in det and 0.0 <= det["confidence"] <= 1.0

    def test_predict_batch_format(self):
        """predict_batch() returns a list of detection lists."""
        model = build_lesion_detector(num_classes=8, pretrained=False)
        dummy_imgs = [np.zeros((100, 100, 3), dtype=np.uint8), np.zeros((100, 100, 3), dtype=np.uint8)]
        batch_out = predict_batch(dummy_imgs, model, confidence_threshold=0.0)
        assert isinstance(batch_out, list)
        assert len(batch_out) == 2

    def test_checkpoint_roundtrip(self):
        """save_checkpoint and load_checkpoint preserve weights and metadata."""
        model = build_lesion_detector(num_classes=8, pretrained=False)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            chk_path = Path(tmp_dir) / "test_checkpoint.pth"
            saved_path = save_checkpoint(
                checkpoint_path=chk_path,
                model=model,
                optimizer=optimizer,
                epoch=2,
                val_metric=0.45,
                config={"batch_size": 1, "learning_rate": 1e-4},
            )
            assert saved_path.is_file()

            # Load into new model
            new_model = build_lesion_detector(num_classes=8, pretrained=False)
            new_opt = optim.AdamW(new_model.parameters(), lr=1e-4)
            meta = load_checkpoint(chk_path, new_model, new_opt)

            assert meta["epoch"] == 2
            assert meta["validation_metric"] == 0.45
            assert "class_mapping" in meta

            # Test inference loading
            loaded = load_model(chk_path, device=torch.device("cpu"), num_classes=8)
            assert loaded is not None


class TestEvaluationMatching:
    """Tests for IoU calculation and bipartite matching."""

    def test_compute_iou_matrix(self):
        """compute_iou_matrix accurately calculates pairwise IoU."""
        box1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        box2 = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)

        ious = compute_iou_matrix(box1, box2)
        assert ious.shape == (1, 2)
        # Exact overlap
        assert abs(ious[0, 0] - 1.0) < 1e-5
        # 5x5 overlap in 10x10 + 10x10 - 25 = 25 / 175 = 1/7 ~= 0.142857
        assert abs(ious[0, 1] - (25.0 / 175.0)) < 1e-4

    def test_match_detections_tp_and_fp(self):
        """match_detections accurately identifies true positives and false positives."""
        gt_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        gt_labels = np.array([2])  # p20_50

        # One perfect match, one stray box
        pred_boxes = np.array([[0, 0, 10, 10], [100, 100, 200, 200]], dtype=np.float32)
        pred_scores = np.array([0.9, 0.8], dtype=np.float32)
        pred_labels = np.array([2, 1])

        tp, fp, fn, matched_ious, per_cls = match_detections(
            pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5
        )

        assert tp == 1
        assert fp == 1
        assert fn == 0
        assert len(matched_ious) == 1
        assert abs(matched_ious[0] - 1.0) < 1e-4
        assert per_cls[2]["tp"] == 1


class TestGPUMovement:
    """GPU-specific tests that run when CUDA is available."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable on this machine")
    def test_cuda_forward_and_backward(self):
        """Model moves to CUDA, performs forward/backward pass, and uses GPU VRAM."""
        device = torch.device("cuda")
        model = build_lesion_detector(num_classes=8, pretrained=False)
        model.to(device)

        alloc_before = torch.cuda.memory_allocated(device)
        assert alloc_before > 0, "Model parameters must allocate VRAM on GPU"

        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        img = torch.rand(3, 320, 320, device=device)
        target = {
            "boxes": torch.tensor([[20.0, 30.0, 80.0, 90.0]], device=device),
            "labels": torch.tensor([5], device=device),
        }

        loss_dict = model([img], [target])
        total_loss = sum(loss for loss in loss_dict.values())
        assert torch.isfinite(total_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        alloc_after = torch.cuda.memory_allocated(device)
        assert alloc_after > 0
