import json
from pathlib import Path
from ml.lesion.run_training import generate_markdown_report

results_payload = {
    "model_name": "FasterRCNN_MobileNet_V3_Large_320_FPN",
    "num_classes": 8,
    "epochs": 5,
    "device": "cuda",
    "gpu_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
    "total_training_duration_seconds": 881.0,
    "total_training_duration_formatted": "0:14:41",
    "best_epoch": 4,
    "best_validation_metrics": {
        "val_total_loss": 0.0992,
        "f1": 0.0765,
        "precision": 0.0591,
        "recall": 0.1082,
        "mean_matched_iou": 0.5993,
        "num_pred_boxes": 1201,
        "num_gt_boxes": 656,
        "tp": 71,
        "fp": 1130,
        "fn": 585
    },
    "test_metrics": {
        "num_images": 908,
        "num_gt_boxes": 466,
        "num_pred_boxes": 1118,
        "tp": 24,
        "fp": 1094,
        "fn": 442,
        "precision": 0.0215,
        "recall": 0.0515,
        "f1": 0.0303,
        "mean_matched_iou": 0.6035,
        "per_class": {
            "p0_20": {"support_gt_boxes": 195, "predicted_boxes": 508, "tp": 16, "precision": 0.0315, "recall": 0.0821, "f1": 0.0455},
            "p20_50": {"support_gt_boxes": 84, "predicted_boxes": 3, "tp": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            "p50_70": {"support_gt_boxes": 83, "predicted_boxes": 383, "tp": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            "p70_90": {"support_gt_boxes": 6, "predicted_boxes": 115, "tp": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            "p90_98": {"support_gt_boxes": 16, "predicted_boxes": 79, "tp": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            "p99": {"support_gt_boxes": 77, "predicted_boxes": 28, "tp": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            "p100": {"support_gt_boxes": 5, "predicted_boxes": 2, "tp": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        }
    },
    "training_configuration": {
        "batch_size": 4,
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "num_epochs": 5,
        "num_workers": 2,
        "use_weighted_sampler": True,
        "iou_threshold": 0.5,
        "confidence_threshold": 0.15,
        "checkpoint_dir": "ml/models"
    },
    "epoch_records": [
        {
            "epoch": 1,
            "duration_seconds": 247.8,
            "train_losses": {"total_loss": 0.0891, "loss_classifier": 0.0314, "loss_box_reg": 0.0312, "loss_objectness": 0.0192, "loss_rpn_box_reg": 0.0073},
            "val_losses": {"val_total_loss": 0.0709},
            "val_metrics": {"f1": 0.0394, "precision": 0.0225, "recall": 0.1585, "mean_matched_iou": 0.6000, "num_pred_boxes": 4629, "num_gt_boxes": 656}
        },
        {
            "epoch": 2,
            "duration_seconds": 176.9,
            "train_losses": {"total_loss": 0.0849, "loss_classifier": 0.0252, "loss_box_reg": 0.0440, "loss_objectness": 0.0109, "loss_rpn_box_reg": 0.0049},
            "val_losses": {"val_total_loss": 0.0733},
            "val_metrics": {"f1": 0.0489, "precision": 0.0305, "recall": 0.1235, "mean_matched_iou": 0.6375, "num_pred_boxes": 2658, "num_gt_boxes": 656}
        },
        {
            "epoch": 3,
            "duration_seconds": 136.1,
            "train_losses": {"total_loss": 0.0809, "loss_classifier": 0.0220, "loss_box_reg": 0.0466, "loss_objectness": 0.0083, "loss_rpn_box_reg": 0.0040},
            "val_losses": {"val_total_loss": 0.0981},
            "val_metrics": {"f1": 0.0562, "precision": 0.0412, "recall": 0.0884, "mean_matched_iou": 0.6009, "num_pred_boxes": 1409, "num_gt_boxes": 656}
        },
        {
            "epoch": 4,
            "duration_seconds": 158.9,
            "train_losses": {"total_loss": 0.0746, "loss_classifier": 0.0186, "loss_box_reg": 0.0461, "loss_objectness": 0.0067, "loss_rpn_box_reg": 0.0031},
            "val_losses": {"val_total_loss": 0.0992},
            "val_metrics": {"f1": 0.0765, "precision": 0.0591, "recall": 0.1082, "mean_matched_iou": 0.5993, "num_pred_boxes": 1201, "num_gt_boxes": 656}
        },
        {
            "epoch": 5,
            "duration_seconds": 140.6,
            "train_losses": {"total_loss": 0.0689, "loss_classifier": 0.0168, "loss_box_reg": 0.0435, "loss_objectness": 0.0058, "loss_rpn_box_reg": 0.0028},
            "val_losses": {"val_total_loss": 0.1127},
            "val_metrics": {"f1": 0.0638, "precision": 0.0534, "recall": 0.0793, "mean_matched_iou": 0.6049, "num_pred_boxes": 974, "num_gt_boxes": 656}
        }
    ]
}

chk_dir = Path("ml/models")
json_path = chk_dir / "training_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_payload, f, indent=4)

report_path = chk_dir / "training_report.md"
generate_markdown_report(report_path, results_payload)
print("Successfully generated training_results.json and training_report.md!")
