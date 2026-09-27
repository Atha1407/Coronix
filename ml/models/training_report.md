# CADICA Lesion Detector Training Report

## 1. Executive Summary & Model Overview
- **Model Architecture**: Faster R-CNN MobileNetV3-Large 320 FPN
- **Pretrained Weights**: Torchvision COCO Default (`FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT`)
- **Total Classes**: 8 (0 = background, 1-7 = CADICA severity categories)
- **Target Device**: NVIDIA GeForce RTX 4050 Laptop GPU (cuda)
- **Total Epochs**: 5
- **Best Epoch**: Epoch 4
- **Total Training Duration**: 0:14:41 (881.0s)
- **Best Validation F1**: 0.0765 (IoU >= 0.5)

---

## 2. Training Hyperparameters
| Parameter | Value |
| :--- | :--- |
| **Batch Size** | 4 |
| **Learning Rate** | 0.0001 |
| **Weight Decay** | 0.0001 |
| **Optimizer** | AdamW with Cosine Annealing LR Scheduler |
| **Weighted Sampling** | True (Rarest present class weighting for p99/p100) |
| **Detection Thresholds** | IoU >= 0.5, Confidence >= 0.15 |

---

## 3. Epoch-by-Epoch Progress
| Epoch | Duration | Train Total Loss | Cls Loss | Box Loss | Val Loss | Val F1 | Val Precision | Val Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 247.8s | 0.0891 | 0.0314 | 0.0312 | 0.0709 | 0.0394 | 0.0225 | 0.1585 |
| 2 | 176.9s | 0.0849 | 0.0252 | 0.0440 | 0.0733 | 0.0489 | 0.0305 | 0.1235 |
| 3 | 136.1s | 0.0809 | 0.0220 | 0.0466 | 0.0981 | 0.0562 | 0.0412 | 0.0884 |
| 4 | 158.9s | 0.0746 | 0.0186 | 0.0461 | 0.0992 | 0.0765 | 0.0591 | 0.1082 |
| 5 | 140.6s | 0.0689 | 0.0168 | 0.0435 | 0.1127 | 0.0638 | 0.0534 | 0.0793 |

---

## 4. Final Test Set Evaluation
*Evaluated strictly on `ml/data/processed/cadica/test.csv` (Patient-isolated test set, zero train/val patient overlap).*

| Metric | Result (IoU >= 0.5) |
| :--- | :--- |
| **Evaluated Images** | 908 |
| **Ground-Truth Boxes** | 466 |
| **Predicted Boxes** | 1118 |
| **True Positives** | 24 |
| **False Positives** | 1094 |
| **False Negatives** | 442 |
| **Detection Precision** | **0.0215** |
| **Detection Recall** | **0.0515** |
| **Detection F1 Score** | **0.0303** |
| **Mean Matched IoU** | **0.6035** |

### Per-Severity-Class Breakdown (Test Set)
| Severity Class | Support (GT) | Predicted | TP | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| p0_20 | 195 | 508 | 16 | 0.0315 | 0.0821 | 0.0455 |
| p20_50 | 84 | 3 | 0 | 0.0000 | 0.0000 | 0.0000 |
| p50_70 | 83 | 383 | 0 | 0.0000 | 0.0000 | 0.0000 |
| p70_90 | 6 | 115 | 0 | 0.0000 | 0.0000 | 0.0000 |
| p90_98 | 16 | 79 | 0 | 0.0000 | 0.0000 | 0.0000 |
| p99 | 77 | 28 | 0 | 0.0000 | 0.0000 | 0.0000 |
| p100 | 5 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 |

---

## 5. Visual Test Predictions
Test set visual predictions showing ground-truth (green) and predicted bounding boxes (red) are saved in:
`ml/models/predictions/`

---

## 6. Disclaimer & Clinical Limitations
> [!IMPORTANT]
> **Research and Hackathon Prototype Only**: This model is an engineering prototype trained on the CADICA research dataset. It is **NOT** clinically validated, **NOT** certified for medical diagnostics, and must **NOT** be used for clinical decision-making or patient management.
