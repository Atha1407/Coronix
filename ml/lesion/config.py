"""Training and inference configuration for CADICA lesion detection."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "ml" / "data"
PROCESSED_CADICA_DIR = DATA_ROOT / "processed" / "cadica"

TRAIN_CSV = PROCESSED_CADICA_DIR / "train.csv"
VAL_CSV = PROCESSED_CADICA_DIR / "val.csv"
TEST_CSV = PROCESSED_CADICA_DIR / "test.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "ml" / "models"

# 8 classes: 0 = background, 1-7 = CADICA severity categories
CLASS_NAMES: List[str] = [
    "background",
    "p0_20",
    "p20_50",
    "p50_70",
    "p70_90",
    "p90_98",
    "p99",
    "p100",
]

NUM_CLASSES = len(CLASS_NAMES)  # 8

CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}


@dataclass
class LesionTrainingConfig:
    """Configurable hyperparameters for CADICA lesion detector training."""

    # Paths (relative to project root)
    data_root: Path = field(default_factory=lambda: DATA_ROOT)
    train_csv: Path = field(default_factory=lambda: TRAIN_CSV)
    val_csv: Path = field(default_factory=lambda: VAL_CSV)
    test_csv: Path = field(default_factory=lambda: TEST_CSV)
    checkpoint_dir: Path = field(default_factory=lambda: CHECKPOINT_DIR)

    # Class parameters
    num_classes: int = NUM_CLASSES
    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))

    # Training parameters
    batch_size: int = 4
    num_epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2  # Safe 2 workers for Windows multiprocessing acceleration

    # Model evaluation / inference thresholds
    confidence_threshold: float = 0.15
    iou_threshold: float = 0.5

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Imbalance handling
    use_weighted_sampler: bool = True

    # Logging
    log_interval: int = 50

    def __post_init__(self):
        """Ensure paths are Path instances and directories exist."""
        self.data_root = Path(self.data_root)
        self.train_csv = Path(self.train_csv)
        self.val_csv = Path(self.val_csv)
        self.test_csv = Path(self.test_csv)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
