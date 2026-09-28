"""DataLoader factory and weighted sampling for CADICA lesion detection."""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ml.data.cadica_dataset import CADICADataset
from ml.data.transforms import get_cadica_transforms
from ml.lesion.config import LesionTrainingConfig, TRAIN_CSV, VAL_CSV, TEST_CSV


def collate_fn(batch: List[Tuple[Any, Dict[str, Any]]]) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Torchvision detection collate function.

    Batches images and target dictionaries into lists rather than stacked tensors,
    as images can have variable numbers of bounding boxes.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


def compute_image_weights(
    dataset: CADICADataset,
    class_weights_override: Optional[Dict[str, float]] = None
) -> List[float]:
    """Calculate sampling weight for each image based on its rarest severity class.

    Addresses CADICA's severe class imbalance (e.g. p99/p100 are extremely rare
    compared to p50_70/p70_90) without duplicating raw files on disk.

    Rule:
    - Multi-lesion images receive the weight of their rarest present class.
    - Non-lesion (negative) frames receive the non-lesion frequency weight.

    Args:
        dataset: CADICADataset instance.
        class_weights_override: Optional dictionary mapping category name to weight.

    Returns:
        List[float]: Per-image sampling weights for WeightedRandomSampler.
    """
    if class_weights_override is not None:
        cat_weights = dict(class_weights_override)
    else:
        # Compute inverse frequency of each class in the dataset
        cat_counts: Counter = Counter()
        for frame in dataset.frames:
            cats = frame.get("categories", [])
            if not cats or not frame.get("is_lesion", False):
                cat_counts["non_lesion"] += 1
            else:
                for c in cats:
                    cat_counts[c] += 1

        total_counts = sum(cat_counts.values()) or 1
        num_categories = max(1, len(cat_counts))
        # Balanced weighting: total / (num_classes * count)
        cat_weights = {}
        for cat, count in cat_counts.items():
            cat_weights[cat] = float(total_counts) / (num_categories * max(1, count))

    # For each image, select the weight of its rarest lesion (highest weight)
    image_weights: List[float] = []
    for frame in dataset.frames:
        cats = frame.get("categories", [])
        if not cats or not frame.get("is_lesion", False):
            w = cat_weights.get("non_lesion", 1.0)
        else:
            w = max(cat_weights.get(c, 1.0) for c in cats)
        image_weights.append(float(w))

    return image_weights


def get_train_loader(
    train_csv: Union[str, Path] = TRAIN_CSV,
    batch_size: int = 1,
    num_workers: int = 0,
    use_weighted_sampler: bool = True,
    include_non_lesions: bool = True,
    cadica_root: Optional[Union[str, Path]] = None,
) -> Tuple[DataLoader, CADICADataset]:
    """Create training DataLoader with optional balanced sampling."""
    dataset = CADICADataset(
        manifest_path=train_csv,
        cadica_root=cadica_root,
        transforms=get_cadica_transforms(is_train=True),
        include_non_lesions=include_non_lesions,
    )

    if use_weighted_sampler and len(dataset) > 0:
        weights = compute_image_weights(dataset)
        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        sampler = WeightedRandomSampler(
            weights=weights_tensor,
            num_samples=len(weights),
            replacement=True
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    return loader, dataset


def get_eval_loader(
    manifest_csv: Union[str, Path],
    batch_size: int = 1,
    num_workers: int = 0,
    include_non_lesions: bool = True,
    cadica_root: Optional[Union[str, Path]] = None,
) -> Tuple[DataLoader, CADICADataset]:
    """Create evaluation DataLoader (val or test) with shuffle=False."""
    dataset = CADICADataset(
        manifest_path=manifest_csv,
        cadica_root=cadica_root,
        transforms=get_cadica_transforms(is_train=False),
        include_non_lesions=include_non_lesions,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset


def create_dataloaders(
    config: Optional[LesionTrainingConfig] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience factory creating train, validation, and test DataLoaders from config."""
    cfg = config or LesionTrainingConfig()

    train_loader, _ = get_train_loader(
        train_csv=cfg.train_csv,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        use_weighted_sampler=cfg.use_weighted_sampler,
        cadica_root=cfg.data_root / "raw" / "CADICA",
    )

    val_loader, _ = get_eval_loader(
        manifest_csv=cfg.val_csv,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        cadica_root=cfg.data_root / "raw" / "CADICA",
    )

    test_loader, _ = get_eval_loader(
        manifest_csv=cfg.test_csv,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        cadica_root=cfg.data_root / "raw" / "CADICA",
    )

    return train_loader, val_loader, test_loader
