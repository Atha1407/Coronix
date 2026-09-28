"""Visual verification script for the preprocessing stages.

Renders and saves a 4-panel comparison:
1. Original image
2. Grayscale image
3. CLAHE-enhanced image
4. Final preprocessed image

Saves output to ml/data/processed/preprocessing_stages_demo.png.
"""

import sys
from pathlib import Path
from typing import Optional

# Ensure repository root is on sys.path for direct script execution
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ml.config import PreprocessingConfig
from ml.preprocessing.image_preprocessing import (
    apply_clahe,
    apply_light_denoising,
    convert_to_grayscale,
    normalize_intensity,
    validate_input_image,
)


def run_preprocessing_visualization(
    input_image_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Path:
    """Generate and save 4-panel preprocessing comparison figure.

    Args:
        input_image_path: Optional path to an actual CADICA frame.
        output_dir: Destination directory for saving the figure.

    Returns:
        Path: Path to the generated figure file.
    """
    repo_root = Path(__file__).resolve().parent.parent
    demo_dir = repo_root / "data" / "demo"
    raw_dir = repo_root / "data" / "raw"
    save_dir = Path(output_dir) if output_dir else (repo_root / "data" / "processed")
    save_dir.mkdir(parents=True, exist_ok=True)

    target_path = None
    if input_image_path and Path(input_image_path).is_file():
        target_path = Path(input_image_path)
    else:
        # Search for any real image in demo or raw
        candidate_images = list(demo_dir.glob("*.png")) + list(demo_dir.glob("*.jpg")) + \
                           list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpg"))
        candidate_images = [p for p in candidate_images if p.name != ".gitkeep"]
        if candidate_images:
            target_path = candidate_images[0]

    cfg = PreprocessingConfig()

    if target_path:
        print(f"[INFO] Using actual angiogram image: {target_path.name}")
        original = cv2.imread(str(target_path))
        title_prefix = f"Angiogram: {target_path.name}"
    else:
        print(
            "[INFO] No actual angiogram frame found in ml/data/demo/ or ml/data/raw/. "
            "Using synthetic test array strictly for software stage verification."
        )
        # Synthetic test pattern: gradient background with thin lines (software test only)
        h, w = 512, 512
        y, x = np.mgrid[0:h, 0:w]
        synthetic = ((x + y) / (h + w) * 180 + 30).astype(np.uint8)
        # Add test grid lines
        synthetic[200:203, :] = 40
        synthetic[:, 250:253] = 45
        original = cv2.cvtColor(synthetic, cv2.COLOR_GRAY2BGR)
        title_prefix = "Synthetic Software Test Frame"

    # Step-by-step intermediate capture
    validated = validate_input_image(original)
    gray = convert_to_grayscale(validated)
    normalized = normalize_intensity(gray)
    clahe_enhanced = apply_clahe(
        normalized,
        clip_limit=cfg.clahe_clip_limit,
        tile_grid_size=cfg.clahe_tile_grid_size,
    )
    final_processed = apply_light_denoising(
        clahe_enhanced,
        method=cfg.denoise_method,
        kernel_size=cfg.gaussian_kernel_size,
        sigma=cfg.gaussian_sigma,
    )

    # Render 4-panel comparison
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f"StenoTrace Preprocessing Pipeline — {title_prefix}", fontsize=14, fontweight="bold")

    # Panel 1: Original
    if original.ndim == 3:
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        axes[0].imshow(original, cmap="gray")
    axes[0].set_title(f"1. Original\nShape: {original.shape}")
    axes[0].axis("off")

    # Panel 2: Grayscale + Normalization
    axes[1].imshow(normalized, cmap="gray")
    axes[1].set_title(f"2. Grayscale Normalized\nRange: [{normalized.min()}, {normalized.max()}]")
    axes[1].axis("off")

    # Panel 3: CLAHE Enhanced
    axes[2].imshow(clahe_enhanced, cmap="gray")
    axes[2].set_title(f"3. CLAHE Enhanced\nclip={cfg.clahe_clip_limit}, grid={cfg.clahe_tile_grid_size}")
    axes[2].axis("off")

    # Panel 4: Final Preprocessed (Denoised)
    axes[3].imshow(final_processed, cmap="gray")
    axes[3].set_title(f"4. Final Preprocessed\nDenoised (Gaussian {cfg.gaussian_kernel_size})")
    axes[3].axis("off")

    plt.tight_layout()
    output_file = save_dir / "preprocessing_stages_demo.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[SUCCESS] Visualization saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    run_preprocessing_visualization()
