"""Command line demonstration script for the integrated StenoTrace ML pipeline."""

import argparse
import json
import sys
from pathlib import Path

from ml.lesion.config import PROJECT_ROOT
from ml.pipeline.ml_pipeline import analyze_image


def main():
    parser = argparse.ArgumentParser(
        description="Run StenoTrace A/B Guided Lesion Analysis with Catheter Calibration"
    )
    default_img = (
        PROJECT_ROOT
        / "ml"
        / "data"
        / "raw"
        / "CADICA"
        / "selectedVideos"
        / "p4"
        / "v2"
        / "input"
        / "p4_v2_00020.png"
    )

    parser.add_argument(
        "--image",
        type=str,
        default=str(default_img),
        help="Path to angiogram image (default: p4_v2_00020.png)",
    )
    parser.add_argument("--ax", type=float, default=180.0, help="Point A x coordinate (default: 180)")
    parser.add_argument("--ay", type=float, default=120.0, help="Point A y coordinate (default: 120)")
    parser.add_argument("--bx", type=float, default=320.0, help="Point B x coordinate (default: 320)")
    parser.add_argument("--by", type=float, default=140.0, help="Point B y coordinate (default: 140)")
    parser.add_argument("--fr", type=float, default=6.0, help="Catheter French size (default: 6.0 Fr)")
    parser.add_argument(
        "--catheter-px",
        type=float,
        default=24.0,
        help="Catheter diameter in image pixels (default: 24.0 px)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Detection confidence threshold (default: 0.15)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("STENOTRACE ML ANALYSIS PIPELINE — CLINICAL DEMO")
    print("=" * 70)
    print(f"Image:                 {args.image}")
    print(f"Point A:               ({args.ax}, {args.ay})")
    print(f"Point B:               ({args.bx}, {args.by})")
    print(f"Catheter French:       {args.fr} Fr")
    print(f"Catheter Diameter (px):{args.catheter_px} px")
    print(f"Confidence Cutoff:     {args.conf}")
    print("-" * 70)

    result = analyze_image(
        image_path=args.image,
        point_a=(args.ax, args.ay),
        point_b=(args.bx, args.by),
        catheter_fr=args.fr,
        catheter_diameter_px=args.catheter_px,
        confidence_threshold=args.conf,
    )

    print("\nANALYSIS RESULTS:")
    print(json.dumps(result, indent=2))
    print("=" * 70)

    if result.get("valid_points"):
        if result.get("lesion_detected"):
            print(f"[+] Lesion Detected: Category {result['severity']} | Conf: {result['confidence']:.1%}")
            print(f"    Physical Size:   {result['lesion_width_mm']} mm x {result['lesion_height_mm']} mm (Length: {result['lesion_length_mm']} mm)")
            print(f"    Scale Factor:    {result['mm_per_pixel']} mm/pixel")
        else:
            print("[+] Vessel Validated: No significant narrowing detected along corridor.")
    else:
        print(f"[-] Analysis Error:  {result.get('error')}")

    print("=" * 70)


if __name__ == "__main__":
    main()
