"""
Audit dataset annotations before training.

Checks for:
1. Malformed annotation lines (non-numeric tokens, missing coordinates).
2. Out-of-bounds coordinates (normalized values not in [0, 1]).
3. Small bounding boxes (< 12px at target training resolution).
4. Aspect ratio distribution (p95 check for elongated boxes).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def audit(images_dir: str | Path, labels_dir: str | Path, imgsz: int = 960) -> dict:
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    image_files = sorted([p for p in images_path.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    total_images = len(image_files)

    malformed_count = 0
    out_of_bounds_count = 0
    small_box_count = 0
    total_boxes = 0
    aspect_ratios = []
    box_heights_px = []

    for img_p in image_files:
        lbl_p = labels_path / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue

        lines = lbl_p.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            # Standard YOLO bbox format: class xc yc w h (5 tokens)
            if len(parts) == 5:
                try:
                    cls_id = int(float(parts[0]))
                    xc, yc, w, h = (float(v) for v in parts[1:5])
                except ValueError:
                    malformed_count += 1
                    continue

                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    out_of_bounds_count += 1

                # Note: Assumes standard square letterbox scaling where normalized height scales directly against target imgsz.
                h_px = h * imgsz
                w_px = w * imgsz
                total_boxes += 1
                box_heights_px.append(h_px)
                if h_px < 12.0:
                    small_box_count += 1
                if min(w_px, h_px) > 0:
                    aspect_ratios.append(max(w_px, h_px) / min(w_px, h_px))
            else:
                malformed_count += 1

    p95_ar = float(np.percentile(aspect_ratios, 95)) if aspect_ratios else 0.0
    under_12_pct = (small_box_count / total_boxes * 100.0) if total_boxes > 0 else 0.0

    print("========================================")
    print("Dataset Audit Summary")
    print("========================================")
    print(f"Total Images:            {total_images}")
    print(f"Total Boxes:             {total_boxes}")
    print(f"Malformed Entries:       {malformed_count}")
    print(f"Out of Bounds Entries:   {out_of_bounds_count}")
    print(f"Boxes < 12px tall:       {small_box_count} ({under_12_pct:.1f}%)")
    print(f"Aspect Ratio (p95):      {p95_ar:.2f}")
    print("========================================")

    return {
        "total_images": total_images,
        "total_boxes": total_boxes,
        "malformed_count": malformed_count,
        "out_of_bounds_count": out_of_bounds_count,
        "small_box_count": small_box_count,
        "small_box_percent": under_12_pct,
        "p95_aspect_ratio": p95_ar,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit YOLO dataset annotations before training.")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--labels", required=True, help="Path to labels directory")
    parser.add_argument("--imgsz", type=int, default=960, help="Target training resolution")
    args = parser.parse_args()

    audit(args.images, args.labels, args.imgsz)


if __name__ == "__main__":
    main()
