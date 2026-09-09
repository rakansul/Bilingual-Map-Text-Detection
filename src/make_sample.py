"""Build a density-spanning sample of dataset tiles and labels."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def make_sample(images_dir: str, labels_dir: str, out_dir: str, n: int = 20):
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    out_path = Path(out_dir)

    out_images = out_path / "images"
    out_labels = out_path / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in images_path.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not image_files:
        print("No images found in source directory.")
        return

    # Count annotations per tile to sample evenly from sparse to dense
    counts = []
    for img_p in image_files:
        lbl_p = labels_path / f"{img_p.stem}.txt"
        cnt = 0
        if lbl_p.exists():
            cnt = len([line for line in lbl_p.read_text(encoding="utf-8").strip().splitlines() if line.strip()])
        counts.append((cnt, img_p, lbl_p))

    # Sort by annotation density and sample at a fixed stride
    counts.sort(key=lambda x: x[0])
    step = max(1, len(counts) // n)
    sampled = counts[::step][:n]

    for cnt, img_p, lbl_p in sampled:
        shutil.copy2(img_p, out_images / img_p.name)
        if lbl_p.exists():
            shutil.copy2(lbl_p, out_labels / lbl_p.name)

    print(f"Sample created: {len(sampled)} items copied to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build a density-spanning dataset sample.")
    parser.add_argument("--images", required=True, help="Source images directory")
    parser.add_argument("--labels", required=True, help="Source labels directory")
    parser.add_argument("--out", default="data/sample", help="Target output directory")
    parser.add_argument("--n", type=int, default=20, help="Number of tiles to sample")
    args = parser.parse_args()

    make_sample(args.images, args.labels, args.out, args.n)


if __name__ == "__main__":
    main()
