"""
Training entry point for YOLOv8 model on bilingual map text.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def train_model(
    data: str,
    model: str = "yolov8s.pt",
    epochs: int = 120,
    imgsz: int = 960,
    batch: int = 8,
    seed: int = 0,
    save_dir: str = "runs/detect/map_text",
):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics is required for training. Install with `pip install ultralytics`.")

    yolo_model = YOLO(model)
    results = yolo_model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=seed,
        project=str(Path(save_dir).parent),
        name=Path(save_dir).name,
        exist_ok=True,
        degrees=0.0,    # Rotation carries semantic signal; do not augment away
        fliplr=0.0,     # Mirrored Arabic text is invalid
        flipud=0.0,
        scale=0.4,
        translate=0.1,
        hsv_h=0.0,      # Fixed basemap palettes
        hsv_s=0.3,
        hsv_v=0.3,
        mosaic=0.6,
        close_mosaic=15,
        plots=True,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model on map text dataset.")
    parser.add_argument("--data", default="configs/data.yaml", help="Path to data.yaml config")
    parser.add_argument("--model", default="yolov8s.pt", help="Base model checkpoint")
    parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image resolution")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save-dir", default="runs/detect/map_text", help="Directory to save run results")
    args = parser.parse_args()

    train_model(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
