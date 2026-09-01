"""
Evaluation and metrics generation.
Evaluates model weights on a dataset split and writes docs/metrics.md and docs/metrics.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(
    weights: str,
    data: str = "configs/data.yaml",
    split: str = "test",
    imgsz: int = 960,
    out_dir: str = "docs",
):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics is required for evaluation. Install with `pip install ultralytics`.")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    metrics = model.val(data=data, split=split, imgsz=imgsz)

    box_metrics = getattr(metrics, "box", metrics)
    map50 = float(getattr(box_metrics, "map50", 0.0))
    map50_95 = float(getattr(box_metrics, "map", 0.0))
    precision = float(getattr(box_metrics, "mp", 0.0))
    recall = float(getattr(box_metrics, "mr", 0.0))

    data_summary = {
        "split": split,
        "imgsz": imgsz,
        "weights": weights,
        "metrics": {
            "mAP50": round(map50, 4),
            "mAP50-95": round(map50_95, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        },
    }

    # Save JSON
    (out_path / "metrics.json").write_text(json.dumps(data_summary, indent=2), encoding="utf-8")

    # Save Markdown matching standard template
    md_content = f"""# Evaluation Metrics

Evaluated on split `{split}` at resolution `{imgsz}x{imgsz}`.

| Metric | Value |
|---|---|
| mAP@50 | {map50:.4f} |
| mAP@50-95 | {map50_95:.4f} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |
"""
    (out_path / "metrics.md").write_text(md_content, encoding="utf-8")
    print(md_content)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO weights on a dataset split.")
    parser.add_argument("--weights", required=True, help="Path to model weights (best.pt)")
    parser.add_argument("--data", default="configs/data.yaml", help="Path to data.yaml config")
    parser.add_argument("--split", default="test", help="Split to evaluate on (test/val)")
    parser.add_argument("--imgsz", type=int, default=960, help="Evaluation image resolution")
    parser.add_argument("--out", default="docs", help="Output directory for metrics files")
    args = parser.parse_args()

    evaluate(args.weights, args.data, args.split, args.imgsz, args.out)


if __name__ == "__main__":
    main()
