"""
Inference entry point.
Generates predictions with resolution-independent bounding box rendering and JSON export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import cv2

from src.draw import DrawStyle, draw_detections, stack_comparison


def run_inference(
    weights: str,
    source: str,
    out: str = "assets/predictions",
    conf: float = 0.3,
    compare: bool = False,
    save_json: bool = False,
    boxes_only: bool = False,
):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics is required for inference. Install with `pip install ultralytics`.")

    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    src_path = Path(source)
    image_files = sorted([p for p in src_path.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    style = DrawStyle()
    draw_labels = not boxes_only

    for img_p in image_files:
        orig_img = cv2.imread(str(img_p))
        if orig_img is None:
            continue

        results = model.predict(str(img_p), conf=conf, verbose=False)[0]

        boxes_xyxy = []
        confidences = []
        class_ids = []

        if hasattr(results, "obb") and results.obb is not None and len(results.obb) > 0:
            for box in results.obb:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                boxes_xyxy.append(xyxy)
                confidences.append(float(box.conf[0].cpu().numpy()))
                class_ids.append(int(box.cls[0].cpu().numpy()))
        elif hasattr(results, "boxes") and results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                boxes_xyxy.append(xyxy)
                confidences.append(float(box.conf[0].cpu().numpy()))
                class_ids.append(int(box.cls[0].cpu().numpy()))

        annotated = draw_detections(
            image=orig_img,
            boxes_xyxy=boxes_xyxy,
            class_ids=class_ids,
            confidences=confidences,
            class_names=["text"],
            style=style,
            draw_labels=draw_labels,
        )

        out_img = stack_comparison(orig_img, annotated) if compare else annotated
        save_name = f"{img_p.stem}_pred.png" if not compare else f"{img_p.stem}_compare.png"
        cv2.imwrite(str(out_path / save_name), out_img)

        if save_json:
            json_data = {
                "image": img_p.name,
                "detections": [
                    {"box": b, "conf": round(c, 4), "class": cid}
                    for b, c, cid in zip(boxes_xyxy, confidences, class_ids)
                ],
            }
            (out_path / f"{img_p.stem}.json").write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    print(f"Processed {len(image_files)} images -> saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference and save predictions.")
    parser.add_argument("--weights", required=True, help="Path to weights file")
    parser.add_argument("--source", required=True, help="Path to input images directory")
    parser.add_argument("--out", default="assets/predictions", help="Directory to save outputs")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--compare", action="store_true", help="Save side-by-side comparison with original")
    parser.add_argument("--save-json", action="store_true", help="Save detection coordinates as JSON")
    parser.add_argument("--boxes-only", action="store_true", help="Render bounding boxes without chip labels")
    args = parser.parse_args()

    run_inference(
        weights=args.weights,
        source=args.source,
        out=args.out,
        conf=args.conf,
        compare=args.compare,
        save_json=args.save_json,
        boxes_only=args.boxes_only,
    )


if __name__ == "__main__":
    main()
