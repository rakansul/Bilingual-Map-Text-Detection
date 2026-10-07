"""Run YOLO OBB inference over map tiles and export visualisations or JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.draw import DrawStyle, draw_obb_detections, stack_comparison

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _extract_detections(results):
    """Extract polygons (flat 8-point corner lists), confidences, classes, and angles."""
    polygons: list[list[float]] = []
    confidences: list[float] = []
    class_ids: list[int] = []
    angles: list[float | None] = []

    obb = getattr(results, "obb", None)
    if obb is not None and len(obb) > 0:
        corners = obb.xyxyxyxy.cpu().numpy().reshape(len(obb), 8)
        confs = obb.conf.cpu().numpy()
        classes = obb.cls.cpu().numpy()
        # xywhr carries rotation in radians at index 4 (used for JSON export)
        rot = obb.xywhr.cpu().numpy()[:, 4] if hasattr(obb, "xywhr") else None
        for i in range(len(obb)):
            polygons.append([float(v) for v in corners[i]])
            confidences.append(float(confs[i]))
            class_ids.append(int(classes[i]))
            angles.append(float(np.degrees(rot[i])) if rot is not None else None)
        return polygons, confidences, class_ids, angles

    boxes = getattr(results, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].cpu().numpy())
            polygons.append([x1, y1, x2, y1, x2, y2, x1, y2])
            confidences.append(float(box.conf[0].cpu().numpy()))
            class_ids.append(int(box.cls[0].cpu().numpy()))
            angles.append(0.0)

    return polygons, confidences, class_ids, angles


def load_model(weights: str):
    """Load an Ultralytics model once so repeated detect() calls can reuse it."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics is required for inference. Install with `pip install ultralytics`.")
    return YOLO(weights)


def _as_model_input(image):
    """Normalise a path, BGR ndarray, or PIL image into something model.predict accepts."""
    if isinstance(image, (str, Path)):
        return str(image)
    if isinstance(image, np.ndarray):
        return image  # Ultralytics treats ndarrays as BGR, matching cv2.imread
    if hasattr(image, "convert"):  # PIL.Image.Image
        return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    raise TypeError(f"Unsupported image type: {type(image).__name__}")


def detect(image, model, imgsz: int = 1024, conf: float = 0.3) -> list[dict]:
    """Detect map text on one image and return plain, JSON-serialisable dicts.

    Args:
        image: file path, BGR numpy array (as from cv2.imread), or PIL image.
        model: object returned by load_model().
        imgsz: inference resolution.
        conf: confidence threshold.

    Returns:
        One dict per detection, with keys polygon (flat list of 8 floats,
        x1,y1,...,x4,y4 in pixel coordinates), angle_deg, confidence,
        class_id and class_name.
    """
    results = model.predict(_as_model_input(image), imgsz=imgsz, conf=conf, verbose=False)[0]
    polygons, confidences, class_ids, angles = _extract_detections(results)
    class_names = getattr(model, "names", {0: "text"})
    return [
        {
            "polygon": [round(v, 2) for v in poly],
            "angle_deg": round(a, 2) if a is not None else None,
            "confidence": round(c, 4),
            "class_id": cid,
            "class_name": class_names.get(cid, str(cid)) if isinstance(class_names, dict) else str(cid),
        }
        for poly, a, c, cid in zip(polygons, angles, confidences, class_ids)
    ]


def run_inference(
    weights: str,
    source: str,
    out: str = "assets/predictions",
    imgsz: int = 1024,
    conf: float = 0.3,
    compare: bool = False,
    save_json: bool = False,
    boxes_only: bool = False,
) -> int:
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)

    model = load_model(weights)
    class_names = getattr(model, "names", {0: "text"})

    src_path = Path(source)
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory not found: {src_path}")

    image_files = sorted([p for p in src_path.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES])
    if not image_files:
        raise FileNotFoundError(f"No .png/.jpg images found in {src_path}")

    style = DrawStyle()
    draw_labels = not boxes_only
    total_detections = 0

    for img_p in image_files:
        orig_img = cv2.imread(str(img_p))
        if orig_img is None:
            print(f"  skipped (unreadable): {img_p.name}")
            continue

        dets = detect(orig_img, model, imgsz=imgsz, conf=conf)
        total_detections += len(dets)

        annotated = draw_obb_detections(
            image=orig_img,
            polygons=[d["polygon"] for d in dets],
            class_ids=[d["class_id"] for d in dets],
            confidences=[d["confidence"] for d in dets],
            class_names=class_names,
            style=style,
            draw_labels=draw_labels,
        )

        out_img = stack_comparison(orig_img, annotated) if compare else annotated
        save_name = f"{img_p.stem}_compare.png" if compare else f"{img_p.stem}_pred.png"
        cv2.imwrite(str(out_path / save_name), out_img)

        print(f"  {img_p.name}: {len(dets)} detection(s)")

        if save_json:
            json_data = {
                "image": img_p.name,
                "imgsz": imgsz,
                "conf_threshold": conf,
                "detections": dets,
            }
            (out_path / f"{img_p.stem}.json").write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    print(f"Processed {len(image_files)} image(s), {total_detections} detection(s) -> saved to {out_path}")

    if total_detections == 0:
        print(
            "WARNING: zero detections across all tiles. Check that the weights "
            "are an OBB checkpoint and that --conf is not set too high."
        )

    return total_detections


def main():
    parser = argparse.ArgumentParser(description="Run YOLO OBB inference on map tiles.")
    parser.add_argument("--weights", required=True, help="Path to trained weights (best.pt)")
    parser.add_argument("--source", required=True, help="Path to input images directory")
    parser.add_argument("--out", default="assets/predictions", help="Directory to save visual/json outputs")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image resolution")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--compare", action="store_true", help="Save side-by-side comparison with original tile")
    parser.add_argument("--save-json", action="store_true", help="Save detection coordinates to JSON")
    parser.add_argument("--boxes-only", action="store_true", help="Render boxes without text label chips")
    args = parser.parse_args()

    run_inference(
        weights=args.weights,
        source=args.source,
        out=args.out,
        imgsz=args.imgsz,
        conf=args.conf,
        compare=args.compare,
        save_json=args.save_json,
        boxes_only=args.boxes_only,
    )


if __name__ == "__main__":
    main()
