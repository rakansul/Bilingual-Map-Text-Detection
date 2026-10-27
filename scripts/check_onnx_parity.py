"""Compare a PyTorch model and its ONNX export on a folder of image tiles.

Exits with 1 if any tile exceeds the tolerance thresholds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _dets(model, path: str, imgsz: int, conf: float):
    """Return (polygons Nx8, confidences N), sorted by confidence descending."""
    obb = model.predict(path, imgsz=imgsz, conf=conf, verbose=False)[0].obb
    if obb is None or len(obb) == 0:
        return np.zeros((0, 8)), np.zeros(0)
    polys = obb.xyxyxyxy.cpu().numpy().reshape(len(obb), 8)
    confs = obb.conf.cpu().numpy()
    order = np.argsort(-confs, kind="stable")
    return polys[order], confs[order]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--source", required=True, help="folder of .png/.jpg tiles")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--max-px", type=float, default=2.0, help="allowed corner difference in pixels")
    ap.add_argument("--max-conf", type=float, default=0.02, help="allowed confidence difference")
    args = ap.parse_args()

    from ultralytics import YOLO

    pt = YOLO(args.pt)
    ox = YOLO(args.onnx, task="obb")

    tiles = sorted(p for p in Path(args.source).glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not tiles:
        print(f"No images found in {args.source}")
        return 1

    failed = 0
    print(f"{'tile':<18}{'pt':>4}{'onnx':>6}{'max px':>9}{'max conf':>10}  result")
    for t in tiles:
        pp, pc = _dets(pt, str(t), args.imgsz, args.conf)
        op, oc = _dets(ox, str(t), args.imgsz, args.conf)
        if len(pp) != len(op):
            failed += 1
            print(f"{t.name:<18}{len(pp):>4}{len(op):>6}{'-':>9}{'-':>10}  FAIL (count differs)")
            continue
        if len(pp) == 0:
            print(f"{t.name:<18}{0:>4}{0:>6}{'-':>9}{'-':>10}  ok (no detections)")
            continue
        px = float(np.abs(pp - op).max())
        cf = float(np.abs(pc - oc).max())
        ok = px <= args.max_px and cf <= args.max_conf
        failed += 0 if ok else 1
        print(f"{t.name:<18}{len(pp):>4}{len(op):>6}{px:>9.3f}{cf:>10.4f}  {'ok' if ok else 'FAIL'}")

    print(f"\n{len(tiles) - failed}/{len(tiles)} tiles within tolerance "
          f"(corners <= {args.max_px}px, confidence <= {args.max_conf}).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())