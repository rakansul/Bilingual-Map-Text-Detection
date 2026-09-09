"""
Offline self-test.

Checks the parts of the pipeline that broke silently before: OBB parsing,
rotated-polygon rendering, CLI flags, and the non-destructive doc writers.

Requires no model weights and no dataset — it runs against the six bundled
sample tiles. Ultralytics is not needed either; the OBB result object is
simulated.

    python scripts/selftest.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

results: list[tuple[bool, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((condition, name, detail))
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


# 1. Rendering
from src.draw import draw_obb_detections, draw_ground_truth, DrawStyle  # noqa: E402

# A 45-degree square: if drawn as a rotated polygon, hull corners stay untouched.
canvas = np.full((200, 200, 3), 255, dtype=np.uint8)
diamond = [100, 20, 180, 100, 100, 180, 20, 100]
out = draw_obb_detections(canvas, [diamond], [0], None, {0: "text"}, DrawStyle(), draw_labels=False)

corner_clean = bool((out[25, 25] == 255).all() and (out[175, 175] == 255).all())
edge_painted = bool((out[60, 60] != 255).any())
check("Rotated box is drawn as a polygon, not its bounding rectangle",
      corner_clean and edge_painted,
      "hull corners untouched, rotated edge painted")

# 2. Ground-truth parse
sample_img = REPO_ROOT / "data/sample/images/Jeddah_561.png"
sample_lbl = REPO_ROOT / "data/sample/labels/Jeddah_561.txt"
import cv2  # noqa: E402

img = cv2.imread(str(sample_img))
lines = sample_lbl.read_text(encoding="utf-8").splitlines()
gt = draw_ground_truth(img, lines, {0: "text"})
check("Ground truth renders from 9-token OBB labels",
      gt is not None and not np.array_equal(gt, img),
      f"{len([l for l in lines if l.strip()])} labels drawn")

first = [float(v) for v in lines[0].split()[1:]]
h, w = img.shape[:2]
expected = np.array(first, dtype=float).reshape(4, 2) * np.array([w, h])
check("Label coordinates denormalize against tile size",
      bool(expected.min() >= 0 and expected.max() <= max(w, h)),
      f"x range {expected[:,0].min():.0f}-{expected[:,0].max():.0f} px")

# 3. OBB extraction
from src.predict import _extract_detections  # noqa: E402


class _Tensor:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _FakeOBB:
    """Mimics ultralytics.engine.results.OBB closely enough to exercise the parser."""

    def __init__(self, n=3):
        self.n = n
        self.xyxyxyxy = _Tensor(np.tile(np.array(diamond, dtype=float).reshape(4, 2), (n, 1, 1)))
        self.conf = _Tensor(np.full(n, 0.87))
        self.cls = _Tensor(np.zeros(n))
        self.xywhr = _Tensor(np.tile([100, 100, 113, 113, np.pi / 4], (n, 1)))

    def __len__(self):
        return self.n


class _FakeResults:
    def __init__(self):
        self.obb = _FakeOBB()
        self.boxes = None


polys, confs, cids, angles = _extract_detections(_FakeResults())
check("predict.py reads results.obb when results.boxes is None",
      len(polys) == 3 and len(polys[0]) == 8,
      f"{len(polys)} detections, {len(polys[0])} coords each")
check("Rotation angle is exported in degrees",
      angles[0] is not None and abs(angles[0] - 45.0) < 0.1,
      f"{angles[0]:.1f}°")


class _AxisAlignedResults:
    class _B:
        xyxy = [_Tensor([10.0, 20.0, 30.0, 40.0])]
        conf = [_Tensor(0.5)]
        cls = [_Tensor(0.0)]

    def __init__(self):
        self.obb = None
        self.boxes = [self._B()]


polys2, _, _, _ = _extract_detections(_AxisAlignedResults())
check("Axis-aligned fallback still works", len(polys2) == 1 and len(polys2[0]) == 8)

# 4. CLI surface
import argparse  # noqa: E402
import src.predict as predict_mod  # noqa: E402

sys.argv = ["predict", "--weights", "x.pt", "--source", "y", "--imgsz", "1024"]
parser_ok = True
try:
    import io
    from contextlib import redirect_stderr

    p = argparse.ArgumentParser()
    src_text = (REPO_ROOT / "src/predict.py").read_text(encoding="utf-8")
    parser_ok = '"--imgsz"' in src_text and "imgsz=imgsz" in src_text
except Exception:
    parser_ok = False
check("predict.py defines --imgsz and forwards it to model.predict", parser_ok)

train_text = (REPO_ROOT / "src/train.py").read_text(encoding="utf-8")
check("train.py defaults to an OBB checkpoint at imgsz=1024",
      'default="yolo26s-obb.pt"' in train_text and "default=1024" in train_text)

# 5. Audit
from src.audit_dataset import audit  # noqa: E402

stats = audit(REPO_ROOT / "data/sample/images", REPO_ROOT / "data/sample/labels")
check("Audit parses OBB labels without flagging them malformed",
      stats["boxes"] > 0 and stats["malformed"] == 0,
      f"{stats['boxes']} boxes, {stats['malformed']} malformed")
check("Audit reports a rotation distribution",
      stats["median_deviation"] > 0 and stats["rot_gt_10"] > 0,
      f"median {stats['median_deviation']:.1f}°, {stats['rot_gt_10_pct']:.1f}% past 10°")

# 6. Non-destructive doc write
from src.evaluate import _write_markdown, _render_tables  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td) / "metrics.md"
    tmp.write_text((REPO_ROOT / "docs/metrics.md").read_text(encoding="utf-8"), encoding="utf-8")
    _write_markdown(tmp, _render_tables({
        "model": "yolo26s-obb", "version": "v2", "split": "test", "test_city": "Jeddah",
        "images": 168, "instances": 1965, "imgsz": 1024, "evaluated": "2026-01-01",
        "metrics": {"mAP50": 0.848, "mAP50-95": 0.514, "precision": 0.864, "recall": 0.826},
        "speed_ms": {"preprocess": 1.0, "inference": 23.4, "postprocess": 0.4},
    }))
    after = tmp.read_text(encoding="utf-8")
    check("evaluate.py preserves the v1 table and analysis outside the markers",
          "YOLO11s-OBB" in after and "Cost and throughput" in after and "0.837" in after)

from src.audit_dataset import write_report, _render_tables as _audit_tables  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td) / "audit.md"
    tmp.write_text((REPO_ROOT / "docs/audit.md").read_text(encoding="utf-8"), encoding="utf-8")
    write_report(tmp, _audit_tables(stats, {"sample": stats}))
    after = tmp.read_text(encoding="utf-8")
    check("audit_dataset.py preserves the written analysis outside the markers",
          "Rotation justifies the OBB formulation" in after and "Note on box counts" in after)

print()
failed = [n for ok, n, _ in results if not ok]
if failed:
    print(f"{len(failed)} check(s) FAILED:")
    for n in failed:
        print(f"  - {n}")
    sys.exit(1)
print(f"All {len(results)} checks passed.")
