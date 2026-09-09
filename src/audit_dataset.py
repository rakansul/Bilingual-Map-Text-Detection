"""Dataset geometry audit for YOLO OBB annotations (rotation, aspect ratio, small boxes, malformed lines)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
SMALL_SIDE_PX = 12.0
AXIS_ALIGNED_DEG = 0.5

BEGIN_MARKER = "<!-- AUDIT:BEGIN -->"
END_MARKER = "<!-- AUDIT:END -->"


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def _measure_polygon(coords: list[float], width: int, height: int) -> tuple[float, float, float]:
    # Denormalize coordinates to tile pixel space
    pts = np.asarray(coords, dtype=float).reshape(4, 2)
    pts[:, 0] *= width
    pts[:, 1] *= height

    # Identify long and short sides from adjacent polygon edges
    edge_a = pts[1] - pts[0]
    edge_b = pts[2] - pts[1]
    len_a = float(np.hypot(*edge_a))
    len_b = float(np.hypot(*edge_b))

    long_edge = edge_a if len_a >= len_b else edge_b
    long_side = max(len_a, len_b)
    short_side = min(len_a, len_b)

    # Long-edge deviation from horizontal (0° = horizontal, 90° = vertical)
    deviation = float(np.degrees(np.arctan2(abs(long_edge[1]), abs(long_edge[0]))))
    return long_side, short_side, deviation


def audit(images_dir: str | Path, labels_dir: str | Path) -> dict:
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    image_files = sorted([p for p in images_path.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES])
    image_stems = {p.stem: p for p in image_files}

    malformed = 0
    out_of_bounds = 0
    orphan_labels = 0
    deviations: list[float] = []
    aspect_ratios: list[float] = []
    short_sides: list[float] = []

    for lbl_p in sorted(labels_path.glob("*.txt")):
        img_p = image_stems.get(lbl_p.stem)
        if img_p is None:
            orphan_labels += 1
            continue

        width, height = _image_size(img_p)

        # Expected format: cls x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
        for line in lbl_p.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 9:
                malformed += 1
                continue
            try:
                coords = [float(v) for v in parts[1:9]]
                int(float(parts[0]))
            except ValueError:
                malformed += 1
                continue

            # Flag normalized coordinates outside [0.0, 1.0]
            if any(c < 0.0 or c > 1.0 for c in coords):
                out_of_bounds += 1

            long_side, short_side, deviation = _measure_polygon(coords, width, height)
            if short_side <= 0.0:
                malformed += 1
                continue

            deviations.append(deviation)
            aspect_ratios.append(long_side / short_side)
            short_sides.append(short_side)

    raw = {
        "deviations": deviations,
        "aspect_ratios": aspect_ratios,
        "short_sides": short_sides,
    }
    return _summarize(len(image_files), malformed, out_of_bounds, orphan_labels, raw)


def _summarize(tiles, malformed, out_of_bounds, orphan_labels, raw) -> dict:
    deviations = raw["deviations"]
    aspect_ratios = raw["aspect_ratios"]
    short_sides = raw["short_sides"]

    total_boxes = len(deviations)
    dev = np.asarray(deviations) if total_boxes else np.zeros(0)
    ars = np.asarray(aspect_ratios) if total_boxes else np.zeros(0)
    sides = np.asarray(short_sides) if total_boxes else np.zeros(0)

    def pct(count: int) -> float:
        return (count / total_boxes * 100.0) if total_boxes else 0.0

    axis_aligned = int((dev < AXIS_ALIGNED_DEG).sum())
    rot10 = int((dev > 10.0).sum())
    rot30 = int((dev > 30.0).sum())
    small = int((sides < SMALL_SIDE_PX).sum())

    return {
        "_raw": raw,
        "tiles": tiles,
        "boxes": total_boxes,
        "malformed": malformed,
        "out_of_bounds": out_of_bounds,
        "orphan_labels": orphan_labels,
        "axis_aligned": axis_aligned,
        "axis_aligned_pct": pct(axis_aligned),
        "rot_gt_10": rot10,
        "rot_gt_10_pct": pct(rot10),
        "rot_gt_30": rot30,
        "rot_gt_30_pct": pct(rot30),
        "median_deviation": float(np.median(dev)) if total_boxes else 0.0,
        "aspect_median": float(np.median(ars)) if total_boxes else 0.0,
        "aspect_p95": float(np.percentile(ars, 95)) if total_boxes else 0.0,
        "short_side_median": float(np.median(sides)) if total_boxes else 0.0,
        "short_side_p05": float(np.percentile(sides, 5)) if total_boxes else 0.0,
        "small_boxes": small,
        "small_boxes_pct": pct(small),
    }


def print_summary(name: str, s: dict) -> None:
    print("=" * 44)
    print(f"Dataset Audit — {name}")
    print("=" * 44)
    print(f"Tiles:                     {s['tiles']}")
    print(f"Boxes:                     {s['boxes']}")
    print(f"Malformed lines:           {s['malformed']}")
    print(f"Out-of-bounds coords:      {s['out_of_bounds']}")
    print(f"Labels w/o image:          {s['orphan_labels']}")
    print(f"Axis-aligned (<0.5 deg):   {s['axis_aligned']} ({s['axis_aligned_pct']:.1f}%)")
    print(f"Rotated > 10 deg:          {s['rot_gt_10']} ({s['rot_gt_10_pct']:.1f}%)")
    print(f"Rotated > 30 deg:          {s['rot_gt_30']} ({s['rot_gt_30_pct']:.1f}%)")
    print(f"Median deviation:          {s['median_deviation']:.1f} deg")
    print(f"Aspect ratio med / p95:    {s['aspect_median']:.2f} / {s['aspect_p95']:.2f}")
    print(f"Short side med / p05 px:   {s['short_side_median']:.1f} / {s['short_side_p05']:.1f}")
    print(f"Short side < 12px:         {s['small_boxes']} ({s['small_boxes_pct']:.1f}%)")
    print("=" * 44)


def _render_tables(overall: dict, splits: dict[str, dict]) -> str:
    lines = ["## All splits", "", "| | Value |", "|---|---|"]
    lines += [
        f"| Tiles | {overall['tiles']:,} |",
        f"| Boxes | {overall['boxes']:,} |",
        f"| Malformed lines | {overall['malformed']:,} |",
        f"| Labels without a matching image | {overall['orphan_labels']:,} |",
        f"| Out-of-bounds coordinates | {overall['out_of_bounds']:,} |",
        f"| Axis-aligned (< 0.5°) | {overall['axis_aligned']:,} ({overall['axis_aligned_pct']:.1f}%) |",
        f"| Rotated > 10° | {overall['rot_gt_10']:,} ({overall['rot_gt_10_pct']:.1f}%) |",
        f"| Rotated > 30° | {overall['rot_gt_30']:,} ({overall['rot_gt_30_pct']:.1f}%) |",
        f"| Median deviation | {overall['median_deviation']:.1f}° |",
        f"| Aspect ratio, median / p95 | {overall['aspect_median']:.2f} / {overall['aspect_p95']:.2f} |",
        f"| Short side px, median / p05 | {overall['short_side_median']:.1f} / {overall['short_side_p05']:.1f} |",
        f"| Short side < 12px | {overall['small_boxes']:,} ({overall['small_boxes_pct']:.1f}%) |",
        "",
    ]

    if len(splits) > 1:
        names = list(splits)
        lines += ["## By split", "", "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        rows = [
            ("Tiles", lambda s: f"{s['tiles']:,}"),
            ("Boxes", lambda s: f"{s['boxes']:,}"),
            ("Rotated > 10°", lambda s: f"{s['rot_gt_10_pct']:.1f}%"),
            ("Rotated > 30°", lambda s: f"{s['rot_gt_30_pct']:.1f}%"),
            ("Median deviation", lambda s: f"{s['median_deviation']:.1f}°"),
            ("Aspect ratio, median / p95", lambda s: f"{s['aspect_median']:.2f} / {s['aspect_p95']:.2f}"),
            ("Short side px, median", lambda s: f"{s['short_side_median']:.1f}"),
            ("Short side < 12px", lambda s: f"{s['small_boxes_pct']:.1f}%"),
        ]
        for label, fn in rows:
            lines.append(f"| {label} | " + " | ".join(fn(splits[n]) for n in names) + " |")
        lines.append("")

    return "\n".join(lines)


def write_report(path: Path, tables: str) -> None:
    # Injects new table while preserving existing analysis text outside markers
    block = f"{BEGIN_MARKER}\n{tables}{END_MARKER}"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if BEGIN_MARKER in text and END_MARKER in text:
            head, rest = text.split(BEGIN_MARKER, 1)
            _, tail = rest.split(END_MARKER, 1)
            path.write_text(head + block + tail, encoding="utf-8")
            print(f"Updated measured tables in {path} (analysis text preserved).")
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Dataset Audit Summary\n\n{block}\n", encoding="utf-8")
    print(f"Wrote {path}.")


def main():
    parser = argparse.ArgumentParser(description="Audit YOLO OBB annotations.")
    parser.add_argument("--root", help="Dataset root containing images/{split} and labels/{split}")
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated splits when using --root")
    parser.add_argument("--images", help="Images directory (single-split mode)")
    parser.add_argument("--labels", help="Labels directory (single-split mode)")
    parser.add_argument("--write", help="Markdown report path, e.g. docs/audit.md")
    args = parser.parse_args()

    splits: dict[str, dict] = {}

    if args.root:
        root = Path(args.root)
        for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
            images = root / "images" / split
            labels = root / "labels" / split
            if not images.exists():
                print(f"Skipping '{split}': {images} not found.")
                continue
            splits[split] = audit(images, labels)
            print_summary(split, splits[split])
    elif args.images and args.labels:
        splits["all"] = audit(args.images, args.labels)
        print_summary("all", splits["all"])
    else:
        parser.error("Provide either --root, or both --images and --labels.")

    if not splits:
        parser.error("No splits were audited.")

    if len(splits) == 1:
        overall = next(iter(splits.values()))
    else:
        merged = {
            "deviations": [v for s in splits.values() for v in s["_raw"]["deviations"]],
            "aspect_ratios": [v for s in splits.values() for v in s["_raw"]["aspect_ratios"]],
            "short_sides": [v for s in splits.values() for v in s["_raw"]["short_sides"]],
        }
        overall = _summarize(
            tiles=sum(s["tiles"] for s in splits.values()),
            malformed=sum(s["malformed"] for s in splits.values()),
            out_of_bounds=sum(s["out_of_bounds"] for s in splits.values()),
            orphan_labels=sum(s["orphan_labels"] for s in splits.values()),
            raw=merged,
        )
        print_summary("all splits", overall)

    if args.write:
        write_report(Path(args.write), _render_tables(overall, splits))


if __name__ == "__main__":
    main()

