"""Evaluation and metrics generation for YOLO OBB models."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

BEGIN_MARKER = "<!-- METRICS:BEGIN -->"
END_MARKER = "<!-- METRICS:END -->"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _count_split_images(data_yaml: str, split: str) -> int:
    """Count images in the given split, resolving paths the way Ultralytics does."""
    try:
        import yaml
    except ImportError:
        return 0
    try:
        cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    except OSError:
        return 0

    rel = cfg.get(split)
    if rel is None:
        return 0
    root = Path(cfg.get("path", "."))
    candidates = [root / rel, Path(rel)]
    for base in candidates:
        if base.is_dir():
            return len([p for p in base.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES])
    return 0


def _render_tables(summary: dict) -> str:
    """Render summary metrics and latency into markdown tables."""
    m = summary["metrics"]
    s = summary.get("speed_ms", {})
    lines = [
        f"## {summary['version']} — {summary['model']} (evaluated {summary['evaluated']})",
        "",
        f"Split `{summary['split']}`"
        + (f", test city {summary['test_city']}" if summary.get("test_city") else "")
        + f" — {summary['images']} tiles, {summary['instances']:,} instances, `imgsz={summary['imgsz']}`.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| mAP@50 | {m['mAP50']:.3f} |",
        f"| mAP@50-95 | {m['mAP50-95']:.3f} |",
        f"| Precision | {m['precision']:.3f} |",
        f"| Recall | {m['recall']:.3f} |",
        "",
    ]
    if s:
        lines += [
            "| Stage | ms/image |",
            "|---|---|",
            f"| Preprocess | {s.get('preprocess', 0.0):.1f} |",
            f"| Inference | {s.get('inference', 0.0):.1f} |",
            f"| Postprocess | {s.get('postprocess', 0.0):.1f} |",
            "",
        ]
    return "\n".join(lines)


def _write_markdown(path: Path, tables: str) -> None:
    """Update or insert generated metric tables between marker tags in markdown documentation."""
    block = f"{BEGIN_MARKER}\n{tables}{END_MARKER}"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if BEGIN_MARKER in text and END_MARKER in text:
            head, rest = text.split(BEGIN_MARKER, 1)
            _, tail = rest.split(END_MARKER, 1)
            path.write_text(head + block + tail, encoding="utf-8")
            print(f"Updated measured tables in {path} (analysis text preserved).")
            return
        backup = path.with_suffix(".md.bak")
        backup.write_text(text, encoding="utf-8")
        print(f"No markers found in {path}; existing content backed up to {backup}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Evaluation Metrics\n\n{block}\n", encoding="utf-8")
    print(f"Wrote {path}.")


def evaluate(
    weights: str,
    data: str = "configs/data.yaml",
    split: str = "test",
    imgsz: int = 1024,
    out_dir: str = "docs",
    model_name: str | None = None,
    version: str | None = None,
    test_city: str | None = None,
    write_markdown: bool = True,
):
    """Run model validation and record performance metrics to JSON and Markdown."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics is required for evaluation. Install with `pip install ultralytics`.")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    metrics = model.val(data=data, split=split, imgsz=imgsz)

    box_metrics = getattr(metrics, "box", metrics)
    nt = getattr(metrics, "nt_per_class", None)
    instances = int(sum(nt)) if nt is not None else 0

    summary = {
        "model": model_name or Path(weights).parent.parent.name or "unknown",
        "version": version or "",
        "split": split,
        "test_city": test_city or "",
        "images": _count_split_images(data, split),
        "instances": instances,
        "imgsz": imgsz,
        "weights": weights,
        "evaluated": date.today().isoformat(),
        "metrics": {
            "mAP50": round(float(getattr(box_metrics, "map50", 0.0)), 4),
            "mAP50-95": round(float(getattr(box_metrics, "map", 0.0)), 4),
            "precision": round(float(getattr(box_metrics, "mp", 0.0)), 4),
            "recall": round(float(getattr(box_metrics, "mr", 0.0)), 4),
        },
        "speed_ms": {k: round(float(v), 2) for k, v in getattr(metrics, "speed", {}).items()},
    }

    (out_path / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_path / 'metrics.json'}.")

    if write_markdown:
        _write_markdown(out_path / "metrics.md", _render_tables(summary))

    print(json.dumps(summary["metrics"], indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO OBB weights on a dataset split.")
    parser.add_argument("--weights", required=True, help="Path to model weights (best.pt)")
    parser.add_argument("--data", default="configs/data.yaml", help="Path to data.yaml config")
    parser.add_argument("--split", default="test", help="Split to evaluate on (test/val)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Evaluation image resolution")
    parser.add_argument("--out", default="docs", help="Output directory for metrics files")
    parser.add_argument("--model-name", help="Model label for the report, e.g. yolo26s-obb")
    parser.add_argument("--version", help="Version label for the report, e.g. v2")
    parser.add_argument("--test-city", help="Held-out city name for the report, e.g. Jeddah")
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Write metrics.json only, leaving docs/metrics.md untouched",
    )
    args = parser.parse_args()

    evaluate(
        weights=args.weights,
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        out_dir=args.out,
        model_name=args.model_name,
        version=args.version,
        test_city=args.test_city,
        write_markdown=not args.no_markdown,
    )


if __name__ == "__main__":
    main()
