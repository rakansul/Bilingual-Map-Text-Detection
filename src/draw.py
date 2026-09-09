"""Detection rendering for map tiles with dynamic resolution scaling and multilingual text."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

# Optional Arabic reshaping and BiDi text backend via Pillow
try:  # pragma: no cover
    from PIL import Image, ImageDraw, ImageFont
    import arabic_reshaper
    from bidi.algorithm import get_display

    _PIL_TEXT_AVAILABLE = True
except Exception:  # noqa: BLE001
    _PIL_TEXT_AVAILABLE = False


_FONT_CANDIDATES = (
    os.environ.get("MTD_FONT_PATH", ""),
    "assets/fonts/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def _resolve_font_path() -> str | None:
    for candidate in _FONT_CANDIDATES:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _has_arabic(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06ff" or "\ufb50" <= ch <= "\ufeff" for ch in text)


@dataclass(frozen=True)
class DrawStyle:
    """Dynamic resolution-scaled drawing parameters."""

    # Cap height of label text as a fraction of image height
    text_height_frac: float = 0.016
    # Box line thickness as a fraction of the image diagonal
    line_thickness_frac: float = 0.0022
    # Padding inside the label chip, as a fraction of text height
    chip_pad_frac: float = 0.35
    # Clamp bounds to ensure legibility across various tile sizes
    min_text_px: int = 10
    max_text_px: int = 48
    min_line_px: int = 1
    max_line_px: int = 6
    # Opacity of the filled label chip (1.0 = solid)
    chip_alpha: float = 0.85
    show_conf: bool = True
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX


_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (60, 76, 231),    # Red
    (219, 152, 52),   # Blue
    (94, 172, 30),    # Green
    (15, 196, 241),   # Amber
    (182, 89, 155),   # Purple
    (43, 57, 192),    # Brick
    (173, 168, 45),   # Teal
    (89, 76, 62),     # Slate
)


def color_for_class(class_id: int) -> tuple[int, int, int]:
    """Return stable BGR color for class index."""
    return _PALETTE_BGR[int(class_id) % len(_PALETTE_BGR)]


def resolve_scales(image_shape: Sequence[int], style: DrawStyle) -> tuple[float, int, int]:
    """Derive font scale, text pixel height, and line thickness for an image."""
    h, w = int(image_shape[0]), int(image_shape[1])
    diag = float(np.hypot(h, w))

    text_px = int(round(h * style.text_height_frac))
    text_px = int(np.clip(text_px, style.min_text_px, style.max_text_px))

    line_px = int(round(diag * style.line_thickness_frac))
    line_px = int(np.clip(line_px, style.min_line_px, style.max_line_px))

    probe = cv2.getTextSize("Hg", style.font_face, 1.0, 1)[0][1]
    font_scale = text_px / float(probe) if probe else 1.0

    return font_scale, text_px, line_px


def _place_chip(
    x1: int,
    y1: int,
    chip_w: int,
    chip_h: int,
    img_w: int,
    img_h: int,
) -> tuple[int, int]:
    """Place label chip above bounding box, clamped inside image boundaries."""
    cx = x1
    cy = y1 - chip_h

    if cy < 0:
        cy = y1

    if cx + chip_w > img_w:
        cx = max(0, img_w - chip_w)
    cx = max(0, cx)

    cy = int(np.clip(cy, 0, max(0, img_h - chip_h)))
    return cx, cy


def _safe_label(text: str) -> str:
    """Fallback for OpenCV Hershey font which does not support Arabic glyphs."""
    if _has_arabic(text):
        return "[ar]"
    return text.encode("ascii", "replace").decode("ascii")


def _as_polygon(box: Sequence[float]) -> np.ndarray:
    """Normalize 4-point (xyxy) or 8-point flat coordinates to a (4, 2) corner array."""
    arr = np.asarray(box, dtype=float).reshape(-1)
    if arr.size == 4:
        x1, y1, x2, y2 = arr
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)
    if arr.size == 8:
        return arr.reshape(4, 2)
    raise ValueError(f"Expected 4 or 8 coordinates per box, got {arr.size}")


def _anchor_vertex(poly: np.ndarray) -> tuple[int, int]:
    """Anchor label chip to topmost corner (ties broken left) to stay clear of rotated box."""
    idx = int(np.lexsort((poly[:, 0], poly[:, 1]))[0])
    return int(round(poly[idx, 0])), int(round(poly[idx, 1]))


def _polygon_area(poly: np.ndarray) -> float:
    """Shoelace formula to filter out degenerate zero-area boxes."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _class_label(
    class_id: int,
    class_names: Sequence[str] | dict[int, str] | None,
) -> str:
    cid = int(class_id)
    if isinstance(class_names, dict):
        return class_names.get(cid, f"class_{cid}")
    if class_names is not None and cid < len(class_names):
        return class_names[cid]
    return f"class_{cid}"


def draw_obb_detections(
    image: np.ndarray,
    polygons: Iterable[Sequence[float]],
    class_ids: Iterable[int] | None = None,
    confidences: Iterable[float] | None = None,
    class_names: Sequence[str] | dict[int, str] | None = None,
    style: DrawStyle | None = None,
    draw_labels: bool = True,
) -> np.ndarray:
    """Draw oriented four-point boxes with optional label chips."""
    style = style or DrawStyle()
    canvas = image.copy()
    h, w = canvas.shape[:2]

    polys = [_as_polygon(b) for b in polygons]
    polys = [p for p in polys if _polygon_area(p) >= 1.0]
    if not polys:
        return canvas

    ids = list(class_ids) if class_ids is not None else [0] * len(polys)
    confs = list(confidences) if confidences is not None else [None] * len(polys)

    font_scale, text_px, line_px = resolve_scales(canvas.shape, style)
    text_thickness = max(1, line_px // 2)
    pad = max(2, int(round(text_px * style.chip_pad_frac)))

    # 1. Stroke rotated box outlines directly on canvas
    for poly, cid in zip(polys, ids):
        pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            canvas,
            [pts],
            isClosed=True,
            color=color_for_class(cid),
            thickness=line_px,
            lineType=cv2.LINE_AA,
        )

    if not draw_labels:
        return canvas

    # 2. Measure label chips and stage background rectangles on overlay
    overlay = canvas.copy()
    chip_records = []

    font_path = _resolve_font_path()
    use_pil = _PIL_TEXT_AVAILABLE and (font_path is not None)
    pil_font = ImageFont.truetype(font_path, text_px) if use_pil and font_path else None

    for poly, cid, conf in zip(polys, ids, confs):
        name = _class_label(cid, class_names)
        label = name
        if style.show_conf and conf is not None:
            label = f"{name} {float(conf):.2f}"

        if use_pil and pil_font:
            shaped = get_display(arabic_reshaper.reshape(label)) if _has_arabic(label) else label
            bbox = pil_font.getbbox(shaped)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            render_label = shaped
            bbox_offset = (bbox[0], bbox[1])
        else:
            render_label = _safe_label(label)
            (tw, th), baseline = cv2.getTextSize(render_label, style.font_face, font_scale, text_thickness)
            th = th + baseline
            bbox_offset = (0, 0)

        chip_w, chip_h = tw + 2 * pad, th + 2 * pad
        ax, ay = _anchor_vertex(poly)
        cx, cy = _place_chip(ax, ay, chip_w, chip_h, w, h)

        cv2.rectangle(overlay, (cx, cy), (cx + chip_w, cy + chip_h), color_for_class(cid), -1)
        chip_records.append((cx, cy, chip_w, chip_h, pad, render_label, bbox_offset, th))

    # 3. Blend overlay so map details remain visible underneath
    if style.chip_alpha < 1.0:
        cv2.addWeighted(overlay, style.chip_alpha, canvas, 1.0 - style.chip_alpha, 0, canvas)
    else:
        canvas[:] = overlay

    # 4. Render label text on top of blended chips
    pil_image = None
    pil_draw = None
    if use_pil and pil_font:
        pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        pil_draw = ImageDraw.Draw(pil_image)

    for cx, cy, chip_w, chip_h, pad, render_label, bbox_offset, th in chip_records:
        if pil_draw and pil_font:
            draw_x = cx + pad - bbox_offset[0]
            draw_y = cy + pad - bbox_offset[1]
            pil_draw.text((draw_x, draw_y), render_label, font=pil_font, fill=(255, 255, 255))
        else:
            cv2.putText(
                canvas,
                render_label,
                (cx + pad, cy + pad + th - 2),
                style.font_face,
                font_scale,
                (255, 255, 255),
                text_thickness,
                lineType=cv2.LINE_AA,
            )

    if pil_image is not None:
        canvas[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return canvas


def draw_detections(
    image: np.ndarray,
    boxes_xyxy: Iterable[Sequence[float]],
    class_ids: Iterable[int] | None = None,
    confidences: Iterable[float] | None = None,
    class_names: Sequence[str] | dict[int, str] | None = None,
    style: DrawStyle | None = None,
    draw_labels: bool = True,
) -> np.ndarray:
    """Axis-aligned convenience wrapper converting (x1, y1, x2, y2) into 4 corners."""
    return draw_obb_detections(
        image=image,
        polygons=boxes_xyxy,
        class_ids=class_ids,
        confidences=confidences,
        class_names=class_names,
        style=style,
        draw_labels=draw_labels,
    )


def draw_ground_truth(
    image: np.ndarray,
    yolo_lines: Iterable[str],
    class_names: Sequence[str] | dict[int, str] | None = None,
    style: DrawStyle | None = None,
) -> np.ndarray:
    """Render normalized YOLO label lines (OBB 9 tokens or horizontal 5 tokens) onto an image."""
    h, w = image.shape[:2]
    polys, ids = [], []

    for line in yolo_lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cid = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            continue

        if len(coords) == 8:
            pts = np.asarray(coords, dtype=float).reshape(4, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            polys.append(pts)
            ids.append(cid)
        elif len(coords) == 4:
            xc, yc, bw, bh = coords
            polys.append(
                np.array(
                    [
                        [(xc - bw / 2) * w, (yc - bh / 2) * h],
                        [(xc + bw / 2) * w, (yc - bh / 2) * h],
                        [(xc + bw / 2) * w, (yc + bh / 2) * h],
                        [(xc - bw / 2) * w, (yc + bh / 2) * h],
                    ],
                    dtype=float,
                )
            )
            ids.append(cid)

    style = style or DrawStyle()
    gt_style = DrawStyle(**{**style.__dict__, "show_conf": False})
    return draw_obb_detections(image, polys, ids, None, class_names, gt_style)


def stack_comparison(left: np.ndarray, right: np.ndarray, gap: int = 12) -> np.ndarray:
    """Join two same-height images side by side with a white gutter."""
    h = max(left.shape[0], right.shape[0])

    def _pad(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == h:
            return img
        out = np.full((h, img.shape[1], 3), 255, dtype=img.dtype)
        out[: img.shape[0]] = img
        return out

    l, r = _pad(left), _pad(right)
    gutter = np.full((h, gap, 3), 255, dtype=l.dtype)
    return np.hstack([l, gutter, r])
