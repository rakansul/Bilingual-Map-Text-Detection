"""
Detection rendering for map tiles.

The problem this module addresses
---------------------------------
Visualising YOLO output on variable-size map tiles often leads to readability issues
when hard-coded font scales and line thicknesses are used: small tiles get obscured
by oversized text, while large map tiles result in illegible microscopic text.

This module derives drawing dimensions dynamically from the input image resolution:
* Font scale scales with image height, constrained by minimum and maximum bounds.
* Line thickness scales with the image diagonal.
* Label chips are measured (via Pillow TrueType bounding box when available, or OpenCV)
  and clamped within image borders.
* Semi-transparent filled chips are rendered behind labels for contrast over complex map features.

Arabic and Multilingual Text
----------------------------
OpenCV Hershey fonts do not natively support Arabic glyphs or right-to-left shaping.
When ``pillow``, ``arabic-reshaper``, and ``python-bidi`` are installed, this module
renders text via Pillow with contextual shaping. If unavailable, it falls back to
OpenCV with safe fallback labeling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Optional Arabic-capable text backend
# --------------------------------------------------------------------------
try:  # pragma: no cover
    from PIL import Image, ImageDraw, ImageFont
    import arabic_reshaper
    from bidi.algorithm import get_display

    _PIL_TEXT_AVAILABLE = True
except Exception:  # noqa: BLE001
    _PIL_TEXT_AVAILABLE = False


# Font search candidates for TrueType rendering
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


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DrawStyle:
    """Dynamic resolution-scaled drawing parameters."""

    # Cap height of label text as a fraction of image height.
    text_height_frac: float = 0.016
    # Box line thickness as a fraction of the image diagonal.
    line_thickness_frac: float = 0.0022
    # Padding inside the label chip, as a fraction of text height.
    chip_pad_frac: float = 0.35
    # Clamp bounds to ensure legibility on both tiny crops and large tiles.
    min_text_px: int = 10
    max_text_px: int = 48
    min_line_px: int = 1
    max_line_px: int = 6
    # Opacity of the filled label chip. 1.0 = fully solid.
    chip_alpha: float = 0.85
    # Draw confidence score alongside class name.
    show_conf: bool = True
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX


# High-contrast palette in BGR
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


# --------------------------------------------------------------------------
# Scale resolution
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Label measurement and placement
# --------------------------------------------------------------------------
def _place_chip(
    x1: int,
    y1: int,
    chip_w: int,
    chip_h: int,
    img_w: int,
    img_h: int,
) -> tuple[int, int]:
    """Place label chip above the bounding box when possible, clamping inside frame."""
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
    """Sanitize text for OpenCV Hershey renderer."""
    if _has_arabic(text):
        return "[ar]"
    return text.encode("ascii", "replace").decode("ascii")


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def draw_detections(
    image: np.ndarray,
    boxes_xyxy: Iterable[Sequence[float]],
    class_ids: Iterable[int] | None = None,
    confidences: Iterable[float] | None = None,
    class_names: Sequence[str] | dict[int, str] | None = None,
    style: DrawStyle | None = None,
    draw_labels: bool = True,
) -> np.ndarray:
    """Draw bounding boxes and optional label chips on an image."""
    style = style or DrawStyle()
    canvas = image.copy()
    h, w = canvas.shape[:2]

    boxes = [tuple(float(v) for v in b) for b in boxes_xyxy]
    if not boxes:
        return canvas

    ids = list(class_ids) if class_ids is not None else [0] * len(boxes)
    confs = list(confidences) if confidences is not None else [None] * len(boxes)

    font_scale, text_px, line_px = resolve_scales(canvas.shape, style)
    text_thickness = max(1, line_px // 2)
    pad = max(2, int(round(text_px * style.chip_pad_frac)))

    # Step 1: Draw bounding box outlines directly on canvas
    for box, cid in zip(boxes, ids):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y2 = int(np.clip(y2, 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        color = color_for_class(cid)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_px, lineType=cv2.LINE_AA)

    if not draw_labels:
        return canvas

    # Step 2: Prepare chip background overlays
    overlay = canvas.copy()
    chip_records = []

    font_path = _resolve_font_path()
    use_pil = _PIL_TEXT_AVAILABLE and (font_path is not None)
    pil_font = ImageFont.truetype(font_path, text_px) if use_pil and font_path else None

    for box, cid, conf in zip(boxes, ids, confs):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y2 = int(np.clip(y2, 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        if class_names is not None:
            if isinstance(class_names, dict):
                name = class_names.get(int(cid), f"class_{int(cid)}")
            elif int(cid) < len(class_names):
                name = class_names[int(cid)]
            else:
                name = f"class_{int(cid)}"
        else:
            name = f"class_{int(cid)}"

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
        cx, cy = _place_chip(x1, y1, chip_w, chip_h, w, h)

        color = color_for_class(cid)
        # Draw the chip rectangle ON OVERLAY ONLY so blending works accurately
        cv2.rectangle(overlay, (cx, cy), (cx + chip_w, cy + chip_h), color, -1)
        chip_records.append((cx, cy, chip_w, chip_h, pad, render_label, bbox_offset, th))

    # Step 3: Blend overlay with canvas
    if style.chip_alpha < 1.0:
        cv2.addWeighted(overlay, style.chip_alpha, canvas, 1.0 - style.chip_alpha, 0, canvas)
    else:
        canvas[:] = overlay

    # Step 4: Render text on top of blended canvas
    pil_image = None
    pil_draw = None
    if use_pil and pil_font:
        pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        pil_draw = ImageDraw.Draw(pil_image)

    for cx, cy, chip_w, chip_h, pad, render_label, bbox_offset, th in chip_records:
        if pil_draw and pil_font:
            # Shift by bbox_offset so text is centered and doesn't get clipped
            draw_x = cx + pad - bbox_offset[0]
            draw_y = cy + pad - bbox_offset[1]
            pil_draw.text((draw_x, draw_y), render_label, font=pil_font, fill=(255, 255, 255))
        else:
            text_org_cv = (cx + pad, cy + pad + th - 2)
            cv2.putText(
                canvas,
                render_label,
                text_org_cv,
                style.font_face,
                font_scale,
                (255, 255, 255),
                text_thickness,
                lineType=cv2.LINE_AA,
            )

    if pil_image is not None:
        canvas[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return canvas


def draw_ground_truth(
    image: np.ndarray,
    yolo_lines: Iterable[str],
    class_names: Sequence[str] | dict[int, str] | None = None,
    style: DrawStyle | None = None,
) -> np.ndarray:
    """Render YOLO-format normalized label lines (cls xc yc w h) on an image."""
    h, w = image.shape[:2]
    boxes, ids = [], []
    for line in yolo_lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        xc, yc, bw, bh = (float(p) for p in parts[1:5])
        boxes.append(((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h))
        ids.append(cid)
    return draw_detections(image, boxes, ids, None, class_names, style)


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
