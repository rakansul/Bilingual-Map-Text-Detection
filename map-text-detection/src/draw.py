"""
Detection rendering for map tiles.

The problem this module solves
------------------------------
The common failure mode when visualising YOLO output with OpenCV is calling
``cv2.putText`` with a hard-coded ``fontScale`` (usually 1.0) and a hard-coded
``thickness``. Those units are absolute, not relative to the image. A tile
rendered at 640x640 gets a legible label; the same call on a 4096x4096 map
export produces text that is invisibly small, while a 256x256 crop gets a label
that covers the entire tile. The detector is fine -- the drawing is not.

Everything here is derived from image size, so a box looks the same at any
resolution:

* font scale is solved so that cap height is a fixed fraction of image height
* line thickness scales with the image diagonal
* label chips are measured with ``cv2.getTextSize`` and clamped inside bounds
* a filled chip is drawn behind text so labels stay readable over dense maps

Arabic labels
-------------
OpenCV's Hershey fonts contain no Arabic glyphs and apply no bidi reordering or
contextual shaping, so ``cv2.putText`` renders Arabic as garbage or blank boxes.
If ``pillow``, ``arabic-reshaper`` and ``python-bidi`` are installed, this
module renders text through Pillow instead and handles shaping correctly.
Otherwise it falls back to OpenCV and transliterates non-Latin class names so
output is never silently corrupted.
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
try:  # pragma: no cover - depends on optional extras
    from PIL import Image, ImageDraw, ImageFont
    import arabic_reshaper
    from bidi.algorithm import get_display

    _PIL_TEXT_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure means fall back
    _PIL_TEXT_AVAILABLE = False


# Font search order for the Pillow backend. Override with MTD_FONT_PATH.
_FONT_CANDIDATES = (
    os.environ.get("MTD_FONT_PATH", ""),
    "assets/fonts/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
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
    """Resolution-independent drawing parameters.

    All sizes are *fractions*, resolved against the image at draw time. This is
    the whole point of the module: nothing here is in absolute pixels.
    """

    # Cap height of label text as a fraction of image height.
    text_height_frac: float = 0.016
    # Box line thickness as a fraction of the image diagonal.
    line_thickness_frac: float = 0.0022
    # Padding inside the label chip, as a fraction of text height.
    chip_pad_frac: float = 0.35
    # Clamp resolved values so tiny and huge tiles both stay sane.
    min_text_px: int = 10
    max_text_px: int = 48
    min_line_px: int = 1
    max_line_px: int = 6
    # Opacity of the filled label chip. 1.0 = solid.
    chip_alpha: float = 0.85
    # Draw the confidence value alongside the class name.
    show_conf: bool = True
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX


# Palette in BGR. Deliberately high-contrast against typical map beige/grey.
_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (60, 76, 231),    # red
    (219, 152, 52),   # blue
    (94, 172, 30),    # green
    (15, 196, 241),   # amber
    (182, 89, 155),   # purple
    (43, 57, 192),    # brick
    (173, 168, 45),   # teal
    (89, 76, 62),     # slate
)


def color_for_class(class_id: int) -> tuple[int, int, int]:
    """Stable BGR colour for a class index."""
    return _PALETTE_BGR[int(class_id) % len(_PALETTE_BGR)]


# --------------------------------------------------------------------------
# Scale resolution
# --------------------------------------------------------------------------
def resolve_scales(image_shape: Sequence[int], style: DrawStyle) -> tuple[float, int, int]:
    """Turn fractional style values into concrete pixel values for one image.

    Returns ``(font_scale, text_px, line_px)`` where ``font_scale`` is the value
    to hand to ``cv2.putText`` and ``text_px`` is the resulting cap height.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    diag = float(np.hypot(h, w))

    text_px = int(round(h * style.text_height_frac))
    text_px = int(np.clip(text_px, style.min_text_px, style.max_text_px))

    line_px = int(round(diag * style.line_thickness_frac))
    line_px = int(np.clip(line_px, style.min_line_px, style.max_line_px))

    # Solve for the fontScale that yields text_px cap height. cv2 text height is
    # linear in fontScale, so one measurement gives the ratio exactly.
    probe = cv2.getTextSize("Hg", style.font_face, 1.0, 1)[0][1]
    font_scale = text_px / float(probe) if probe else 1.0

    return font_scale, text_px, line_px


# --------------------------------------------------------------------------
# Label placement
# --------------------------------------------------------------------------
def _measure(text: str, style: DrawStyle, font_scale: float, thickness: int) -> tuple[int, int]:
    (tw, th), baseline = cv2.getTextSize(text, style.font_face, font_scale, thickness)
    return tw, th + baseline


def _place_chip(
    x1: int,
    y1: int,
    x2: int,
    chip_w: int,
    chip_h: int,
    img_w: int,
    img_h: int,
) -> tuple[int, int]:
    """Choose the chip origin so it never leaves the frame.

    Preference is above the box; if that would clip the top of the image the
    chip flips to sit inside the box instead. Horizontal position is clamped so
    long labels on right-edge detections stay visible.
    """
    cx = x1
    cy = y1 - chip_h

    if cy < 0:  # not enough room above -> put it inside the box
        cy = y1

    if cx + chip_w > img_w:  # would overflow right edge
        cx = max(0, img_w - chip_w)
    cx = max(0, cx)

    cy = int(np.clip(cy, 0, max(0, img_h - chip_h)))
    return cx, cy


def _draw_text_cv2(
    canvas: np.ndarray,
    text: str,
    org: tuple[int, int],
    style: DrawStyle,
    font_scale: float,
    thickness: int,
    color: tuple[int, int, int],
) -> None:
    cv2.putText(
        canvas,
        text,
        org,
        style.font_face,
        font_scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def _draw_text_pil(
    canvas: np.ndarray,
    text: str,
    org: tuple[int, int],
    text_px: int,
    color: tuple[int, int, int],
) -> bool:
    """Render shaped/bidi text via Pillow. Returns False if unavailable."""
    if not _PIL_TEXT_AVAILABLE:
        return False
    font_path = _resolve_font_path()
    if font_path is None:
        return False

    shaped = get_display(arabic_reshaper.reshape(text)) if _has_arabic(text) else text

    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    drawer = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, text_px)
    drawer.text(org, shaped, font=font, fill=(color[2], color[1], color[0]))
    canvas[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return True


def _safe_label(text: str) -> str:
    """Strip glyphs OpenCV cannot render, so labels degrade visibly not silently."""
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
    class_names: Sequence[str] | None = None,
    style: DrawStyle | None = None,
    draw_labels: bool = True,
) -> np.ndarray:
    """Draw detections on a copy of ``image`` and return it.

    Parameters
    ----------
    image:
        BGR image as returned by ``cv2.imread``.
    boxes_xyxy:
        Iterable of ``(x1, y1, x2, y2)`` in absolute pixel coordinates.
    class_ids, confidences:
        Optional parallel iterables. Missing values default to class 0 and no
        confidence display.
    class_names:
        Index-aligned display names. Falls back to ``class_<id>``.
    style:
        Overrides for :class:`DrawStyle`.
    draw_labels:
        Set False to render boxes only. Useful for very dense tiles where the
        chips would occlude the map more than they inform.
    """
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

    overlay = canvas.copy() if style.chip_alpha < 1.0 else canvas

    for box, cid, conf in zip(boxes, ids, confs):
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
            continue

        name = (
            class_names[int(cid)]
            if class_names is not None and int(cid) < len(class_names)
            else f"class_{int(cid)}"
        )
        label = name
        if style.show_conf and conf is not None:
            label = f"{name} {float(conf):.2f}"

        use_pil = _PIL_TEXT_AVAILABLE and _resolve_font_path() is not None
        render_label = label if use_pil else _safe_label(label)

        tw, th = _measure(render_label, style, font_scale, text_thickness)
        chip_w, chip_h = tw + 2 * pad, th + 2 * pad
        cx, cy = _place_chip(x1, y1, x2, chip_w, chip_h, w, h)

        cv2.rectangle(overlay, (cx, cy), (cx + chip_w, cy + chip_h), color, -1)
        if overlay is not canvas:
            cv2.rectangle(canvas, (cx, cy), (cx + chip_w, cy + chip_h), color, -1)

        text_org_cv = (cx + pad, cy + chip_h - pad - 1)
        if not (use_pil and _draw_text_pil(canvas, label, (cx + pad, cy + pad), text_px, (255, 255, 255))):
            _draw_text_cv2(
                canvas, render_label, text_org_cv, style, font_scale, text_thickness, (255, 255, 255)
            )

    if overlay is not canvas and style.chip_alpha < 1.0:
        cv2.addWeighted(overlay, 1.0 - style.chip_alpha, canvas, style.chip_alpha, 0, canvas)

    return canvas


def draw_ground_truth(
    image: np.ndarray,
    yolo_lines: Iterable[str],
    class_names: Sequence[str] | None = None,
    style: DrawStyle | None = None,
) -> np.ndarray:
    """Render YOLO-format label lines (``cls xc yc w h``, normalised) on an image.

    Used to sanity-check that annotations themselves are correct before blaming
    the model for poor output.
    """
    h, w = image.shape[:2]
    boxes, ids = [], []
    for line in yolo_lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        xc, yc, bw, bh = (float(p) for p in parts[1:5])
        boxes.append(((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h))
        ids.append(cid)
    return draw_detections(image, boxes, ids, None, class_names, style)


def stack_comparison(left: np.ndarray, right: np.ndarray, gap: int = 12) -> np.ndarray:
    """Join two same-height images side by side with a white gutter.

    Handy for the README hero and the infographic 'before / after' panel.
    """
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
