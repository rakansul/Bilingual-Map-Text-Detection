# Bilingual Map Text Detection

Detecting and localising Arabic and English text labels on rendered map tiles.

![Detection example](assets/hero.png)

*Left: input map tile. Right: detected text labels with confidence scores.*

---

## The problem

Rendered map tiles carry their meaning in text — street names, district labels,
points of interest — but that text is baked into the raster. There is no layer
to query. Recovering it means finding the text first.

This is not the same problem as scene text detection on photographs, and models
trained on that task transfer poorly:

- **Labels are small and thin.** A district name may be 14 pixels tall on a
  1024-pixel tile. Standard 640-pixel training downsamples it past the point of
  being learnable.
- **Text follows map geometry.** Street labels rotate to run along roads and
  curve to follow them.
- **Two scripts, opposite directions.** Arabic is cursive, right-to-left, and
  connects its glyphs; English is discrete and left-to-right. They share tiles
  and sometimes share a single label.
- **The background is adversarial.** Road casings, contour lines, and hatching
  are thin dark strokes on a light ground — visually, exactly what text is.

## Dataset

| | |
|---|---|
| Tiles | ~1,200 |
| Format | YOLO (`class xc yc w h`, normalised) & OBB 8-point polygon |
| Classes | 1 (`text`) |
| Scripts | Arabic, English, mixed |
| Source | Rendered basemap tiles (Riyadh & Jeddah) |
| Split | Geographic region split: Riyadh (Train/Val), Jeddah (Test) |

A 20-tile sample lives in [`data/sample`](data/sample) so the repo runs on a
fresh clone.

> **Splitting note.** Splits are by geographic region, not random. Tiles from
> one city share basemap styling, fonts, and label vocabulary, so a random split
> puts near-duplicates on both sides and inflates validation scores.

## Approach

Single-class YOLO detection and Oriented Bounding Box (OBB) formulation. Every text label is one box, regardless of script.

Choices worth explaining:

- **Training at `imgsz=1024` / `960`, not 640.** Set by the box-size audit: at 640, a
  large share of labels fell below roughly 12 pixels tall. Raising input
  resolution mattered more than model capacity.
- **Rotation and shear augmentation disabled.** Rotating a tile arbitrarily produces label noise, as text angle directly aligns with road geometry.
- **Horizontal flipping disabled.** It mirrors text. Reversed Arabic glyphs are invalid.
- **Single class, not one per script.** Locating text and identifying its script
  are separable problems. Merging them into one head would have made a small
  dataset smaller per class. Script classification belongs downstream.

## Results

Evaluated on the held-out **Jeddah** test split at `imgsz=1024`. Full output:
[`docs/metrics.md`](docs/metrics.md).

| Metric | Value |
|---|---|
| mAP@50 | 0.9412 |
| mAP@50-95 | 0.6640 |
| Precision | 0.8925 |
| Recall | 0.9444 |

### Observations & Failure Modes
The detector performs reliably across both Arabic and Latin scripts on straight and angled roads. Degradation occurs primarily in:
1. **Extremely dense urban junctions** where multiple overlapping district and road names lead to NMS suppression of closely adjacent boxes.
2. **Low-contrast labels** running across textured terrain or shaded green park fills.
3. **Sharp curved road labels** where a single box approximates a compound multi-segment arc.

## Quickstart

```bash
git clone https://github.com/rakansul/map-text-detection.git
cd map-text-detection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run detection on the bundled sample tiles:

```bash
# Download weights from the Releases page or local path
python src/predict.py --weights best.pt --source data/sample/images \
    --out assets/predictions --compare --save-json
```

Reproduce training from the full dataset:

```bash
python src/audit_dataset.py --images datasets/map_text/images/train \
    --labels datasets/map_text/labels/train --imgsz 960
python src/train.py --data configs/data.yaml --model yolov8s.pt --epochs 120
python src/evaluate.py --weights runs/detect/map_text/weights/best.pt --split test
```

Full command sequence: [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Repository layout

```
├── configs/
│   ├── data.yaml            # full dataset config
│   └── data.sample.yaml     # bundled sample, for smoke tests
├── src/
│   ├── __init__.py
│   ├── draw.py              # resolution-independent box/label rendering
│   ├── audit_dataset.py     # dataset validation — run before training
│   ├── train.py             # training entry point
│   ├── evaluate.py          # metrics → docs/metrics.md
│   ├── predict.py           # inference → annotated images + JSON
│   └── make_sample.py       # carve a committable sample from the full set
├── data/sample/             # sample tiles + labels, runnable on clone
├── assets/                  # figures used in this README
├── docs/                    # runbook, metrics, notes
├── scripts/                 # verification scripts
├── requirements.txt
├── LICENSE
└── README.md
```

## A note on the visualisation

Early output looked like a failed model: boxes with labels so large they covered
the tile. The detector was fine. The renderer was calling `cv2.putText` with a
hard-coded `fontScale=1.0` — an absolute value, unrelated to image size. The
same call yields 22-pixel text whether the tile is 256 pixels or 2048.

[`src/draw.py`](src/draw.py) derives every drawing dimension from the image
instead. Cap height is a fixed fraction of image height, line thickness scales
with the diagonal, and label chips are measured with `cv2.getTextSize` and
clamped inside the frame.

## Limitations

- **Curved text is approximated.** Labels following sharp bends are approximated by a single bounding box rather than a polygon chain.
- **Single class.** Output says *where* text is, not what script it is in.
- **Detection only.** No recognition — this locates text, it does not transcribe it.
- **Basemap styling dependency.** Trained on specific tile styling; other renderers using different font weights and halos may require fine-tuning.

## Future work

- **Script classification.** Add multi-head classification (`text_ar`, `text_en`, `text_mixed`) to route cropped detections to language-specific OCR engines.
- **End-to-end OCR Recognition.** Crop detected boxes and pass them to Arabic and Latin OCR pipelines.
- **FastAPI Web Service.** A lightweight inference endpoint accepting map tiles and returning JSON detection geometries.

## License

Code is MIT — see [LICENSE](LICENSE).

## Author

Rakan Al-Wahaibi — Computer Engineering, King Saud University.
