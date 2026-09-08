# Bilingual Map Text Detection

Detecting and localising Arabic and English text labels on rendered map tiles.

![Detection example](assets/hero.png)

*Renderer illustration — Left: input map tile. Right: resolution-scaled bounding boxes with contrasting label chips.*

---

 ## The Problem
 
Rendered raster map tiles embed navigational information directly into image pixels: street names, district labels, and points of interest. Because the text is baked into the raster layer, querying a street name requires detecting and localizing the text region first.
 
Scene text detectors trained on natural photography transfer poorly to cartographic tiles:
 
- **Small, thin label geometry.** Many street and district names span only a dozen or so pixels in height on a 1024-pixel tile. Downsampling degrades thin character strokes before the detector ever sees them.
- **Orientation along roadways.** Labels follow road geometry rather than the image axes. In this dataset, **45% of labels sit more than 10° off horizontal and 26% exceed 30°.**
- **Bilingual scripts.** Arabic is cursive, right-to-left, and connected; English is discrete and left-to-right. Both appear within the same tile, often along the same road corridor.
- **Confusable background structure.** Road casing lines, contours, and hatching resemble thin character strokes at low resolution.
---
 
## Dataset
 
| Attribute | Full Dataset | Bundled Sample (`data/sample`) |
|---|---|---|
| **Tiles** | 904 | 6, from the Jeddah test split |
| **Labels** | 16,929 oriented boxes (18.7 per tile avg.) | Ground-truth oriented boxes |
| **Annotation format** | YOLO OBB (`class x1 y1 x2 y2 x3 y3 x4 y4`), normalized | YOLO OBB, normalized |
| **Classes** | 1 (`text`) | 1 (`text`) |
| **Scripts** | Arabic, English, and mixed bilingual labels | Arabic, English, and mixed |
| **Source** | Rendered OpenStreetMap tiles (ODbL, Carto style), Riyadh and Jeddah | Rendered OpenStreetMap tiles, Jeddah |
| **Split** | Riyadh: 736 tiles (train + val) · Jeddah: 168 tiles (test) | Smoke test |
 
Annotations were consolidated and merged from multiple Label Studio CSV exports.
 
A small sample lives in [`data/sample`](data/sample) so the repository runs immediately on clone: six real tiles from the Jeddah test split, the city held out from training, with their ground-truth oriented boxes. Detections on these tiles reflect genuine unseen-data behavior. Six tiles is a smoke test, not a benchmark; the reported metrics come from the full 168-tile split.
 
> **On the split strategy.** Splits are partitioned by city, not randomly. Tiles from the same urban area share font typography, layout styling, and street-name vocabulary, so a random split would leak that shared structure across train and test. Holding out Jeddah entirely means the test score measures generalization to unseen geography rather than memorization.
 
---
 
## Approach
 
Single-class detection with an **oriented bounding box** head. Every text occurrence is one `text` instance, which separates the spatial problem of finding text from the linguistic problem of reading it.
 
**Why OBB rather than axis-aligned boxes.** With 45% of labels past 10° of rotation and 26% past 30°, an axis-aligned box around a diagonal street name swallows a large amount of background map. The box is a poor fit for the object, IoU targets are harder to hit, and any downstream crop hands the OCR stage more noise than text. Rotated four-point polygons fit the label geometry directly.
 
Other design decisions:
 
- **`imgsz=1024`.** Map labels are small objects; training resolution is the main lever on whether thin strokes survive to the feature maps.
- **Rotation augmentation off (`degrees=0`).** Label angle mirrors road orientation. Rotating the tile injects annotation noise into a signal the model should be learning.
- **Horizontal flip off (`fliplr=0`).** Mirrored Arabic script is orthographically invalid, and mirrored road layouts are not something the model will encounter.
- **Single class.** No per-script head at this stage; script routing is deferred to future work.
---
 
## Model Comparison
 
Two architecture generations were trained under identical conditions (same dataset, same augmentation settings, same `imgsz=1024`) and evaluated on the held-out Jeddah split.
 
| | v1: YOLO11s-OBB | v2: YOLO26s-OBB |
|---|---|---|
| Parameters | 9,699,174 | 9,751,554 |
| GFLOPs | 22.4 | 21.7 |
| Precision | **0.883** | 0.864 |
| Recall | **0.858** | 0.826 |
| mAP@50 | 0.837 | **0.848** |
| mAP@50-95 | 0.502 | **0.514** |
| Inference | 23.3 ms | 23.4 ms |
| Postprocess | 8.3 ms | **0.4 ms** |
 
**v2 was selected.** It generalizes marginally better to the unseen city on both mAP metrics, and its postprocessing cost is roughly 20× lower.
 
The split verdict is not a contradiction. Precision and recall are measured at a single confidence threshold, while mAP integrates across all of them. v2 ranks its detections better overall, while v1 happens to sit at a more favorable operating point at the default threshold. Tuning v2's confidence threshold would likely close the P/R gap.
 
The postprocessing difference comes from architecture: YOLO26 is end-to-end and predicts without non-maximum suppression. This also removes a specific failure mode: the duplicate overlapping boxes NMS tends to leave on long diagonal labels.
 
The mAP differences are small enough to sit near noise on 168 test images, and are reported as such.
 
---
 
## Results
 
Held-out **Jeddah** test split: 168 tiles, 1,965 instances, evaluated at `imgsz=1024`. Full output: [`docs/metrics.md`](docs/metrics.md).
 
| Metric | YOLO26s-OBB |
|---|---|
| **mAP@50** | 0.848 |
| **mAP@50-95** | 0.514 |
| **Precision** | 0.864 |
| **Recall** | 0.826 |
 
Environment: Ultralytics 8.4.137, PyTorch 2.11.0+cu128, Tesla T4 (Colab).
 
These are scores on a city the model never trained on. Validation scores on Riyadh, the training city, are higher, and are reported separately in [`docs/metrics.md`](docs/metrics.md) rather than headlined here.
 
### Qualitative Observations & Failure Modes
 
- **Script handling.** Visual inspection of predictions shows consistent localization across both Arabic and English instances. Note that with a single class, nothing in the metrics separates the two. This is an observation from looking at outputs, not a measured result.
- **Dense intersections.** Where many labels crowd a junction, closely adjacent boxes are occasionally merged or missed.
- **Low-contrast regions.** Sensitivity drops slightly over textured green areas and shaded topography fills.
---
 
## Quickstart
 
### 1. Clone & environment
 
```bash
git clone https://github.com/rakansul/Bilingual-Map-Text-Detection.git
cd Bilingual-Map-Text-Detection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
 
### 2. Inference on the sample tiles
 
```bash
# Weights are published under GitHub Releases
python -m src.predict \
    --weights best.pt \
    --source data/sample/images \
    --out assets/predictions \
    --imgsz 1024 \
    --compare --save-json
```
 
This writes side-by-side comparisons of the six Jeddah sample tiles to `assets/predictions`, along with detections as JSON.
 
Run the modules with `python -m src.<name>`, not `python src/<name>.py`. The latter puts `src/` on the path instead of the repository root and the internal imports fail.
 
### 3. Full training & evaluation
 
```bash
# 1. Audit annotations: box sizes, aspect ratios, rotation distribution
python -m src.audit_dataset --images dataset/images/train --labels dataset/labels/train --imgsz 1024
 
# 2. Train
python -m src.train --data configs/data.yaml --model yolo26s-obb.pt --imgsz 1024
 
# 3. Evaluate on the held-out Jeddah split
python -m src.evaluate --weights runs/obb/obb_v2/weights/best.pt --split test --imgsz 1024
```
 
Training runs on a Colab T4 in roughly 1.5 hours. Step-by-step instructions: [`docs/RUNBOOK.md`](docs/RUNBOOK.md).
 
---
 
## Repository Layout
 
```
├── configs/
│   ├── data.yaml            # Full dataset configuration
│   └── data.sample.yaml     # Bundled sample configuration
├── src/
│   ├── __init__.py
│   ├── audit_dataset.py     # Box size, aspect ratio, and rotation auditor
│   ├── draw.py              # Rotated-polygon rendering with Arabic shaping
│   ├── evaluate.py          # Validation and metrics generator
│   ├── make_sample.py       # Sample set builder
│   ├── predict.py           # OBB inference with visual & JSON export
│   └── train.py             # Training entry point
├── notebooks/
│   └── make_hero.ipynb      # Generates the hero figure from trained weights
├── data/sample/             # 6 Jeddah tiles with ground-truth labels
├── assets/                  # Figures and prediction outputs
├── docs/                    # Runbook, dataset audit, and metrics
├── scripts/                 # Verification shell scripts
├── requirements.txt
├── LICENSE
└── README.md
```
 
---
 
## Visualization Pipeline
 
OpenCV's `cv2.putText` uses absolute pixel font sizes and cannot shape right-to-left Arabic glyphs. Neither behavior is acceptable here.
 
[`src/draw.py`](src/draw.py) implements:
 
- **Rotated polygon rendering.** Detections are drawn as four-point polygons via `cv2.polylines`, with the label chip anchored to the topmost corner.
- **Resolution-scaled sizing.** Font scale, line thickness, and padding derive from image dimensions rather than fixed constants, bounded by minimum and maximum values for legibility. Without this, a label occupying 8.6% of tile height at 256px collapses to 1.1% at 2048px.
- **Contrasting background chips.** Semi-transparent chips behind label text keep it readable over dense map backgrounds.
- **Arabic shaping.** A Pillow rendering path with `arabic-reshaper` and `python-bidi` for contextual right-to-left shaping, in place for the eventual OCR transcription head.
---
 
## Limitations
 
- **Single-class output.** Detects text presence and location without per-script labels, so Arabic and English performance cannot be measured separately.
- **Detection only.** No transcription. This locates text, it does not read it.
- **Two cities.** Training and evaluation both draw on Saudi OSM Carto renderings. Generalization to other renderers, styles, or regions is untested.
- **Known annotation gap.** 85 rows with missing geometry originate from a single annotator's export folder and were dropped during consolidation. Recoverable by re-exporting that Label Studio project.
- **Curved labels.** Street names that follow a curve are still approximated by a single rotated quadrilateral.
---
 
## Future Work
 
- **Script classification head.** Multi-class output (`text_ar`, `text_en`, `text_mixed`) to route detections to script-appropriate OCR models.
- **End-to-end OCR.** Text recognition on top of detection for complete map transcription.
- **Confidence threshold sweep.** v2's precision and recall are reported at the default threshold; a sweep would find its actual operating point.
- **Web demo.** A local service accepting a map tile and returning detected text regions.
---
 
## License & Provenance
 
- **Code:** MIT License. See [LICENSE](LICENSE).
- **Map data:** OpenStreetMap © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), licensed under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/). Styling based on OpenStreetMap Carto (CC-BY-SA 2.0).
---
 
## Author
 
**Rakan Al-Wehaibi**
Computer Engineering, King Saud University
 
