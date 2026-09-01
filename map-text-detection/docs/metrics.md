# Evaluation Metrics

Evaluated on the held-out test split (Jeddah map tiles) at resolution `1024x1024` using YOLO-OBB.

| Metric | Value |
|---|---|
| mAP@50 | 0.9412 |
| mAP@50-95 | 0.6640 |
| Precision | 0.8925 |
| Recall | 0.9444 |

### Metric Breakdown and Analysis
- **mAP@50 (94.12%)**: Strong localization accuracy across standard horizontal and diagonal street names.
- **mAP@50-95 (66.40%)**: Reflects tighter IoU thresholds where extremely dense or tightly curved text lines encounter subtle boundary alignment variations.
- **Precision (89.25%) & Recall (94.44%)**: High recall ensures nearly all street and POI labels are captured, while precision reflects occasional false positives over high-frequency background road casings.
