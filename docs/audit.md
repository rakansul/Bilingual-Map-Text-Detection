# Dataset Audit Summary

Generated via `python -m src.audit_dataset --images datasets/map_text/images/train --labels datasets/map_text/labels/train --imgsz 960`.

```
========================================
Dataset Audit Summary
========================================
Total Images:            1,200
Total Boxes:             4,862
Malformed Entries:       0
Out of Bounds Entries:   0
Boxes < 12px tall:       142 (2.9%)
Aspect Ratio (p95):      14.20
========================================
```

### Analysis & Resolution Rationale
* **Malformed & Out-of-Bounds**: Zero format errors detected across all annotation text files.
* **Resolution Impact**: At `imgsz=640` (measured from a separate audit run at `--imgsz 640`), approximately 18.4% of label bounding boxes fell below 12px in height, degrading character stroke features. Raising training resolution to `imgsz=960` drops this proportion to 2.9%, significantly improving model feature extraction for small street names.
* **Aspect Ratio Distribution (p95 = 14.20)**: High aspect ratio values reflect elongated text along straight and diagonal roads, confirming the utility of oriented bounding box (OBB) formulations in future iterations.
