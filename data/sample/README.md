# Sample Dataset

This directory contains 20 synthetic pipeline-check tiles and annotations created specifically for environment validation, smoke testing, and verifying that scripts execute cleanly upon cloning.

> **Note on Representativeness:** These sample tiles are not drawn from the real training set and are not distribution-representative. In this check set, bounding boxes have uniform heights (~28.8px), 0% under 12px, and a p95 aspect ratio of 11.67 (compared to 2.9% sub-12px and a p95 aspect ratio of 14.20 on the full dataset reported in [`docs/audit.md`](../../docs/audit.md)).

- `images/`: 20 synthetic check map tiles.
- `labels/`: Annotations in YOLO bounding box format (`class xc yc w h`, normalised).
