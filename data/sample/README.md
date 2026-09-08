# Sample Tiles
 
Six real map tiles from the Jeddah test split, bundled so the repository runs immediately on clone without downloading the full dataset. Jeddah was held out entirely from training, so detections here reflect genuine unseen-data behavior.
 
- `images/`: 6 rendered OpenStreetMap tiles containing Arabic, English, and mixed labels.
- `labels/`: Ground-truth annotations in YOLO OBB format (`class x1 y1 x2 y2 x3 y3 x4 y4`, normalized).
This is a smoke test, not a benchmark. Reported results in the top-level [README](../../README.md) come from the full 168-tile Jeddah split.
 
```bash
python -m src.predict --weights best.pt --source data/sample/images \
    --out assets/predictions --imgsz 1024 --compare --save-json
```
 
Tiles rendered from OpenStreetMap data, © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), ODbL. Styling based on OpenStreetMap Carto (CC-BY-SA 2.0).
 
