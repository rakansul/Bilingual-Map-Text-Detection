# Runbook

Step-by-step reproduction and publishing runbook for Bilingual Map Text Detection.

---

## 0. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verify dependencies: `python -c "import ultralytics, cv2, PIL; print('Environment ready')"`

---

## 1. Dataset Configuration

Organize your full dataset in the structure defined in `configs/data.yaml`:

```
datasets/map_text/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

**Split by geographic region**: Use Riyadh tiles for training/validation and Jeddah tiles for held-out testing to prevent spatial and styling leakage.

---

## 2. Dataset Audit

Validate annotations and bounding box dimensions:

```bash
python -m src.audit_dataset \
    --images datasets/map_text/images/train \
    --labels datasets/map_text/labels/train \
    --imgsz 960
```

Refer to [`docs/audit.md`](audit.md) for expected distributions.

---

## 3. Training

Train the single-class YOLOv8s detector:

```bash
python -m src.train \
    --data configs/data.yaml \
    --model yolov8s.pt \
    --epochs 120 \
    --imgsz 960 \
    --batch 8 \
    --seed 0
```

Weights will be saved to `runs/detect/map_text/weights/best.pt`.

---

## 4. Evaluation

Evaluate performance on the held-out test split:

```bash
python -m src.evaluate \
    --weights runs/detect/map_text/weights/best.pt \
    --data configs/data.yaml \
    --split test \
    --imgsz 960
```

This updates [`docs/metrics.md`](metrics.md) and [`docs/metrics.json`](metrics.json).

---

## 5. Visual Inference & Predictions

Run inference across sample tiles:

```bash
python -m src.predict \
    --weights runs/detect/map_text/weights/best.pt \
    --source data/sample/images \
    --out assets/predictions \
    --compare --save-json
```

---

## 6. Build Sample Subset (Optional)

The committed `data/sample/` folder contains synthetic check tiles for clone smoke tests. To replace them with a real stratified sample carved from your full training set:

```bash
python -m src.make_sample \
    --images datasets/map_text/images/train \
    --labels datasets/map_text/labels/train \
    --out data/sample \
    --n 20
```

---

## 7. Repository Verification & Publication

Run the local verification script:

```bash
./scripts/verify_repo.sh
```

Initialize git and push to GitHub:

```bash
git init
git add .
git commit -m "Initial release: bilingual map text detection repository"
git branch -M main
git remote add origin https://github.com/rakansul/Bilingual-Map-Text-Detection.git
git push -u origin main
```

Navigate to **Releases** on GitHub, draft tag `v1.0`, and attach `best.pt`.
