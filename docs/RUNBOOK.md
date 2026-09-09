# Runbook

Reproduction and publishing steps for Bilingual Map Text Detection.

The annotated dataset is not published with this repository, so steps 1 through 4
require your own tiles in the layout described below. Steps 5 onward run against
the bundled sample.

---

## 0. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verify dependencies: `python -c "import ultralytics, cv2, PIL; print('Environment ready')"`

Training and evaluation for this project ran on a Colab T4 with Ultralytics
8.4.137 and PyTorch 2.11.0+cu128. A full training run takes roughly 1.5 hours.

---

## 1. Dataset Configuration

Organize the dataset in the structure defined in `configs/data.yaml`, whose
`path` key resolves to `datasets/map_text`:

```
datasets/map_text/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

Labels are YOLO OBB format: `class x1 y1 x2 y2 x3 y3 x4 y4`, normalized.

**Split by city, not at random.** Riyadh tiles fill train and val; Jeddah tiles
are held out entirely as test. Tiles from one urban area share font typography,
layout styling, and street-name vocabulary, so a random split leaks that shared
structure across train and test and the resulting score measures memorization.

---

## 2. Dataset Audit

Validate annotations, box dimensions, and rotation distribution:

```bash
python -m src.audit_dataset \
    --images datasets/map_text/images/train \
    --labels datasets/map_text/labels/train \
    --imgsz 1024
```

This writes [`docs/audit.md`](audit.md). Re-run it whenever the annotation set
changes, so the audit numbers stay consistent with the dataset table in the
README.

---

## 3. Training

Train the single-class OBB detector:

```bash
python -m src.train \
    --data configs/data.yaml \
    --model yolo26s-obb.pt \
    --imgsz 1024 \
    --epochs <EPOCHS> \
    --batch <BATCH> \
    --seed <SEED>
```

Weights are saved to `runs/obb/obb_v2/weights/best.pt`.

Substitute `yolo11s-obb.pt` to reproduce v1. Both versions were trained under
identical conditions so the comparison in the README holds.

**Augmentation constraints.** `degrees=0` and `fliplr=0` are deliberate, not
oversights. Label angle mirrors road orientation, so rotating a tile injects
noise into the signal the model should be learning; mirrored Arabic script is
orthographically invalid.

---

## 4. Evaluation

Evaluate on the held-out Jeddah test split:

```bash
python -m src.evaluate \
    --weights runs/obb/obb_v2/weights/best.pt \
    --data configs/data.yaml \
    --split test \
    --imgsz 1024
```

This updates [`docs/metrics.md`](metrics.md) and [`docs/metrics.json`](metrics.json).
Confirm the headline figures in the README match the regenerated files before
publishing.

---

## 5. Visual Inference & Predictions

Run inference over the bundled sample tiles:

```bash
python -m src.predict \
    --weights runs/obb/obb_v2/weights/best.pt \
    --source data/sample/images \
    --out assets/predictions \
    --imgsz 1024 \
    --compare --save-json
```

Run modules with `python -m src.<name>`, not `python src/<name>.py`. The latter
puts `src/` on the path instead of the repository root and the internal imports
fail.

---

## 6. Build Sample Subset (Optional)

The committed `data/sample/` folder holds six real tiles from the Jeddah test
split with their ground-truth oriented boxes, so the repository runs immediately
on clone against genuine unseen data. To rebuild it as a stratified sample from
your own tiles:

```bash
python -m src.make_sample \
    --images datasets/map_text/images/test \
    --labels datasets/map_text/labels/test \
    --out data/sample \
    --n 6
```

---

## 7. Repository Verification & Publication

Run the local verification script:

```bash
./scripts/verify_repo.sh
```

Push to GitHub:

```bash
git add .
git commit -m "Update documentation and metrics"
git push
```

Trained weights are distributed through GitHub **Releases** rather than committed
to the repository. Draft the tag, then attach `best.pt`.
