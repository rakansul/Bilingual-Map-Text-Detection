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

**On `path` resolution.** Ultralytics resolves a relative `path` against its own
`datasets_dir` setting, not the current working directory. Either place the
dataset under that directory, set an absolute `path` in `configs/data.yaml`, or
point the setting at the repository root:

```bash
yolo settings datasets_dir="$(pwd)"
```

**Split by city, not at random.** Riyadh tiles fill train and val; Jeddah tiles
are held out entirely as test. Tiles from one urban area share font typography,
layout styling, and street-name vocabulary, so a random split leaks that shared
structure across train and test and the resulting score measures memorization.

---

## 2. Dataset Audit

Validate annotations, box dimensions, and rotation distribution:

```bash
python -m src.audit_dataset \
    --root datasets/map_text \
    --splits train,val,test \
    --write docs/audit.md
```

Each polygon is denormalized against its own tile before measurement, so the
pixel figures reflect the rendered tile rather than an assumed size.

`--write` refreshes only the tables between the `<!-- AUDIT:BEGIN -->` and
`<!-- AUDIT:END -->` markers in [`docs/audit.md`](audit.md); the analysis text
around them is preserved. Omit `--write` to print the summary without touching
the file. Re-run whenever the annotation set changes, so the audit numbers stay
consistent with the dataset table in the README.

A single split can be audited directly:

```bash
python -m src.audit_dataset --images data/sample/images --labels data/sample/labels
```

---

## 3. Training

Train the single-class OBB detector:

```bash
python -m src.train \
    --data configs/data.yaml \
    --model yolo26s-obb.pt \
    --imgsz 1024 \
    --save-dir runs/obb/obb_v2 \
    --epochs <EPOCHS> \
    --batch <BATCH> \
    --seed <SEED>
```

Weights land in `<save-dir>/weights/best.pt`, so the command above writes
`runs/obb/obb_v2/weights/best.pt`.

To reproduce v1, substitute `--model yolo11s-obb.pt --save-dir runs/obb/obb_v1`.
Both versions were trained under identical conditions so the comparison in the
README holds.

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
    --imgsz 1024 \
    --model-name yolo26s-obb --version v2 --test-city Jeddah
```

This rewrites [`docs/metrics.json`](metrics.json) in full, and refreshes only the
region between the `<!-- METRICS:BEGIN -->` and `<!-- METRICS:END -->` markers in
[`docs/metrics.md`](metrics.md). The v1 table, the cost comparison, and the
written analysis sit outside those markers and survive the run. Pass
`--no-markdown` to update the JSON alone.

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
on clone against genuine unseen data. To rebuild it from your own tiles — sorted
by annotation density and sampled at a fixed stride, so the result spans sparse
through dense tiles:

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

It checks required files, that every sample tile has a matching 9-token OBB label
file, that no document still points at the pre-rename sample folder, that the
generated-section markers in `docs/` are intact, that `src/` compiles, and it
runs the offline self-test.

The self-test can also be run on its own. It needs neither weights nor the full
dataset — the OBB result object is simulated and the checks run against the six
bundled tiles:

```bash
python scripts/selftest.py
```

It verifies that rotated boxes are rendered as polygons rather than their
bounding rectangles, that `predict.py` reads `results.obb` when `results.boxes`
is `None`, that the audit parses 9-token labels, and that the metrics and audit
writers leave hand-written analysis intact.

The hero figure is a `src.predict` output, not a separate artefact. Regenerate it
with:

```bash
python -m src.predict --weights runs/obb/obb_v2/weights/best.pt \
    --source data/sample/images --out assets/predictions --imgsz 1024 --compare
cp assets/predictions/Jeddah_569_compare.png assets/hero.png
```

Push to GitHub:

```bash
git add .
git commit -m "Update documentation and metrics"
git push
```

Trained weights are distributed through GitHub **Releases** rather than committed
to the repository. Draft the tag, then attach `best.pt`.
