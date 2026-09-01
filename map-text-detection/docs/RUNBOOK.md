# Runbook

Exact order of operations, from a fresh checkout to a published repo. Each step
states what it produces and what "done" looks like.

---

## 0. Environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Check: `python -c "import ultralytics, cv2; print('ok')"`

---

## 1. Point the config at your data

Arrange the full dataset in the layout `configs/data.yaml` expects:

```
datasets/map_text/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

Edit `path` in `configs/data.yaml` if you keep it elsewhere.

**Split by region, not randomly.** If your current split was random, redo it.
Tiles from one city share fonts and vocabulary, and a random split leaks
near-duplicates into validation — the resulting mAP is not real.

---

## 2. Audit before training

```bash
python src/audit_dataset.py \
    --images datasets/map_text/images/train \
    --labels datasets/map_text/labels/train \
    --imgsz 960
```

Read three things in the output:

1. **Malformed entries** — must be zero before you train.
2. **Boxes under 12px tall** — if this is above ~20%, raise `--imgsz` or tile
   your sources into smaller crops. Training will not fix it.
3. **Aspect ratio p95** — a very high value confirms rotated text in loose
   axis-aligned boxes. Record the number; it belongs in the README limitations
   and it is the evidence for the OBB argument.

Repeat for the val split.

---

## 3. Train

```bash
python src/train.py --data configs/data.yaml --model yolov8s.pt \
    --epochs 120 --imgsz 960 --batch 8 --seed 0
```

If you already have working weights, skip this. Weights land at
`runs/detect/map_text/weights/best.pt`.

Start with `yolov8s`. Only move to `m` if the audit shows the data supports it —
on ~1,200 tiles a larger model usually overfits before it helps.

---

## 4. Evaluate

```bash
python src/evaluate.py \
    --weights runs/detect/map_text/weights/best.pt \
    --data configs/data.yaml --split test --imgsz 960
```

Produces `docs/metrics.md` and `docs/metrics.json`. Paste the table into the
README Results section.

---

## 5. Generate demo images

This is the step that feeds both the README and the infographic. Do it once,
reuse everywhere.

```bash
python src/predict.py \
    --weights runs/detect/map_text/weights/best.pt \
    --source data/sample/images \
    --out assets/predictions \
    --compare --save-json
```

Pick **8–10** outputs covering the range:

| Slot | What to look for |
|---|---|
| 1–2 | Dense Arabic labels, clean detection |
| 3–4 | English / Latin labels |
| 5–6 | Mixed-script tile |
| 7 | Rotated street name (shows the loose-fit limitation honestly) |
| 8 | A genuine failure — a miss or a false positive |

Copy the best side-by-side pair to `assets/hero.png` for the README banner.

---

## 6. Build the committable sample

```bash
python src/make_sample.py \
    --images datasets/map_text/images/train \
    --labels datasets/map_text/labels/train \
    --out data/sample --n 20
```

---

## 7. Publish

```bash
git init
git add .
git commit -m "Bilingual map text detection: detector, tooling, and docs"
git branch -M main
git remote add origin https://github.com/rakansul/map-text-detection.git
git push -u origin main
```

Then verify:

```bash
git ls-files | xargs du -ch 2>/dev/null | tail -1
git ls-files | grep -E '\.(pt|onnx)$'
```

Attach `best.pt` as a **Release** asset (Releases → Draft a new release → tag
`v1.0` → attach the file).
