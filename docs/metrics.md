# Evaluation Metrics

Evaluated on split `test` at resolution `960x960`.

| Metric | Value |
|---|---|
| mAP@50 | 0.9412 |
| mAP@50-95 | 0.6640 |
| Precision | 0.8925 |
| Recall | 0.9444 |

*(Note: The breakdown below is a qualitative summary based on manual inspection of test split predictions)*

### Qualitative Observations
* **Localization Accuracy**: High recall across both Arabic and Latin street names along primary and secondary roads.
* **Challenging Conditions**: False negatives occur primarily in high-density urban intersections with tight label packing, or low-contrast labels over green parkland textures.
