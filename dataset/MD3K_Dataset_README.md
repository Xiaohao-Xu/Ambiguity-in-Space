# MD-3K Dataset README

MD-3K is the dataset used in **One Scene, Two Depths: Probing Geometric Ambiguity in Monocular Foundation Models**. It is a sparse ordinal benchmark for studying how monocular depth models behave when a transparent scene admits two valid ray-wise depth layers.

This README documents the **current released annotation format**. The current `annotations.json` intentionally uses a compact schema with one `label` field and one ordered point pair. It does **not** contain separate `foreground_label` and `background_label` fields. The compact schema is sufficient to recover both layer-specific ordinal relations.

## Scope

MD-3K is intended for analyzing:

- depth-layer preference under transparent-scene ambiguity,
- single-output agreement with the transparent foreground layer, measured by `SRA(1)`,
- single-output agreement with the visible background layer, measured by `SRA(2)`,
- paired RGB/LVP complementarity, measured by `ML-SRA`.

MD-3K is **not** dense metric multi-layer depth ground truth. It should not be used to claim dense physical two-layer depth recovery unless additional dense ground truth and evaluation are provided.

## Dataset contents

MD-3K contains:

- 3,161 RGB images from GDD,
- ambiguous-region masks,
- sparse point-pair annotations,
- one ordered point pair per image in the released annotation file,
- two layer-specific ordinal relations per point pair:
  - **Layer 1:** transparent foreground surface,
  - **Layer 2:** visible background behind the transparent surface.

The benchmark is partitioned into:

- **Same subset:** foreground and background layers induce the same ordinal relation.
- **Reverse subset:** foreground and background layers induce conflicting ordinal relations.

The Reverse subset is the key diagnostic setting because one duplicated single-depth map cannot satisfy both layer-specific ordinal relations there.

## Suggested local layout

```text
data/MD-3K/
├── images/
│   ├── ...
├── masks/
│   ├── ...
└── annotations.json
```

## Released compact annotation schema

The current release uses the following compact schema:

```json
{
  "./images/example.png": [
    {
      "position": [[x1, y1], [x2, y2]],
      "label": 2
    }
  ]
}
```

Some older internal scripts used `point1` and `point2`. In the released compact schema, these correspond to:

```python
p1 = annotation["position"][0]
p2 = annotation["position"][1]
```

Coordinates are stored as image coordinates `[x, y]`. When indexing a NumPy/OpenCV depth map, use `[y, x]`:

```python
d1 = depth[int(y1), int(x1)]
d2 = depth[int(y2), int(x2)]
```

## Meaning of `position` and `label`

The released compact schema is canonicalized so that the foreground-layer relation is always:

```text
Layer 1 / foreground target: p1 is nearer than p2
```

The `label` field specifies whether the background-layer relation is the same or reversed:

| `label` | subset | Layer 1 / foreground target | Layer 2 / background target |
|---:|---|---|---|
| `1` | Same | `p1` nearer than `p2` | `p1` nearer than `p2` |
| `2` | Reverse | `p1` nearer than `p2` | `p2` nearer than `p1` |

Equivalently, the compact schema expands to:

```python
foreground_label = "p1_nearer_than_p2"

if label == 1:
    subset = "same"
    background_label = "p1_nearer_than_p2"
elif label == 2:
    subset = "reverse"
    background_label = "p2_nearer_than_p1"
else:
    raise ValueError(f"Unexpected MD-3K label: {label}")
```

Do **not** interpret `label = 1` as `SRA(1)` or `label = 2` as `SRA(2)`. The integer `label` identifies the Same/Reverse relation pattern. `SRA(1)` and `SRA(2)` are evaluation metrics for Layer 1 and Layer 2.

## Depth convention before evaluation

Before comparing a predicted depth map to MD-3K labels, verify the output convention.

Some models save **depth-like** outputs, where larger values mean farther points. Other models save **inverse-depth/disparity-like** outputs, where larger values mean nearer points. The ordinal comparison must be made under a consistent near/far convention.

The helper below returns a signed predicted relation:

- `+1`: predicted `p1` nearer than `p2`,
- `-1`: predicted `p2` nearer than `p1`,
- `0`: tie or invalid value; count as incorrect unless explicitly handled otherwise.

```python
import math


def predicted_relation(depth, p1, p2, *, larger_value_means_farther=True):
    """Return +1 if p1 is predicted nearer, -1 if p2 is predicted nearer, 0 for tie/invalid."""
    x1, y1 = p1
    x2, y2 = p2

    d1 = float(depth[int(y1), int(x1)])
    d2 = float(depth[int(y2), int(x2)])

    if not (math.isfinite(d1) and math.isfinite(d2)) or d1 == d2:
        return 0

    if larger_value_means_farther:
        return +1 if d1 < d2 else -1
    else:
        return +1 if d1 > d2 else -1
```

## Computing SRA(1) and SRA(2)

For each annotation:

```python
p1, p2 = annotation["position"]
label = annotation["label"]

fg_target = +1
bg_target = +1 if label == 1 else -1

r = predicted_relation(depth, p1, p2, larger_value_means_farther=True)

correct_sra1 = (r == fg_target)
correct_sra2 = (r == bg_target)
```

Aggregate over the chosen split:

```python
SRA1 = mean(correct_sra1 for all evaluated pairs)
SRA2 = mean(correct_sra2 for all evaluated pairs)
alpha = SRA2 - SRA1
```

Interpretation:

- positive `alpha`: stronger agreement with the visible background layer,
- negative `alpha`: stronger agreement with the transparent foreground layer.

For the Same subset (`label = 1`), `fg_target` and `bg_target` are identical, so `SRA(1)` and `SRA(2)` measure the same ordinal relation. For the Reverse subset (`label = 2`), they are conflicting targets.

## Worked example

Suppose an annotation is:

```json
{
  "position": [[120, 300], [450, 310]],
  "label": 2
}
```

Then:

```text
p1 = [120, 300]
p2 = [450, 310]
label = 2 → Reverse subset

Layer 1 / foreground target: p1 nearer than p2
Layer 2 / background target: p2 nearer than p1
```

If a predicted depth map gives `p1` nearer than `p2`, then:

```text
correct for SRA(1)
incorrect for SRA(2)
```

If a predicted depth map gives `p2` nearer than `p1`, then:

```text
incorrect for SRA(1)
correct for SRA(2)
```

## Computing ML-SRA for an RGB/LVP candidate pair

`ML-SRA` evaluates whether a **pair** of candidate predictions can jointly satisfy the two layer-specific targets. The assignment between candidates and layers is fixed once over the evaluated split; it is **not** chosen per image.

For each annotation, compute relations for both predictions:

```python
r_rgb = predicted_relation(depth_rgb, p1, p2, larger_value_means_farther=True)
r_lvp = predicted_relation(depth_lvp, p1, p2, larger_value_means_farther=True)

fg_target = +1
bg_target = +1 if label == 1 else -1
```

Evaluate the two possible fixed assignments:

```python
# Assignment A: RGB -> foreground, LVP -> background
correct_rgb_fg = (r_rgb == fg_target) and (r_lvp == bg_target)

# Assignment B: LVP -> foreground, RGB -> background
correct_lvp_fg = (r_lvp == fg_target) and (r_rgb == bg_target)
```

The benchmark-level paired score is computed with one fixed assignment across all images. A convenient implementation is:

```python
ml_sra_rgb_fg = mean(correct_rgb_fg for all evaluated pairs)
ml_sra_lvp_fg = mean(correct_lvp_fg for all evaluated pairs)

ML_SRA = max(ml_sra_rgb_fg, ml_sra_lvp_fg)
assignment = "RGB->foreground, LVP->background" if ml_sra_rgb_fg >= ml_sra_lvp_fg else "LVP->foreground, RGB->background"
```

Report the selected assignment together with the score. This is a dataset-level assignment, not per-image oracle layer selection.

For the worked example above (`label = 2`):

```text
Foreground target: p1 nearer than p2
Background target: p2 nearer than p1
```

If RGB predicts `p1` nearer than `p2` and LVP predicts `p2` nearer than `p1`, then the assignment `RGB -> foreground, LVP -> background` is correct for this pair. If both predictions produce the same ordinal relation, the pair cannot satisfy both layers on this Reverse-subset example.

## Minimal evaluator skeleton

```python
from statistics import mean


def targets_from_label(label):
    if label == 1:
        return +1, +1
    if label == 2:
        return +1, -1
    raise ValueError(f"Unexpected MD-3K label: {label}")


def evaluate_single(depth_by_image, annotations, *, larger_value_means_farther=True):
    sra1_hits, sra2_hits = [], []
    for image_path, anns in annotations.items():
        depth = depth_by_image[image_path]
        for ann in anns:
            p1, p2 = ann["position"]
            fg_target, bg_target = targets_from_label(ann["label"])
            r = predicted_relation(depth, p1, p2, larger_value_means_farther=larger_value_means_farther)
            sra1_hits.append(r == fg_target)
            sra2_hits.append(r == bg_target)
    sra1 = mean(sra1_hits)
    sra2 = mean(sra2_hits)
    return {"SRA(1)": sra1, "SRA(2)": sra2, "alpha": sra2 - sra1}


def evaluate_pair(depth_rgb_by_image, depth_lvp_by_image, annotations, *, larger_value_means_farther=True):
    rgb_fg_hits, lvp_fg_hits = [], []
    for image_path, anns in annotations.items():
        depth_rgb = depth_rgb_by_image[image_path]
        depth_lvp = depth_lvp_by_image[image_path]
        for ann in anns:
            p1, p2 = ann["position"]
            fg_target, bg_target = targets_from_label(ann["label"])
            r_rgb = predicted_relation(depth_rgb, p1, p2, larger_value_means_farther=larger_value_means_farther)
            r_lvp = predicted_relation(depth_lvp, p1, p2, larger_value_means_farther=larger_value_means_farther)
            rgb_fg_hits.append((r_rgb == fg_target) and (r_lvp == bg_target))
            lvp_fg_hits.append((r_lvp == fg_target) and (r_rgb == bg_target))

    ml_rgb_fg = mean(rgb_fg_hits)
    ml_lvp_fg = mean(lvp_fg_hits)
    if ml_rgb_fg >= ml_lvp_fg:
        return {"ML-SRA": ml_rgb_fg, "assignment": "RGB->foreground, LVP->background"}
    return {"ML-SRA": ml_lvp_fg, "assignment": "LVP->foreground, RGB->background"}
```

## Recommended reporting

When reporting MD-3K results, specify:

- whether results are on Overall, Same, or Reverse subset,
- whether the model output is depth-like or inverse-depth-like,
- the candidate inputs being evaluated, e.g. RGB, LVP, Gaussian, Sobel,
- `SRA(1)`, `SRA(2)`, and `alpha` for single predictions,
- `ML-SRA` and the fixed assignment for paired predictions.

## Citation

```bibtex
@inproceedings{xu2026onescenetwodepths,
  title={One Scene, Two Depths: Probing Geometric Ambiguity in Monocular Foundation Models},
  author={Xu, Xiaohao and Xue, Feng and Li, Xiang and Li, Haowei and Yang, Shusheng and Zhang, Tianyi and Johnson-Roberson, Matthew and Huang, Xiaonan},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```
