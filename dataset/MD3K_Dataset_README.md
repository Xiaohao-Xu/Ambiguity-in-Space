# MultiDepth-3k (MD-3k) Benchmark

MD-3k is a real-world transparent-scene benchmark for measuring how monocular depth models resolve layered geometry.

In transparent scenes, a single camera ray may contain two visually present and geometrically valid surfaces: a transparent foreground surface and the visible background behind it. Since a standard monocular depth model outputs one scalar depth per pixel, MD-3k evaluates which valid layer the model reports and whether paired hypotheses can jointly satisfy both layer-specific ordinal relations.

## Scope

MD-3k is a focused diagnostic benchmark for transparent-scene ambiguity. It is intended for analyzing depth-layer preference and multi-layer ordinal behavior, not for dense metric multi-layer depth estimation or exhaustive coverage of all transparency or material categories.

## Dataset contents

MD-3k contains:

- 3,161 RGB images from GDD.
- Ambiguous-region masks.
- Sparse point-pair annotations.
- Two layer-specific ordinal relations per point pair:
  - **Layer 1**: transparent foreground surface.
  - **Layer 2**: visible background behind the transparent surface.

The benchmark is partitioned into:

- **Same subset**: foreground and background layers induce the same ordinal relation.
- **Reverse subset**: foreground and background layers induce conflicting ordinal relations.

The Reverse subset is the key diagnostic setting where a duplicated single-depth hypothesis cannot satisfy both valid layer relations.

## Suggested layout

```text
data/MD-3K/
├── images/
│   ├── ...
├── masks/
│   ├── ...
└── annotations.json
```

## Annotation schema

A recommended explicit schema is:

```json
{
  "images/example.png": [
    {
      "point1": [x1, y1],
      "point2": [x2, y2],
      "foreground_label": "near",
      "background_label": "far",
      "subset": "reverse"
    }
  ]
}
```

If using a compact legacy schema with a single `label` field, document the mapping explicitly. Avoid describing `label = 1` as SRA(1) and `label = 2` as SRA(2), because SRA(1) and SRA(2) are evaluation targets for the foreground and background layers, not dataset subset identifiers.

## Depth convention

Before comparing a model output to MD-3k ordinal labels, verify the definition of the saved prediction.

Some models save depth-like outputs, where larger values indicate farther points. Other models save inverse-depth or disparity-like outputs, where larger values indicate nearer points. MD-3k evaluation requires a consistent near or far convention. If the prediction is inverse depth or disparity, convert it to depth, or flip the ordinal comparison, before computing SRA(1), SRA(2), depth-layer preference, or ML-SRA.

## Metrics

### SRA(1)

Spatial Relationship Accuracy with respect to the transparent foreground layer.

### SRA(2)

Spatial Relationship Accuracy with respect to the visible background layer.

### Depth-layer preference

```text
alpha = SRA(2) - SRA(1)
```

Positive `alpha` indicates stronger background-layer preference. Negative `alpha` indicates stronger foreground-layer preference.

### ML-SRA

Multi-Layer Spatial Relationship Accuracy evaluates whether a paired output, such as RGB and LVP predictions, jointly satisfies both layer-specific ordinal relations.

In the paper protocol, RGB and LVP are assigned according to the model's RGB depth-layer preference at the benchmark level:

- If RGB prefers layer 1, assign RGB to layer 1 and LVP to layer 2.
- If RGB prefers layer 2, assign RGB to layer 2 and LVP to layer 1.

This is not a per-image oracle and should not be described as automatic layer selection. Alternative pairing strategies may be useful future improvements, but they should be reported separately.

## Recommended use

Use MD-3k to:

- characterize model-intrinsic depth-layer preference,
- compare RGB and LVP behavior,
- evaluate RGB/LVP paired-hypothesis complementarity,
- analyze Same and Reverse subsets separately.

Do not use MD-3k to claim full dense metric multi-layer depth recovery unless additional dense ground truth and evaluation are provided.

## Citation

```bibtex
@inproceedings{xu2026onescenetwodepths,
  title={One Scene, Two Depths: Probing Geometric Ambiguity in Monocular Foundation Models},
  author={Xu, Xiaohao and Xue, Feng and Li, Xiang and Li, Haowei and Yang, Shusheng and Zhang, Tianyi and Johnson-Roberson, Matthew and Huang, Xiaonan},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```
