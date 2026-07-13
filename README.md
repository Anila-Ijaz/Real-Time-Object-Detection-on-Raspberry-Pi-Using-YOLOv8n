# Real-Time Object Detection on Raspberry Pi Using YOLOv8n

Lightweight real-time shape detection on Raspberry Pi. The project starts from a **classical OpenCV computer-vision baseline** (edge detection + contour analysis) that re-implements, and extends it into a **learning-based YOLOv8-nano pipeline** that covers dataset generation, auto-labeling, training/fine-tuning, inference, and quantitative evaluation.

The task in both pipelines is the same: **detect geometric objects in an image and classify each one as `triangular`, `rectangular`, or `polygon`** (any shape with more than four edges).

---

## Table of Contents

- [Motivation](#motivation)
- [Repository Structure](#repository-structure)
- [The Two Pipelines](#the-two-pipelines)
  - [1. Baseline — Classical OpenCV Pipeline](#1-baseline--classical-opencv-pipeline)
  - [2. Extension — YOLOv8n Pipeline](#2-extension--yolov8n-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Methodology](#evaluation-methodology)
  - [Confusion Matrix](#confusion-matrix)
  - [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Baseline vs. YOLOv8n — Comparison](#baseline-vs-yolov8n--comparison)
- [Class Encoding](#class-encoding)
- [References](#references)

---

## Motivation

Classical detection pipelines (Canny edges → morphology → contour approximation) are fast and require **no training data**, but they are brittle: they depend on hand-tuned thresholds, assume clean high-contrast scenes, and cannot report a confidence score or recover from occlusion, noise, or overlapping objects.

The extension replaces the hand-crafted decision logic with a **YOLOv8-nano** detector — the smallest model in the YOLOv8 family — chosen because it is compact enough to run in real time on a Raspberry Pi while still learning shape appearance directly from data. The classical pipeline is not discarded: it is reused to **auto-label** a training set, bootstrapping the learned model without manual annotation.

---

## Repository Structure

```
.
├── Baseline_pipeline.py    # Classical OpenCV shape-detection pipeline
├── Yolo_extension.py       # YOLOv8n dataset prep, training, inference & evaluation
└── README.md
```

---

## The Two Pipelines

### 1. Baseline — Classical OpenCV Pipeline

[`Baseline_pipeline.py`](Baseline_pipeline.py) is a deterministic image-processing chain with no learned parameters:

| Step | Operation | Function |
|------|-----------|----------|
| 1 | Image acquisition (`raspistill`) or load from disk | `capture_image`, `load_image` |
| 2 | Grayscale conversion | `convert_to_grayscale` |
| 3 | Gaussian blur (3×3, σ=0) | `apply_gaussian_blur` |
| 4 | Canny edge detection (min=10, max=250) | `apply_canny_edge_detection` |
| 5 | Morphological closing (7×7 kernel) | `apply_morphological_operations` |
| 6 | External contour detection | `detect_contours` |
| 7 | Shape classification by vertex count | `classify_shapes` |
| 8 | Draw, count, and report | `draw_contours_and_classify` |

Classification is purely geometric: the contour is approximated with `approxPolyDP`, and the vertex count decides the class (3 → triangular, 4 → rectangular, otherwise → polygon).

**Strengths:** zero training data, very low compute, fully interpretable.
**Weaknesses:** threshold-sensitive, no confidence scores, fails on noise/occlusion/overlap, and over-counts spurious contours.

### 2. Extension — YOLOv8n Pipeline

[`Yolo_extension.py`](Yolo_extension.py) wraps [Ultralytics YOLOv8](https://docs.ultralytics.com/) into an end-to-end workflow:

| Stage | What it does | Function |
|-------|--------------|----------|
| Auto-labeling | Reuses the classical contour logic to generate YOLO-format labels for a numeric image range | `autolabel_range` |
| Dataset split | Copies images/labels into train/val folders (default 80/20) | `prepare_dataset_split` |
| Config | Writes the `data.yaml` describing paths and class names | `build_data_yaml` |
| Training | Fine-tunes `yolov8n.pt` on the shape dataset | `train_yolo` |
| Inference | Runs prediction with configurable conf/IoU thresholds | `run_yolo` |
| Visualization | Draws boxes with class + confidence labels and counts objects | `process_and_display` |
| Evaluation | Matches predictions to ground truth and computes metrics | `compute_tp_fp_fn`, `compute_metrics` |

**Strengths:** learns shape appearance, outputs per-detection confidence, robust to noise and overlap, real-time capable on Pi.
**Weaknesses:** needs labeled data and a training step; inference cost is higher than the baseline.

---

## Installation

```bash
# Python 3.8+ recommended
pip install ultralytics opencv-python numpy
```

- `ultralytics` pulls in PyTorch and the YOLOv8 weights.
- On a Raspberry Pi, image capture uses `raspistill` (part of the legacy camera stack); on other machines, pass an existing image with `--image`.

---

## Usage

### Baseline

```bash
# Run detection on an image
python Baseline_pipeline.py --image my_image.jpg

# Show each processing step and save the annotated output
python Baseline_pipeline.py --image my_image.jpg --show-steps --save
```

### YOLOv8n Extension

```bash
# 1. Auto-label a numeric image range using the classical pipeline
python Yolo_extension.py --autolabel --data shapes_dataset --start 0 --end 817

# 2. Fine-tune YOLOv8n (expects shapes_dataset/train/images + shapes_dataset/train/labels)
python Yolo_extension.py --finetune --epochs 50 --imgsz 640

# 3. Run inference with the trained weights
python Yolo_extension.py --image my_image.jpg --model runs/train/custom_shape_detection/weights/best.pt --save

# 4. Run inference AND evaluate against a ground-truth label file
python Yolo_extension.py --image my_image.jpg \
    --model runs/train/custom_shape_detection/weights/best.pt \
    --gt path/to/my_image.txt
```

Key inference flags: `--conf` (confidence threshold, default 0.25), `--iou` (NMS IoU threshold, default 0.7).

---

## Evaluation Methodology

The YOLO pipeline includes a self-contained evaluation module. When a ground-truth label file is passed with `--gt`, each prediction is matched to a ground-truth box using **Intersection-over-Union (IoU)** with a matching threshold of **0.5**, and only same-class matches are allowed (see `compute_tp_fp_fn`).

### Confusion Matrix

Because matching is one-to-one and class-aware, every prediction and every ground-truth box falls into one of three outcomes. This forms the confusion matrix for detection:

|                       | **Ground Truth: Object present** | **Ground Truth: No object** |
|-----------------------|:--------------------------------:|:---------------------------:|
| **Predicted: Object** | ✅ **True Positive (TP)** — correct class, IoU ≥ 0.5 | ❌ **False Positive (FP)** — spurious / wrong-class / low-IoU box |
| **Predicted: None**   | ❌ **False Negative (FN)** — missed object | ⬜ True Negative (N/A) — not counted in detection |

- **True Positive (TP):** a predicted box whose class matches a still-unmatched ground-truth box with IoU ≥ 0.5.
- **False Positive (FP):** a predicted box that matches no ground-truth box (wrong class, too little overlap, or a duplicate/hallucinated detection).
- **False Negative (FN):** a ground-truth box that no prediction matched (a miss).
- **True Negative (TN):** not meaningful for object detection — the "background" has effectively infinite negative windows — so it is excluded, which is why accuracy below is defined without TN.

For a **per-class (multi-class) confusion matrix**, the same matching is grouped by class label. Off-diagonal cells indicate class confusion — e.g. a `polygon` predicted where the ground truth was `rectangular`:

|  Actual \ Predicted | Triangular | Rectangular | Polygon | Missed (FN) |
|---------------------|:----------:|:-----------:|:-------:|:-----------:|
| **Triangular**      |  TP        | confusion   | confusion | FN        |
| **Rectangular**     | confusion  | TP          | confusion | FN        |
| **Polygon**         | confusion  | confusion   | TP        | FN        |
| **Background (FP)** | FP         | FP          | FP       | —          |

> Ultralytics also generates a normalized confusion matrix automatically during validation (`runs/train/.../confusion_matrix.png`), which can be used to cross-check the values reported by this pipeline.

### Evaluation Metrics

From the TP/FP/FN counts, `compute_metrics` derives the following (all in `Yolo_extension.py`):

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision** | `TP / (TP + FP)` | Of everything the model detected, how much was correct. Penalizes false alarms. |
| **Recall** | `TP / (TP + FN)` | Of everything that was actually there, how much the model found. Penalizes misses. |
| **F1 Score** | `2·P·R / (P + R)` | Harmonic mean of precision and recall — a single balanced score. |
| **Accuracy** | `TP / (TP + FP + FN)` | Detection accuracy excluding true negatives (a.k.a. the Jaccard/critical success index for detection). |

Example console output from `--gt`:

```
====================================================
  EVALUATION METRICS
====================================================
TP=12  FP=2  FN=1
Accuracy:  0.800
Precision: 0.857
Recall:    0.923
F1 Score:  0.889
```

---

## Results

Both pipelines were evaluated on the **same Raspberry Pi hardware** across four scenarios designed to stress inter-object spacing, background contrast, shadow, and camera distance.

> **Research question:** *Can YOLOv8n on the same Raspberry Pi hardware overcome the limitations of inter-object spacing, background contrast, shadow, and camera distance to achieve robust real-time object detection?*

### Per-Scenario Accuracy

| Scenario | Condition | Baseline Pipeline Accuracy | YOLOv8n Accuracy |
|----------|-----------|:--------------------------:|:----------------:|
| **S1** | Single shape | 100 % | 100 % |
| **S2** | Multiple freehand objects | 92.31 % | 94.4 % |
| **S3** | High spacing, sharp contrast | 100 % | 100 % |
| **S4** | Close spacing / low contrast | **58.72 %** | **94.4 %** |
| **Average** | — | **≈ 90.21 %** | **≈ 97.2 %** |

The decisive difference is **S4 (close spacing / low contrast)**: the classical baseline collapses to **58.72 %** because merged edges and weak contrast break its contour separation, while YOLOv8n holds at **94.4 %**. In easy scenarios (S1, S3) both reach 100 %, confirming the extension loses nothing on the cases the baseline already handled. Overall accuracy improves from **≈ 90.21 %** to **≈ 97.2 %**.

### Detection Metrics per Scenario

Across all four scenarios, the YOLOv8n pipeline achieves **Accuracy, Precision, Recall, and F1 scores at or near 1.0**, with the lowest values appearing in S2 and S4 (the multi-object / low-contrast cases) — consistent with the accuracy table above. This confirms the model answers the research question: it stays robust precisely where the baseline degrades.

---

## Baseline vs. YOLOv8n — Comparison

| Aspect | Baseline (Classical OpenCV) | Extension (YOLOv8n) |
|--------|-----------------------------|---------------------|
| **Approach** | Hand-crafted edges + contour geometry | Learned single-stage CNN detector |
| **Training data** | None required | Requires labeled dataset (auto-labeled here) |
| **Localization output** | Contour polygons | Axis-aligned bounding boxes |
| **Confidence scores** | ❌ None | ✅ Per-detection confidence |
| **Robustness to noise/occlusion/overlap** | Low — threshold-sensitive | High — learned features |
| **Parameters to tune** | Canny thresholds, blur, kernel, `epsilon` | conf, IoU, epochs, imgsz (learned weights otherwise) |
| **Interpretability** | Fully transparent | Black-box (mitigated by confidence + confusion matrix) |
| **Compute cost** | Very low | Higher, but real-time on Pi with the nano model |
| **Failure mode** | Over-/under-counts on messy scenes | Needs representative training data to generalize |
| **Evaluation support** | Object counts only | TP/FP/FN, precision, recall, F1, accuracy, confusion matrix |

**Summary.** The baseline is an excellent zero-data, low-power starting point and is genuinely useful for bootstrapping labels. The YOLOv8n extension trades a one-time training cost for a detector that is measurably more robust — raising average accuracy from **≈ 90.21 %** to **≈ 97.2 %**, and from **58.72 %** to **94.4 %** on the hardest scenario (close spacing / low contrast) — while producing confidence-scored boxes that can be **quantitatively evaluated** with a confusion matrix and standard detection metrics rather than judged only by eye.

---

## Class Encoding

| ID | Class | Vertices | Box / Contour Color (BGR) |
|----|-------|----------|---------------------------|
| 0 | `triangular` | 3 | Green `(0, 255, 0)` |
| 1 | `rectangular` | 4 | Red `(0, 0, 255)` |
| 2 | `polygon` | > 4 | Blue `(255, 0, 0)` |

---

## References

- Abdulhamid, M., Odondi, N., & Al-Rawi, M. (2020). *Comparative study of image processing techniques for object detection* (baseline re-implemented in [`Baseline_pipeline.py`](Baseline_pipeline.py)).
- Jocher, G. et al. *Ultralytics YOLOv8*. https://docs.ultralytics.com/
