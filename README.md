# Real-Time Object Detection on Raspberry Pi Using YOLOv8n

Shape detection on Raspberry Pi, built in two stages: a **classical OpenCV baseline** (Canny edges + contour geometry) and a **YOLOv8-nano extension** covering auto-labeling, training, inference, and evaluation.

Both pipelines solve the same task — detect objects in an image and classify each as `triangular`, `rectangular`, or `polygon` (more than four edges).

**Research question:** Can YOLOv8n on the same Raspberry Pi hardware overcome the limitations of inter-object spacing, background contrast, shadow, and camera distance to achieve robust real-time detection?

**Answer:** Yes — average accuracy rises from **90.21 %** to **97.2 %**, and from **58.72 %** to **94.4 %** on the hardest scenario.

---

## Results

Evaluated on the same Raspberry Pi hardware across four scenarios:

| Scenario | Condition | Baseline | YOLOv8n |
|----------|-----------|:--------:|:-------:|
| S1 | Single shape | 100 % | 100 % |
| S2 | Multiple freehand objects | 92.31 % | 94.4 % |
| S3 | High spacing, sharp contrast | 100 % | 100 % |
| S4 | Close spacing / low contrast | **58.72 %** | **94.4 %** |
| **Average** | — | **≈ 90.21 %** | **≈ 97.2 %** |

S1 and S3 are ties — the extension gives up nothing on cases the baseline already handled. The gap opens at **S4**, where merged edges and weak contrast break the baseline's contour separation (58.72 %) while YOLOv8n holds at 94.4 %. Precision, recall, and F1 stay at or near 1.0 across all four scenarios, with the lowest values in S2 and S4.

---

## Pipelines

### Baseline — [`Baseline_pipeline.py`](Baseline_pipeline.py)

Deterministic, no learned parameters:

`load_image` → `convert_to_grayscale` → `apply_gaussian_blur` (3×3) → `apply_canny_edge_detection` (10/250) → `apply_morphological_operations` (7×7 close) → `detect_contours` → `classify_shapes` → `draw_contours_and_classify`

Classification is purely geometric: `approxPolyDP` reduces each contour to vertices — 3 → triangular, 4 → rectangular, else → polygon.

### Extension — [`Yolo_extension.py`](Yolo_extension.py)

Wraps [Ultralytics YOLOv8](https://docs.ultralytics.com/) end to end:

| Stage | Function |
|-------|----------|
| Auto-label images by reusing the classical contour logic | `autolabel_range` |
| Split into train/val (default 80/20) | `prepare_dataset_split` |
| Write `data.yaml` | `build_data_yaml` |
| Fine-tune `yolov8n.pt` | `train_yolo` |
| Inference with conf/IoU thresholds | `run_yolo` |
| Draw boxes, count objects | `process_and_display` |
| Match to ground truth, compute metrics | `compute_tp_fp_fn`, `compute_metrics` |

The baseline isn't discarded — it bootstraps the training labels, so no manual annotation was needed.

---

## Setup

```bash
pip install ultralytics opencv-python numpy    # Python 3.8+
```

On Raspberry Pi, capture uses `raspistill`; elsewhere pass an existing image with `--image`.

---

## Usage

```bash
# Baseline
python Baseline_pipeline.py --image my_image.jpg [--show-steps] [--save]

# 1. Auto-label a numeric image range
python Yolo_extension.py --autolabel --data shapes_dataset --start 0 --end 817

# 2. Fine-tune YOLOv8n
python Yolo_extension.py --finetune --epochs 50 --imgsz 640

# 3. Inference
python Yolo_extension.py --image my_image.jpg \
    --model runs/train/custom_shape_detection/weights/best.pt --save

# 4. Inference + evaluation against ground truth
python Yolo_extension.py --image my_image.jpg \
    --model runs/train/custom_shape_detection/weights/best.pt --gt labels/my_image.txt
```

Inference flags: `--conf` (default 0.25), `--iou` (NMS, default 0.7).

---

## Evaluation

Predictions are matched to ground truth by **IoU ≥ 0.5**, one-to-one and class-aware (`compute_tp_fp_fn`).

### Confusion Matrix

|  | **GT: object present** | **GT: no object** |
|---|:---:|:---:|
| **Predicted: object** | **TP** — correct class, IoU ≥ 0.5 | **FP** — spurious, wrong class, or low IoU |
| **Predicted: none** | **FN** — missed object | TN — not counted |

TN is excluded: background contains effectively infinite negative windows, which is why accuracy below omits it.

Grouping the same matches by class gives the per-class matrix, where off-diagonal cells are class confusion (e.g. `polygon` predicted where truth was `rectangular`):

| Actual \ Predicted | Triangular | Rectangular | Polygon | Missed |
|---|:---:|:---:|:---:|:---:|
| **Triangular** | TP | confusion | confusion | FN |
| **Rectangular** | confusion | TP | confusion | FN |
| **Polygon** | confusion | confusion | TP | FN |
| **Background** | FP | FP | FP | — |

Ultralytics also writes a normalized confusion matrix during validation (`runs/train/.../confusion_matrix.png`) for cross-checking.

### Metrics

`compute_metrics` derives:

| Metric | Formula | Penalizes |
|--------|---------|-----------|
| Precision | `TP / (TP + FP)` | False alarms |
| Recall | `TP / (TP + FN)` | Misses |
| F1 | `2·P·R / (P + R)` | Imbalance between the two |
| Accuracy | `TP / (TP + FP + FN)` | Both (TN excluded) |

Output from `--gt`:

```
TP=12  FP=2  FN=1
Accuracy:  0.800
Precision: 0.857
Recall:    0.923
F1 Score:  0.889
```

---

## Baseline vs. YOLOv8n

| Aspect | Baseline | YOLOv8n |
|--------|----------|---------|
| Approach | Hand-crafted edges + contour geometry | Learned single-stage CNN |
| Training data | None | Required (auto-labeled here) |
| Output | Contour polygons | Boxes + confidence scores |
| Low contrast / close spacing | Fails (58.72 %) | Robust (94.4 %) |
| Tuning | Canny thresholds, blur, kernel, epsilon | conf, IoU, epochs, imgsz |
| Interpretability | Fully transparent | Black-box, offset by confidence scores |
| Compute | Very low | Higher, still real-time on Pi |
| Evaluation | Object counts only | TP/FP/FN, P/R/F1, confusion matrix |

The baseline remains a good zero-data, low-power starting point and is genuinely useful for bootstrapping labels. The extension trades a one-time training cost for measurably better robustness and results that can be quantitatively evaluated rather than judged by eye.

---

## Class Encoding

| ID | Class | Vertices | Color (BGR) |
|----|-------|----------|-------------|
| 0 | `triangular` | 3 | Green `(0, 255, 0)` |
| 1 | `rectangular` | 4 | Red `(0, 0, 255)` |
| 2 | `polygon` | > 4 | Blue `(255, 0, 0)` |

---

## Future Work

- Explore newer YOLO versions (v9/v10/v11) for higher accuracy and improved metrics.
- Add real-time live detection on the Raspberry Pi — streaming camera inference rather than single-image capture.
- Train on more 3D images for robust 3D detection under varied viewpoints and lighting.

---

## References

- Abdulhamid, M., Odondi, N., & Al-Rawi, M. (2020). *Comparative study of image processing techniques for object detection* — baseline re-implemented in [`Baseline_pipeline.py`](Baseline_pipeline.py).
- Jocher, G. et al. *Ultralytics YOLOv8*. https://docs.ultralytics.com/
