import cv2
import argparse
import os
import shutil
import random
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image file: {image_path}")
    print(f"\nImage loaded: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")
    return image

def prepare_dataset_split(raw_images_dir, raw_labels_dir, output_dir, train_split=0.8):
    
    raw_images_dir = Path(raw_images_dir)
    raw_labels_dir = Path(raw_labels_dir)
    output_dir = Path(output_dir)   
    train_img_dir = output_dir / 'images' / 'train'
    val_img_dir = output_dir / 'images' / 'val'
    train_lbl_dir = output_dir / 'labels' / 'train'
    val_lbl_dir = output_dir / 'labels' / 'val'   
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    images = sorted(
        [p for p in raw_images_dir.iterdir()
         if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    )
    print(f"Found {len(images)} images to split")
    
    random.shuffle(images)
    split_idx = int(len(images) * train_split)
    train_images = images[:split_idx]
    val_images = images[split_idx:]  
    print(f"  Train: {len(train_images)} images")
    print(f"  Val: {len(val_images)} images")
    
    for img in train_images:
        shutil.copy(img, train_img_dir / img.name)
        lbl = raw_labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy(lbl, train_lbl_dir / lbl.name)
    
    for img in val_images:
        shutil.copy(img, val_img_dir / img.name)
        lbl = raw_labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy(lbl, val_lbl_dir / lbl.name)
    
    print(f"Dataset split complete at: {output_dir}")
    return str(output_dir)

def build_data_yaml(data_dir, classes, output_path=None):
    data_dir = Path(data_dir).resolve()
    if output_path is None:
        output_path = data_dir / "data.yaml"
    else:
        output_path = Path(output_path).resolve()

    train_images = data_dir / "images" / "train"
    val_images = data_dir / "images" / "val"
    train_labels = data_dir / "labels" / "train"
    val_labels = data_dir / "labels" / "val"

    if not train_images.exists() or not val_images.exists():
        raise FileNotFoundError(
            f"Expected training and validation image folders under {data_dir / 'images'}"
        )
    if not train_labels.exists() or not val_labels.exists():
        raise FileNotFoundError(
            f"Expected training and validation label folders under {data_dir / 'labels'}"
        )

    yaml_content = (
        f"path: {data_dir}\n"
        f"train: {train_images.relative_to(data_dir)}\n"
        f"val: {val_images.relative_to(data_dir)}\n"
        f"nc: {len(classes)}\n"
        "names: ["
        + ", ".join(f'"{c}"' for c in classes)
        + "]\n"
    )
    output_path.write_text(yaml_content, encoding="utf-8")
    print(f"Created dataset YAML at: {output_path}")
    return str(output_path)


def train_yolo(data_dir, model_name="yolov8n.pt", epochs=50, imgsz=640, classes=None):
    
    if classes is None:
        classes = ["triangular", "rectangular", "polygon"]
    yaml_path = build_data_yaml(data_dir, classes)
    print(f"Starting YOLOv8 training with base model: {model_name}")
    model = YOLO(model_name)
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        project="runs/train",
        name="custom_shape_detection",
        exist_ok=True,
    )
    print("Training completed. Use the generated best.pt file for inference.")
    return model

def autolabel_range(images_dir, labels_dir, start=None, end=None, min_area=200):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    #imgs = sorted(images_dir.glob('*.jpg'))
    imgs = sorted(
      p for p in images_dir.iterdir()
      if p.suffix.lower() in ('.jpg', '.jpeg', '.png')
    )
    processed = 0
    for img in imgs:
        try:
            stem = img.stem
            num = int(stem)
        except Exception:
            continue
        if start is not None and num < start:  # only process images within the specified range
            continue
        if end is not None and num > end:
            continue
        # read grayscale, threshold bright shapes
        gray = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Could not read image: {img}")
            continue
        h, w = gray.shape[:2]
        _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lines = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            verts = len(approx)
            if verts == 3:
                cls = 0
            elif verts == 4:
                cls = 1
            else:
                cls = 2
            x, y, bw, bh = cv2.boundingRect(cnt)
            x_c = (x + bw / 2) / w
            y_c = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {nw:.6f} {nh:.6f}\n")

        (labels_dir / f"{stem}.txt").write_text(''.join(lines), encoding='utf-8')
        print(f"Wrote {stem}.txt -> {len(lines)} objects")
        processed += 1

    print(f"Autolabel complete. Processed {processed} images.")
    return processed
# ------------------------------------
# YOLOV8 Inference and Detection
# 0 = triangular, 1 = rectangular, 2 = polygon
# -------------------------------------

def run_yolo(model, image_path, conf=0.25, imgsz=640, iou=0.5):

    results = model.predict(
        source=image_path,
        conf=conf,
        imgsz=imgsz,
        iou=iou,
        verbose=False
    )
    #results = model(image_path, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id    = int(box.cls[0])
            cls_name  = model.names[cls_id]
            score     = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((cls_name, (x1, y1, x2, y2), score))
    return detections
# ---------------------------------------------------------
# DRAW DETECTIONS AND COUNT
# Colors: green = triangular, red = rectangular, blue = polygon
# ---------------------------------------------------------
def process_and_display(image, detections, output_path=None):
    output = image.copy()
    triangles  = 0
    rectangles = 0
    polygons   = 0
    for cls_name, (x1, y1, x2, y2), conf in detections:
        if cls_name == "triangular":
            color = (0, 255, 0)
            triangles += 1
        elif cls_name == "rectangular":
            color = (0, 0, 255)
            rectangles += 1
        else:
            color = (255, 0, 0)
            polygons += 1
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        # label each box with class name + confidence score
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ly = max(y1, th + 4)
        cv2.rectangle(output, (x1, ly - th - 4), (x1 + tw, ly), color, -1)
        cv2.putText(output, label, (x1, ly - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    # ----------------------------------------------
                  # Terminal Output
    # ----------------------------------------------
    total = triangles + rectangles + polygons
    print("\n" + "=" * 52)
    print("  DETECTION RESULTS (YOLOv8 nano)")
    print("=" * 52)
    if triangles == 0:
        print("There isn't any triangular object in the image.")
    elif triangles == 1:
        print("There is 1 triangular object in the image.")
    else:
        print(f"There are {triangles} triangular objects in the image.")
    if rectangles == 0:
        print("There isn't any rectangular object in the image.")
    elif rectangles == 1:
        print("There is 1 rectangular object in the image.")
    else:
        print(f"There are {rectangles} rectangular objects in the image.")
    if polygons == 0:
        print("There isn't any object with more than four edges in the image.")
    elif polygons == 1:
        print("There is 1 object with more than four edges in the image.")
    else:
        print(f"There are {polygons} objects with more than four edges in the image.")
    if total == 1:
        print(f"\nI am glad to let you know that there is 1 object in this image.")
    else:
        print(f"\nI am glad to let you know that there are {total} objects in this image.")
    print("=" * 52)

    if output_path:
        cv2.imwrite(output_path, output)
        print(f"\nOutput saved to: {output_path}")

    return output, triangles, rectangles, polygons

def load_yolo_labels(txt_path):
    txt_path = Path(txt_path)
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        cls_id, xc, yc, w, h = line.split()
        boxes.append((int(float(cls_id)), float(xc), float(yc), float(w), float(h)))
    return boxes
 
 
def yolo_to_xyxy(box, img_w, img_h):
    cls_id, xc, yc, w, h = box
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return cls_id, (x1, y1, x2, y2)
 
 
def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0
 
 
CLASS_NAME_TO_ID = {"triangular": 0, "rectangular": 1, "polygon": 2}
 
 
def compute_tp_fp_fn(detections, gt_boxes, img_w, img_h, iou_thresh=0.5):
    gt_xyxy = [yolo_to_xyxy(b, img_w, img_h) for b in gt_boxes]
    matched_gt = set()
    tp = 0
    fp = 0
    for cls_name, pred_box, conf in detections:
        pred_cls_id = CLASS_NAME_TO_ID.get(cls_name, -1)
        best_iou, best_idx = 0.0, -1
        for idx, (gt_cls_id, gt_box) in enumerate(gt_xyxy):
            if idx in matched_gt or gt_cls_id != pred_cls_id:
                continue
            i = iou(pred_box, gt_box)
            if i > best_iou:
                best_iou, best_idx = i, idx
        if best_idx != -1 and best_iou >= iou_thresh:
            matched_gt.add(best_idx)
            tp += 1
        else:
            fp += 1
 
    fn = len(gt_xyxy) - len(matched_gt)
    return tp, fp, fn
 
def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

# Main code part
def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 nano shape detection"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="my_image.jpg",
        help="Path to input image"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base model for training or trained weights for inference"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output image to file"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a new YOLOv8 model from a custom dataset"
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Fine-tune YOLOv8 on shapes_dataset (00000-00817) with automatic train/val split"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset root containing images/train, images/val, labels/train, labels/val"
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Train/val split ratio for --finetune (default 0.8 = 80%% train, 20%% val)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["triangular", "rectangular", "polygon"],
        help="Class names for training"
    )
    parser.add_argument(
        "--autolabel",
        action="store_true",
        help="Auto-generate YOLO labels for a numeric image range"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index (inclusive) for autolabeling, e.g. 800"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (inclusive) for autolabeling, e.g. 817"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence threshold for YOLO detection"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for Non-Maximum Suppression"
   )
    parser.add_argument(
        "--gt", 
        type=str, 
        default=None,
        help="Path to ground-truth YOLO-format .txt file for this image")
    args = parser.parse_args()
    print("\n" + "=" * 52)
    print("  YOLOv8 Nano — Shape Detection")
    print("=" * 52)
    if args.train:
        if not args.data:
            raise ValueError("Please provide --data with the dataset root folder")
        train_yolo(
            data_dir=args.data,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            classes=args.classes,
        )
        return
    if args.finetune:
        data_root = Path("shapes_dataset")
        raw_images = data_root / 'train' / 'images'
        raw_labels = data_root / 'train' / 'labels'
        
        if not raw_images.exists() or not raw_labels.exists():
            raise FileNotFoundError(
                f"Expected {raw_images} and {raw_labels} to exist.\n"
                f"Please ensure your dataset has the structure:\n"
                f"  shapes_dataset/train/images/ (00000.jpg - 00817.jpg)\n"
                f"  shapes_dataset/train/labels/ (00000.txt - 00817.txt)"
            )
        
        print("\nPreparing dataset with train/val split...")
        split_root = prepare_dataset_split(
            raw_images_dir=raw_images,
            raw_labels_dir=raw_labels,
            output_dir=data_root / 'split',
            train_split=args.split
        )
        
        print("\nStarting fine-tuning on shapes dataset (YOLOv8 Nano)...")
        train_yolo(
            data_dir=split_root,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            classes=args.classes,
        )
        print("\nFine-tuning complete! Trained model saved in runs/train/custom_shape_detection/")
        return

    if args.autolabel:
        if not args.data:
            raise ValueError("Please provide --data pointing to your dataset root (shapes_dataset)")
        images_dir = Path(args.data) / 'images' / 'train'
        labels_dir = Path(args.data) / 'labels' / 'train'
        autolabel_range(images_dir, labels_dir, start=args.start, end=args.end)
        return
    # determine output path
    output_path = None
    if args.save:
        name = os.path.splitext(args.image)[0]
        output_path = f"{name}_yolo_output.jpg"
    
    # image acquisition
    image_path = args.image
    print("\nLoading image...")
    try:
        image = load_image(image_path)
        h, w = image.shape[:2] 
    except FileNotFoundError as exc:
        print(f"\nError: {exc}")
        return
    # load model
    print(f"Loading YOLOv8 nano model from: {args.model}")
    model = YOLO(args.model)
    print("Running YOLOv8 inference...")
    detections = run_yolo( model,image_path,conf=args.conf,imgsz=args.imgsz,iou=args.iou)
    print(f"        {len(detections)} detections found")
    for cls_name, (x1, y1, x2, y2), score in detections:
        print(f"          - {cls_name:<12} conf={score:.2f}  box=({x1},{y1},{x2},{y2})")

    if args.gt:
        h, w = image.shape[:2]
        gt_boxes = load_yolo_labels(args.gt)
        tp, fp, fn = compute_tp_fp_fn(detections, gt_boxes, img_w=w, img_h=h, iou_thresh=0.5)
        metrics = compute_metrics(tp, fp, fn)
        print("\n" + "=" * 52)
        print("  EVALUATION METRICS")
        print("=" * 52)
        print(f"TP={tp}  FP={fp}  FN={fn}")
        print(f"Accuracy:  {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall:    {metrics['recall']:.3f}")
        print(f"F1 Score:  {metrics['f1_score']:.3f}")
    # display results in a resizable window
    output, tri, rect, poly = process_and_display(image, detections, output_path)
    win = "Output - YOLOv8 Detection"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    h, w = output.shape[:2]
    max_w, max_h = 1280, 900
    scale = min(max_w / w, max_h / h, 1.0)
    cv2.resizeWindow(win, int(w * scale), int(h * scale))
    cv2.imshow(win, output)
    print("\nDrag the window edges to zoom. Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
