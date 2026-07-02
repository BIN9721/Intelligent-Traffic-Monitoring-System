"""
AIC21 Vehicle Counting -> YOLO Auto-Labeler
============================================
Auto-labels 31 MP4 videos from the AIC21 Vehicle Counting dataset
using YOLO11x (pretrained on COCO) to produce YOLO TXT annotations.

The AIC21 counting GT only has vehicle-count lines (no bounding boxes),
so we treat this dataset the same as the Vietnam videos: run inference
and use the predictions as pseudo-ground-truth labels.

Class mapping (COCO -> Custom 4-class):
  COCO 2 (car)        -> 0  car
  COCO 3 (motorcycle) -> 1  motorcycle
  COCO 5 (bus)        -> 2  bus
  COCO 7 (truck)      -> 3  truck

Split: videos are shuffled (seed=42) then split 70/15/15 at video level.
Output: aic21_dataset/{train,val,test}/{images,labels}/
"""

import cv2
import random
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

# ──────────────────────────── CONFIG ────────────────────────────
PROJECT_ROOT = Path("/home/hoangthang/Intelligent Traffic Monitoring System")
VIDEO_DIR    = PROJECT_ROOT / "Datasets" / "Vehicle Counting" / "Dataset_A"
OUTPUT_DIR   = PROJECT_ROOT / "aic21_dataset"
MODEL_PATH   = PROJECT_ROOT / "yolo11x.pt"

FRAME_INTERVAL_SEC = 1.0   # 1 frame per second
MSE_THRESHOLD      = 10.0  # skip near-duplicate frames
CONF_THRESH        = 0.15
IOU_THRESH         = 0.45
IMG_SIZE           = 1280
RANDOM_SEED        = 42

COCO_TO_CUSTOM = {2: 0, 3: 1, 5: 2, 7: 3}
CLASS_NAMES    = ["car", "motorcycle", "bus", "truck"]
COLORS         = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # BGR

# ──────────────────────────── HELPERS ───────────────────────────
def draw_sample_images(output_dir: Path, split="train", n=5):
    """Vẽ bounding box lên n ảnh mẫu ngẫu nhiên và lưu vào visual_results/."""
    visual_dir = output_dir / "visual_results"
    visual_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / split / "images"
    lbl_dir = output_dir / split / "labels"

    img_files = sorted(img_dir.glob("*.jpg"))
    if not img_files:
        print("  [WARN] Không có ảnh để visualize.")
        return
    random.shuffle(img_files)

    for img_path in img_files[:n]:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                color = COLORS[cls_id % len(COLORS)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = CLASS_NAMES[cls_id]
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        out_path = visual_dir / f"visualized_{img_path.name}"
        cv2.imwrite(str(out_path), img)
        print(f"  [VIS] {out_path.name}")


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))

def downscaled_gray(frame: np.ndarray, size=(32, 32)) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

# ──────────────────────────── MAIN ──────────────────────────────
def main():
    print("=" * 60)
    print("AIC21 VEHICLE COUNTING — AUTO LABELER")
    print("=" * 60)

    # Gather videos
    video_exts = {".mp4", ".avi", ".mkv", ".MOV"}
    videos = sorted([p for p in VIDEO_DIR.iterdir() if p.suffix in video_exts])
    print(f"Found {len(videos)} videos in {VIDEO_DIR}")
    if not videos:
        print("No videos found. Exiting.")
        return

    # 70 / 15 / 15 split at video level
    random.seed(RANDOM_SEED)
    shuffled = videos.copy()
    random.shuffle(shuffled)
    n         = len(shuffled)
    train_end = int(n * 0.70)
    val_end   = train_end + int(n * 0.15)
    splits = {}
    for i, v in enumerate(shuffled):
        splits[v] = "train" if i < train_end else ("val" if i < val_end else "test")

    for split in ["train", "val", "test"]:
        cnt = sum(1 for s in splits.values() if s == split)
        print(f"  {split}: {cnt} videos")

    # Create output dirs
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading YOLO11x from {MODEL_PATH} on {device} ...")
    model = YOLO(str(MODEL_PATH))

    stats = {s: {"frames": 0, "boxes": [0, 0, 0, 0]} for s in ["train", "val", "test"]}
    total_extracted = 0
    total_skipped   = 0

    for v_idx, video_path in enumerate(shuffled):
        split = splits[video_path]
        print(f"\n[{v_idx+1}/{n}] {video_path.name} -> {split}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [WARN] Cannot open {video_path.name}")
            continue

        fps         = cap.get(cv2.CAP_PROP_FPS) or 1
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step  = max(1, int(fps * FRAME_INTERVAL_SEC))
        print(f"  FPS={fps:.1f}  Total={frame_total}  Step={frame_step}")

        prev_gray  = None
        frame_idx  = 0
        saved      = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                cur_gray = downscaled_gray(frame)

                # Duplicate-frame detection
                if prev_gray is not None and mse(cur_gray, prev_gray) < MSE_THRESHOLD:
                    total_skipped += 1
                    frame_idx += 1
                    continue

                prev_gray = cur_gray

                stem = f"{video_path.stem}_f{frame_idx:06d}"
                img_path = OUTPUT_DIR / split / "images" / f"{stem}.jpg"
                lbl_path = OUTPUT_DIR / split / "labels" / f"{stem}.txt"

                results = model(
                    frame, verbose=False, device=device,
                    imgsz=IMG_SIZE, conf=CONF_THRESH,
                    iou=IOU_THRESH, agnostic_nms=True,
                )
                valid_boxes = []
                for box in results[0].boxes:
                    cid = int(box.cls[0])
                    if cid in COCO_TO_CUSTOM and float(box.conf[0]) >= CONF_THRESH:
                        custom = COCO_TO_CUSTOM[cid]
                        xywhn  = box.xywhn[0].tolist()
                        valid_boxes.append((custom, xywhn))

                if valid_boxes:
                    cv2.imwrite(str(img_path), frame)
                    with open(lbl_path, "w") as f:
                        for cls_id, (cx, cy, w, h) in valid_boxes:
                            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                            stats[split]["boxes"][cls_id] += 1
                    stats[split]["frames"] += 1
                    total_extracted += 1
                    saved += 1

            frame_idx += 1

        cap.release()
        print(f"  Saved {saved} frames from {video_path.name}")

    # Write YAML
    yaml_content = f"""path: {OUTPUT_DIR}
train: train/images
val: val/images
test: test/images

names:
  0: car
  1: motorcycle
  2: bus
  3: truck
"""
    yaml_path = OUTPUT_DIR / "aic21_data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\nCreated {yaml_path}")

    # Print stats
    total_frames = sum(s["frames"] for s in stats.values())
    total_boxes  = [sum(s["boxes"][c] for s in stats.values()) for c in range(4)]

    print("\n" + "=" * 60)
    print("AIC21 DATASET GENERATION STATISTICS")
    print("=" * 60)
    print(f"Total Frames Extracted  : {total_frames:,}")
    print(f"Total Bounding Boxes    : {sum(total_boxes):,}")
    print(f"Duplicate Frames Skipped: {total_skipped:,}")
    print("-" * 60)
    for split in ["train", "val", "test"]:
        s = stats[split]
        print(f"Split: {split.upper()}")
        print(f"  Frames : {s['frames']:,}")
        for c, name in enumerate(CLASS_NAMES):
            if s["boxes"][c] > 0:
                print(f"  {name:<12}: {s['boxes'][c]:,}")
        print("-" * 60)
    print("=" * 60)

    # ── Visualize mẫu ──
    print("\n[Generating sample visualizations...]")
    draw_sample_images(OUTPUT_DIR, "train", n=5)
    print("Done.")


if __name__ == "__main__":
    main()
