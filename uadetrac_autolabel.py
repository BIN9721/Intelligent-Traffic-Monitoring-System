"""
UA-DETRAC Auto-Labeler (YOLO11x)
=================================
Thay thế GT XML bằng YOLO11x pseudo-labels để đảm bảo:
  - 4 class đầy đủ (car, motorcycle, bus, truck)
  - Coverage toàn frame, không bỏ sót xe rìa/khuất
  - Pipeline nhất quán với Vietnam dataset

Nguồn ảnh: uadetrac_dataset/{train,val,test}/images/  (đã extract sẵn)
Ghi đè:    uadetrac_dataset/{train,val,test}/labels/  (thay GT cũ)

Class mapping (COCO -> Custom):
  2 (car)        -> 0
  3 (motorcycle) -> 1
  5 (bus)        -> 2
  7 (truck)      -> 3

Frame sampling: dùng ảnh đã có sẵn (mỗi 5 frame annotated từ bước trước),
không cần trích thêm. Chỉ chạy inference và ghi lại label mới.
"""

import cv2
import random
import torch
from pathlib import Path
from ultralytics import YOLO

# ──────────────────────────── CONFIG ────────────────────────────
PROJECT_ROOT = Path("/home/hoangthang/Intelligent Traffic Monitoring System")
DATASET_DIR  = PROJECT_ROOT / "uadetrac_dataset"
MODEL_PATH   = PROJECT_ROOT / "yolo11x.pt"

CONF_THRESH = 0.15
IOU_THRESH  = 0.45
IMG_SIZE    = 1280

COCO_TO_CUSTOM = {2: 0, 3: 1, 5: 2, 7: 3}
CLASS_NAMES    = ["car", "motorcycle", "bus", "truck"]
COLORS         = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # BGR
SPLITS         = ["train", "val", "test"]

# ──────────────────────────── HELPERS ───────────────────────────
def draw_sample_images(split="train", n=5):
    """Vẽ bounding box lên n ảnh mẫu ngẫu nhiên và lưu vào visual_results/."""
    visual_dir = DATASET_DIR / "visual_results"
    visual_dir.mkdir(parents=True, exist_ok=True)
    img_dir = DATASET_DIR / split / "images"
    lbl_dir = DATASET_DIR / split / "labels"

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


# ──────────────────────────── MAIN ──────────────────────────────
def main():
    print("=" * 60)
    print("UA-DETRAC — YOLO11x AUTO LABELER (overwrite GT labels)")
    print("=" * 60)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading YOLO11x from {MODEL_PATH} on {device} ...")
    model = YOLO(str(MODEL_PATH))

    stats = {s: {"frames_kept": 0, "frames_empty": 0, "boxes": [0, 0, 0, 0]}
             for s in SPLITS}

    for split in SPLITS:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"
        lbl_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted(img_dir.glob("*.jpg"))
        total = len(img_files)
        print(f"\n[{split.upper()}] {total} images")

        for i, img_path in enumerate(img_files):
            if (i + 1) % 500 == 0 or i == 0:
                print(f"  [{i+1}/{total}] {img_path.name}")

            results = model(
                str(img_path), verbose=False, device=device,
                imgsz=IMG_SIZE, conf=CONF_THRESH,
                iou=IOU_THRESH, agnostic_nms=True,
            )

            valid_boxes = []
            for box in results[0].boxes:
                cid = int(box.cls[0])
                if cid in COCO_TO_CUSTOM and float(box.conf[0]) >= CONF_THRESH:
                    custom = COCO_TO_CUSTOM[cid]
                    cx, cy, w, h = box.xywhn[0].tolist()
                    valid_boxes.append((custom, cx, cy, w, h))

            lbl_path = lbl_dir / f"{img_path.stem}.txt"

            if valid_boxes:
                with open(lbl_path, "w") as f:
                    for cls_id, cx, cy, w, h in valid_boxes:
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                        stats[split]["boxes"][cls_id] += 1
                stats[split]["frames_kept"] += 1
            else:
                # Ghi file rỗng để tránh lỗi khi training
                open(lbl_path, "w").close()
                stats[split]["frames_empty"] += 1

    # ── In thống kê ──
    total_frames = sum(s["frames_kept"] for s in stats.values())
    total_boxes  = [sum(s["boxes"][c] for s in stats.values()) for c in range(4)]

    print("\n" + "=" * 60)
    print("UA-DETRAC AUTO-LABEL STATISTICS")
    print("=" * 60)
    print(f"Total Frames Labeled    : {total_frames:,}")
    print(f"Total Bounding Boxes    : {sum(total_boxes):,}")
    print("-" * 60)
    for split in SPLITS:
        s = stats[split]
        print(f"Split: {split.upper()}")
        print(f"  Frames with labels : {s['frames_kept']:,}")
        print(f"  Frames empty       : {s['frames_empty']:,}")
        for c, name in enumerate(CLASS_NAMES):
            if s["boxes"][c] > 0:
                print(f"  {name:<12}: {s['boxes'][c]:,}")
        print("-" * 60)
    print(f"\nClass totals:")
    for c, name in enumerate(CLASS_NAMES):
        pct = total_boxes[c] / max(1, sum(total_boxes)) * 100
        print(f"  {name:<12}: {total_boxes[c]:,}  ({pct:.1f}%)")
    print("=" * 60)

    # ── Visualize mẫu ──
    print("\n[Generating sample visualizations...]")
    draw_sample_images("train", n=5)
    print("Done.")


if __name__ == "__main__":
    main()
