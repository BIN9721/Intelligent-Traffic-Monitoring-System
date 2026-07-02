"""
CityFlowV2 (AIC22) Auto-Labeler (YOLO11x)
==========================================
Trích frame từ vdo.avi và gán nhãn bằng YOLO11x — nhất quán với pipeline
Vietnam dataset. GT MOT (gt.txt) không được dùng làm nhãn.

Nguồn: Datasets/CityFlowV2/{train,validation}/*/vdo.avi
Output: cityflow_dataset/{train,val}/{images,labels}/

Split mapping:
  CityFlowV2 train/      -> YOLO train
  CityFlowV2 validation/ -> YOLO val
  CityFlowV2 test/       -> Bỏ qua (không có GT, không cần)

Frame sampling: 1 frame/giây + MSE duplicate filter (giống Vietnam)

Class mapping (COCO -> Custom):
  2 (car)        -> 0
  3 (motorcycle) -> 1
  5 (bus)        -> 2
  7 (truck)      -> 3
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ──────────────────────────── CONFIG ────────────────────────────
PROJECT_ROOT = Path("/home/hoangthang/Intelligent Traffic Monitoring System")
CITYFLOW_DIR = PROJECT_ROOT / "Datasets" / "CityFlowV2"
OUTPUT_DIR   = PROJECT_ROOT / "cityflow_dataset"
MODEL_PATH   = PROJECT_ROOT / "yolo11x.pt"

FRAME_INTERVAL_SEC = 1.0   # 1 frame/giây
MSE_THRESHOLD      = 10.0  # lọc frame trùng lặp
CONF_THRESH        = 0.15
IOU_THRESH         = 0.45
IMG_SIZE           = 1280

COCO_TO_CUSTOM = {2: 0, 3: 1, 5: 2, 7: 3}
CLASS_NAMES    = ["car", "motorcycle", "bus", "truck"]
COLORS         = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # BGR

SPLIT_MAP = {
    "train":      "train",
    "validation": "val",
}

# ──────────────────────────── HELPERS ───────────────────────────
def draw_sample_images(split="train", n=5):
    """Vẽ bounding box lên n ảnh mẫu ngẫu nhiên và lưu vào visual_results/."""
    import random
    visual_dir = OUTPUT_DIR / "visual_results"
    visual_dir.mkdir(parents=True, exist_ok=True)
    img_dir = OUTPUT_DIR / split / "images"
    lbl_dir = OUTPUT_DIR / split / "labels"

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

def process_video(vdo_path: Path, cam_id: str, yolo_split: str,
                  model, device: str, stats: dict) -> int:
    """Trích frame từ 1 vdo.avi, chạy YOLO11x, ghi label. Trả về số frame lưu."""
    cap = cv2.VideoCapture(str(vdo_path))
    if not cap.isOpened():
        print(f"  [WARN] Không mở được: {vdo_path}")
        return 0

    fps        = cap.get(cv2.CAP_PROP_FPS) or 10
    frame_step = max(1, int(fps * FRAME_INTERVAL_SEC))

    out_img = OUTPUT_DIR / yolo_split / "images"
    out_lbl = OUTPUT_DIR / yolo_split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    prev_gray = None
    frame_idx = 0
    saved     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            cur_gray = downscaled_gray(frame)

            # Lọc frame trùng (tĩnh / kẹt)
            if prev_gray is not None and mse(cur_gray, prev_gray) < MSE_THRESHOLD:
                frame_idx += 1
                continue
            prev_gray = cur_gray

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
                    cx, cy, w, h = box.xywhn[0].tolist()
                    valid_boxes.append((custom, cx, cy, w, h))

            if valid_boxes:
                stem    = f"{cam_id}_f{frame_idx:06d}"
                img_out = out_img / f"{stem}.jpg"
                lbl_out = out_lbl / f"{stem}.txt"

                cv2.imwrite(str(img_out), frame)
                with open(lbl_out, "w") as f:
                    for cls_id, cx, cy, w, h in valid_boxes:
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                        stats[yolo_split]["boxes"][cls_id] += 1
                stats[yolo_split]["frames"] += 1
                saved += 1

        frame_idx += 1

    cap.release()
    return saved

# ──────────────────────────── MAIN ──────────────────────────────
def main():
    print("=" * 60)
    print("CITYFLOW V2 — YOLO11x AUTO LABELER")
    print("=" * 60)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading YOLO11x from {MODEL_PATH} on {device} ...")
    model = YOLO(str(MODEL_PATH))

    stats = {
        "train": {"frames": 0, "boxes": [0, 0, 0, 0]},
        "val":   {"frames": 0, "boxes": [0, 0, 0, 0]},
    }

    for cf_split, yolo_split in SPLIT_MAP.items():
        split_dir = CITYFLOW_DIR / cf_split
        if not split_dir.exists():
            print(f"[WARN] {split_dir} không tồn tại, bỏ qua.")
            continue

        # Tìm tất cả vdo.avi trong split
        vdo_files = sorted(split_dir.rglob("vdo.avi"))
        print(f"\n[{cf_split.upper()} -> {yolo_split}] {len(vdo_files)} cameras")

        for i, vdo_path in enumerate(vdo_files):
            # cam_id: e.g. S01_c001
            cam_id = f"{vdo_path.parent.parent.name}_{vdo_path.parent.name}"
            saved  = process_video(vdo_path, cam_id, yolo_split, model, device, stats)
            print(f"  [{i+1}/{len(vdo_files)}] {cam_id}: {saved} frames")

    # ── YAML ──
    yaml_content = f"""path: {OUTPUT_DIR}
train: train/images
val: val/images

names:
  0: car
  1: motorcycle
  2: bus
  3: truck
"""
    yaml_path = OUTPUT_DIR / "cityflow_data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\nCreated {yaml_path}")

    # ── Thống kê ──
    total_frames = sum(s["frames"] for s in stats.values())
    total_boxes  = [sum(s["boxes"][c] for s in stats.values()) for c in range(4)]

    print("\n" + "=" * 60)
    print("CITYFLOWV2 AUTO-LABEL STATISTICS")
    print("=" * 60)
    print(f"Total Frames Extracted  : {total_frames:,}")
    print(f"Total Bounding Boxes    : {sum(total_boxes):,}")
    print("-" * 60)
    for split in ["train", "val"]:
        s = stats[split]
        print(f"Split: {split.upper()}")
        print(f"  Frames : {s['frames']:,}")
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
