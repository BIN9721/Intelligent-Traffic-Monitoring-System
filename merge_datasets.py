"""
Merge All 4 Datasets -> combined_dataset/
==========================================
Merges Vietnam, UA-DETRAC, AIC21, and CityFlowV2 YOLO datasets into one
unified dataset for training. Each source dataset keeps its prefix in the
output filename to avoid collisions.

Source datasets:
  vietnam_dataset/   -> prefix "vn"
  uadetrac_dataset/  -> prefix "ua"
  aic21_dataset/     -> prefix "aic"
  cityflow_dataset/  -> prefix "cf"

Output structure:
  combined_dataset/
    train/images/   train/labels/
    val/images/     val/labels/
    test/images/    test/labels/
    combined_data.yaml

Only TRAIN and VAL from CityFlowV2 (no GT for test).
"""

import shutil
from pathlib import Path

# ──────────────────────────── CONFIG ────────────────────────────
PROJECT_ROOT = Path("/home/hoangthang/Intelligent Traffic Monitoring System")
OUTPUT_DIR   = PROJECT_ROOT / "combined_dataset"

SOURCES = [
    # (dataset_dir, prefix, splits_available)
    (PROJECT_ROOT / "vietnam_dataset",  "vn",  ["train", "val", "test"]),
    (PROJECT_ROOT / "uadetrac_dataset", "ua",  ["train", "val", "test"]),
    (PROJECT_ROOT / "aic21_dataset",    "aic", ["train", "val", "test"]),
    (PROJECT_ROOT / "cityflow_dataset", "cf",  ["train", "val"]),
]

SPLITS = ["train", "val", "test"]

# ──────────────────────────── HELPERS ───────────────────────────
def copy_split(src_dir: Path, prefix: str, split: str, stats: dict):
    """Copy all images + labels from src_dir/split to combined_dataset/split."""
    src_img = src_dir / split / "images"
    src_lbl = src_dir / split / "labels"
    if not src_img.exists():
        return

    dst_img = OUTPUT_DIR / split / "images"
    dst_lbl = OUTPUT_DIR / split / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    img_files = sorted(src_img.glob("*.jpg")) + sorted(src_img.glob("*.png"))
    copied = 0
    skipped = 0

    for img_path in img_files:
        new_stem   = f"{prefix}_{img_path.stem}"
        dst_i = dst_img / f"{new_stem}{img_path.suffix}"
        dst_l = dst_lbl / f"{new_stem}.txt"

        lbl_path = src_lbl / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            skipped += 1
            continue

        shutil.copy2(str(img_path), str(dst_i))
        shutil.copy2(str(lbl_path), str(dst_l))
        copied += 1

    stats[split]["frames"] += copied
    print(f"    {split}: {copied} frames copied ({skipped} skipped — no label)")


# ──────────────────────────── MAIN ──────────────────────────────
def main():
    print("=" * 60)
    print("MERGING ALL 4 DATASETS -> combined_dataset/")
    print("=" * 60)

    stats = {s: {"frames": 0} for s in SPLITS}

    for src_dir, prefix, available_splits in SOURCES:
        if not src_dir.exists():
            print(f"\n[WARN] {src_dir.name} not found, skipping.")
            continue
        print(f"\n[{src_dir.name}] prefix='{prefix}'")
        for split in SPLITS:
            if split in available_splits:
                copy_split(src_dir, prefix, split, stats)
            else:
                print(f"    {split}: skipped (not in this source)")

    # Write combined YAML
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
    yaml_path = OUTPUT_DIR / "combined_data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\nCreated {yaml_path}")

    # Count boxes per class across all splits
    box_counts = {s: [0, 0, 0, 0] for s in SPLITS}
    class_names = ["car", "motorcycle", "bus", "truck"]
    total_frames_actual = 0
    for split in SPLITS:
        lbl_dir = OUTPUT_DIR / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl_path in lbl_dir.glob("*.txt"):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        if 0 <= cls_id < 4:
                            box_counts[split][cls_id] += 1
        total_frames_actual += stats[split]["frames"]

    total_boxes = [sum(box_counts[s][c] for s in SPLITS) for c in range(4)]

    print("\n" + "=" * 60)
    print("COMBINED DATASET STATISTICS")
    print("=" * 60)
    print(f"Total Frames  : {total_frames_actual:,}")
    print(f"Total Boxes   : {sum(total_boxes):,}")
    print("-" * 60)
    for split in SPLITS:
        if stats[split]["frames"] == 0:
            continue
        print(f"Split: {split.upper()}")
        print(f"  Frames : {stats[split]['frames']:,}")
        for c, name in enumerate(class_names):
            if box_counts[split][c] > 0:
                print(f"  {name:<12}: {box_counts[split][c]:,}")
        print("-" * 60)
    print(f"\nClass totals across all splits:")
    for c, name in enumerate(class_names):
        pct = total_boxes[c] / max(1, sum(total_boxes)) * 100
        print(f"  {name:<12}: {total_boxes[c]:,}  ({pct:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
