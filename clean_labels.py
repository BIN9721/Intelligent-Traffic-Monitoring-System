import os
import glob
from pathlib import Path

def clean_label_file(lbl_path: Path):
    if not lbl_path.exists() or os.path.getsize(lbl_path) == 0:
        return 0, 0, 0 # modified, deleted, kept

    with open(lbl_path) as f:
        lines = f.readlines()

    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            boxes.append({
                'cls': cls_id,
                'cx': cx,
                'cy': cy,
                'w': w,
                'h': h,
                'x1': cx - w/2,
                'y1': cy - h/2,
                'x2': cx + w/2,
                'y2': cy + h/2
            })

    kept_boxes = []
    deleted_count = 0
    modified_count = 0

    # Separate large and small boxes for grouping checks
    large_indices = []
    small_indices = []
    for i, box in enumerate(boxes):
        if box['cls'] in [2, 3]: # bus, truck
            large_indices.append(i)
        elif box['cls'] in [0, 1]: # car, motorcycle
            small_indices.append(i)

    discard_indices = set()

    # Rule 1: Grouping / Containment Filter
    for l_idx in large_indices:
        l_box = boxes[l_idx]
        l_area = l_box['w'] * l_box['h']
        if l_area == 0:
            continue

        contained_area = 0
        contained_count = 0

        for s_idx in small_indices:
            s_box = boxes[s_idx]
            # Intersection coordinates
            ix1 = max(l_box['x1'], s_box['x1'])
            iy1 = max(l_box['y1'], s_box['y1'])
            ix2 = min(l_box['x2'], s_box['x2'])
            iy2 = min(l_box['y2'], s_box['y2'])

            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                small_area = s_box['w'] * s_box['h']
                if small_area > 0 and inter_area / small_area > 0.80:
                    contained_area += inter_area
                    contained_count += 1

        if contained_count >= 2 and contained_area / l_area > 0.60:
            discard_indices.add(l_idx)
            deleted_count += 1

    # Rule 4: Multi-Class Duplicate NMS (IoU > 0.40)
    active_indices = [i for i in range(len(boxes)) if i not in discard_indices]
    for i in range(len(active_indices)):
        idx1 = active_indices[i]
        if idx1 in discard_indices:
            continue
        box1 = boxes[idx1]
        area1 = box1['w'] * box1['h']
        
        for j in range(i + 1, len(active_indices)):
            idx2 = active_indices[j]
            if idx2 in discard_indices:
                continue
            box2 = boxes[idx2]
            area2 = box2['w'] * box2['h']
            
            # Intersection coordinates
            ix1 = max(box1['x1'], box2['x1'])
            iy1 = max(box1['y1'], box2['y1'])
            ix2 = min(box1['x2'], box2['x2'])
            iy2 = min(box1['y2'], box2['y2'])
            
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                
                if iou > 0.40:
                    # Keep the smaller area (tighter box), discard the larger duplicate
                    if area1 < area2:
                        discard_indices.add(idx2)
                        deleted_count += 1
                    else:
                        discard_indices.add(idx1)
                        deleted_count += 1
                        break

    # Apply other rules on the remaining boxes
    for i, box in enumerate(boxes):
        if i in discard_indices:
            continue

        cls_id = box['cls']
        w = box['w']
        h = box['h']
        if h == 0:
            deleted_count += 1
            continue

        aspect_ratio = (w * 16.0) / (h * 9.0) # Assumed 16:9 aspect ratio
        area = w * h

        # Rule 5: Geometric/Perspective Constraint Filter
        if box['cy'] < 0.35 and area > 0.05:
            deleted_count += 1
            continue

        # Rule 2: Motorcycle Aspect Ratio Filter
        if cls_id == 1: # motorcycle
            if aspect_ratio > 1.1:
                # Discard horizontal motorcycle boxes (false positives)
                deleted_count += 1
                continue

        # Rule 3: Truck/Bus Size & Aspect Ratio Filter
        if cls_id in [2, 3]: # bus, truck
            if area < 0.001:
                # Far distance: change to car (safer default)
                cls_id = 0
                modified_count += 1
            elif aspect_ratio < 0.70:
                # Vertical truck/bus box is impossible -> discard
                deleted_count += 1
                continue

        kept_boxes.append((cls_id, box['cx'], box['cy'], w, h))

    # Write cleaned labels back to file
    if kept_boxes:
        with open(lbl_path, "w") as f:
            for c, cx, cy, w, h in kept_boxes:
                f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    else:
        # Keep an empty file to prevent training issues
        open(lbl_path, "w").close()

    return modified_count, deleted_count, len(kept_boxes)

def main():
    print("=" * 60)
    print("AUTOMATIC LABEL CLEANER")
    print("=" * 60)

    project_root = Path("/home/hoangthang/Intelligent Traffic Monitoring System")
    datasets = [
        "vietnam_dataset",
        "uadetrac_dataset",
        "aic21_dataset",
        "cityflow_dataset"
    ]

    total_modified = 0
    total_deleted = 0
    total_kept = 0

    for dataset_name in datasets:
        dataset_dir = project_root / dataset_name
        if not dataset_dir.exists():
            print(f"[WARN] {dataset_name} does not exist. Skipping.")
            continue

        print(f"Cleaning dataset: {dataset_name} ...")
        label_files = glob.glob(str(dataset_dir / "**" / "labels" / "*.txt"), recursive=True)
        print(f"  Found {len(label_files)} label files.")

        ds_modified = 0
        ds_deleted = 0
        ds_kept = 0

        for f_path in label_files:
            m, d, k = clean_label_file(Path(f_path))
            ds_modified += m
            ds_deleted += d
            ds_kept += k

        print(f"  Results for {dataset_name}:")
        print(f"    Modified: {ds_modified:,} labels")
        print(f"    Deleted : {ds_deleted:,} labels")
        print(f"    Kept    : {ds_kept:,} labels")
        
        total_modified += ds_modified
        total_deleted += ds_deleted
        total_kept += ds_kept

    print("\n" + "=" * 60)
    print("OVERALL CLEANING STATISTICS")
    print("=" * 60)
    print(f"Total Modified (Reclassified) : {total_modified:,}")
    print(f"Total Deleted (False Positives): {total_deleted:,}")
    print(f"Total Kept Labels              : {total_kept:,}")
    print("=" * 60)

if __name__ == "__main__":
    main()
