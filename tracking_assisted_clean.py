import cv2
import numpy as np
import torch
import glob
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict, Counter

def process_video_tracking(video_path: Path, dataset_dir: Path, model, coco_to_custom, device):
    print(f"\nProcessing Video: {video_path.name} ...")
    
    # 1. Run ByteTrack tracking on the video
    results = model.track(
        source=str(video_path), persist=True, imgsz=1280,
        conf=0.15, iou=0.45, agnostic_nms=True,
        stream=True, verbose=False, device=device
    )
    
    # Store tracked boxes for each frame index
    # frame_idx -> list of {'track_id': t_id, 'box': [cx, cy, w, h]}
    frame_tracks = defaultdict(list)
    track_classes = defaultdict(list)  # track_id -> list of class_ids
    
    for frame_idx, r in enumerate(results):
        if r.boxes is not None and r.boxes.id is not None:
            track_ids = r.boxes.id.int().tolist()
            class_ids = r.boxes.cls.int().tolist()
            xywhn = r.boxes.xywhn.tolist()
            
            for t_id, c_id, box in zip(track_ids, class_ids, xywhn):
                if c_id in coco_to_custom:
                    custom_c_id = coco_to_custom[c_id]
                    frame_tracks[frame_idx].append({
                        'track_id': t_id,
                        'box': box  # [cx, cy, w, h] normalized
                    })
                    track_classes[t_id].append(custom_c_id)
                    
    # 2. Compute majority class for each track ID
    track_majority = {}
    for t_id, class_list in track_classes.items():
        if class_list:
            majority_cls = Counter(class_list).most_common(1)[0][0]
            track_majority[t_id] = majority_cls
            
    # 3. Update the label files for the video
    corrected_count = 0
    total_checked = 0
    
    for split in ["train", "val", "test"]:
        lbl_dir = dataset_dir / split / "labels"
        if not lbl_dir.exists():
            continue
            
        # Search for all labels matching this video stem: {video_stem}_f*.txt
        lbl_files = sorted(list(lbl_dir.glob(f"{video_path.stem}_f*.txt")))
        for lbl_path in lbl_files:
            filename = lbl_path.name
            try:
                f_part = filename.split("_f")[-1].replace(".txt", "")
                frame_idx = int(f_part)
            except ValueError:
                continue
                
            if frame_idx not in frame_tracks:
                continue
                
            with open(lbl_path) as f:
                lines = f.readlines()
                
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                total_checked += 1
                
                # Match this box to the tracking results at this frame
                best_match = None
                best_dist = 0.05  # threshold normalized center distance
                
                for track in frame_tracks[frame_idx]:
                    tcx, tcy, tw, th = track['box']
                    dist = np.sqrt((cx - tcx)**2 + (cy - tcy)**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = track
                        
                if best_match is not None:
                    t_id = best_match['track_id']
                    majority_cls = track_majority.get(t_id, cls_id)
                    if majority_cls != cls_id:
                        cls_id = majority_cls
                        corrected_count += 1
                        
                new_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                
            with open(lbl_path, "w") as f:
                f.writelines(new_lines)
                
    if total_checked > 0:
        pct = (corrected_count / total_checked) * 100
        print(f"  Result: Corrected {corrected_count:,} out of {total_checked:,} boxes ({pct:.1f}%)")
    else:
        print("  Result: No boxes checked.")
        
    return corrected_count, total_checked

import os

def get_processed_videos(log_path):
    processed = set()
    if not log_path.exists():
        return processed
    try:
        with open(log_path) as f:
            content = f.read()
        parts = content.split("Processing Video: ")
        for part in parts[1:]:
            lines = part.strip().split("\n")
            if not lines:
                continue
            video_name = lines[0].split(" ...")[0].strip()
            has_result = False
            for line in lines[1:]:
                if "Result: " in line:
                    has_result = True
                    break
            if has_result:
                processed.add(video_name)
    except Exception as e:
        print(f"Error reading log checkpoint: {e}")
    return processed

def main():
    print("=" * 60)
    print("TRACKING-ASSISTED LABEL SANITIZER")
    print("=" * 60)

    project_root = Path("/home/hoangthang/Intelligent Traffic Monitoring System")
    model_path = project_root / "yolo11x.pt"
    log_path = project_root / "tracking_run.log"
    
    # Load already processed videos
    processed_videos = get_processed_videos(log_path)
    print(f"Loaded {len(processed_videos)} already processed videos from log.")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading YOLO11x on {device} ...")
    model = YOLO(str(model_path))
    model.to(device)

    coco_to_custom = {2: 0, 3: 1, 5: 2, 7: 3}
    
    targets = [
        {
            "name": "vietnam_dataset",
            "video_dir": project_root / "Datasets" / "VietNam",
            "dataset_dir": project_root / "vietnam_dataset"
        },
        {
            "name": "aic21_dataset",
            "video_dir": project_root / "Datasets" / "Vehicle Counting" / "Dataset_A",
            "dataset_dir": project_root / "aic21_dataset"
        }
    ]

    global_corrected = 0
    global_checked = 0

    for t in targets:
        name = t["name"]
        video_dir = t["video_dir"]
        dataset_dir = t["dataset_dir"]
        
        if not video_dir.exists() or not dataset_dir.exists():
            print(f"\n[WARN] Directories not found for {name}. Skipping.")
            continue
            
        print(f"\nProcessing Dataset: {name}")
        video_exts = {".mp4", ".avi", ".mkv", ".MOV"}
        videos = sorted([p for p in video_dir.iterdir() if p.suffix in video_exts])
        print(f"Found {len(videos)} videos.")
        
        for i, video_path in enumerate(videos):
            if video_path.name in processed_videos:
                print(f"  [{i+1}/{len(videos)}] Skipping {video_path.name} (already processed)")
                continue
            
            print(f"  [{i+1}/{len(videos)}] ...")
            corr, chk = process_video_tracking(video_path, dataset_dir, model, coco_to_custom, device)
            global_corrected += corr
            global_checked += chk

    print("\n" + "=" * 60)
    print("TRACKING-ASSISTED CLEANING STATISTICS")
    print("=" * 60)
    print(f"Total Bounding Boxes Checked   : {global_checked:,}")
    print(f"Total Bounding Boxes Corrected : {global_corrected:,}")
    if global_checked > 0:
        print(f"Overall Correction Rate        : {global_corrected/global_checked*100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
