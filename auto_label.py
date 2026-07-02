import os
import cv2
import numpy as np
import random
from pathlib import Path
from ultralytics import YOLO

def calculate_mse(img1, img2):
    """Calculates Mean Squared Error between two images."""
    return np.mean((img1 - img2) ** 2)

def get_downscaled_gray(frame, size=(32, 32)):
    """Converts a frame to grayscale and resizes it for similarity comparison."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return resized

def main():
    print("=" * 60)
    print("INTELLIGENT TRAFFIC MONITORING SYSTEM - AUTO LABELING TOOL")
    print("=" * 60)

    # 1. Config paths
    project_root = Path("/home/hoangthang/Intelligent Traffic Monitoring System")
    video_dir = project_root / "Datasets" / "VietNam"
    output_dir = project_root / "vietnam_dataset"
    
    # Target classes mapping: COCO class ID -> Custom Class ID
    # COCO: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
    coco_to_custom = {
        2: 0,  # car
        3: 1,  # motorcycle
        5: 2,  # bus
        7: 3   # truck
    }
    class_names = ["car", "motorcycle", "bus", "truck"]
    
    # 2. Find all videos
    video_extensions = (".mp4", ".avi", ".mkv", ".MOV")
    video_files = [p for p in video_dir.iterdir() if p.suffix in video_extensions]
    print(f"Found {len(video_files)} videos in {video_dir}")
    
    if not video_files:
        print("No videos found! Exiting.")
        return
        
    # Process ALL videos in the dataset directory
    num_to_process = len(video_files)
    selected_videos = sorted(video_files)  # Sort for reproducible ordering
    print(f"Processing ALL {num_to_process} videos:")
    for v in selected_videos:
        print(f"  - {v.name}")

    # 3. Create train/val/test splits (video-level to avoid data leakage)
    # Split: 70% Train, 15% Val, 15% Test
    import random
    random.seed(42)
    shuffled_videos = selected_videos.copy()
    random.shuffle(shuffled_videos)
    selected_videos = shuffled_videos
    n = len(selected_videos)
    train_end = int(n * 0.7)
    val_end = train_end + int(n * 0.15)
    
    splits = {}
    for i, v in enumerate(selected_videos):
        if i < train_end:
            splits[v] = "train"
        elif i < val_end:
            splits[v] = "val"
        else:
            splits[v] = "test"
            
    print(f"\nSplit distribution:")
    print(f"  Train: {len([k for k, v in splits.items() if v == 'train'])} videos")
    print(f"  Val: {len([k for k, v in splits.items() if v == 'val'])} videos")
    print(f"  Test: {len([k for k, v in splits.items() if v == 'test'])} videos")

    # 4. Create directories
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # 5. Load YOLO model for auto-labeling
    print("\nLoading YOLO11 model for auto-labeling...")
    # Load model on CPU or GPU based on CUDA availability
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = YOLO("yolo11x.pt")  # will auto-download yolo11x.pt if not present
    
    # 6. Process each video
    frame_interval_seconds = 1.0  # Extract 1 frame per second
    mse_threshold = 10.0           # DMSE threshold for duplicate detection
    
    stats = {
        "train": {"frames": 0, "boxes": [0, 0, 0, 0]},
        "val": {"frames": 0, "boxes": [0, 0, 0, 0]},
        "test": {"frames": 0, "boxes": [0, 0, 0, 0]}
    }
    
    total_extracted = 0
    total_skipped_duplicate = 0
    
    for v_idx, video_path in enumerate(selected_videos):
        split = splits[video_path]
        print(f"\n[{v_idx+1}/{num_to_process}] Processing {video_path.name} -> Split: {split}")
        print(f"  Progress: {v_idx+1}/{num_to_process} ({(v_idx+1)/num_to_process*100:.1f}%)")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  Error: Could not open {video_path.name}")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps == 0 or frame_count == 0:
            print("  Error: Invalid FPS or frame count")
            cap.release()
            continue
            
        frame_step = int(fps * frame_interval_seconds)
        if frame_step == 0:
            frame_step = 1
            
        print(f"  FPS: {fps:.2f}, Total Frames: {frame_count}, Frame Step: {frame_step}")
        
        prev_downscaled = None
        video_frame_idx = 0
        saved_in_video = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only process at selected intervals
            if video_frame_idx % frame_step == 0:
                current_downscaled = get_downscaled_gray(frame)
                
                # Check for duplicate frame
                if prev_downscaled is not None:
                    mse = calculate_mse(current_downscaled, prev_downscaled)
                    if mse < mse_threshold:
                        total_skipped_duplicate += 1
                        video_frame_idx += 1
                        continue
                
                # If unique, save and update reference
                prev_downscaled = current_downscaled
                
                # Generate unique filename
                base_name = f"{video_path.stem}_f{video_frame_idx}"
                img_filename = f"{base_name}.jpg"
                txt_filename = f"{base_name}.txt"
                
                img_path = output_dir / split / "images" / img_filename
                txt_path = output_dir / split / "labels" / txt_filename
                
                # Run YOLO prediction for labeling with YOLO11x (imgsz=1280, conf=0.15, iou=0.45, agnostic_nms=True)
                results = model(frame, verbose=False, device=device, imgsz=1280, conf=0.15, iou=0.45, agnostic_nms=True)
                boxes = results[0].boxes
                
                # Find matching target classes
                valid_boxes = []
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in coco_to_custom:
                        conf = float(box.conf[0])
                        custom_cls = coco_to_custom[cls_id]
                        # Keep only confident boxes (> 0.15)
                        if conf >= 0.15:
                            xywh = box.xywhn[0].tolist() # normalized coordinates
                            valid_boxes.append((custom_cls, xywh))
                
                # If we have valid labels, save frame and labels
                if valid_boxes:
                    cv2.imwrite(str(img_path), frame)
                    
                    # Write label file in YOLO format
                    with open(txt_path, "w") as f_lbl:
                        for custom_cls, (x, y, w, h) in valid_boxes:
                            f_lbl.write(f"{custom_cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                            stats[split]["boxes"][custom_cls] += 1
                            
                    stats[split]["frames"] += 1
                    saved_in_video += 1
                    total_extracted += 1
                    
            video_frame_idx += 1
            
        cap.release()
        print(f"  Extracted {saved_in_video} unique annotated frames from {video_path.name}")

    # 7. Write data.yaml config
    yaml_content = f"""path: {output_dir}
train: train/images
val: val/images
test: test/images

names:
  0: car
  1: motorcycle
  2: bus
  3: truck
"""
    with open(output_dir / "vietnam_data.yaml", "w") as f_yaml:
        f_yaml.write(yaml_content)
    print(f"\nCreated {output_dir / 'vietnam_data.yaml'}")

    # 8. Print final statistics
    print("\n" + "=" * 50)
    print("DATASET GENERATION STATISTICS")
    print("=" * 50)
    print(f"Total Unique Frames Extracted: {total_extracted}")
    print(f"Total Static Frames Skipped (DMSE): {total_skipped_duplicate}")
    print("-" * 50)
    for split in ["train", "val", "test"]:
        s_data = stats[split]
        print(f"Split: {split.upper()}")
        print(f"  Extracted Frames: {s_data['frames']}")
        print(f"  Boxes count:")
        for c_idx, c_name in enumerate(class_names):
            print(f"    - {c_name}: {s_data['boxes'][c_idx]}")
        print("-" * 50)
    print("=" * 50)

if __name__ == "__main__":
    main()
