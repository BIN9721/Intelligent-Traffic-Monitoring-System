import os
import shutil
from pathlib import Path

def extract_suspicious_frames():
    project_root = Path("/home/hoangthang/Intelligent Traffic Monitoring System")
    output_dir = project_root / "suspicious_val_test_labels"
    
    # Create output directories for YOLO format compatibility
    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    
    datasets = ["vietnam_dataset", "aic21_dataset"]
    splits = ["val", "test"]
    
    # Keywords indicating challenging conditions (night, dawn, rain, snow)
    target_keywords = ["18h", "19h", "dawn", "rain", "night", "snow", "17h30", "17h05", "17h"]
    
    print("=" * 60)
    print("EXTRACTING SUSPICIOUS NIGHT/RAIN LABELS FOR VAL/TEST SETS")
    print("=" * 60)
    
    total_copied = 0
    total_images_scanned = 0
    
    for ds_name in datasets:
        ds_dir = project_root / ds_name
        if not ds_dir.exists():
            continue
            
        print(f"Scanning dataset: {ds_name} ...")
        
        for split in splits:
            img_dir = ds_dir / split / "images"
            lbl_dir = ds_dir / split / "labels"
            
            if not img_dir.exists() or not lbl_dir.exists():
                continue
                
            img_files = list(img_dir.glob("*.jpg"))
            for img_path in img_files:
                filename = img_path.name
                total_images_scanned += 1
                
                # Check 1: Filename condition (weather or time)
                is_target_condition = any(kw in filename.lower() for kw in target_keywords)
                if not is_target_condition:
                    continue
                    
                # Check 2: Label file exists and contains class 2 (bus) or 3 (truck)
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    continue
                    
                has_bus_or_truck = False
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            if cls_id in [2, 3]: # bus, truck
                                has_bus_or_truck = True
                                break
                                
                if has_bus_or_truck:
                    # Copy image and label with dataset split prefix to avoid naming conflicts
                    new_stem = f"{ds_name}_{split}_{img_path.stem}"
                    new_img_name = f"{new_stem}.jpg"
                    new_lbl_name = f"{new_stem}.txt"
                    
                    shutil.copy2(img_path, out_images / new_img_name)
                    shutil.copy2(lbl_path, out_labels / new_lbl_name)
                    
                    total_copied += 1
                    
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total Val/Test Images Scanned      : {total_images_scanned:,}")
    print(f"Suspicious Night/Rain Frames Copied: {total_copied:,}")
    print(f"Output folder: {output_dir}")
    print("  - Images saved to: suspicious_val_test_labels/images/")
    print("  - Labels saved to: suspicious_val_test_labels/labels/")
    print("=" * 60)

if __name__ == "__main__":
    extract_suspicious_frames()
