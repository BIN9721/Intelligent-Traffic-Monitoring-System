import os
import cv2
import random
from pathlib import Path

def draw_boxes(image, label_path):
    h, w = image.shape[:2]
    display_img = image.copy()
    
    class_names = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}
    class_colors = {
        0: (0, 255, 0),    # green (car)
        1: (255, 255, 0),  # cyan (motorcycle)
        2: (0, 165, 255),  # orange (bus)
        3: (0, 0, 255)     # red (truck)
    }
    
    if not label_path.exists():
        return display_img
        
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                color = class_colors.get(cls_id, (255, 255, 255))
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                
                label = class_names.get(cls_id, str(cls_id))
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(display_img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                cv2.putText(display_img, label, (x1, y1 - 3), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                
    return display_img

def main():
    project_root = Path("/home/hoangthang/Intelligent Traffic Monitoring System")
    combined_dir = project_root / "combined_dataset"
    output_dir = Path("/home/hoangthang/.gemini/antigravity/brain/802d35f5-8e24-4d43-893d-3656c6ff424e/alignment_check")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not combined_dir.exists():
        print("[ERROR] combined_dataset directory not found!")
        return
        
    print("=" * 60)
    print("GENERATING VISUAL PREVIEWS FOR TAXONOMY ALIGNMENT VERIFICATION")
    print("=" * 60)
    
    datasets = {
        "vn": "vietnam_dataset",
        "ua": "uadetrac_dataset",
        "aic": "aic21_dataset",
        "cf": "cityflow_dataset"
    }
    
    splits = ["train", "val", "test"]
    
    # Gather all images grouped by prefix
    prefix_images = {pref: [] for pref in datasets}
    
    for split in splits:
        img_dir = combined_dir / split / "images"
        lbl_dir = combined_dir / split / "labels"
        if not img_dir.exists():
            continue
            
        for img_path in img_dir.glob("*.jpg"):
            # Check prefix
            parts = img_path.name.split("_")
            if parts:
                pref = parts[0]
                if pref in prefix_images:
                    lbl_path = lbl_dir / f"{img_path.stem}.txt"
                    prefix_images[pref].append((img_path, lbl_path))
                    
    # Select 10 random images for each prefix and draw boxes
    for pref, dataset_fullname in datasets.items():
        candidates = prefix_images[pref]
        print(f"Dataset '{dataset_fullname}' ({pref}_): found {len(candidates)} total images.")
        
        if len(candidates) == 0:
            continue
            
        selected = random.sample(candidates, min(len(candidates), 10))
        ds_out_dir = output_dir / dataset_fullname
        ds_out_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, (img_path, lbl_path) in enumerate(selected):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            annotated = draw_boxes(image, lbl_path)
            out_img_path = ds_out_dir / f"check_{idx+1}_{img_path.name}"
            cv2.imwrite(str(out_img_path), annotated)
            
        print(f"  Generated 10 validation images for {dataset_fullname} in alignment_check/{dataset_fullname}/")
        
    print("\n[SUCCESS] Completed generation of alignment verification previews!")

if __name__ == "__main__":
    main()
