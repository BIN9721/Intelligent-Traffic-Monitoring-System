import os
import cv2
import numpy as np
from pathlib import Path

def draw_boxes(image, boxes, active_idx):
    h, w, _ = image.shape
    display_img = image.copy()
    
    # Class names mapping
    class_names = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}
    class_colors = {
        0: (0, 255, 0),    # green
        1: (255, 255, 0),  # cyan
        2: (0, 165, 255),  # orange (bus)
        3: (0, 0, 255)     # red (truck)
    }

    for idx, box in enumerate(boxes):
        cls_id, cx, cy, bw, bh = box
        
        # Convert normalized to pixel coordinates
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        
        color = class_colors.get(cls_id, (255, 255, 255))
        thickness = 2
        
        # Highlight the box currently being edited
        if idx == active_idx:
            thickness = 4
            cv2.rectangle(display_img, (x1-2, y1-2), (x2+2, y2+2), (255, 255, 255), 2) # white border outer
            
        cv2.rectangle(display_img, (x1, y1), (x2, y2), color, thickness)
        
        label = f"{class_names.get(cls_id, str(cls_id))}"
        if idx == active_idx:
            label = f"-> {label.upper()} <-"
            
        cv2.putText(display_img, label, (x1, max(y1 - 10, 15)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    return display_img

def main():
    project_root = Path("/home/hoangthang/Intelligent Traffic Monitoring System")
    suspicious_dir = project_root / "suspicious_val_test_labels"
    images_dir = suspicious_dir / "images"
    labels_dir = suspicious_dir / "labels"
    
    if not images_dir.exists() or not labels_dir.exists():
        print("[ERROR] Suspicious directory does not exist. Please run extract_suspicious.py first.")
        return
        
    img_paths = sorted(list(images_dir.glob("*.jpg")))
    total_imgs = len(img_paths)
    
    if total_imgs == 0:
        print("[INFO] No suspicious images to review.")
        return
        
    print("=" * 60)
    print("LOCAL INTERACTIVE LABEL REFINER")
    print("=" * 60)
    print("Keyboard Controls:")
    print("  [2] or [B] : Change selected target to BUS")
    print("  [3] or [T] : Change selected target to TRUCK")
    print("  [0] or [C] : Change selected target to CAR")
    print("  [Space]    : Save changes and skip to next image")
    print("  [Q]        : Save and Quit")
    print("=" * 60)
    
    cv2.namedWindow("Label Editor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Label Editor", 1280, 720)
    
    img_idx = 0
    while img_idx < total_imgs:
        img_path = img_paths[img_idx]
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        
        if not lbl_path.exists():
            img_idx += 1
            continue
            
        image = cv2.imread(str(img_path))
        if image is None:
            img_idx += 1
            continue
            
        # Read boxes
        boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                    
        # Filter boxes that are bus (2) or truck (3) to edit
        candidate_indices = [i for i, b in enumerate(boxes) if b[0] in [2, 3]]
        
        if not candidate_indices:
            img_idx += 1
            continue
            
        active_candidate_ptr = 0
        changed = False
        
        while active_candidate_ptr < len(candidate_indices):
            active_box_idx = candidate_indices[active_candidate_ptr]
            
            # Show image
            display_img = draw_boxes(image, boxes, active_box_idx)
            
            # Overlay metadata text
            cv2.putText(display_img, f"Image {img_idx+1}/{total_imgs} | Targets in frame: {len(candidate_indices)}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_img, f"Press: [2/B]=Bus | [3/T]=Truck | [0/C]=Car | [Space]=Next Image | [Q]=Quit", 
                        (20, display_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow("Label Editor", display_img)
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("[INFO] Exiting...")
                cv2.destroyAllWindows()
                return
                
            elif key in [ord('2'), ord('b'), ord('B')]:
                boxes[active_box_idx][0] = 2
                changed = True
                active_candidate_ptr += 1 # Move to next target in this image
                
            elif key in [ord('3'), ord('t'), ord('T')]:
                boxes[active_box_idx][0] = 3
                changed = True
                active_candidate_ptr += 1
                
            elif key in [ord('0'), ord('c'), ord('C')]:
                boxes[active_box_idx][0] = 0
                changed = True
                active_candidate_ptr += 1
                
            elif key == ord(' '):  # Space key -> skip rest of this image
                break
                
        if changed:
            # Write modified labels back to the file
            with open(lbl_path, "w") as f:
                for box in boxes:
                    f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
            print(f"Saved modifications for: {img_path.name}")
            
        img_idx += 1
        
    print("\n[SUCCESS] Completed all images!")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
