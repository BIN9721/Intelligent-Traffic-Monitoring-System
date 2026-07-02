import os
import shutil
from pathlib import Path

def apply_refined_labels():
    project_root = Path("/home/hoangthang/Intelligent Traffic Monitoring System")
    suspicious_dir = project_root / "suspicious_val_test_labels"
    labels_dir = suspicious_dir / "labels"
    
    if not labels_dir.exists():
        print("[ERROR] Modified labels directory not found.")
        return
        
    refined_files = list(labels_dir.glob("*.txt"))
    print("=" * 60)
    print("APPLYING MODIFIED LABELS BACK TO ORIGINAL DATASETS")
    print("=" * 60)
    
    applied_count = 0
    
    for ref_path in refined_files:
        filename = ref_path.name
        
        # Parse the dataset, split and original filename from prefix
        # Example format: vietnam_dataset_val_vn_18h.7.9.22_f183.txt
        parts = filename.split("_")
        if len(parts) < 3:
            continue
            
        dataset_name = parts[0] + "_" + parts[1] # e.g. "vietnam_dataset" or "aic21_dataset"
        split = parts[2]                         # e.g. "val" or "test"
        
        # The rest of the parts make up the original label filename
        original_lbl_name = "_".join(parts[3:])
        
        target_path = project_root / dataset_name / split / "labels" / original_lbl_name
        
        if target_path.parent.exists():
            shutil.copy2(ref_path, target_path)
            applied_count += 1
            
    print(f"\n[SUCCESS] Successfully copied back {applied_count} modified labels!")
    print("=" * 60)

if __name__ == "__main__":
    apply_refined_labels()
