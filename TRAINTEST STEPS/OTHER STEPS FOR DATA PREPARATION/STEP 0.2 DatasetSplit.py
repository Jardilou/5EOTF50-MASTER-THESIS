import os
import shutil
import random
from pathlib import Path

def organize_dataset(image_source, label_source, output_root, split_ratio=0.8):
    # 1. Setup Directories
    dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for d in dirs:
        os.makedirs(os.path.join(output_root, d), exist_ok=True)

    # 2. Get list of all labels (using the annotation as the source of truth)
    # We look for .txt files
    label_files = [f for f in os.listdir(label_source) if f.endswith('.txt')]
    
    # Shuffle for random split
    random.seed(42)
    random.shuffle(label_files)
    
    # 3. Process files
    matched_count = 0
    missing_images = 0
    
    # Calculate split index
    split_idx = int(len(label_files) * split_ratio)
    
    print(f"Organizing {len(label_files)} annotations...")
    
    for i, label_file in enumerate(label_files):
        # Determine strict association: 3.txt -> 3.jpg
        base_name = os.path.splitext(label_file)[0]
        image_name = base_name + ".jpg" # Assuming .jpg, change if .png
        
        src_image_path = os.path.join(image_source, image_name)
        src_label_path = os.path.join(label_source, label_file)
        
        # Check if the corresponding Left Image exists
        if not os.path.exists(src_image_path):
            missing_images += 1
            continue
            
        # Determine if Train or Val
        subset = 'train' if i < split_idx else 'val'
        
        # Copy Image
        dst_image_path = os.path.join(output_root, 'images', subset, image_name)
        shutil.copy(src_image_path, dst_image_path)
        
        # Copy Label
        dst_label_path = os.path.join(output_root, 'labels', subset, label_file)
        shutil.copy(src_label_path, dst_label_path)
        
        matched_count += 1

    print("------------------------------------------------")
    print(f"Processing Complete.")
    print(f"Total Pairs Created: {matched_count}")
    print(f"Train Set: {len(os.listdir(os.path.join(output_root, 'images/train')))}")
    print(f"Val Set:   {len(os.listdir(os.path.join(output_root, 'images/val')))}")
    print(f"Missing Images: {missing_images} (Labels without images)")
    print(f"Output Directory: {output_root}")

# --- CONFIGURATION ---
# Update these paths to match your actual folders
img_path = r"dataset2/dataset2/split_results/left"            # Your corrected Left images
lbl_path = r"dataset2/dataset2/labels"          # Your text labels
out_path = r"dataset2/dataset2/final_dataset"   # Where the ready-to-train data goes

# Run
organize_dataset(img_path, lbl_path, out_path)