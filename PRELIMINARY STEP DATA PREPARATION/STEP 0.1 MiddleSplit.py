import cv2
import numpy as np
import os

def debug_stereo_split(input_dir, output_dir, manual_fix=0):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "left"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "right"), exist_ok=True)

    print(f"Processing images from {input_dir}...")
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
            
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None: continue

        h, w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Detect Black Band Width (k)
        col_averages = np.mean(gray, axis=0)
        threshold = 20
        # Find first column that is NOT black
        k = np.argmax(col_averages > threshold)
        
        # Validation: Band should be reasonable (e.g., 50px to 400px)
        if k < 10 or k > w//3:
            # If detection fails, use the 'manual_fix' override or safe default
            # You can tweak this value if the red line is consistently off
            k = manual_fix if manual_fix > 0 else 0 
            
        # 2. Calculate the TRUE Split Point
        # The file contains: [Black Band (k)] + [Left View (Width)] + [Right View (Width)]
        # Total Width W = k + 2*ViewWidth
        # ViewWidth = (W - k) / 2
        # Split Point = k + ViewWidth
        
        view_width = (w - k) // 2
        split_point = k + view_width + (k//2)
        
        # 3. Extract Views
        # Left: From end of band (k) -> split_point
        # Right: From split_point -> end
        left_img = img[:, k : split_point]
        right_img = img[:, split_point : split_point + view_width]
        
        # 4. DEBUG: Save a visualization for the first few images
        # We draw a GREEN line at the detected start of content
        # We draw a RED line at the detected split point
        debug_vis = img.copy()
        cv2.line(debug_vis, (k, 0), (k, h), (0, 255, 0), 3)       # Green: Start of Left
        cv2.line(debug_vis, (split_point, 0), (split_point, h), (0, 0, 255), 3) # Red: Split
        
        cv2.imwrite(os.path.join(output_dir, f"DEBUG_{filename}"), debug_vis)
        
        # 5. Save Final Split Images
        cv2.imwrite(os.path.join(output_dir, "left", filename), left_img)
        cv2.imwrite(os.path.join(output_dir, "right", filename), right_img)
        
        print(f"File: {filename} | Band: {k}px | Split at: {split_point}px")

# Usage
input_folder = "dataset2/dataset2/rbg"
output_folder = "dataset2/dataset2/split_results"
debug_stereo_split(input_folder, output_folder)