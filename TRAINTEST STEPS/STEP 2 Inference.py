# import cv2
# import numpy as np
# from ultralytics import YOLO
# import os

# def draw_cubic_bezier(img, points, color=(0, 255, 255), thickness=2):
#     """
#     Takes 4 control points and draws a smooth cubic Bézier curve.
#     Formula: B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
#     """
#     if len(points) != 4:
#         return img
        
#     # Generate 100 points along the curve (t goes from 0.0 to 1.0)
#     curve_points = []
#     for t in np.linspace(0, 1, 100):
#         # The Cubic Bézier formula
#         p = ( (1-t)**3 * points[0] + 
#               3 * (1-t)**2 * t * points[1] + 
#               3 * (1-t) * t**2 * points[2] + 
#               t**3 * points[3] )
#         curve_points.append([int(p[0]), int(p[1])])
        
#     # Draw the curve
#     pts = np.array(curve_points, np.int32).reshape((-1, 1, 2))
#     cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)
#     return img

# def main():
#     # 1. Load your newly trained weights 
#     # (Since val=False, we use last.pt instead of best.pt)
#     weight_path = r"ultralytics/runs/pose/BezierFusion/eamrf_training_run9/weights/last.pt"
    
#     if not os.path.exists(weight_path):
#         print("Training hasn't finished or weights not found!")
#         return
        
#     model = YOLO(weight_path)
    
#     # 2. Point it to your Test/Val images folder
#     test_images_dir = r"dataset2\dataset2\train-test\images\val"
#     output_dir = r"Inference_results_THIRD_TRY_v2"
#     os.makedirs(output_dir, exist_ok=True)

#     # 3. Run Inference on all images in the folder
#     print("Running inference...")
#     results = model.predict(source=test_images_dir, conf=0.4) # Only trust predictions > 50%

#     # 4. Process and Draw
#     for i, result in enumerate(results):
#         # Load the original image array
#         img = result.orig_img.copy()
        
#         # Check if the model actually detected any fish in this image
#         if result.keypoints is not None and result.keypoints.xy is not None:
#             # Loop through every fish detected in the image
#             for fish_idx in range(len(result.keypoints.xy)):
#                 # Extract the 4 control points (x, y) for this specific fish
#                 kpts = result.keypoints.xy[fish_idx].cpu().numpy()
                
#                 # Make sure we got exactly 4 points
#                 if len(kpts) == 4:
#                     # Extract the 4 control points (x, y)
#                     # Extract points and box
#                     kpts = result.keypoints.xy[fish_idx].cpu().numpy()
#                     box = result.boxes.xyxy[fish_idx].cpu().numpy().astype(int)

#                     # Clamp keypoints strictly inside the bounding box
#                     for pt in kpts:
#                         pt[0] = np.clip(pt[0], box[0], box[2]) # Clip X between x_min and x_max
#                         pt[1] = np.clip(pt[1], box[1], box[3]) # Clip Y between y_min and y_max
                    
                    
#                 ###################################################################################################
#                 # --- OPTIONAL FILTER: Check if the fish is facing the camera (head and tail are close together) ---
#                     if len(kpts) == 4:
#                         head = kpts[0]
#                         tail = kpts[3]
                        
#                         # Calculate the 2D distance between head and tail
#                         head_to_tail_dist = np.linalg.norm(head - tail)
                        
#                         # Get the bounding box diagonal size for scale
#                         box = result.boxes.xyxy[fish_idx].cpu().numpy()
#                         box_diagonal = np.linalg.norm([box[2]-box[0], box[3]-box[1]])
                        
#                         # The Filter: If the head and tail don't span at least 60% of the box, 
#                         # the fish is facing the camera. Skip it!
#                         if head_to_tail_dist < (0.60 * box_diagonal):
#                             print("Fish is facing the camera. Ignoring length measurement.")
#                             continue # Skips drawing the curve and moves to the next fish
#                 ###################################################################################################
#                     # Draw the bounding box
#                     box = result.boxes.xyxy[fish_idx].cpu().numpy().astype(int)
#                     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                    
#                     # Draw the smooth Bézier curve (Yellow)
#                     img = draw_cubic_bezier(img, kpts, color=(0, 255, 255), thickness=3)
                    
#                     # Draw the 4 individual control points (Red) so you can see what the AI targeted
#                     for pt in kpts:
#                         cv2.circle(img, (int(pt[0]), int(pt[1])), radius=4, color=(0, 0, 255), thickness=-1)

#         # Save the drawn image to your results folder
#         save_path = os.path.join(output_dir, f"result_{i}.jpg")
#         cv2.imwrite(save_path, img)
#         print(f"Saved {save_path}")

#     print(f"\n✅ All done! Check the '{output_dir}' folder to see your AI's predictions.")

# if __name__ == '__main__':
#     main()
    
    
import cv2
import numpy as np
from ultralytics import YOLO
import os

def draw_cubic_bezier(img, points, color=(0, 255, 255), thickness=2):
    """Draws a smooth cubic Bézier curve from 4 control points."""
    if len(points) != 4:
        return img
    curve_points = []
    for t in np.linspace(0, 1, 100):
        p = ( (1-t)**3 * points[0] + 
              3 * (1-t)**2 * t * points[1] + 
              3 * (1-t) * t**2 * points[2] + 
              t**3 * points[3] )
        curve_points.append([int(p[0]), int(p[1])])
        
    pts = np.array(curve_points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)
    return img

def main():
    weight_path = r"ultralytics/runs/pose/BezierFusion/eamrf_training_run7/weights/last.pt"    
    if not os.path.exists(weight_path):
        print("Weights not found! Check your path.")
        return
        
    model = YOLO(weight_path)
    test_images_dir = r"dataset2\dataset2\train-test\images\val"
    output_dir = r"Inference_results_FIRST_TRY"
    os.makedirs(output_dir, exist_ok=True)

    print("Running inference...")
    
    # FIX #3: Lowered confidence from 0.5 to 0.25 to catch "missing" stereo twins
    results = model.predict(source=test_images_dir, conf=0.4) 
    facing_camera_count = 0
    for i, result in enumerate(results):
        img = result.orig_img.copy()
        h_img, w_img = img.shape[:2] # Get image dimensions to prevent drawing outside the frame
        
        if result.keypoints is not None and result.keypoints.xy is not None:
            for fish_idx in range(len(result.keypoints.xy)):
                kpts = result.keypoints.xy[fish_idx].cpu().numpy()
                
                if len(kpts) == 4:
                    ###################################################################################################
            # --- OPTIONAL FILTER: Check if the fish is facing the camera (head and tail are close together) ---
                    head = kpts[0]
                    tail = kpts[3]
                    
                    # Calculate the 2D distance between head and tail
                    head_to_tail_dist = np.linalg.norm(head - tail)
                    
                    # Get the bounding box diagonal size for scale
                    box = result.boxes.xyxy[fish_idx].cpu().numpy()
                    box_diagonal = np.linalg.norm([box[2]-box[0], box[3]-box[1]])
                    
                    # The Filter: If the head and tail don't span at least 60% of the box, 
                    # the fish is facing the camera. Skip it!
                    if head_to_tail_dist < (0.60 * box_diagonal):
                        print("Fish is facing the camera. Ignoring length measurement.")
                        facing_camera_count += 1
                        continue # Skips drawing the curve and moves to the next fish
            ###################################################################################################
            
                    # Get original tight bounding box
                    box = result.boxes.xyxy[fish_idx].cpu().numpy()
                    x1, y1, x2, y2 = box
                    
                    # FIX #4: Add 5% Padding to the Bounding Box
                    pad_w = (x2 - x1) * 0.05
                    pad_h = (y2 - y1) * 0.05
                    
                    # Apply padding while ensuring the box doesn't go off the edge of the photo
                    x1 = max(0, int(x1 - pad_w))
                    y1 = max(0, int(y1 - pad_h))
                    x2 = min(w_img, int(x2 + pad_w))
                    y2 = min(h_img, int(y2 + pad_h))
                    
                    # FIX #2: Clamp the keypoints safely inside the new, roomier padded box
                    for pt in kpts:
                        pt[0] = np.clip(pt[0], x1, x2)
                        pt[1] = np.clip(pt[1], y1, y2)
                    
                    # Draw the new Padded Bounding Box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Draw the Bézier curve and control points
                    img = draw_cubic_bezier(img, kpts, color=(0, 255, 255), thickness=3)
                    for pt in kpts:
                        cv2.circle(img, (int(pt[0]), int(pt[1])), radius=4, color=(0, 0, 255), thickness=-1)

        save_path = os.path.join(output_dir, f"result_{i}.jpg")
        cv2.imwrite(save_path, img)

    print(f"\n✅ Done! Check the '{output_dir}' folder for the padded boxes and clamped points.")
    print(f"\n  {facing_camera_count} fish were facing the camera and skipped.")


if __name__ == '__main__':
    main()