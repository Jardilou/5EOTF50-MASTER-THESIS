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

# # def main():
# #     # 1. Load your newly trained weights
# #     # (Since val=False, we use last.pt instead of best.pt)
# #     weight_path = r"ultralytics/runs/pose/BezierFusion/eamrf_training_run9/weights/last.pt"

# #     if not os.path.exists(weight_path):
# #         print("Training hasn't finished or weights not found!")
# #         return

# #     model = YOLO(weight_path)

# #     # 2. Point it to your Test/Val images folder
# #     test_images_dir = r"dataset2\dataset2\train-test\images\val"
# #     output_dir = r"Inference_results_THIRD_TRY_v2"
# #     os.makedirs(output_dir, exist_ok=True)

# #     # 3. Run Inference on all images in the folder
# #     print("Running inference...")
# #     results = model.predict(source=test_images_dir, conf=0.4) # Only trust predictions > 50%

# #     # 4. Process and Draw
# #     for i, result in enumerate(results):
# #         # Load the original image array
# #         img = result.orig_img.copy()

# #         # Check if the model actually detected any fish in this image
# #         if result.keypoints is not None and result.keypoints.xy is not None:
# #             # Loop through every fish detected in the image
# #             for fish_idx in range(len(result.keypoints.xy)):
# #                 # Extract the 4 control points (x, y) for this specific fish
# #                 kpts = result.keypoints.xy[fish_idx].cpu().numpy()

# #                 # Make sure we got exactly 4 points
# #                 if len(kpts) == 4:
# #                     # Extract the 4 control points (x, y)
# #                     # Extract points and box
# #                     kpts = result.keypoints.xy[fish_idx].cpu().numpy()
# #                     box = result.boxes.xyxy[fish_idx].cpu().numpy().astype(int)

# #                     # Clamp keypoints strictly inside the bounding box
# #                     for pt in kpts:
# #                         pt[0] = np.clip(pt[0], box[0], box[2]) # Clip X between x_min and x_max
# #                         pt[1] = np.clip(pt[1], box[1], box[3]) # Clip Y between y_min and y_max


# #                 ###################################################################################################
# #                 # --- OPTIONAL FILTER: Check if the fish is facing the camera (head and tail are close together) ---
# #                     if len(kpts) == 4:
# #                         head = kpts[0]
# #                         tail = kpts[3]

# #                         # Calculate the 2D distance between head and tail
# #                         head_to_tail_dist = np.linalg.norm(head - tail)

# #                         # Get the bounding box diagonal size for scale
# #                         box = result.boxes.xyxy[fish_idx].cpu().numpy()
# #                         box_diagonal = np.linalg.norm([box[2]-box[0], box[3]-box[1]])

# #                         # The Filter: If the head and tail don't span at least 60% of the box,
# #                         # the fish is facing the camera. Skip it!
# #                         if head_to_tail_dist < (0.60 * box_diagonal):
# #                             print("Fish is facing the camera. Ignoring length measurement.")
# #                             continue # Skips drawing the curve and moves to the next fish
# #                 ###################################################################################################
# #                     # Draw the bounding box
# #                     box = result.boxes.xyxy[fish_idx].cpu().numpy().astype(int)
# #                     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

# #                     # Draw the smooth Bézier curve (Yellow)
# #                     img = draw_cubic_bezier(img, kpts, color=(0, 255, 255), thickness=3)

# #                     # Draw the 4 individual control points (Red) so you can see what the AI targeted
# #                     for pt in kpts:
# #                         cv2.circle(img, (int(pt[0]), int(pt[1])), radius=4, color=(0, 0, 255), thickness=-1)

# #         # Save the drawn image to your results folder
# #         save_path = os.path.join(output_dir, f"result_{i}.jpg")
# #         cv2.imwrite(save_path, img)
# #         print(f"Saved {save_path}")

# #     print(f"\nAll done! Check the '{output_dir}' folder to see your AI's predictions.")

# # if __name__ == '__main__':
# #     main()


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

# def main():
#     weight_path = r"ultralytics/runs/pose/BezierFusion/eamrf_training_run12/weights/last.pt"
#     if not os.path.exists(weight_path):
#         print("Weights not found! Check your path.")
#         return

#     model = YOLO(weight_path)
#     test_images_dir = r"DATASETS\Bézierfusion Dataset 2 Reannotated\train-test\images\val"
#     output_dir = r"INFERENCE_RESULTS\Inference_results_SIXTH_TRY"
#     os.makedirs(output_dir, exist_ok=True)

#     print("Running inference...")

#     # FIX #3: Lowered confidence from 0.5 to 0.25 to catch "missing" stereo twins
#     results = model.predict(source=test_images_dir, conf=0.25)
#     facing_camera_count = 0
#     for i, result in enumerate(results):
#         img = result.orig_img.copy()
#         h_img, w_img = img.shape[:2] # Get image dimensions to prevent drawing outside the frame

#         if result.keypoints is not None and result.keypoints.xy is not None:
#             for fish_idx in range(len(result.keypoints.xy)):
#                 kpts = result.keypoints.xy[fish_idx].cpu().numpy()

#                 if len(kpts) == 4:
#                     ###################################################################################################
#             # --- OPTIONAL FILTER: Check if the fish is facing the camera (head and tail are close together) ---
#                     head = kpts[0]
#                     tail = kpts[3]

#                     # Calculate the 2D distance between head and tail
#                     head_to_tail_dist = np.linalg.norm(head - tail)

#                     # Get the bounding box diagonal size for scale
#                     box = result.boxes.xyxy[fish_idx].cpu().numpy()
#                     box_diagonal = np.linalg.norm([box[2]-box[0], box[3]-box[1]])

#                     # The Filter: If the head and tail don't span at least 60% of the box,
#                     # the fish is facing the camera. Skip it!
#                     if head_to_tail_dist < (0.60 * box_diagonal):
#                         print("Fish is facing the camera. Ignoring length measurement.")
#                         facing_camera_count += 1
#                         continue # Skips drawing the curve and moves to the next fish
#             ###################################################################################################

#                     # Get original tight bounding box
#                     box = result.boxes.xyxy[fish_idx].cpu().numpy()
#                     x1, y1, x2, y2 = box

#                     # FIX #4: Add 5% Padding to the Bounding Box
#                     pad_w = (x2 - x1) * 0.05
#                     pad_h = (y2 - y1) * 0.05

#                     # Apply padding while ensuring the box doesn't go off the edge of the photo
#                     x1 = max(0, int(x1 - pad_w))
#                     y1 = max(0, int(y1 - pad_h))
#                     x2 = min(w_img, int(x2 + pad_w))
#                     y2 = min(h_img, int(y2 + pad_h))

#                     # FIX #2: Clamp the keypoints safely inside the new, roomier padded box
#                     for pt in kpts:
#                         pt[0] = np.clip(pt[0], x1, x2)
#                         pt[1] = np.clip(pt[1], y1, y2)

#                     # Draw the new Padded Bounding Box
#                     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                     # NEW: Get and Draw Confidence Score
#                     # ==============================================================
#                     # 1. Extract the confidence score for this specific fish
#                     conf = result.boxes.conf[fish_idx].item()

#                     # 2. Format it to 2 decimal places (e.g., "0.85")
#                     conf_text = f"{conf:.2f}"

#                     # 3. Draw the text slightly below the bottom-left corner of the box (x1, y2 + 20)
#                     cv2.putText(
#                         img,
#                         conf_text,
#                         (x1, min(h_img - 5, y2 + 20)), # min() ensures text doesn't fall off the bottom of the image
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6,               # Font scale (size)
#                         (255, 0, 0),       # Color (Blue to match the box)
#                         2                  # Thickness
#                     )
#                     # ==============================================================
#                     # Draw the Bézier curve and control points
#                     img = draw_cubic_bezier(img, kpts, color=(0, 255, 255), thickness=3)
#                     for pt in kpts:
#                         cv2.circle(img, (int(pt[0]), int(pt[1])), radius=4, color=(0, 0, 255), thickness=-1)

#         save_path = os.path.join(output_dir, f"result_{i}.jpg")
#         cv2.imwrite(save_path, img)

#     print(f"\nDone! Check the '{output_dir}' folder for the padded boxes and clamped points.")
#     print(f"\n  {facing_camera_count} fish were facing the camera and skipped.")





# # def main():
# #     weight_path = r"ultralytics/runs/pose/BezierFusion/eamrf_training_run11/weights/last.pt"
# #     if not os.path.exists(weight_path):
# #         print("Weights not found! Check your path.")
# #         return

# #     model = YOLO(weight_path)
# #     test_images_dir = r"DATASETS\Bézierfusion Dataset 2 Reannotated\train-test\images\val"
# #     output_dir = r"INFERENCE_RESULTS\Inference_results_FIFTH_TRY"
# #     os.makedirs(output_dir, exist_ok=True)

# #     print("Running inference...")

# #     Set confidence to 0.3 so YOLO hands us every prediction above 30%
# #     results = model.predict(source=test_images_dir, conf=0.3)
# #     facing_camera_count = 0

# #     for i, result in enumerate(results):
# #         img = result.orig_img.copy()
# #         h_img, w_img = img.shape[:2]

# #         1. Iterate through the Bounding Boxes FIRST (not the keypoints)
# #         if result.boxes is not None:
# #             for fish_idx in range(len(result.boxes)):

# #                 Get the box and confidence score
# #                 box = result.boxes.xyxy[fish_idx].cpu().numpy()
# #                 conf = result.boxes.conf[fish_idx].item()
# #                 x1, y1, x2, y2 = box

# #                 Add 5% padding
# #                 pad_w = (x2 - x1) * 0.05
# #                 pad_h = (y2 - y1) * 0.05
# #                 x1 = max(0, int(x1 - pad_w))
# #                 y1 = max(0, int(y1 - pad_h))
# #                 x2 = min(w_img, int(x2 + pad_w))
# #                 y2 = min(h_img, int(y2 + pad_h))

# #                 ==========================================================
# #                 ALWAYS DRAW THE BOX AND SCORE (Even if curve is skipped)
# #                 ==========================================================
# #                 cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
# #                 conf_text = f"{conf:.2f}"
# #                 cv2.putText(
# #                     img,
# #                     conf_text,
# #                     (x1, min(h_img - 5, y2 + 20)),
# #                     cv2.FONT_HERSHEY_SIMPLEX,
# #                     0.6, (255, 0, 0), 2
# #                 )

# #                 2. Now, check if this specific fish has valid keypoints
# #                 if result.keypoints is not None and len(result.keypoints.xy) > fish_idx:
# #                     kpts = result.keypoints.xy[fish_idx].cpu().numpy()

# #                     if len(kpts) == 4:
# #                         --- OPTIONAL FILTER: Check if facing camera ---
# #                         head = kpts[0]
# #                         tail = kpts[3]
# #                         head_to_tail_dist = np.linalg.norm(head - tail)

# #                         Use the padded box diagonal for the straightness check
# #                         box_diagonal = np.linalg.norm([x2 - x1, y2 - y1])

# #                         if head_to_tail_dist < (0.60 * box_diagonal):
# #                             print(f"Image {i}, Fish {fish_idx}: Facing the camera. Ignoring curve.")
# #                             facing_camera_count += 1
# #                             continue # Skips drawing the curve, but box/score are already drawn!

# #                         Clamp the keypoints safely inside the padded box
# #                         for pt in kpts:
# #                             pt[0] = np.clip(pt[0], x1, x2)
# #                             pt[1] = np.clip(pt[1], y1, y2)

# #                         Draw the Bézier curve and control points
# #                         img = draw_cubic_bezier(img, kpts, color=(0, 255, 255), thickness=3)
# #                         for pt in kpts:
# #                             cv2.circle(img, (int(pt[0]), int(pt[1])), radius=4, color=(0, 0, 255), thickness=-1)

# #         save_path = os.path.join(output_dir, f"result_{i}.jpg")
# #         cv2.imwrite(save_path, img)

# #     print(f"\nDone! Check the '{output_dir}' folder.")
# #     print(f"  {facing_camera_count} fish were facing the camera (boxes drawn, curves skipped).")

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
    weight_path = r"ultralytics/runs/pose/BezierFusion/eamrf_training_run16/weights/best.pt"
    if not os.path.exists(weight_path):
        print("Weights not found! Check your path.")
        return

    model = YOLO(weight_path)
    
    # 1. Create a LIST of all the folders you want to run inference on
    test_image_dirs = [
        r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\DATASETS\DeepFish\First_batch_Train_Test\images\val",
        r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\DATASETS\BézierFusion Dataset 1\CORRECT IMAGES\train test split\images\val",
        r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\DATASETS\Bézierfusion Dataset 2 Reannotated\train-test\images\val"
    ]
    
    base_output_dir = r"INFERENCE_RESULTS\Inference_results_run17"

    # 2. Loop through each dataset folder one by one
    for dataset_idx, test_images_dir in enumerate(test_image_dirs):
        print(f"\n=======================================================")
        print(f"Running inference on Dataset {dataset_idx + 1}...")
        print(f"Path: {test_images_dir}")
        print(f"=======================================================")
        
        # Create a unique sub-folder for this dataset so files don't overwrite each other
        output_dir = os.path.join(base_output_dir, f"Dataset_{dataset_idx + 1}")
        os.makedirs(output_dir, exist_ok=True)

        # Run prediction on this specific folder
        results = model.predict(source=test_images_dir, conf=0.6)
        facing_camera_count = 0
        
        for i, result in enumerate(results):
            img = result.orig_img.copy()
            h_img, w_img = img.shape[:2] 

            if result.keypoints is not None and result.keypoints.xy is not None:
                for fish_idx in range(len(result.keypoints.xy)):
                    kpts = result.keypoints.xy[fish_idx].cpu().numpy()

                    if len(kpts) == 4:
                        # --- OPTIONAL FILTER: Check if the fish is facing the camera ---
                        head = kpts[0]
                        tail = kpts[3]
                        head_to_tail_dist = np.linalg.norm(head - tail)

                        box = result.boxes.xyxy[fish_idx].cpu().numpy()
                        box_diagonal = np.linalg.norm([box[2]-box[0], box[3]-box[1]])

                        if head_to_tail_dist < (0.60 * box_diagonal):
                            print(f"  -> Fish {fish_idx} in image {i} is facing the camera. Ignoring.")
                            facing_camera_count += 1
                            continue 

                        # Get original tight bounding box
                        box = result.boxes.xyxy[fish_idx].cpu().numpy()
                        x1, y1, x2, y2 = box

                        # Add 5% Padding to the Bounding Box
                        pad_w = (x2 - x1) * 0.05
                        pad_h = (y2 - y1) * 0.05

                        x1 = max(0, int(x1 - pad_w))
                        y1 = max(0, int(y1 - pad_h))
                        x2 = min(w_img, int(x2 + pad_w))
                        y2 = min(h_img, int(y2 + pad_h))

                        # Clamp the keypoints
                        for pt in kpts:
                            pt[0] = np.clip(pt[0], x1, x2)
                            pt[1] = np.clip(pt[1], y1, y2)

                        # Draw the new Padded Bounding Box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        
                        # Get and Draw Confidence Score
                        conf = result.boxes.conf[fish_idx].item()
                        conf_text = f"{conf:.2f}"
                        cv2.putText(
                            img,
                            conf_text,
                            (x1, min(h_img - 5, y2 + 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, 
                            (255, 0, 0), 
                            2 
                        )
                        
                        # Draw the Bézier curve and control points
                        img = draw_cubic_bezier(img, kpts, color=(0, 255, 255), thickness=3)
                        for pt in kpts:
                            cv2.circle(img, (int(pt[0]), int(pt[1])), radius=4, color=(0, 0, 255), thickness=-1)

            save_path = os.path.join(output_dir, f"result_{i}.jpg")
            cv2.imwrite(save_path, img)

        print(f"\nDone with Dataset {dataset_idx + 1}!")
        print(f"Results saved to: '{output_dir}'")
        print(f"{facing_camera_count} fish were facing the camera and skipped.\n")



if __name__ == '__main__':
    main()


