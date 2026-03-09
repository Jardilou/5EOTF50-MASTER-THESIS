# import xml.etree.ElementTree as ET
# import os
# import numpy as np

# def convert_cvat_xml_to_yolo(xml_path, output_dir_labels):
#     print("Reading XML file...")
#     tree = ET.parse(xml_path)
#     root = tree.getroot()

#     os.makedirs(output_dir_labels, exist_ok=True)
#     processed_fish = 0
#     processed_images = 0

#     # 1. Loop through every image in the XML
#     for image in root.findall('image'):
#         filename = image.get('name')
#         w_img = float(image.get('width'))
#         h_img = float(image.get('height'))

#         # 2. Extract manual bounding boxes
#         boxes = []
#         for box in image.findall('box'):
#             if box.get('label') == 'fish-box':
#                 xtl = float(box.get('xtl'))
#                 ytl = float(box.get('ytl'))
#                 xbr = float(box.get('xbr'))
#                 ybr = float(box.get('ybr'))
#                 # Convert to [x_min, y_min, width, height]
#                 boxes.append([xtl, ytl, xbr - xtl, ybr - ytl])

#         # 3. Extract skeletons and sort points properly
#         spines = []
#         for skel in image.findall('skeleton'):
#             if skel.get('label') == 'fish-curve':
#                 pt_dict = {}
                
#                 # Fetch each individual point from inside the skeleton
#                 for pt in skel.findall('points'):
#                     lbl = pt.get('label')
#                     coords = pt.get('points')
#                     if coords:
#                         x, y = map(float, coords.split(','))
#                         pt_dict[lbl] = [x, y]
                
#                 # Ensure we have all 4 required points, and put them in exact order
#                 if all(k in pt_dict for k in ['Head', 'Mid1', 'Mid2', 'Tail']):
#                     ordered_pts = [
#                         pt_dict['Head'],
#                         pt_dict['Mid1'],
#                         pt_dict['Mid2'],
#                         pt_dict['Tail']
#                     ]
#                     spines.append(np.array(ordered_pts))

#         yolo_lines = []
#         assigned_spines = set()

#         # 4. OVERLAP-SAFE MATCHING LOGIC (Box + Spine)
#         for bx, by, bw, bh in boxes:
#             box_center = np.array([bx + bw/2, by + bh/2])
            
#             best_spine = None
#             best_score = -float('inf')
#             best_spine_idx = -1
            
#             for i, pts_xy in enumerate(spines):
#                 if i in assigned_spines: continue

#                 # Check if the points fall inside this manual box
#                 points_inside = 0
#                 for px, py in pts_xy:
#                     if (bx <= px <= bx + bw) and (by <= py <= by + bh):
#                         points_inside += 1
                
#                 # If less than 2 points are inside, skip
#                 if points_inside < 2: 
#                     continue

#                 spine_center = np.mean(pts_xy, axis=0)
#                 dist = np.linalg.norm(box_center - spine_center)
#                 score = (points_inside * 1000) - dist
                
#                 if score > best_score:
#                     best_score = score
#                     best_spine = pts_xy
#                     best_spine_idx = i

#             # 5. FORMAT AND SAVE
#             if best_spine is not None:
#                 assigned_spines.add(best_spine_idx)
                
#                 # Normalize Box coordinates for YOLO [0, 1]
#                 cx_norm = box_center[0] / w_img
#                 cy_norm = box_center[1] / h_img
#                 nw = bw / w_img
#                 nh = bh / h_img
                
#                 # Convert the 4 ordered points to strings
#                 kpt_str = ""
#                 for px, py in best_spine:
#                     kpt_str += f"{px/w_img:.6f} {py/h_img:.6f} "
                
#                 # Format: class cx cy w h P1x P1y P2x P2y P3x P3y P4x P4y
#                 line = f"0 {cx_norm:.6f} {cy_norm:.6f} {nw:.6f} {nh:.6f} {kpt_str}"
#                 yolo_lines.append(line.strip())
#                 processed_fish += 1

#         # Write to txt file if we successfully paired any fish in this image
#         if yolo_lines:
#             txt_name = os.path.splitext(filename)[0] + ".txt"
#             with open(os.path.join(output_dir_labels, txt_name), 'w') as f:
#                 f.write("\n".join(yolo_lines))
#             processed_images += 1

#     print("--- Parsing & Pairing Complete ---")
#     print(f"Total Text Files Created: {processed_images}")
#     print(f"Total Fish Instances Successfully Paired: {processed_fish}")
    
    
import xml.etree.ElementTree as ET
import os
import numpy as np

def convert_cvat_xml_to_yolo_final(xml_path, output_dir_labels):
    print("Reading XML file...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    os.makedirs(output_dir_labels, exist_ok=True)
    processed_fish = 0
    processed_images = 0

    # 1. Loop through every image
    for image in root.findall('image'):
        filename = image.get('name')
        w_img = float(image.get('width'))
        h_img = float(image.get('height'))

        # 2. Extract your manual bounding boxes
        boxes = []
        for box in image.findall('box'):
            if box.get('label') == 'fish-box':
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                boxes.append([xtl, ytl, xbr - xtl, ybr - ytl])

        # 3. Extract your skeletons and enforce exact point order
        spines = []
        for skel in image.findall('skeleton'):
            if skel.get('label') == 'fish-curve':
                pt_dict = {}
                
                # Dig into the skeleton to find the named points
                for pt in skel.findall('points'):
                    lbl = pt.get('label')
                    coords = pt.get('points')
                    if coords:
                        x, y = map(float, coords.split(','))
                        pt_dict[lbl] = [x, y]
                
                # Ensure all 4 points exist, and lock them in Head -> Tail order
                if all(k in pt_dict for k in ['Head', 'Mid1', 'Mid2', 'Tail']):
                    ordered_pts = [
                        pt_dict['Head'],
                        pt_dict['Mid1'],
                        pt_dict['Mid2'],
                        pt_dict['Tail']
                    ]
                    spines.append(np.array(ordered_pts))

        yolo_lines = []
        assigned_spines = set()

        # 4. OVERLAP-SAFE MATCHING
        for bx, by, bw, bh in boxes:
            box_center = np.array([bx + bw/2, by + bh/2])
            
            best_spine = None
            best_score = -float('inf')
            best_spine_idx = -1
            
            for i, pts_xy in enumerate(spines):
                if i in assigned_spines: continue

                # Count how many points fall strictly inside this bounding box
                points_inside = sum(1 for px, py in pts_xy if bx <= px <= bx + bw and by <= py <= by + bh)
                
                if points_inside < 2: 
                    continue # Ignore if spine doesn't belong to this box

                spine_center = np.mean(pts_xy, axis=0)
                dist = np.linalg.norm(box_center - spine_center)
                score = (points_inside * 1000) - dist # Maximize containment, minimize distance
                
                if score > best_score:
                    best_score = score
                    best_spine = pts_xy
                    best_spine_idx = i

            # 5. FORMAT FOR YOLO (The Visibility Fix)
            if best_spine is not None:
                assigned_spines.add(best_spine_idx)
                
                cx_norm = box_center[0] / w_img
                cy_norm = box_center[1] / h_img
                nw = bw / w_img
                nh = bh / h_img
                
                kpt_str = ""
                for px, py in best_spine[:4]:
                    # CRITICAL FIX: We add '2' after every x,y pair so YOLO parses correctly
                    kpt_str += f"{px/w_img:.6f} {py/h_img:.6f} 2 "
                
                line = f"0 {cx_norm:.6f} {cy_norm:.6f} {nw:.6f} {nh:.6f} {kpt_str}"
                yolo_lines.append(line.strip())
                processed_fish += 1

        # Save to text file
        if yolo_lines:
            txt_name = os.path.splitext(filename)[0] + ".txt"
            with open(os.path.join(output_dir_labels, txt_name), 'w') as f:
                f.write("\n".join(yolo_lines))
            processed_images += 1

    print("--- Parsing & Pairing Complete ---")
    print(f"Total Text Files Created: {processed_images}")
    print(f"Total Fish Instances Successfully Paired: {processed_fish}")



# --- USAGE ---
xml_file = "annotations.xml" # Update to your actual path
output_folder = "yolo_labels_overlap_safe" 

convert_cvat_xml_to_yolo_final(xml_file, output_folder)
