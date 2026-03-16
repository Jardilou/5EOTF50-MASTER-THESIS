from ultralytics import YOLO

def main():
    # 1. Load your custom architecture YAML
    # (Do NOT load a pre-trained .pt file here, because the pre-trained 
    # weights don't have the EAMRF module and it will crash)
    model = YOLO('yolov8_Bézier.yaml') 
    
    # 2. Train the model using the paper's specifications
    results = model.train(
        data='fish_bezier.yaml', # Your dataset config
        epochs=500,              # Paper used 500 epochs
        batch=16,                 # Paper used batch size 8
        imgsz=640,
        optimizer='AdamW',       # Paper used AdamW
        lr0=1e-3,                # Initial learning rate
        weight_decay=0.0005,     # Explicitly set weight decay to 5e-4 (L2 Regularization)
        cos_lr=True,             # Enables Cosine Annealing learning rate scheduler
        lrf=0.01,                # Final learning rate fraction (stops it from hitting absolute 0)
        
        # --- 1. GEOMETRIC AUGMENTATIONS ---
        scale=0.5,        # Random scaling (+/- 50%) to handle size variations
        translate=0.1,    # Random translation (+/- 10%) to handle position variations
        fliplr=0.5,       # 50% chance to flip left-right 
        
        # --- 2. PHOTOMETRIC AUGMENTATIONS ---
        hsv_h=0.015,      # Slight hue shifts (water color changes)
        hsv_s=0.7,        # Saturation adjustments
        hsv_v=0.4,        # Value/Brightness adjustments (simulates deep/dark water vs shallow)
        
        # =========================================================
        # CUSTOM LOSS WEIGHTS (The "Gains")
        # =========================================================
        box=7.5,     # Weight for Bounding Box CIoU loss (Default is 7.5)
        dfl=1.5,     # Weight for Bounding Box Edge/DFL loss (Default is 1.5)
        pose=12.0,   # Weight for your custom Bézier Geometry loss (Default is 12.0)
        cls=3.0,     # Weight for Classifying it as a "fish" (Default is 0.5)
        # ========================================================
        
        project='BezierFusion',
        name='eamrf_training_run',
        val=False                 # Disable validation during training
    )

if __name__ == '__main__':
    main()

# from ultralytics import YOLO
# import os

# def main():
#     # =========================================================
#     # PHASE 1: The AdamW Sprint (First 450 Epochs)
#     # =========================================================
#     print("Starting Phase 1: Rapid convergence with AdamW...")
    
#     # Load your custom architecture YAML (no pre-trained weights)
#     model_phase1 = YOLO('yolov8_Bézier.yaml') 
    
#     model_phase1.train(
#         data='fish_bezier.yaml', 
#         epochs=450,              # 90% of the total 500 epochs
#         batch=8,                 
#         imgsz=640,
#         optimizer='AdamW',       # Switched to AdamW for rapid, adaptive learning
#         lr0=1e-3,                # Initial learning rate
#         weight_decay=0.0005,     
#         cos_lr=True,             
#         lrf=0.01,                
        
#         # --- GEOMETRIC AUGMENTATIONS ---
#         scale=0.5, translate=0.1, fliplr=0.5, 
        
#         # --- PHOTOMETRIC AUGMENTATIONS ---
#         hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, 
        
#         # =========================================================
#         # CUSTOM LOSS WEIGHTS
#         # =========================================================
#         box=7.5,     # Weight for Bounding Box CIoU loss (Default is 7.5)
#         dfl=1.5,     # Weight for Bounding Box Edge/DFL loss (Default is 1.5)
#         pose=12.0,   # Weight for your custom Bézier Geometry loss (Default is 12.0)
#         cls=3.0,     # Weight for Classifying it as a "fish" (Default is 0.5)
#         # ========================================================
        
#         project=r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\ultralytics\runs\pose\BezierFusion",
#         name='Phase1_AdamW_run2',
#         val=False                # No validation = no best.pt, only last.pt generated
#     )

#     # =========================================================
#     # PHASE 2: The SGD Marathon (Final 50 Epochs)
#     # =========================================================
#     print("\n🐌 Starting Phase 2: Fine-tuning and generalization with SGD...")
    
#     # Grab the final weights from Phase 1 to act as our new starting point
#     phase1_weights = os.path.join(r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\ultralytics\runs\pose\BezierFusion\Phase1_AdamW_run2\weights\last.pt")
    
#     if not os.path.exists(phase1_weights):
#         print(f"Error: Could not find Phase 1 weights at {phase1_weights}")
#         return
        
#     model_phase2 = YOLO(phase1_weights)
    
#     model_phase2.train(
#         data='fish_bezier.yaml', 
#         epochs=50,               # The remaining 10%
#         batch=5,                 
#         imgsz=640,
#         optimizer='SGD',         # Switch to SGD to explore flatter, generalized minima
#         lr0=1e-4,                # CRITICAL: Drop learning rate 10x (from 1e-3 to 1e-4) to prevent shock
#         weight_decay=0.0005,     
#         cos_lr=True,             
#         lrf=0.01,
        
#         # Keep all augmentations and loss weights identical so the environment doesn't shift
#         scale=0.5, translate=0.1, fliplr=0.5,
#         hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
#         box=7.5, dfl=1.5, pose=12.0, cls=3.0,
        
#         project=r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\ultralytics\runs\pose\BezierFusion",
#         name='Phase2_SGD_run2',
#         val=False
#     )
    
#     print("\n Two-Phase Training Complete! Your final, fine-tuned weights are in Phase2_SGD.")

# if __name__ == '__main__':
#     main()