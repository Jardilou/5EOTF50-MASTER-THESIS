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
        batch=8,                 # Paper used batch size 8
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
        cls=0.5,     # Weight for Classifying it as a "fish" (Default is 0.5)
        # ========================================================
        
        project='BezierFusion',
        name='eamrf_training_run',
        val=False                 # Disable validation during training
    )

if __name__ == '__main__':
    main()