

from ultralytics import YOLO

def predict_multiple_val_sets():
    # 1. Load your best weights
    model = YOLO(r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\ultralytics\runs\pose\BezierFusion\eamrf_training_run17\weights\best.pt")

    # 2. List all your validation folders here
    val_folders = [
        r"C:\Users\Work Mode Big Dog\OneDrive - ECAM\Bureau\ERASMUS\PROJECT\CODE\DATASETS\DeepFish\First_batch_Train_Test\images\val",

    ]

    # 3. Loop through them and let Ultralytics do the work
    for i, folder in enumerate(val_folders):
        print(f"\n🚀 Processing Validation Set {i + 1}...")
        
        model.predict(
            source=folder, 
            save=True, 
            conf=0.35,          # Ignore background noise
            iou=0.45,           # Fix the double-box NMS issue
            agnostic_nms=True,  # Merge boxes across classes
            line_width=2,
            project=r"Inferences/run17_Inference_V2", # Main output folder
            name=f"Dataset_{i + 1}"               # Sub-folder for this specific set
        )

    print("\n✅ All validation sets processed perfectly!")

if __name__ == '__main__':
    predict_multiple_val_sets()