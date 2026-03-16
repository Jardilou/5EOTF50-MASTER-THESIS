import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison(run_folders, metric='train/pose_loss'):
    plt.figure(figsize=(10, 6))
    
    for label, folder in run_folders.items():
        csv_path = os.path.join(folder, "results.csv")
        
        if not os.path.exists(csv_path):
            print(f"Could not find {csv_path}. Skipping...")
            continue
            
        # Read the CSV
        df = pd.read_csv(csv_path)
        
        # YOLO saves column names with leading spaces, so we strip them
        df.columns = df.columns.str.strip()
        
        # Plot Epoch vs the chosen metric
        plt.plot(df['epoch'], df[metric], label=label, linewidth=2)

    plt.title(f"Model Comparison: {metric}", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss (Lower is better)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the graph
    save_path = "model_comparison_graph.png"
    plt.savefig(save_path, dpi=300)
    print(f"Graph saved as {save_path}")
    plt.show()

# --- USAGE ---
# Provide a dictionary of the runs you want to compare
# Format: {"Name you want on the graph": "Path to the run folder"}
runs_to_compare = {
    "Baseline (No EAMRF)": r"C:\Users\Work Mode Big Dog\...\runs\pose\baseline_run",
    "Bézierfusion (With EAMRF)": r"C:\Users\Work Mode Big Dog\...\runs\pose\eamrf_run"
}

# You can plot 'train/box_loss', 'train/pose_loss', or 'train/cls_loss'
plot_comparison(runs_to_compare, metric='train/pose_loss')