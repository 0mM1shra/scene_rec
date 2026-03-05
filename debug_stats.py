import torch
import numpy as np
from dataset import RadarOccupancyDataset
import os
import sys

def check_dataset_stats():
    dataroot = r"C:\Users\DELL\Documents\Exploratory\radar_occupancy_project\data\nuscenes"
    version = "v1.0-mini"
    
    if not os.path.exists(dataroot):
        print(f"Error: path {dataroot} does not exist.")
        return

    print("Initializing dataset...")
    try:
        ds = RadarOccupancyDataset(dataroot=dataroot, version=version, split='train')
    except Exception as e:
        print(f"Failed to init dataset: {e}")
        return

    print(f"Dataset length: {len(ds)}")
    
    # Check first few samples
    for i in range(min(3, len(ds))):
        print(f"\n--- Sample {i} ---")
        try:
            input_grid, label_grid = ds[i]
            
            # Input Stats
            input_np = input_grid.numpy()
            print(f"Input Grid Shape: {input_np.shape}")
            print(f"Input Grid Range: [{input_np.min()}, {input_np.max()}]")
            print(f"Input Non-zero count: {np.count_nonzero(input_np)}")
            
            # Label Stats
            label_np = label_grid.numpy()
            # 0: Free, 1: Occupied, 2: Unobserved, 255: Ignore
            print(f"Label Grid Shape: {label_np.shape}")
            unique, counts = np.unique(label_np, return_counts=True)
            print(f"Label Class Counts: {dict(zip(unique, counts))}")
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    check_dataset_stats()
