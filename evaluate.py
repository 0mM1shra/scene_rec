import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from dataset import RadarOccupancyDataset, DummyDataset
from model import RadarOccupancyNet
from baselines import OccupancyGridMap
from torch.utils.data import DataLoader

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def per_class_iou(hist):
    intersection = np.diag(hist)
    union = hist.sum(1) + hist.sum(0) - np.diag(hist)
    iou = intersection / (union + 1e-10)
    return iou

def save_prediction(img_gt, img_in, img_ism, img_pred, method, idx, save_dir='results'):
    """
    Save images for qualitative comparison as in Figure 1.
    (a) Ground truth. (b) Aggregated radar input. (c) Classic ISM result (d) Occupancy net output
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    axs[0].imshow(img_gt)
    axs[0].set_title("(a) Ground truth")
    axs[0].axis('off')
    
    axs[1].imshow(img_in)
    axs[1].set_title("(b) Aggregated radar input")
    axs[1].axis('off')
    
    axs[2].imshow(img_ism)
    axs[2].set_title("(c) Classic ISM result")
    axs[2].axis('off')
    
    axs[3].imshow(img_pred)
    axs[3].set_title(f"(d) Occupancy net ({method})")
    axs[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fig1_{idx:04d}.png"))
    plt.close()

def decode_seg(mask):
    """ Helper to map class indices to colors """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # 0: Free -> Black
    rgb[mask == 0] = [0, 0, 0]
    # 1: Occupied -> White
    rgb[mask == 1] = [255, 255, 255]
    # 2: Unobserved -> Light Gray
    rgb[mask == 2] = [170, 170, 170] # Light Gray
    # 255: Ignore -> Dark Gray
    rgb[mask == 255] = [50, 50, 50]
    return rgb

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {device} using method: {args.method}")
    
    # Dataset
    if args.dummy:
        val_ds = DummyDataset()
    else:
        val_ds = RadarOccupancyDataset(dataroot=args.dataroot, version=args.version, split='val')
    
    # Important: Sequential loader for Baselines (cannot shuffle)
    batch_size = 1 if args.method in ['delta', 'gaussian'] else args.batch_size
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Model Setup
    model = None
    if args.method == 'model':
        if not args.checkpoint:
            print("Warning: No checkpoint provided for model evaluation. Using random weights.")
        model = RadarOccupancyNet(n_channels=1, n_classes=3).to(device)
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        
    # Baseline Setup
    grid_map = None
    current_scene_token = None
    
    # Metrics
    n_classes = 3 # 0: Free, 1: Occupied, 2: Unobserved
    hist = np.zeros((n_classes, n_classes))
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs_dev = inputs.to(device)
            targets_np = targets.cpu().numpy() # (B, H, W)
            
            # Prediction
            preds = None
            
            if args.method == 'model':
                outputs = model(inputs_dev)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
            elif args.method == 'raytrace':
                 # Sec 4.2: "instead of feeding it to the network we perform ray tracing in a similar manner to what is described in section 4.1"
                 # input is (B, 1, H, W)
                 input_grid = inputs.cpu().numpy().squeeze(1) # (B, H, W)
                 preds = np.full_like(input_grid, 2, dtype=int) # Default Unobserved (2)
                 
                 for b in range(input_grid.shape[0]):
                     h, w = input_grid.shape[1], input_grid.shape[2]
                     origin_x, origin_y = 0, int(w / 2)
                     occ_xs, occ_ys = np.where(input_grid[b] > 0.5)
                     
                     from skimage.draw import line
                     occ_mask = np.zeros((h, w), dtype=bool)
                     for ox, oy in zip(occ_xs, occ_ys):
                         rr, cc = line(origin_x, origin_y, ox, oy)
                         v = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                         preds[b, rr[v], cc[v]] = 0 # Free
                         occ_mask[ox, oy] = True
                     preds[b, occ_mask] = 1 # Occupied
            
            elif args.method in ['delta', 'gaussian']:
                sample = val_ds.samples[i]
                scene_token = sample['scene_token']
                
                if scene_token != current_scene_token:
                    grid_map = OccupancyGridMap(val_ds.config)
                    current_scene_token = scene_token
                
                occ_idxs = torch.nonzero(inputs[0, 0] > 0.5, as_tuple=False).cpu().numpy()
                pts_x = occ_idxs[:, 0] * val_ds.config['grid_res']
                pts_y = occ_idxs[:, 1] * val_ds.config['grid_res'] - (val_ds.config['grid_w'] * val_ds.config['grid_res'] / 2)
                radar_points = np.vstack((pts_x, pts_y))
                
                grid_map.update(radar_points, method=args.method)
                preds = grid_map.get_map()
                preds = preds[np.newaxis, :, :] # (1, H, W)
            
            # Update Histogram
            # Flatten but filter out 255 (Ignore) in targets
            t_flat = targets_np.flatten()
            p_flat = preds.flatten()
            
            # Fix: Lovasz loss handles ignore via masking in training, but in Metric we usually filter it too?
            # Or is 255 mapped to 'Unobserved'?
            # The 'fast_hist' takes 'n_classes'. 255 will crash it.
            # Filter valid
            valid_mask = (t_flat != 255)
            hist += fast_hist(t_flat[valid_mask], p_flat[valid_mask], n_classes)
            
            # VISUALIZATION LOGIC
            if args.visualize and i < 20:
                # 1. Ground Truth
                img_gt = decode_seg(targets_np[0])
                
                # 2. Radar Input (Aggregated)
                in_np = inputs[0,0].cpu().numpy()
                img_in = np.zeros((in_np.shape[0], in_np.shape[1], 3), dtype=np.uint8)
                img_in[in_np > 0] = [255, 255, 255]
                
                # 3. Classic ISM (Delta)
                # If we are already running 'delta' method, we have it in preds.
                # If we are running 'model', we need to compute it once for visualization.
                if args.method == 'delta':
                    img_ism = decode_seg(preds[0])
                else:
                    # Temporary ISM update for visualization
                    # We need to know the scene_token to maintain state, or just show per-frame update?
                    # Figure 1(c) usually shows the result of the OGM filter. 
                    # Let's check scene continuity.
                    sample = val_ds.samples[i]
                    scene_token = sample['scene_token']
                    
                    if 'viz_grid_map' not in locals() or scene_token != viz_current_scene:
                        viz_grid_map = OccupancyGridMap(val_ds.config)
                        viz_current_scene = scene_token
                    
                    occ_idxs = torch.nonzero(inputs[0, 0] > 0.5, as_tuple=False).cpu().numpy()
                    pts_x = occ_idxs[:, 0] * val_ds.config['grid_res']
                    pts_y = occ_idxs[:, 1] * val_ds.config['grid_res'] - (val_ds.config['grid_w'] * val_ds.config['grid_res'] / 2)
                    radar_points = np.vstack((pts_x, pts_y))
                    
                    viz_grid_map.update(radar_points, method='delta')
                    ism_preds = viz_grid_map.get_map()
                    img_ism = decode_seg(ism_preds)
                
                # 4. Occupancy net output
                img_pred = decode_seg(preds[0])
                
                save_prediction(img_gt, img_in, img_ism, img_pred, args.method, i)

            if i % 10 == 0:
                print(f"Propagated {i}/{len(val_loader)} steps...", end='\r')
            
    # Compute IoU
    ious = per_class_iou(hist)
    miou = np.nanmean(ious)
    
    print("\nEvaluation Results:")
    print(f"{'Method':<15}: {args.method}")
    print("-" * 40)
    print(f"IoU Free       : {ious[0]:.4f}")
    print(f"IoU Occupied   : {ious[1]:.4f}")
    print(f"IoU Unobserved : {ious[2]:.4f}")
    print("-" * 40)
    print(f"mIoU Total     : {miou:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--dummy', action='store_true')
    parser.add_argument('--dataroot', type=str, default=r'C:\Users\DELL\Documents\Exploratory\radar_occupancy_project\data\nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--baseline', action='store_true', help='Legacy flag (mapped to raytrace)')
    parser.add_argument('--method', type=str, default='model', choices=['model', 'raytrace', 'delta', 'gaussian'], help='Evaluation method')
    parser.add_argument('--visualize', action='store_true', help='Save visualization images')
    
    args = parser.parse_args()
    if args.baseline: args.method = 'raytrace'
    
    evaluate(args)
