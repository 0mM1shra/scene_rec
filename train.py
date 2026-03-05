import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import RadarOccupancyDataset, DummyDataset
from model import RadarOccupancyNet
from losses import LovaszLoss
import torch.nn as nn
import os
import numpy as np

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    if args.dummy:
        print("Using Dummy Dataset")
        train_ds = DummyDataset()
        val_ds = DummyDataset()
    else:
        print(f"Loading NuScenes from {args.dataroot}...")
        train_ds = RadarOccupancyDataset(dataroot=args.dataroot, version=args.version, split='train')
        val_ds = RadarOccupancyDataset(dataroot=args.dataroot, version=args.version, split='val')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = RadarOccupancyNet(n_channels=1, n_classes=3).to(device)
    
    # Optimizer (SGD with Momentum as per paper)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    # Loss
    if args.loss == 'lovasz':
        criterion = LovaszLoss(ignore_index=None) 
    elif args.loss == 'ce':
        # Weights to handle class imbalance (Paper mentions weighting)
        # 0: Free (Frequent), 1: Occupied (Rare), 2: Unobserved (Frequent)
        weights = torch.tensor([0.1, 1.0, 0.1]).to(device) 
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=255)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")
    
    # Scheduler (Decay by 0.9 when mIoU plateaus for 2 epochs)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=2)
    
    # Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Step {i}, Loss: {loss.item():.4f}")
                
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation (mIoU)
        model.eval()
        from evaluate import fast_hist, per_class_iou
        hist = np.zeros((3, 3))
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                targets_np = targets.numpy()
                
                # Filter ignore index 255
                valid_mask = (targets_np != 255)
                hist += fast_hist(targets_np[valid_mask], preds[valid_mask], 3)
        
        ious = per_class_iou(hist)
        miou = np.nanmean(ious)
        print(f"Epoch {epoch+1} Validation mIoU: {miou:.4f} (Free: {ious[0]:.4f}, Occ: {ious[1]:.4f}, Unobs: {ious[2]:.4f})")
        
        # Step Scheduler
        scheduler.step(miou)

    # Save
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f'checkpoints/model_epoch_{args.epochs}.pth')
    print("Training Complete. Model saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--loss', type=str, default='lovasz', choices=['lovasz', 'ce'], help='Loss function')
    parser.add_argument('--dummy', action='store_true', help='Use dummy data')
    parser.add_argument('--dataroot', type=str, default=r'C:\Users\DELL\Documents\Exploratory\radar_occupancy_project\data\nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-mini')
    
    args = parser.parse_args()
    train(args)
